#!/usr/bin/env python3
"""Local bridge that lets Kiro IDE use Ollama models next to its cloud models.

Kiro reads two undocumented settings that replace its backend hosts:
    codewhisperer.config.krsEndpoints -> runtime.<region>.kiro.dev    (chat)
    codewhisperer.config.cpsEndpoints -> management.<region>.kiro.dev (model list)
Point them at this bridge. It forwards everything to the real Kiro backend,
except:
  * ListAvailableModels (POST / + X-Amz-Target, or GET /ListAvailableModels)
        -> real list + local Ollama models appended
  * POST /v1/responses           (CreateResponse transport, SSE)
  * POST /generateAssistantResponse (legacy transport, AWS event stream)
        -> for a "local-..." model id, translated to Ollama /api/chat and
           streamed back in the format Kiro parses for that transport

Stdlib only. Binds to 127.0.0.1. Never logs the Authorization header.

    python3 kiro_ollama_bridge.py                     # runtime :8787, management :8788
    python3 kiro_ollama_bridge.py --num-ctx 131072 --think on --show-thinking
"""
import argparse
import http.client
import json
import re
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCAL_PREFIX = "local-"
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
              "accept-encoding", "date", "server"}
CHARS_PER_TOKEN = 3.5   # rough estimate used only for the context-window guard
THINK_CLOSE = "</think>"
HOLD_CHARS = 3000       # how much leading text to hold back while checking for THINK_CLOSE

CONFIG = {}            # filled in main()
LOCAL_MODELS = {}      # kiro model id -> ollama model name
_models_lock = threading.Lock()


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# --------------------------------------------------------------------------
# Ollama helpers
# --------------------------------------------------------------------------

def refresh_local_models() -> list:
    """Return Kiro-style model entries for chat-capable Ollama models."""
    try:
        with urllib.request.urlopen(CONFIG["ollama"] + "/api/tags", timeout=5) as r:
            tags = json.load(r).get("models", [])
    except Exception as e:  # Ollama down: just show no local models
        log(f"ollama unreachable: {e}")
        return []
    entries, mapping = [], {}
    for m in tags:
        caps = m.get("capabilities") or []
        if "completion" not in caps:
            continue
        name = m["name"]
        if CONFIG["skip_cloud"] and name.endswith(":cloud"):
            continue
        kiro_id = LOCAL_PREFIX + re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-")
        mapping[kiro_id] = name
        entries.append({
            "modelId": kiro_id,
            "modelName": f"Local {name}",
            "description": f"Ollama model {name} via kiro_ollama_bridge"
                           + ("" if "tools" in caps else " (no tool support)"),
            "modelProvider": None,
            "rateMultiplier": 0.0,
            "rateUnit": "Credit",
            "tokenLimits": {"maxInputTokens": CONFIG["num_ctx"], "maxOutputTokens": 32000},
            "supportedInputTypes": ["TEXT"],
            "additionalModelRequestFieldsSchema": None,
            "availableOrigins": None, "consentDescription": None, "consentStatus": None,
            "disclaimers": None, "promptCaching": None, "refusalFallbackModels": None,
            "status": None, "termsUrl": None,
        })
    with _models_lock:
        LOCAL_MODELS.clear()
        LOCAL_MODELS.update(mapping)
    return entries


def resolve_local(model_id: str):
    with _models_lock:
        name = LOCAL_MODELS.get(model_id)
    if not name:
        refresh_local_models()
        with _models_lock:
            name = LOCAL_MODELS.get(model_id)
    return name


def ollama_chat_stream(model: str, messages: list, tools: list, think):
    """Yield ("thinking", str) | ("text", str) | ("tool", name, args_json) | ("done", stats)."""
    req = {"model": model, "messages": messages, "stream": True,
           "options": {"num_ctx": CONFIG["num_ctx"]}}
    if tools:
        req["tools"] = tools
    if think is not None:
        req["think"] = think
    r = urllib.request.urlopen(urllib.request.Request(
        CONFIG["ollama"] + "/api/chat", data=json.dumps(req).encode(),
        headers={"Content-Type": "application/json"}), timeout=900)
    # Some models (e.g. qwen with think=false) leak their reasoning into the
    # content followed by a stray "</think>". Hold back the start of the text
    # until we know whether that happens, then drop everything up to the tag.
    held, settled = "", False
    for line in r:
        if not line.strip():
            continue
        chunk = json.loads(line)
        if chunk.get("error"):
            raise RuntimeError(chunk["error"])
        msg = chunk.get("message") or {}
        if msg.get("thinking"):
            yield ("thinking", msg["thinking"])
        if msg.get("content"):
            if settled:
                yield ("text", msg["content"])
            else:
                held += msg["content"]
                if THINK_CLOSE in held:
                    leaked, held = held.split(THINK_CLOSE, 1)
                    yield ("thinking", leaked)
                    held, settled = held.lstrip(), True
                    if held:
                        yield ("text", held)
                elif len(held) > HOLD_CHARS:
                    settled = True
                    yield ("text", held)
        if msg.get("tool_calls") and held and not settled:
            settled = True
            yield ("text", held)
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments")
            yield ("tool", fn.get("name"), args if isinstance(args, str) else json.dumps(args or {}))
        if chunk.get("done"):
            if held and not settled:
                yield ("text", held)
            yield ("done", chunk)


def context_overflow(messages: list, tools: list):
    """Return an explanation if the prompt clearly can't fit in num_ctx."""
    est = int(len(json.dumps(messages)) + len(json.dumps(tools))) / CHARS_PER_TOKEN
    if est > CONFIG["num_ctx"] * 0.95:
        return (f"[kiro_ollama_bridge] This prompt is about {int(est):,} tokens, but the "
                f"local context window is {CONFIG['num_ctx']:,}. Ollama would silently drop "
                f"the start of the conversation. Start a new session, or restart the bridge "
                f"with a larger --num-ctx if the model and your RAM allow it.")
    return None


def _parse_args(value):
    if isinstance(value, str):
        try:
            return json.loads(value) if value.strip() else {}
        except json.JSONDecodeError:
            return {"_raw": value}
    return value or {}


def _flush_open(messages: list) -> list:
    for m in messages:
        m.pop("_open", None)
    return messages


# --------------------------------------------------------------------------
# CreateResponse (/v1/responses) -> Ollama /api/chat
# --------------------------------------------------------------------------

def responses_to_ollama(req: dict):
    messages, names, dropped = [], {}, 0
    if req.get("instructions"):
        messages.append({"role": "system", "content": req["instructions"]})
    for item in req.get("input") or []:
        kind = item.get("type")
        if kind == "reasoning":          # encrypted cloud reasoning: unusable locally
            continue
        if kind == "function_call":
            names[item.get("call_id")] = item.get("name")
            call = {"function": {"name": item.get("name"),
                                 "arguments": _parse_args(item.get("arguments"))}}
            if messages and messages[-1]["role"] == "assistant" and messages[-1].get("_open"):
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": [call],
                                 "_open": True})
            continue
        if kind == "function_call_output":
            out = item.get("output")
            messages.append({"role": "tool", "tool_name": names.get(item.get("call_id"), "tool"),
                             "content": out if isinstance(out, str) else json.dumps(out)})
            continue
        role = item.get("role")
        if role not in ("user", "assistant", "system", "developer"):
            continue
        content = item.get("content")
        if isinstance(content, list):
            texts = []
            for part in content:
                if part.get("type") in ("input_text", "output_text", "text"):
                    texts.append(part.get("text", ""))
                else:
                    dropped += 1        # images / files: local entries are TEXT only
            content = "\n".join(texts)
        msg = {"role": "system" if role == "developer" else role, "content": content or ""}
        if role == "assistant":
            msg["_open"] = True
        messages.append(msg)
    tools = [{"type": "function", "function": {
        "name": t["name"], "description": t.get("description", ""),
        "parameters": t.get("parameters") or {"type": "object"}}}
        for t in req.get("tools") or [] if t.get("type") == "function" and t.get("name")]
    effort = (req.get("reasoning") or {}).get("effort")
    think = False if effort == "none" else CONFIG["think"]
    return _flush_open(messages), tools, think, dropped


# --------------------------------------------------------------------------
# GenerateAssistantResponse -> Ollama /api/chat
# --------------------------------------------------------------------------

def _tool_result_text(result: dict) -> str:
    parts = []
    for c in result.get("content") or []:
        if "text" in c:
            parts.append(c["text"])
        elif "json" in c:
            parts.append(json.dumps(c["json"]))
    text = "\n".join(parts)
    return text if result.get("status", "success") == "success" else f"[tool error]\n{text}"


def gar_to_ollama(state: dict):
    messages, names = [], {}

    def add_user(msg: dict):
        ctx = msg.get("userInputMessageContext") or {}
        for r in ctx.get("toolResults") or []:
            messages.append({"role": "tool", "tool_name": names.get(r.get("toolUseId"), "tool"),
                             "content": _tool_result_text(r)})
        if msg.get("content"):
            messages.append({"role": "user", "content": msg["content"]})

    for entry in state.get("history") or []:
        if "userInputMessage" in entry:
            add_user(entry["userInputMessage"])
        elif "assistantResponseMessage" in entry:
            a = entry["assistantResponseMessage"]
            m = {"role": "assistant", "content": a.get("content") or ""}
            calls = []
            for tu in a.get("toolUses") or []:
                names[tu.get("toolUseId")] = tu.get("name")
                calls.append({"function": {"name": tu.get("name"),
                                           "arguments": _parse_args(tu.get("input"))}})
            if calls:
                m["tool_calls"] = calls
            messages.append(m)
    current = state["currentMessage"]["userInputMessage"]
    add_user(current)
    ctx = current.get("userInputMessageContext") or {}
    tools = [{"type": "function", "function": {
        "name": t["toolSpecification"]["name"],
        "description": t["toolSpecification"].get("description", ""),
        "parameters": (t["toolSpecification"].get("inputSchema") or {}).get("json")
        or {"type": "object"}}}
        for t in ctx.get("tools") or [] if t.get("toolSpecification")]
    return messages, tools


def encode_event(event_type: str, payload: dict) -> bytes:
    """One AWS event-stream frame (application/vnd.amazon.eventstream)."""
    headers = b""
    for name, value in ((":event-type", event_type), (":content-type", "application/json"),
                        (":message-type", "event")):
        n, v = name.encode(), value.encode()
        headers += bytes([len(n)]) + n + bytes([7]) + struct.pack(">H", len(v)) + v
    body = json.dumps(payload).encode()
    prelude = struct.pack(">II", 12 + len(headers) + len(body) + 4, len(headers))
    msg = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + body
    return msg + struct.pack(">I", zlib.crc32(msg))


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------

class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = ""   # set per server
    label = ""      # "runtime" / "management"

    def log_message(self, fmt, *args):
        pass

    # ---- plumbing -------------------------------------------------------
    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _op(self) -> str:
        target = self.headers.get("X-Amz-Target") or ""
        path = self.path.split("?")[0]
        return f"{path} [{target.split('.')[-1]}]" if target else path

    def _start_chunked(self, status: int, headers: dict):
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, data: bytes):
        if data:
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

    def _end_chunked(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _send_full(self, status: int, headers: dict, body: bytes):
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _upstream_request(self, body: bytes):
        u = urllib.parse.urlsplit(self.upstream)
        conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=330)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        headers["Host"] = u.hostname
        if body:
            headers["Content-Length"] = str(len(body))
        conn.request(self.command, self.path, body=body or None, headers=headers)
        return conn, conn.getresponse()

    def _forward(self, body: bytes):
        conn, resp = self._upstream_request(body)
        try:
            out = {k: v for k, v in resp.getheaders() if k.lower() not in HOP_BY_HOP}
            self._start_chunked(resp.status, out)
            while True:
                data = resp.read1(65536)
                if not data:
                    break
                self._chunk(data)
            self._end_chunked()
        finally:
            conn.close()
        log(f"[{self.label}] {self.command} {self._op()} -> upstream {resp.status}")

    # ---- routing --------------------------------------------------------
    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch()

    def do_PUT(self):
        self._dispatch()

    def do_DELETE(self):
        self._dispatch()

    def _dispatch(self):
        try:
            body = self._read_body()
            path = self.path.split("?")[0]
            target = self.headers.get("X-Amz-Target") or ""
            if target.endswith(".ListAvailableModels") or \
                    path in ("/ListAvailableModels", "/List-Available-Models"):
                return self._list_models(body)
            if self.command == "POST" and body and path in ("/v1/responses",
                                                            "/generateAssistantResponse"):
                req = json.loads(body)
                if path == "/v1/responses":
                    model_id = req.get("model")
                else:
                    model_id = ((req.get("conversationState") or {}).get("currentMessage", {})
                                .get("userInputMessage", {}).get("modelId"))
                if model_id and model_id.startswith(LOCAL_PREFIX):
                    if path == "/v1/responses":
                        return self._local_responses(req, model_id, len(body))
                    return self._local_gar(req["conversationState"], model_id, len(body))
            return self._forward(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"[{self.label}] ERROR {self.command} {self._op()}: {e!r}")
            try:
                self._send_full(502, {"Content-Type": "application/json"},
                                json.dumps({"message": f"kiro_ollama_bridge: {e}"}).encode())
            except Exception:
                pass

    def _list_models(self, body: bytes):
        conn, resp = self._upstream_request(body)
        try:
            raw = resp.read()
        finally:
            conn.close()
        headers = {k: v for k, v in resp.getheaders() if k.lower() not in HOP_BY_HOP}
        note = ""
        if resp.status == 200:
            data = json.loads(raw)
            if not data.get("nextToken"):   # append on the last page only
                local = refresh_local_models()
                data["models"] = (data.get("models") or []) + local
                raw = json.dumps(data).encode()
                note = f", {len(data['models'])} models ({len(local)} local)"
        else:
            note = f", body: {raw[:200]!r}"
        self._send_full(resp.status, headers, raw)
        log(f"[{self.label}] {self.command} {self._op()} -> upstream {resp.status}{note}")

    # ---- local chat: CreateResponse (SSE) -------------------------------
    def _sse(self, event: str, data: dict):
        payload = json.dumps({"type": event, **data})
        self._chunk(f"event: {event}\ndata: {payload}\n\n".encode())

    def _local_responses(self, req: dict, model_id: str, body_len: int):
        ollama_model = resolve_local(model_id)
        messages, tools, think, dropped = responses_to_ollama(req)
        rid = "resp_" + uuid.uuid4().hex
        self._start_chunked(200, {"Content-Type": "text/event-stream",
                                  "x-amzn-requestid": str(uuid.uuid4())})
        self._sse("response.created", {"response": {"id": rid, "status": "in_progress",
                                                    "model": model_id}})
        log(f"[{self.label}] local /v1/responses {model_id} -> {ollama_model}: "
            f"{len(messages)} msgs, {len(tools)} tools, body {body_len // 1024} KB"
            + (f", {dropped} non-text parts dropped" if dropped else ""))
        problem = (f"[kiro_ollama_bridge] Unknown local model {model_id}."
                   if not ollama_model else context_overflow(messages, tools))
        out_index, stats, started, n_text, n_tools = 0, {}, time.time(), 0, 0
        text_open = False
        if problem:
            log(f"[{self.label}] refused: {problem}")
            self._sse("response.output_text.delta", {"output_index": 0, "content_index": 0,
                                                     "delta": problem})
        else:
            try:
                for ev in ollama_chat_stream(ollama_model, messages, tools, think):
                    if ev[0] == "thinking" and CONFIG["show_thinking"]:
                        self._sse("response.reasoning_text.delta",
                                  {"output_index": out_index, "delta": ev[1]})
                    elif ev[0] == "text":
                        text_open = True
                        n_text += len(ev[1])
                        self._sse("response.output_text.delta",
                                  {"output_index": out_index, "content_index": 0, "delta": ev[1]})
                    elif ev[0] == "tool":
                        if text_open:
                            out_index, text_open = out_index + 1, False
                        call_id = "call_" + uuid.uuid4().hex[:24]
                        item = {"type": "function_call", "id": "fc_" + uuid.uuid4().hex[:24],
                                "call_id": call_id, "name": ev[1], "arguments": ""}
                        self._sse("response.output_item.added",
                                  {"output_index": out_index, "item": item})
                        self._sse("response.output_item.done",
                                  {"output_index": out_index,
                                   "item": {**item, "arguments": ev[2], "status": "completed"}})
                        out_index += 1
                        n_tools += 1
                    elif ev[0] == "done":
                        stats = ev[1]
            except Exception as e:
                log(f"[{self.label}] ollama error: {e!r}")
                self._sse("response.output_text.delta",
                          {"output_index": out_index, "content_index": 0,
                           "delta": f"\n[kiro_ollama_bridge] Ollama error: {e}"})
        p_tok, o_tok = stats.get("prompt_eval_count") or 0, stats.get("eval_count") or 0
        usage = {"input_tokens": p_tok, "output_tokens": o_tok, "total_tokens": p_tok + o_tok}
        if p_tok:
            usage["kiro:context_usage_percentage"] = round(100 * p_tok / CONFIG["num_ctx"], 2)
        self._sse("response.completed", {"response": {"id": rid, "status": "completed",
                                                      "model": model_id, "usage": usage}})
        self._end_chunked()
        log(f"[{self.label}] local done in {time.time() - started:.1f}s: {n_text} chars, "
            f"{n_tools} tool calls, prompt {p_tok} tok, output {o_tok} tok")

    # ---- local chat: GenerateAssistantResponse (AWS event stream) -------
    def _local_gar(self, state: dict, model_id: str, body_len: int):
        ollama_model = resolve_local(model_id)
        messages, tools = gar_to_ollama(state)
        self._start_chunked(200, {"Content-Type": "application/vnd.amazon.eventstream",
                                  "x-amzn-requestid": str(uuid.uuid4())})
        log(f"[{self.label}] local /generateAssistantResponse {model_id} -> {ollama_model}: "
            f"{len(messages)} msgs, {len(tools)} tools, body {body_len // 1024} KB")
        problem = (f"[kiro_ollama_bridge] Unknown local model {model_id}."
                   if not ollama_model else context_overflow(messages, tools))
        stats, started, n_text, n_tools = {}, time.time(), 0, 0
        if problem:
            log(f"[{self.label}] refused: {problem}")
            self._chunk(encode_event("assistantResponseEvent", {"content": problem}))
        else:
            try:
                for ev in ollama_chat_stream(ollama_model, messages, tools, CONFIG["think"]):
                    if ev[0] == "thinking" and CONFIG["show_thinking"]:
                        self._chunk(encode_event("reasoningContentEvent", {"text": ev[1]}))
                    elif ev[0] == "text":
                        n_text += len(ev[1])
                        self._chunk(encode_event("assistantResponseEvent", {"content": ev[1]}))
                    elif ev[0] == "tool":
                        tid = "tooluse_" + uuid.uuid4().hex[:22]
                        self._chunk(encode_event("toolUseEvent",
                                                 {"toolUseId": tid, "name": ev[1], "input": ev[2]}))
                        self._chunk(encode_event("toolUseEvent",
                                                 {"toolUseId": tid, "name": ev[1], "stop": True}))
                        n_tools += 1
                    elif ev[0] == "done":
                        stats = ev[1]
            except Exception as e:
                log(f"[{self.label}] ollama error: {e!r}")
                self._chunk(encode_event("assistantResponseEvent",
                                         {"content": f"\n[kiro_ollama_bridge] Ollama error: {e}"}))
        p_tok = stats.get("prompt_eval_count") or 0
        if p_tok:
            self._chunk(encode_event("contextUsageEvent", {
                "contextUsagePercentage": round(100 * p_tok / CONFIG["num_ctx"], 2)}))
        self._end_chunked()
        log(f"[{self.label}] local done in {time.time() - started:.1f}s: {n_text} chars, "
            f"{n_tools} tool calls, prompt {p_tok} tok, output {stats.get('eval_count', 0)} tok")


def serve(port: int, upstream: str, label: str) -> ThreadingHTTPServer:
    handler = type("Handler", (BridgeHandler,), {"upstream": upstream, "label": label})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"[{label}] listening on http://127.0.0.1:{port} -> {upstream}")
    return server


def main():
    p = argparse.ArgumentParser(description="Kiro <-> Ollama local bridge")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--runtime-port", type=int, default=8787)
    p.add_argument("--management-port", type=int, default=8788)
    p.add_argument("--ollama", default="http://127.0.0.1:11434")
    p.add_argument("--num-ctx", type=int, default=65536,
                   help="Ollama context window; also reported to Kiro as maxInputTokens")
    p.add_argument("--think", choices=["on", "off", "default"], default="off",
                   help="Ollama thinking (default off: much faster for agent turns)")
    p.add_argument("--show-thinking", action="store_true",
                   help="Forward model thinking to Kiro as reasoning")
    p.add_argument("--include-cloud", action="store_true",
                   help="Also list Ollama ':cloud' models (they don't run locally)")
    a = p.parse_args()
    CONFIG.update(ollama=a.ollama.rstrip("/"), num_ctx=a.num_ctx,
                  think={"on": True, "off": False, "default": None}[a.think],
                  show_thinking=a.show_thinking, skip_cloud=not a.include_cloud)
    log(f"local models: {[m['modelId'] for m in refresh_local_models()]}")
    serve(a.runtime_port, f"https://runtime.{a.region}.kiro.dev", "runtime")
    serve(a.management_port, f"https://management.{a.region}.kiro.dev", "management")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
