# kiro-ollama-bridge

Use local [Ollama](https://ollama.com) models in Kiro IDE, from Kiro's own model picker, next to your subscription's cloud models.

![Kiro model picker listing Local Ollama models at 0x credit below the cloud models](docs/kiro-model-picker-local-models.png)

*Kiro's model picker with the bridge running. The "Local …" entries are Ollama models on this machine.*

> **Unofficial.** This project is not affiliated with or endorsed by Amazon or Kiro. It relies on undocumented Kiro settings that can change or disappear in any release. Verified with Kiro IDE **1.2.37** and Ollama **0.40** on macOS (November 2026).

## Requirements

- Kiro IDE, signed in. Cloud models keep using your normal Kiro login and subscription.
- Ollama running locally (`http://127.0.0.1:11434`) with at least one chat model. For agent work, choose a model with the `tools` capability.
- Python 3.9+. The bridge uses only the standard library, so there's nothing to install.

## How it works

Kiro reads two settings that aren't in its Settings UI. Each replaces one of Kiro's backend hosts for a region:

| Setting | Replaces | Used for |
|---|---|---|
| `codewhisperer.config.krsEndpoints` | `runtime.<region>.kiro.dev` | chat (`/v1/responses`, `/generateAssistantResponse`), feature config, MCP |
| `codewhisperer.config.cpsEndpoints` | `management.<region>.kiro.dev` | model list (`ListAvailableModels`), usage, profile |

The bridge listens on both and forwards every request to the real Kiro host unchanged, with three exceptions:

- **Model list:** it appends your Ollama models as `local-<name>` entries (0 credits, text input).
- **Chat on `/v1/responses`** for a `local-…` model: translated to Ollama `/api/chat` and streamed back as the Responses-API SSE events Kiro parses.
- **Chat on `/generateAssistantResponse`** for a `local-…` model: same, streamed back as an AWS event stream.

A server-side flag decides which chat path Kiro uses for your account, so the bridge supports both. Tool calls and tool results are translated in both directions, so local models can use Kiro's tools.

## Setup

1. Start the bridge and leave it running:
   ```bash
   nohup python3 kiro_ollama_bridge.py > bridge.log 2>&1 &
   tail -f bridge.log
   ```
2. Copy `vscode-settings.example.jsonc` into `.vscode/settings.json` of the project you want to use it in. This limits the bridge to windows that open that folder. Kiro reads workspace values for these settings only in a **trusted** workspace. To use the bridge everywhere, put the same two keys in your user settings instead.
3. Reload the Kiro window: Command Palette → **Developer: Reload Window**. Kiro reads the endpoint settings only at startup.
4. In a **new** session, open the model picker and choose a **Local …** model. Kiro refreshes the model list every 5 minutes; the log shows `ListAvailableModels -> N models (M local)` when it does.

**To undo,** delete the two keys and reload. Keep the bridge running whenever the settings are set; otherwise all chat in that window fails.

## Options

| Flag | Default | Notes |
|---|---|---|
| `--num-ctx` | 65536 | Ollama context window, also reported to Kiro as the model's input limit. |
| `--think on\|off\|default` | off | Ollama thinking. Off is much faster for agent turns. |
| `--show-thinking` | off | Forward model thinking to Kiro as reasoning. |
| `--include-cloud` | off | Also list Ollama `:cloud` models, which don't run locally. |
| `--slim` | off | Replace Kiro's system prompt and tool set with a small one and trim old turns. Needed for small context windows. See [Slim mode](#slim-mode). |
| `--slim-tools` | 8 core tools | Comma-separated tools to keep in slim mode. Default: `read_file,fs_write,str_replace,fs_append,grep_search,file_search,list_directory,execute_bash`. |
| `--region` | us-east-1 | Which Kiro region to forward to. Use the region of your Kiro profile, and match it in the settings. |
| `--runtime-port` / `--management-port` | 8787 / 8788 | Must match the settings. |
| `--ollama` | `http://127.0.0.1:11434` | Ollama base URL. |

## Slim mode

Kiro sends about **20,000 tokens on every turn, even in a new session**: a long system prompt (about 18k tokens) plus 23 tool definitions (about 19k tokens of JSON, though Ollama counts somewhat fewer). A local model has to read all of it before it writes anything. A 27B model on Apple silicon reads roughly 100 tokens per second, so that's 3–4 minutes of silence per turn, and `--num-ctx` can't go below about 24k.

`--slim` shrinks the request before it reaches Ollama:

- **System prompt:** Kiro's prompt is replaced by a short coding-assistant prompt. It keeps the workspace root, OS and date lines from Kiro's prompt.
- **Tools:** only the `--slim-tools` allow-list is kept, with one-line descriptions and schemas stripped of their descriptions.
- **Conversation:** the oldest turns are dropped, then oversized tool outputs are truncated, until the request fits `--num-ctx` with room left for the reply.

```bash
nohup python3 kiro_ollama_bridge.py --slim --num-ctx 8192 > bridge.log 2>&1 &
```

In testing, a Kiro-sized request went from about 37,000 estimated tokens to about 650. The first answer arrived in 11 s instead of several minutes. The log shows the size of every request before and after slimming.

**What slim mode loses.** The local model no longer sees Kiro's own instructions: steering files, specs, hooks guidance, and the rules for Kiro's other tools. It also loses any tools outside the allow-list, and anything said in turns that were dropped. It's a plain coding assistant with file and shell tools, not Kiro's full agent. Slim mode only affects `local-…` models; cloud models are untouched.

## Limits

- **Context size.** Without `--slim`, Kiro's system prompt and tool definitions alone need about 20k tokens, and long sessions reach hundreds of thousands. When a prompt clearly won't fit `--num-ctx`, the bridge answers with an explanation instead of letting Ollama silently cut the start of the conversation. Use `--slim`, or new sessions with a larger `--num-ctx`.
- **"Paused" or stalled turns.** Kiro marks a stream as stalled after 60 s without an event and aborts it after 300 s. The bridge sends a keepalive event every 15 s while Ollama is still reading the prompt, so slow models aren't cut off. Long waits are still normal without `--slim`.
- **One request at a time.** Ollama processes requests for a model one after another, and loading a second large model can evict the first. Several sessions using local models at once will queue behind each other. Stick to one local model if your RAM can't hold two.
- **Text only.** Images and file attachments are dropped for local models.
- **Agent quality depends on the model.** Small models often handle Kiro's tool protocol poorly.
- **No Kiro-side features for local turns.** Metering, prompt caching and reasoning replay apply only to cloud models.

## Security

- **The bridge sees your Kiro token.** Kiro sends its bearer token on every request, and the bridge forwards it to Kiro's real hosts over HTTPS. The bridge binds to `127.0.0.1` only and never logs the token. Don't change the bind address.
- **Kiro warns about plain HTTP.** Kiro logs "bearer token will be sent over an unencrypted connection" because the endpoint is `http://`. The token only travels over loopback to the bridge.
- **Local requests never reach Kiro's servers.** Requests for `local-…` models go only to Ollama.
