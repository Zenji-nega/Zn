# Zn

A pocket dev workshop for Android, written entirely in Rust with [egui](https://github.com/emilk/egui).
The goal: describe an app, game or website in a sentence and have an AI agent build it, all on the phone.

## Status: early skeleton

What works today (covered by unit tests, desktop-verified):

| Area | State |
|------|-------|
| Projects | Create App / Game (Godot 4) / Website projects from templates, list and open them |
| Editor | Open, edit, save and add files inside a project (paths cannot escape the project folder) |
| Agent | Real AI agent: a tool-using loop (`create_project`, `list_files`, `read_file`, `write_file`, optional `run_command`) on a worker thread, with Stop and a step cap. Works with **Anthropic**, any **OpenAI-compatible** server, or an **on-device model** (see below). Cloud providers need your own API key |
| Terminal | Runs one `sh -c` command at a time in the project directory, streams stdout/stderr |
| Settings | Dark mode, UI size, GitHub username, AI provider/model/key, and a switch for agent shell commands (off by default) |

Not built yet: streaming replies, GPU/NPU acceleration, Godot 4 export, web preview/publishing, GitHub integration, a bundled Linux environment.

## Project layout

```
src/lib.rs        entry points: `android_main` (Android) and `run` (shared)
src/main.rs       desktop entry point for fast development
src/app.rs        app state, tab bar, per-frame polling
src/ui/           one file per tab
src/project.rs    templates, listing, safe file access
src/agent.rs      agent loop + threaded AgentSession
src/llm.rs        Anthropic / OpenAI-compatible adapters (pure request/response code)
src/tools.rs      the tools the agent can call, scoped to the open project
src/local/        on-device models: prompt format (prompt.rs), candle GGUF engine (engine.rs), downloads (models.rs)
src/secrets.rs    API keys (separate file, owner-only permissions)
src/shell.rs      in-app terminal
src/settings.rs   persisted settings
```

## Develop on desktop

```sh
# Linux needs: libxkbcommon-dev libwayland-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev
cargo run --bin zn-desktop
cargo test
```

## Build the Android APK

Locally (needs the Android SDK + NDK and `ANDROID_HOME` / `ANDROID_NDK_ROOT` set):

```sh
rustup target add aarch64-linux-android
cargo install cargo-apk --locked
cargo apk build --lib          # APK in target/debug/apk/
cargo apk run --lib            # install and launch on a connected device
```

Or run the **Android APK** workflow from the repository's Actions tab and download the `zn-apk` artifact.


## Using the agent

1. Open **Settings**, choose a provider, enter the model name and your API key.
2. Go to **Agent** and describe what you want. The agent creates a project if none is open, writes files, and shows each tool call (tap one to see its output).
3. Open the **Editor** tab to see and tweak the result.

## On-device (offline) models

Pick **On-device (offline)** in Settings, download a model, and the agent runs it on the phone's CPU with [candle](https://github.com/huggingface/candle), pure Rust and no internet needed afterwards.

- Supported: quantized **GGUF** models of the `qwen2` (Qwen2.5, Qwen2.5-Coder) and `qwen3` architectures. Three Qwen2.5-Coder presets are built in; "Download by link" accepts any other compatible GGUF plus its `tokenizer.json`.
- Downloads resume if interrupted. Models live in the app's private `models/` folder.
- Expect it to be **slow**: a 1.5B model on a recent phone reads the prompt at tens of tokens per second and writes a few per second. The attention cache is kept between steps, so only new text is reprocessed. Models of 0.5B to 3B are also much less reliable at tool use than cloud models, so keep requests small and concrete.
- The conversation is limited to 8192 tokens; start a new chat when you hit it.
- "Free memory" unloads the model.

## Known limitations

- egui does not shape Arabic or lay out right-to-left text, so the UI is English for now. Arabic characters render through system fonts but joined letters will look disconnected.
- `accesskit` (screen reader support) is disabled on Android because eframe cannot combine it with `native-activity`.
- The terminal uses the Android system shell; there is no bundled Linux userland yet.
- API keys are saved unencrypted in the app's private folder (`secrets.json`, owner-only). Android Keystore support would be a later upgrade. GitHub tokens are not stored at all yet.
- With "Let the agent run shell commands" on, the agent runs commands without asking each time. Leave it off unless you want that. On a desktop build the commands run with your user's permissions.
- The on-device engine was verified with a tiny synthetic model in unit tests, not yet with real Qwen weights on a phone.
- The model sees the whole conversation on every request; there is no history trimming yet, so very long chats will get slow and costly.
