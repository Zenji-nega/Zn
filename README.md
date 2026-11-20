# Zn

A pocket dev workshop for Android, written entirely in Rust with [egui](https://github.com/emilk/egui).
The goal: describe an app, game or website in a sentence and have an AI agent build it, all on the phone.

## Status: early skeleton

What works today (covered by unit tests, desktop-verified):

| Area | State |
|------|-------|
| Projects | Create App / Game (Godot 4) / Website projects from templates, list and open them |
| Editor | Open, edit, save and add files inside a project (paths cannot escape the project folder) |
| Agent | Chat UI and a background-thread `AgentBackend` trait. Only a **placeholder** backend exists, no AI model is connected yet |
| Terminal | Runs one `sh -c` command at a time in the project directory, streams stdout/stderr |
| Settings | Dark mode, UI size, GitHub username (saved as JSON) |

Not built yet: real AI backends (remote API or on-device model), Godot 4 export, web preview/publishing, GitHub integration, a bundled Linux environment.

## Project layout

```
src/lib.rs        entry points: `android_main` (Android) and `run` (shared)
src/main.rs       desktop entry point for fast development
src/app.rs        app state, tab bar, per-frame polling
src/ui/           one file per tab
src/project.rs    templates, listing, safe file access
src/agent.rs      AgentBackend trait + threaded AgentSession
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

> The Android build has not been run yet. The code was only compiled for desktop, so the first CI run may need small fixes to the `cargo-apk` metadata in `Cargo.toml`.

## Known limitations

- egui does not shape Arabic or lay out right-to-left text, so the UI is English for now. Arabic characters render through system fonts but joined letters will look disconnected.
- `accesskit` (screen reader support) is disabled on Android because eframe cannot combine it with `native-activity`.
- The terminal uses the Android system shell; there is no bundled Linux userland yet.
- Settings never store secrets. API keys and GitHub tokens need a proper secure-storage design before they are added.
