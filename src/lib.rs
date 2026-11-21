//! Zn: a pocket dev workshop for Android, written entirely in Rust.
//!
//! The crate is built twice from the same code:
//! * as a `cdylib` loaded by Android's `NativeActivity` (see `android_main`);
//! * as a regular desktop binary (`zn-desktop`) for fast iteration and tests.

pub mod agent;
pub mod app;
pub mod diag;
pub mod filepicker;
pub mod llm;
pub mod local;
pub mod platform;
pub mod project;
pub mod secrets;
pub mod settings;
pub mod shell;
pub mod tools;
mod ui;

use std::path::PathBuf;

/// Where projects and settings live when the platform gives us no better hint.
///
/// On Android the app passes its private internal storage instead.
pub fn default_data_dir() -> PathBuf {
    let from_env = |var: &str| {
        std::env::var_os(var)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    if let Some(dir) = from_env("XDG_DATA_HOME").or_else(|| from_env("APPDATA")) {
        return dir.join("zn");
    }
    if let Some(home) = from_env("HOME").or_else(|| from_env("USERPROFILE")) {
        return home.join(".local").join("share").join("zn");
    }
    std::env::temp_dir().join("zn")
}

/// Starts the UI and blocks until the window or activity closes.
pub fn run(data_dir: PathBuf, options: eframe::NativeOptions) -> eframe::Result {
    let previous_session = diag::init(&data_dir);
    local::configure_threads(settings::Settings::load(&data_dir).local_threads);
    eframe::run_native(
        "Zn",
        options,
        Box::new(move |cc| Ok(Box::new(app::ZnApp::new(cc, data_dir, previous_session)?))),
    )
}

#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
fn android_main(app: android_activity::AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );
    platform::init_android(&app);
    diag::install_thread_altstack();
    let data_dir = app
        .internal_data_path()
        .unwrap_or_else(|| PathBuf::from("/data/local/tmp/zn"));
    let options = eframe::NativeOptions {
        android_app: Some(app),
        ..Default::default()
    };
    if let Err(err) = run(data_dir, options) {
        log::error!("Zn exited with an error: {err}");
    }
}
