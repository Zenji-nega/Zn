//! User settings, stored as JSON next to the projects.
//!
//! Secrets (API keys, GitHub tokens) are deliberately not stored here yet.

use std::{fs, io, path::Path};

use serde::{Deserialize, Serialize};

const FILE_NAME: &str = "settings.json";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub dark_mode: bool,
    /// UI zoom factor, clamped to `MIN_SCALE..=MAX_SCALE`.
    pub font_scale: f32,
    pub github_user: String,
}

impl Settings {
    pub const MIN_SCALE: f32 = 0.8;
    pub const MAX_SCALE: f32 = 2.0;

    /// Loads settings; a missing or corrupt file yields the defaults.
    pub fn load(dir: &Path) -> Self {
        let mut settings: Self = fs::read_to_string(dir.join(FILE_NAME))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        settings.font_scale = settings.font_scale.clamp(Self::MIN_SCALE, Self::MAX_SCALE);
        settings
    }

    pub fn save(&self, dir: &Path) -> io::Result<()> {
        fs::create_dir_all(dir)?;
        let text = serde_json::to_string_pretty(self).map_err(io::Error::other)?;
        fs::write(dir.join(FILE_NAME), text)
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            dark_mode: true,
            font_scale: 1.2,
            github_user: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let settings = Settings {
            dark_mode: false,
            font_scale: 1.5,
            github_user: "zenji".into(),
        };
        settings.save(dir.path()).unwrap();
        assert_eq!(Settings::load(dir.path()), settings);
    }

    #[test]
    fn missing_or_corrupt_file_gives_defaults() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(Settings::load(dir.path()), Settings::default());
        fs::write(dir.path().join(FILE_NAME), "{ not json").unwrap();
        assert_eq!(Settings::load(dir.path()), Settings::default());
    }

    #[test]
    fn scale_is_clamped_and_unknown_fields_are_filled() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(FILE_NAME), r#"{"font_scale": 99.0}"#).unwrap();
        let loaded = Settings::load(dir.path());
        assert_eq!(loaded.font_scale, Settings::MAX_SCALE);
        assert!(loaded.dark_mode);
    }
}
