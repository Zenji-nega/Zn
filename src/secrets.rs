//! API keys, kept apart from the ordinary settings.
//!
//! The file lives in the app's private data directory (on Android that storage
//! is sandboxed per app) and is created readable by the owner only. It is
//! *not* encrypted; Android Keystore integration is a possible later upgrade.

use std::{fs, io, path::Path};

use serde::{Deserialize, Serialize};

const FILE_NAME: &str = "secrets.json";

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Secrets {
    pub anthropic_key: String,
    pub openai_key: String,
}

// Deliberately not derived, so keys can never leak through `{:?}` logging.
impl std::fmt::Debug for Secrets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secrets { .. }")
    }
}

impl Secrets {
    pub fn load(dir: &Path) -> Self {
        fs::read_to_string(dir.join(FILE_NAME))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, dir: &Path) -> io::Result<()> {
        use std::io::Write;

        fs::create_dir_all(dir)?;
        let text = serde_json::to_string_pretty(self).map_err(io::Error::other)?;
        let path = dir.join(FILE_NAME);

        let mut options = fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&path)?;
        // A file created earlier with looser permissions must be tightened too.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(0o600))?;
        }
        file.write_all(text.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(Secrets::load(dir.path()), Secrets::default());
        let secrets = Secrets {
            anthropic_key: "a".into(),
            openai_key: "b".into(),
        };
        secrets.save(dir.path()).unwrap();
        assert_eq!(Secrets::load(dir.path()), secrets);
    }

    #[test]
    fn debug_output_hides_keys() {
        let secrets = Secrets {
            anthropic_key: "sk-secret".into(),
            openai_key: String::new(),
        };
        assert!(!format!("{secrets:?}").contains("sk-secret"));
    }

    #[cfg(unix)]
    #[test]
    fn file_is_owner_only_even_if_it_existed_before() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILE_NAME);
        fs::write(&path, "{}").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        Secrets::default().save(dir.path()).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}
