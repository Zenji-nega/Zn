//! A small folder browser for choosing model files (the logic; the drawing is in `ui/`).

use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PickerEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
}

/// Browses folders, listing sub-folders and `.gguf` files only.
pub struct FilePicker {
    pub dir: PathBuf,
    pub entries: Vec<PickerEntry>,
    /// Why the current folder could not be listed, if it could not.
    pub error: Option<String>,
    /// The address bar: a path the user may type.
    pub path_text: String,
}

impl FilePicker {
    pub fn open(start: &Path) -> Self {
        let mut picker = Self {
            dir: start.to_path_buf(),
            entries: Vec::new(),
            error: None,
            path_text: String::new(),
        };
        picker.reload();
        picker
    }

    /// Goes to `dir`. If it cannot be read, the folder is still shown (empty) with the reason.
    pub fn go(&mut self, dir: PathBuf) {
        self.dir = dir;
        self.reload();
    }

    pub fn enter(&mut self, name: &str) {
        let next = self.dir.join(name);
        self.go(next);
    }

    pub fn up(&mut self) {
        if let Some(parent) = self.dir.parent() {
            let parent = parent.to_path_buf();
            self.go(parent);
        }
    }

    pub fn file_path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    /// Jumps to what was typed in the address bar: a folder, or the folder of a typed file path.
    pub fn go_to_text(&mut self) {
        let typed = PathBuf::from(self.path_text.trim());
        if typed.as_os_str().is_empty() {
            return;
        }
        self.go(typed);
    }

    fn reload(&mut self) {
        self.path_text = self.dir.display().to_string();
        self.entries.clear();
        self.error = None;
        let read = match fs::read_dir(&self.dir) {
            Ok(read) => read,
            Err(err) => {
                self.error = Some(format!("Cannot open this folder: {err}"));
                return;
            }
        };
        for entry in read.filter_map(Result::ok) {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            // Symlinks to folders (common on phones) count as folders.
            let is_dir = kind.is_dir() || (kind.is_symlink() && entry.path().is_dir());
            let is_model = Path::new(&name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
            if is_dir || is_model {
                let size = if is_dir {
                    0
                } else {
                    entry.metadata().map_or(0, |m| m.len())
                };
                self.entries.push(PickerEntry { name, is_dir, size });
            }
        }
        self.entries
            .sort_by_key(|e| (!e.is_dir, e.name.to_lowercase()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(picker: &FilePicker) -> Vec<(&str, bool)> {
        picker
            .entries
            .iter()
            .map(|e| (e.name.as_str(), e.is_dir))
            .collect()
    }

    #[test]
    fn lists_folders_first_and_only_gguf_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("Zeta")).unwrap();
        fs::create_dir(dir.path().join("alpha")).unwrap();
        fs::create_dir(dir.path().join(".hidden")).unwrap();
        fs::write(dir.path().join("b.GGUF"), b"GGUF").unwrap();
        fs::write(dir.path().join("a.gguf"), b"GGUFxx").unwrap();
        fs::write(dir.path().join("notes.txt"), b"x").unwrap();
        fs::write(dir.path().join(".secret.gguf"), b"x").unwrap();

        let picker = FilePicker::open(dir.path());
        assert_eq!(
            names(&picker),
            [
                ("alpha", true),
                ("Zeta", true),
                ("a.gguf", false),
                ("b.GGUF", false)
            ]
        );
        assert_eq!(picker.entries[2].size, 6);
        assert!(picker.error.is_none());
    }

    #[test]
    fn navigates_into_folders_and_back_up() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("models/small")).unwrap();
        fs::write(dir.path().join("models/small/m.gguf"), b"GGUF").unwrap();

        let mut picker = FilePicker::open(dir.path());
        picker.enter("models");
        picker.enter("small");
        assert_eq!(names(&picker), [("m.gguf", false)]);
        assert_eq!(
            picker.file_path("m.gguf"),
            dir.path().join("models/small/m.gguf")
        );
        picker.up();
        picker.up();
        assert_eq!(picker.dir, dir.path());
        assert_eq!(picker.path_text, dir.path().display().to_string());
    }

    #[test]
    fn unreadable_folders_report_why_and_can_be_left() {
        let dir = tempfile::tempdir().unwrap();
        let mut picker = FilePicker::open(dir.path());
        picker.go(dir.path().join("missing"));
        assert!(picker.entries.is_empty());
        assert!(
            picker
                .error
                .as_deref()
                .unwrap()
                .contains("Cannot open this folder")
        );
        picker.up();
        assert!(picker.error.is_none());
    }

    #[test]
    fn the_address_bar_jumps_to_a_typed_path() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("x")).unwrap();
        let mut picker = FilePicker::open(Path::new("/"));
        picker.path_text = format!("  {}  ", dir.path().join("x").display());
        picker.go_to_text();
        assert_eq!(picker.dir, dir.path().join("x"));
        picker.path_text = "   ".into();
        picker.go_to_text(); // blank input is ignored
        assert_eq!(picker.dir, dir.path().join("x"));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_folders_are_browsable() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("real")).unwrap();
        std::os::unix::fs::symlink(dir.path().join("real"), dir.path().join("link")).unwrap();
        let picker = FilePicker::open(dir.path());
        assert!(picker.entries.iter().any(|e| e.name == "link" && e.is_dir));
    }
}
