//! Projects on disk: creation from templates, listing, and safe file access.

use std::{
    fs, io,
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};

const META_FILE: &str = ".zn.json";
const MAX_FILES: usize = 2000;
const MAX_DEPTH: usize = 8;
const SKIPPED_DIRS: [&str; 4] = [".git", ".godot", "node_modules", "target"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectKind {
    App,
    Game,
    Website,
}

impl ProjectKind {
    pub const ALL: [Self; 3] = [Self::App, Self::Game, Self::Website];

    pub fn label(self) -> &'static str {
        match self {
            Self::App => "App",
            Self::Game => "Game (Godot 4)",
            Self::Website => "Website",
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Meta {
    kind: ProjectKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Project {
    pub name: String,
    pub kind: ProjectKind,
    pub path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct ProjectStore {
    root: PathBuf,
}

impl ProjectStore {
    /// Uses `<data_dir>/projects`, creating it if needed.
    pub fn new(data_dir: &Path) -> io::Result<Self> {
        let root = data_dir.join("projects");
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn list(&self) -> io::Result<Vec<Project>> {
        let mut projects = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let path = entry?.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()).map(str::to_owned) else {
                continue;
            };
            let Ok(meta_text) = fs::read_to_string(path.join(META_FILE)) else {
                continue;
            };
            if let Ok(meta) = serde_json::from_str::<Meta>(&meta_text) {
                projects.push(Project {
                    name,
                    kind: meta.kind,
                    path,
                });
            }
        }
        projects.sort_by_key(|p| p.name.to_lowercase());
        Ok(projects)
    }

    pub fn create(&self, name: &str, kind: ProjectKind) -> io::Result<Project> {
        let name = sanitize_name(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Use letters, digits, '-' or '_' (max 64)",
            )
        })?;
        let path = self.root.join(&name);
        if path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "A project with that name exists",
            ));
        }
        fs::create_dir_all(&path)?;
        let meta = serde_json::to_string(&Meta { kind }).map_err(io::Error::other)?;
        fs::write(path.join(META_FILE), meta)?;
        for (file, contents) in template(kind, &name) {
            fs::write(path.join(file), contents)?;
        }
        Ok(Project { name, kind, path })
    }

    /// Relative paths of the project's files, sorted, without internal metadata.
    pub fn files(&self, project: &Project) -> io::Result<Vec<PathBuf>> {
        let mut out = Vec::new();
        collect(&project.path, Path::new(""), 0, &mut out)?;
        out.sort();
        Ok(out)
    }

    pub fn read(&self, project: &Project, rel: &Path) -> io::Result<String> {
        fs::read_to_string(safe_join(&project.path, rel)?)
    }

    pub fn write(&self, project: &Project, rel: &Path, contents: &str) -> io::Result<()> {
        let full = safe_join(&project.path, rel)?;
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(full, contents)
    }
}

fn collect(base: &Path, rel: &Path, depth: usize, out: &mut Vec<PathBuf>) -> io::Result<()> {
    if depth > MAX_DEPTH {
        return Ok(());
    }
    for entry in fs::read_dir(base.join(rel))? {
        let entry = entry?;
        let name = entry.file_name();
        let rel_path = rel.join(&name);
        if entry.file_type()?.is_dir() {
            if !SKIPPED_DIRS.iter().any(|skip| name == *skip) {
                collect(base, &rel_path, depth + 1, out)?;
            }
        } else if name != META_FILE && out.len() < MAX_FILES {
            out.push(rel_path);
        }
    }
    Ok(())
}

/// Turns user input into a safe single directory name, or `None` if impossible.
pub fn sanitize_name(input: &str) -> Option<String> {
    let name: String = input
        .trim()
        .chars()
        .map(|c| if c.is_whitespace() { '-' } else { c })
        .collect();
    let valid = !name.is_empty()
        && name.chars().count() <= 64
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_');
    valid.then_some(name)
}

/// Joins `rel` onto `base`, rejecting absolute paths and any `..` or root components.
fn safe_join(base: &Path, rel: &Path) -> io::Result<PathBuf> {
    let escapes = rel
        .components()
        .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir));
    if escapes || rel.as_os_str().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Path must stay inside the project",
        ));
    }
    Ok(base.join(rel))
}

fn template(kind: ProjectKind, name: &str) -> Vec<(&'static str, String)> {
    match kind {
        ProjectKind::App => vec![(
            "README.md",
            format!("# {name}\n\nDescribe your app here.\n"),
        )],
        ProjectKind::Website => vec![
            (
                "index.html",
                format!(
                    "<!doctype html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"utf-8\">\n  \
                     <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n  \
                     <title>{name}</title>\n  <link rel=\"stylesheet\" href=\"style.css\">\n</head>\n\
                     <body>\n  <h1>{name}</h1>\n  <script src=\"script.js\"></script>\n</body>\n</html>\n"
                ),
            ),
            (
                "style.css",
                "body { font-family: sans-serif; margin: 2rem; }\n".to_owned(),
            ),
            ("script.js", "console.log('ready');\n".to_owned()),
        ],
        ProjectKind::Game => vec![
            (
                "project.godot",
                format!(
                    "; Engine configuration file.\nconfig_version=5\n\n[application]\n\
                     config/name=\"{name}\"\nrun/main_scene=\"res://main.tscn\"\n\
                     config/features=PackedStringArray(\"4.3\")\n"
                ),
            ),
            (
                "main.tscn",
                "[gd_scene format=3]\n\n[node name=\"Main\" type=\"Node2D\"]\n".to_owned(),
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_accepts_and_rejects() {
        assert_eq!(sanitize_name("  my game "), Some("my-game".into()));
        assert_eq!(sanitize_name("مشروع_1"), Some("مشروع_1".into()));
        for bad in ["", "   ", ".hidden", "a/b", "..", "a\\b", &"x".repeat(65)] {
            assert_eq!(sanitize_name(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn safe_join_blocks_escapes() {
        let base = Path::new("/p");
        assert!(safe_join(base, Path::new("src/main.rs")).is_ok());
        for bad in ["../x", "/etc/passwd", "a/../../x", ""] {
            assert!(safe_join(base, Path::new(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn create_list_read_write() {
        let dir = tempfile::tempdir().unwrap();
        let store = ProjectStore::new(dir.path()).unwrap();
        let site = store.create("My Site", ProjectKind::Website).unwrap();
        store.create("Zed", ProjectKind::Game).unwrap();
        assert_eq!(site.name, "My-Site");
        assert!(store.create("My Site", ProjectKind::App).is_err());

        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0], site);

        let files = store.files(&site).unwrap();
        assert!(files.contains(&PathBuf::from("index.html")));
        assert!(!files.contains(&PathBuf::from(META_FILE)));

        store
            .write(&site, Path::new("js/app.js"), "let x = 1;")
            .unwrap();
        assert_eq!(
            store.read(&site, Path::new("js/app.js")).unwrap(),
            "let x = 1;"
        );
        assert!(store.write(&site, Path::new("../evil"), "x").is_err());
        assert!(store.read(&site, Path::new("/etc/passwd")).is_err());
    }
}
