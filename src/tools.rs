//! The tools the AI agent can call, all scoped to the open project.

use std::{path::Path, time::Duration};

use serde_json::{Value, json};

use crate::{
    llm::{ToolCall, ToolResult, ToolSpec},
    project::{Project, ProjectKind, ProjectStore, sanitize_name},
    shell,
};

const MAX_READ_BYTES: usize = 100_000;
const MAX_LISTED_FILES: usize = 500;
const COMMAND_OUTPUT_CHARS: usize = 16_000;
const DEFAULT_TIMEOUT_SECS: u64 = 60;
const MAX_TIMEOUT_SECS: u64 = 600;

pub struct Toolbox {
    store: ProjectStore,
    project: Option<Project>,
    allow_commands: bool,
}

impl Toolbox {
    pub fn new(store: ProjectStore, project: Option<Project>, allow_commands: bool) -> Self {
        Self {
            store,
            project,
            allow_commands,
        }
    }

    pub fn project(&self) -> Option<&Project> {
        self.project.as_ref()
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        let mut specs = vec![
            ToolSpec {
                name: "create_project",
                description: "Create a new project from a template and make it the open project \
                              (or open it if a project with that name already exists).",
                schema: json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Letters, digits, '-' or '_'" },
                        "kind": { "type": "string", "enum": ["app", "game", "website"] }
                    },
                    "required": ["name", "kind"]
                }),
            },
            ToolSpec {
                name: "list_files",
                description: "List all files in the open project.",
                schema: json!({ "type": "object", "properties": {} }),
            },
            ToolSpec {
                name: "read_file",
                description: "Read a text file from the open project.",
                schema: json!({
                    "type": "object",
                    "properties": { "path": { "type": "string", "description": "Relative to the project root" } },
                    "required": ["path"]
                }),
            },
            ToolSpec {
                name: "write_file",
                description: "Create or overwrite a text file in the open project. \
                              Parent folders are created automatically.",
                schema: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Relative to the project root" },
                        "content": { "type": "string", "description": "The complete new file contents" }
                    },
                    "required": ["path", "content"]
                }),
            },
        ];
        if self.allow_commands {
            specs.push(ToolSpec {
                name: "run_command",
                description: "Run a shell command (sh -c) in the project directory and return its \
                              output and exit code.",
                schema: json!({
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" },
                        "timeout_secs": { "type": "integer", "description": "Default 60, maximum 600" }
                    },
                    "required": ["command"]
                }),
            });
        }
        specs
    }

    pub fn execute(&mut self, call: &ToolCall) -> ToolResult {
        let outcome = self.dispatch(&call.name, &call.input);
        let (content, is_error) = match outcome {
            Ok(text) => (text, false),
            Err(err) => (format!("Error: {err}"), true),
        };
        ToolResult {
            id: call.id.clone(),
            content,
            is_error,
        }
    }

    fn dispatch(&mut self, name: &str, input: &Value) -> Result<String, String> {
        if !input.is_object() {
            return Err("the arguments must be a JSON object".into());
        }
        match name {
            "create_project" => self.create_project(input),
            "list_files" => self.list_files(),
            "read_file" => self.read_file(input),
            "write_file" => self.write_file(input),
            "run_command" if self.allow_commands => self.run_command(input),
            "run_command" => Err("running commands is disabled in Settings".into()),
            other => Err(format!("unknown tool \"{other}\"")),
        }
    }

    fn open_project(&self) -> Result<&Project, String> {
        self.project
            .as_ref()
            .ok_or_else(|| "no project is open; call create_project first".to_owned())
    }

    fn create_project(&mut self, input: &Value) -> Result<String, String> {
        let name = str_arg(input, "name")?;
        let kind = match str_arg(input, "kind")? {
            "app" => ProjectKind::App,
            "game" => ProjectKind::Game,
            "website" => ProjectKind::Website,
            other => {
                return Err(format!(
                    "unknown kind \"{other}\" (use app, game or website)"
                ));
            }
        };
        let existing = sanitize_name(name).and_then(|clean| {
            self.store
                .list()
                .ok()?
                .into_iter()
                .find(|p| p.name == clean)
        });
        let (project, verb) = match existing {
            Some(project) => (project, "Opened existing"),
            None => (
                self.store.create(name, kind).map_err(|e| e.to_string())?,
                "Created",
            ),
        };
        let message = format!(
            "{verb} project \"{}\" ({}).",
            project.name,
            project.kind.label()
        );
        self.project = Some(project);
        Ok(message)
    }

    fn list_files(&self) -> Result<String, String> {
        let project = self.open_project()?;
        let files = self.store.files(project).map_err(|e| e.to_string())?;
        if files.is_empty() {
            return Ok("(the project has no files)".into());
        }
        let mut lines: Vec<String> = files
            .iter()
            .take(MAX_LISTED_FILES)
            .map(|f| f.display().to_string())
            .collect();
        if files.len() > MAX_LISTED_FILES {
            lines.push(format!("... and {} more", files.len() - MAX_LISTED_FILES));
        }
        Ok(lines.join("\n"))
    }

    fn read_file(&self, input: &Value) -> Result<String, String> {
        let project = self.open_project()?;
        let text = self
            .store
            .read(project, Path::new(str_arg(input, "path")?))
            .map_err(|e| e.to_string())?;
        if text.len() <= MAX_READ_BYTES {
            return Ok(text);
        }
        let mut end = MAX_READ_BYTES;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        Ok(format!(
            "{}\n[... file truncated at {MAX_READ_BYTES} bytes ...]",
            &text[..end]
        ))
    }

    fn write_file(&self, input: &Value) -> Result<String, String> {
        let project = self.open_project()?;
        let path = str_arg(input, "path")?;
        let content = str_arg(input, "content")?;
        self.store
            .write(project, Path::new(path), content)
            .map_err(|e| e.to_string())?;
        Ok(format!("Wrote {} bytes to {path}.", content.len()))
    }

    fn run_command(&self, input: &Value) -> Result<String, String> {
        let project = self.open_project()?;
        let command = str_arg(input, "command")?;
        let secs = input
            .get("timeout_secs")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .clamp(1, MAX_TIMEOUT_SECS);
        let got = shell::run_capture(
            command,
            &project.path,
            Duration::from_secs(secs),
            COMMAND_OUTPUT_CHARS,
        )
        .map_err(|e| format!("could not start the command: {e}"))?;
        let status = if got.timed_out {
            format!("[timed out after {secs}s and was killed]")
        } else {
            got.code.map_or("[terminated by signal]".to_owned(), |c| {
                format!("[exit code {c}]")
            })
        };
        Ok(format!("{}\n{status}", got.output.trim_end()))
    }
}

fn str_arg<'a>(input: &'a Value, key: &str) -> Result<&'a str, String> {
    input
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or invalid \"{key}\" (a string is required)"))
}

/// One-line description of a call, for the chat log.
pub fn describe_call(call: &ToolCall) -> String {
    let arg = |key: &str| call.input.get(key).and_then(Value::as_str).unwrap_or("?");
    match call.name.as_str() {
        "create_project" => format!("create_project {}", arg("name")),
        "read_file" | "write_file" => format!("{} {}", call.name, arg("path")),
        "run_command" => format!("run_command {}", arg("command")),
        other => other.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, input: Value) -> ToolCall {
        ToolCall {
            id: "id".into(),
            name: name.into(),
            input,
        }
    }

    fn toolbox(allow_commands: bool) -> (tempfile::TempDir, Toolbox) {
        let dir = tempfile::tempdir().unwrap();
        let store = ProjectStore::new(dir.path()).unwrap();
        (dir, Toolbox::new(store, None, allow_commands))
    }

    #[test]
    fn tools_need_an_open_project_until_one_is_created() {
        let (_dir, mut tools) = toolbox(false);
        let result = tools.execute(&call("list_files", json!({})));
        assert!(result.is_error && result.content.contains("create_project"));

        let created = tools.execute(&call(
            "create_project",
            json!({"name": "demo", "kind": "website"}),
        ));
        assert!(!created.is_error, "{}", created.content);
        assert_eq!(tools.project().unwrap().name, "demo");

        let listed = tools.execute(&call("list_files", json!({})));
        assert!(listed.content.contains("index.html"));
    }

    #[test]
    fn create_project_reopens_an_existing_one() {
        let (_dir, mut tools) = toolbox(false);
        tools.execute(&call(
            "create_project",
            json!({"name": "demo", "kind": "app"}),
        ));
        let again = tools.execute(&call(
            "create_project",
            json!({"name": "demo", "kind": "app"}),
        ));
        assert!(!again.is_error && again.content.starts_with("Opened existing"));
        let bad = tools.execute(&call(
            "create_project",
            json!({"name": "../x", "kind": "app"}),
        ));
        assert!(bad.is_error);
        let kind = tools.execute(&call(
            "create_project",
            json!({"name": "ok", "kind": "robot"}),
        ));
        assert!(kind.is_error && kind.content.contains("unknown kind"));
    }

    #[test]
    fn write_then_read_and_paths_cannot_escape() {
        let (_dir, mut tools) = toolbox(false);
        tools.execute(&call("create_project", json!({"name": "p", "kind": "app"})));
        let wrote = tools.execute(&call(
            "write_file",
            json!({"path": "src/a.txt", "content": "hi"}),
        ));
        assert!(!wrote.is_error, "{}", wrote.content);
        let read = tools.execute(&call("read_file", json!({"path": "src/a.txt"})));
        assert_eq!(read.content, "hi");
        for path in ["../evil.txt", "/etc/passwd", "a/../../b"] {
            let w = tools.execute(&call("write_file", json!({"path": path, "content": "x"})));
            assert!(w.is_error, "{path}");
            let r = tools.execute(&call("read_file", json!({"path": path})));
            assert!(r.is_error, "{path}");
        }
    }

    #[test]
    fn bad_arguments_are_reported_not_panicked() {
        let (_dir, mut tools) = toolbox(false);
        tools.execute(&call("create_project", json!({"name": "p", "kind": "app"})));
        assert!(
            tools
                .execute(&call("write_file", json!({"path": "a"})))
                .is_error
        );
        assert!(
            tools
                .execute(&call("read_file", Value::String("{oops".into())))
                .is_error
        );
        assert!(tools.execute(&call("nope", json!({}))).is_error);
    }

    #[test]
    fn run_command_is_hidden_and_refused_unless_enabled() {
        let (_dir, mut off) = toolbox(false);
        assert!(off.specs().iter().all(|s| s.name != "run_command"));
        off.execute(&call("create_project", json!({"name": "p", "kind": "app"})));
        let refused = off.execute(&call("run_command", json!({"command": "echo hi"})));
        assert!(refused.is_error && refused.content.contains("disabled"));

        let (_dir, on) = toolbox(true);
        assert!(on.specs().iter().any(|s| s.name == "run_command"));
    }

    #[cfg(unix)]
    #[test]
    fn run_command_runs_in_the_project_directory() {
        let (_dir, mut tools) = toolbox(true);
        tools.execute(&call("create_project", json!({"name": "p", "kind": "app"})));
        let result = tools.execute(&call(
            "run_command",
            json!({"command": "cat README.md; exit 4"}),
        ));
        assert!(!result.is_error);
        assert!(result.content.contains("# p") && result.content.ends_with("[exit code 4]"));
    }

    #[test]
    fn long_files_are_truncated_on_a_char_boundary() {
        let (_dir, mut tools) = toolbox(false);
        tools.execute(&call("create_project", json!({"name": "p", "kind": "app"})));
        let big = "é".repeat(MAX_READ_BYTES);
        tools.execute(&call(
            "write_file",
            json!({"path": "big.txt", "content": big}),
        ));
        let read = tools.execute(&call("read_file", json!({"path": "big.txt"})));
        assert!(read.content.contains("file truncated"));
    }

    #[test]
    fn descriptions_are_short() {
        assert_eq!(
            describe_call(&call("write_file", json!({"path": "a.js"}))),
            "write_file a.js"
        );
        assert_eq!(describe_call(&call("list_files", json!({}))), "list_files");
    }
}
