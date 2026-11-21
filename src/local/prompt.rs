//! ChatML prompts with Hermes-style tool calls, as used by the Qwen family.
//!
//! Rendering and parsing are pure string work, so they are fully unit tested
//! without loading a model.

use serde::Serialize;
use serde_json::{Value, json, ser::Formatter};

use crate::llm::{Completion, Message, ToolCall, ToolSpec};

/// Name given to a tool call the model wrote but that could not be parsed.
/// The toolbox answers it with an explanation so the model can retry.
pub const INVALID_CALL: &str = "invalid_tool_call";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Template {
    Qwen2,
    /// Like Qwen2, but replies start with an empty `<think>` block so the model
    /// answers directly instead of spending tokens on reasoning.
    Qwen3,
}

impl Template {
    fn reply_prefix(self) -> &'static str {
        match self {
            Self::Qwen2 => "",
            Self::Qwen3 => "<think>\n\n</think>\n\n",
        }
    }
}

/// Serializes JSON the way Python's `json.dumps` does (`", "` and `": "`),
/// which is what these models were trained on and what they write back.
struct PythonStyle;

impl Formatter for PythonStyle {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

fn to_json(value: &Value) -> String {
    let mut out = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, PythonStyle);
    match value.serialize(&mut serializer) {
        Ok(()) => String::from_utf8(out).unwrap_or_default(),
        Err(_) => value.to_string(),
    }
}

/// Short instructions for small on-device models: every prompt token costs seconds on a phone.
pub const COMPACT_SYSTEM: &str = "You are Zn, a coding assistant in a phone app. Build what the user asks by calling tools. \
Create a project first if none is open. Websites: plain HTML/CSS/JS with index.html. Games: Godot 4 (GDScript). \
Write complete files. Keep replies short and use the user's language.";

/// Drops parameter descriptions and keeps only the first sentence of the tool description.
fn compact_spec(tool: &ToolSpec) -> Value {
    let mut schema = tool.schema.clone();
    if let Some(props) = schema.get_mut("properties").and_then(Value::as_object_mut) {
        for prop in props.values_mut() {
            if let Some(obj) = prop.as_object_mut() {
                obj.remove("description");
            }
        }
    }
    let first = tool
        .description
        .split_inclusive(". ")
        .next()
        .unwrap_or(tool.description)
        .trim();
    json!({ "type": "function", "function": { "name": tool.name, "description": first, "parameters": schema } })
}

fn system_block(system: &str, tools: &[ToolSpec], compact: bool) -> String {
    if tools.is_empty() {
        return system.to_owned();
    }
    let mut text = format!(
        "{system}\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    );
    for tool in tools {
        let spec = if compact {
            compact_spec(tool)
        } else {
            json!({
                "type": "function",
                "function": { "name": tool.name, "description": tool.description, "parameters": tool.schema }
            })
        };
        text.push('\n');
        text.push_str(&to_json(&spec));
    }
    text.push_str(
        "\n</tools>\n\nFor each function call, return a json object with function name and arguments \
         within <tool_call></tool_call> XML tags:\n<tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );
    text
}

/// Renders the whole conversation, ending with the cue for the model's next reply.
pub fn render(
    template: Template,
    system: &str,
    conversation: &[Message],
    tools: &[ToolSpec],
) -> String {
    render_with(template, system, conversation, tools, false)
}

/// Like [`render`] but with trimmed tool descriptions, for slow on-device models.
pub fn render_compact(
    template: Template,
    system: &str,
    conversation: &[Message],
    tools: &[ToolSpec],
) -> String {
    render_with(template, system, conversation, tools, true)
}

fn render_with(
    template: Template,
    system: &str,
    conversation: &[Message],
    tools: &[ToolSpec],
    compact: bool,
) -> String {
    let mut out = format!(
        "<|im_start|>system\n{}<|im_end|>\n",
        system_block(system, tools, compact)
    );
    for message in conversation {
        match message {
            Message::User(text) => out.push_str(&format!("<|im_start|>user\n{text}<|im_end|>\n")),
            Message::Assistant { text, calls } => {
                out.push_str("<|im_start|>assistant\n");
                out.push_str(template.reply_prefix());
                out.push_str(text.trim());
                for call in calls {
                    let body = json!({ "name": call.name, "arguments": call.input });
                    if !out.ends_with('\n') {
                        out.push('\n');
                    }
                    out.push_str(&format!("<tool_call>\n{}\n</tool_call>", to_json(&body)));
                }
                out.push_str("<|im_end|>\n");
            }
            Message::ToolResults(results) => {
                out.push_str("<|im_start|>user");
                for result in results {
                    out.push_str(&format!(
                        "\n<tool_response>\n{}\n</tool_response>",
                        result.content
                    ));
                }
                out.push_str("<|im_end|>\n");
            }
        }
    }
    out.push_str("<|im_start|>assistant\n");
    out.push_str(template.reply_prefix());
    out
}

/// Splits the model's raw reply into visible text and tool calls.
/// `next_id` numbers the calls so each gets a unique id.
pub fn parse_output(raw: &str, next_id: &mut usize) -> Completion {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";

    let raw = strip_think(raw);
    let mut text = String::new();
    let mut calls = Vec::new();
    let mut rest = raw.as_str();
    while let Some(start) = rest.find(OPEN) {
        text.push_str(&rest[..start]);
        let after = &rest[start + OPEN.len()..];
        let (body, remaining) = match after.find(CLOSE) {
            Some(end) => (&after[..end], &after[end + CLOSE.len()..]),
            None => (after, ""), // cut off before the closing tag
        };
        calls.push(parse_call(body.trim(), next_id));
        rest = remaining;
    }
    text.push_str(rest);
    Completion {
        text: text.trim().to_owned(),
        calls,
    }
}

fn parse_call(body: &str, next_id: &mut usize) -> ToolCall {
    let id = format!("local-{}", *next_id);
    *next_id += 1;
    let parsed = serde_json::from_str::<Value>(body).ok().and_then(|v| {
        let name = v.get("name")?.as_str()?.to_owned();
        let arguments = match v.get("arguments") {
            None => json!({}),
            // Some models double-encode the arguments as a string.
            Some(Value::String(s)) => serde_json::from_str(s).ok()?,
            Some(other) => other.clone(),
        };
        Some((name, arguments))
    });
    match parsed {
        Some((name, input)) => ToolCall { id, name, input },
        None => ToolCall {
            id,
            name: INVALID_CALL.to_owned(),
            input: Value::String(body.to_owned()),
        },
    }
}

fn strip_think(raw: &str) -> String {
    let mut text = raw.to_owned();
    while let Some(start) = text.find("<think>") {
        match text[start..].find("</think>") {
            Some(len) => text.replace_range(start..start + len + "</think>".len(), ""),
            None => {
                text.truncate(start);
                break;
            }
        }
    }
    text.replace("</think>", "")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ToolResult;

    fn spec() -> ToolSpec {
        ToolSpec {
            name: "read_file",
            description: "Read a file",
            schema: json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
        }
    }

    #[test]
    fn python_style_json_matches_what_the_models_write() {
        let value = json!({"name": "x", "arguments": {"path": "a", "list": [1, 2]}});
        assert_eq!(
            to_json(&value),
            r#"{"name": "x", "arguments": {"path": "a", "list": [1, 2]}}"#
        );
        assert_eq!(to_json(&json!({})), "{}");
        assert_eq!(to_json(&json!("é\n")), "\"é\\n\"");
    }

    #[test]
    fn renders_system_tools_and_turns() {
        let conv = vec![
            Message::User("hi".into()),
            Message::Assistant {
                text: "Looking.".into(),
                calls: vec![ToolCall {
                    id: "1".into(),
                    name: "read_file".into(),
                    input: json!({"path": "a"}),
                }],
            },
            Message::ToolResults(vec![
                ToolResult {
                    id: "1".into(),
                    content: "hello".into(),
                    is_error: false,
                },
                ToolResult {
                    id: "2".into(),
                    content: "bye".into(),
                    is_error: false,
                },
            ]),
        ];
        let prompt = render(Template::Qwen2, "SYS", &conv, &[spec()]);
        assert!(prompt.starts_with("<|im_start|>system\nSYS\n\n# Tools"));
        assert!(
            prompt.contains(
                "<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"read_file\""
            )
        );
        assert!(prompt.contains("<|im_start|>user\nhi<|im_end|>\n"));
        assert!(prompt.contains(
            "<|im_start|>assistant\nLooking.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"}}\n</tool_call><|im_end|>\n"
        ));
        assert!(prompt.contains(
            "<|im_start|>user\n<tool_response>\nhello\n</tool_response>\n<tool_response>\nbye\n</tool_response><|im_end|>\n"
        ));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn no_tools_means_a_plain_system_prompt() {
        let prompt = render(Template::Qwen2, "SYS", &[Message::User("x".into())], &[]);
        assert!(prompt.starts_with("<|im_start|>system\nSYS<|im_end|>\n"));
        assert!(!prompt.contains("<tools>"));
    }

    #[test]
    fn a_call_without_text_follows_the_role_line_directly() {
        let conv = vec![Message::Assistant {
            text: String::new(),
            calls: vec![ToolCall {
                id: "1".into(),
                name: "list_files".into(),
                input: json!({}),
            }],
        }];
        let prompt = render(Template::Qwen2, "S", &conv, &[]);
        assert!(prompt.contains("<|im_start|>assistant\n<tool_call>\n{\"name\": \"list_files\", \"arguments\": {}}\n</tool_call><|im_end|>"));
    }

    #[test]
    fn qwen3_skips_thinking_in_every_reply() {
        let conv = vec![Message::Assistant {
            text: "ok".into(),
            calls: vec![],
        }];
        let prompt = render(Template::Qwen3, "S", &conv, &[]);
        assert!(prompt.contains("<|im_start|>assistant\n<think>\n\n</think>\n\nok<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn parses_text_and_multiple_calls() {
        let raw = "I'll do it.\n<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"a\", \"content\": \"x\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"list_files\", \"arguments\": {}}\n</tool_call>";
        let mut id = 0;
        let done = parse_output(raw, &mut id);
        assert_eq!(done.text, "I'll do it.");
        assert_eq!(done.calls.len(), 2);
        assert_eq!(done.calls[0].name, "write_file");
        assert_eq!(done.calls[0].input, json!({"path": "a", "content": "x"}));
        assert_eq!(done.calls[1].id, "local-1");
        assert_eq!(id, 2);
    }

    #[test]
    fn plain_answers_have_no_calls() {
        let done = parse_output("  Just text. ", &mut 0);
        assert_eq!((done.text.as_str(), done.calls.len()), ("Just text.", 0));
    }

    #[test]
    fn broken_calls_become_invalid_calls_the_model_can_see() {
        let mut id = 0;
        let bad = parse_output("<tool_call>\n{not json}\n</tool_call>", &mut id);
        assert_eq!(bad.calls[0].name, INVALID_CALL);
        assert_eq!(bad.calls[0].input, Value::String("{not json}".into()));

        let cut = parse_output(
            "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"pa",
            &mut id,
        );
        assert_eq!(cut.calls[0].name, INVALID_CALL);

        let nameless = parse_output("<tool_call>{\"arguments\": {}}</tool_call>", &mut id);
        assert_eq!(nameless.calls[0].name, INVALID_CALL);
    }

    #[test]
    fn string_encoded_arguments_are_accepted() {
        let raw =
            r#"<tool_call>{"name": "read_file", "arguments": "{\"path\": \"a\"}"}</tool_call>"#;
        let done = parse_output(raw, &mut 0);
        assert_eq!(done.calls[0].input, json!({"path": "a"}));
    }

    #[test]
    fn thinking_is_removed() {
        let done = parse_output("<think>hmm\nlots</think>\n\nAnswer", &mut 0);
        assert_eq!(done.text, "Answer");
        let cut = parse_output("Hi <think>never closed", &mut 0);
        assert_eq!(cut.text, "Hi");
    }

    #[test]
    fn what_the_model_writes_renders_back_identically() {
        // Keeping the re-rendered history byte-identical lets the engine reuse its KV cache.
        let written = "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"a.html\", \"content\": \"<h1>Hi</h1>\\n\"}}\n</tool_call>";
        let done = parse_output(written, &mut 0);
        let conv = vec![Message::Assistant {
            text: done.text,
            calls: done.calls,
        }];
        let prompt = render(Template::Qwen2, "S", &conv, &[]);
        assert!(prompt.contains(&format!("<|im_start|>assistant\n{written}<|im_end|>")));
    }

    #[test]
    fn compact_prompt_is_much_shorter_and_keeps_tool_names() {
        let tools = vec![
            ToolSpec {
                name: "write_file",
                description: "Create or overwrite a file. Parent folders are created. Be complete.",
                schema: json!({"type": "object", "properties": {
                    "path": {"type": "string", "description": "Relative to the project root"}
                }, "required": ["path"]}),
            },
            spec(),
        ];
        let conv = vec![Message::User("hi".into())];
        let full = render(Template::Qwen2, COMPACT_SYSTEM, &conv, &tools);
        let short = render_compact(Template::Qwen2, COMPACT_SYSTEM, &conv, &tools);
        assert!(short.len() < full.len());
        assert!(short.contains("\"name\": \"write_file\""));
        assert!(short.contains("Create or overwrite a file."));
        assert!(!short.contains("Parent folders"));
        assert!(!short.contains("Relative to the project root"));
        assert!(short.ends_with("<|im_start|>assistant\n"));
    }
}
