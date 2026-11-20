//! Talking to language models.
//!
//! The agent works with provider-neutral [`Message`]s. Each [`Provider`]
//! translates them to its wire format; the request builders and response
//! parsers are pure functions so they can be tested without a network.

use std::{
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};

use serde_json::{Value, json};

pub const MAX_OUTPUT_TOKENS: u32 = 8192;

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolResult {
    pub id: String,
    pub content: String,
    pub is_error: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    User(String),
    Assistant { text: String, calls: Vec<ToolCall> },
    ToolResults(Vec<ToolResult>),
}

#[derive(Clone, Debug)]
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    /// JSON Schema (`{"type":"object", ...}`) describing the arguments.
    pub schema: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Completion {
    pub text: String,
    pub calls: Vec<ToolCall>,
}

pub trait Provider: Send {
    fn name(&self) -> String;

    /// Gives long-running providers (on-device models) the agent's stop flag.
    fn attach_cancel(&mut self, _cancel: Arc<AtomicBool>) {}

    /// Blocking: sends the conversation and returns the model's next turn.
    fn complete(
        &self,
        system: &str,
        conversation: &[Message],
        tools: &[ToolSpec],
    ) -> Result<Completion, String>;
}

// ---------------------------------------------------------------- HTTP

pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

pub trait Http: Send + Sync {
    fn post_json(
        &self,
        url: &str,
        headers: &[(&str, String)],
        body: &Value,
    ) -> Result<HttpResponse, String>;
}

/// Real HTTP client. TLS roots are bundled (no dependence on the OS store).
pub struct UreqHttp {
    agent: ureq::Agent,
}

impl UreqHttp {
    pub fn new() -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(180)))
            .http_status_as_error(false)
            .build();
        Self {
            agent: ureq::Agent::new_with_config(config),
        }
    }
}

impl Default for UreqHttp {
    fn default() -> Self {
        Self::new()
    }
}

impl Http for UreqHttp {
    fn post_json(
        &self,
        url: &str,
        headers: &[(&str, String)],
        body: &Value,
    ) -> Result<HttpResponse, String> {
        let mut request = self
            .agent
            .post(url)
            .header("content-type", "application/json");
        for (name, value) in headers {
            request = request.header(*name, value.as_str());
        }
        let mut response = request
            .send(body.to_string())
            .map_err(|err| format!("network error: {err}"))?;
        let status = response.status().as_u16();
        let body = response
            .body_mut()
            .read_to_string()
            .map_err(|err| format!("could not read the response: {err}"))?;
        Ok(HttpResponse { status, body })
    }
}

/// Turns a non-2xx response into a readable error, preferring the API's own message.
fn http_error(response: &HttpResponse) -> String {
    let detail = serde_json::from_str::<Value>(&response.body)
        .ok()
        .and_then(|v| {
            v.pointer("/error/message")
                .or_else(|| v.get("message"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| response.body.chars().take(300).collect());
    format!("HTTP {}: {}", response.status, detail.trim())
}

fn parse_body(response: &HttpResponse) -> Result<Value, String> {
    if !(200..300).contains(&response.status) {
        return Err(http_error(response));
    }
    serde_json::from_str(&response.body)
        .map_err(|err| format!("invalid JSON from the server: {err}"))
}

fn nonempty(text: &str) -> String {
    if text.trim().is_empty() {
        "(no output)".to_owned()
    } else {
        text.to_owned()
    }
}

// ------------------------------------------------------------ Anthropic

pub struct Anthropic {
    pub http: Arc<dyn Http>,
    pub api_key: String,
    pub model: String,
    pub base_url: String,
}

impl Anthropic {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.anthropic.com";
    pub const DEFAULT_MODEL: &'static str = "claude-sonnet-5-5";
    const VERSION: &'static str = "2023-06-01";
}

pub fn anthropic_request(
    model: &str,
    system: &str,
    conversation: &[Message],
    tools: &[ToolSpec],
) -> Value {
    let messages: Vec<Value> = conversation
        .iter()
        .map(|message| match message {
            Message::User(text) => json!({ "role": "user", "content": text }),
            Message::Assistant { text, calls } => {
                let mut blocks = Vec::new();
                if !text.trim().is_empty() {
                    blocks.push(json!({ "type": "text", "text": text }));
                }
                for call in calls {
                    blocks.push(json!({
                        "type": "tool_use", "id": call.id, "name": call.name, "input": call.input
                    }));
                }
                json!({ "role": "assistant", "content": blocks })
            }
            Message::ToolResults(results) => {
                let blocks: Vec<Value> = results
                    .iter()
                    .map(|r| {
                        json!({
                            "type": "tool_result", "tool_use_id": r.id,
                            "content": nonempty(&r.content), "is_error": r.is_error
                        })
                    })
                    .collect();
                json!({ "role": "user", "content": blocks })
            }
        })
        .collect();

    let mut request = json!({
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": system,
        "messages": messages,
    });
    if !tools.is_empty() {
        request["tools"] = tools
            .iter()
            .map(|t| json!({ "name": t.name, "description": t.description, "input_schema": t.schema }))
            .collect();
    }
    request
}

pub fn anthropic_parse(body: &Value) -> Result<Completion, String> {
    let blocks = body
        .get("content")
        .and_then(Value::as_array)
        .ok_or("the response has no content")?;
    let mut text = String::new();
    let mut calls = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(t) = block.get("text").and_then(Value::as_str) {
                    text.push_str(t);
                }
            }
            Some("tool_use") => calls.push(ToolCall {
                id: block
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                name: block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                input: block.get("input").cloned().unwrap_or_else(|| json!({})),
            }),
            _ => {}
        }
    }
    if body.get("stop_reason").and_then(Value::as_str) == Some("max_tokens") && calls.is_empty() {
        text.push_str("\n\n[The reply was cut off: output limit reached.]");
    }
    Ok(Completion { text, calls })
}

impl Provider for Anthropic {
    fn name(&self) -> String {
        format!("Anthropic · {}", self.model)
    }

    fn complete(
        &self,
        system: &str,
        conversation: &[Message],
        tools: &[ToolSpec],
    ) -> Result<Completion, String> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let headers = [
            ("x-api-key", self.api_key.clone()),
            ("anthropic-version", Self::VERSION.to_owned()),
        ];
        let request = anthropic_request(&self.model, system, conversation, tools);
        let response = self.http.post_json(&url, &headers, &request)?;
        anthropic_parse(&parse_body(&response)?)
    }
}

// -------------------------------------------------------- OpenAI-compatible

pub struct OpenAiCompat {
    pub http: Arc<dyn Http>,
    /// May be empty for local servers that need no key.
    pub api_key: String,
    pub model: String,
    pub base_url: String,
}

impl OpenAiCompat {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";
}

pub fn openai_request(
    model: &str,
    system: &str,
    conversation: &[Message],
    tools: &[ToolSpec],
) -> Value {
    let mut messages = vec![json!({ "role": "system", "content": system })];
    for message in conversation {
        match message {
            Message::User(text) => messages.push(json!({ "role": "user", "content": text })),
            Message::Assistant { text, calls } => {
                let mut value = json!({
                    "role": "assistant",
                    "content": if text.trim().is_empty() { Value::Null } else { json!(text) },
                });
                if !calls.is_empty() {
                    value["tool_calls"] = calls
                        .iter()
                        .map(|c| {
                            json!({
                                "id": c.id, "type": "function",
                                "function": { "name": c.name, "arguments": c.input.to_string() }
                            })
                        })
                        .collect();
                }
                messages.push(value);
            }
            Message::ToolResults(results) => {
                for r in results {
                    messages.push(json!({
                        "role": "tool", "tool_call_id": r.id, "content": nonempty(&r.content)
                    }));
                }
            }
        }
    }

    let mut request = json!({ "model": model, "messages": messages });
    if !tools.is_empty() {
        request["tools"] = tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": { "name": t.name, "description": t.description, "parameters": t.schema }
                })
            })
            .collect();
    }
    request
}

pub fn openai_parse(body: &Value) -> Result<Completion, String> {
    let message = body
        .pointer("/choices/0/message")
        .ok_or("the response has no choices")?;
    let text = match message.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    };
    let calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .map(|c| {
                    let raw = c
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}");
                    ToolCall {
                        id: c
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        name: c
                            .pointer("/function/name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        // Keep unparsable arguments as a string; the toolbox reports them to the model.
                        input: if raw.trim().is_empty() {
                            json!({})
                        } else {
                            serde_json::from_str(raw)
                                .unwrap_or_else(|_| Value::String(raw.to_owned()))
                        },
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(Completion { text, calls })
}

impl Provider for OpenAiCompat {
    fn name(&self) -> String {
        format!("OpenAI-compatible · {}", self.model)
    }

    fn complete(
        &self,
        system: &str,
        conversation: &[Message],
        tools: &[ToolSpec],
    ) -> Result<Completion, String> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let mut headers = Vec::new();
        if !self.api_key.is_empty() {
            headers.push(("authorization", format!("Bearer {}", self.api_key)));
        }
        let request = openai_request(&self.model, system, conversation, tools);
        let response = self.http.post_json(&url, &headers, &request)?;
        openai_parse(&parse_body(&response)?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Mutex;

    use super::*;

    pub(crate) fn sample_tools() -> Vec<ToolSpec> {
        vec![ToolSpec {
            name: "read_file",
            description: "Read a file",
            schema: json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] }),
        }]
    }

    fn sample_conversation() -> Vec<Message> {
        vec![
            Message::User("make a page".into()),
            Message::Assistant {
                text: "Reading.".into(),
                calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "read_file".into(),
                    input: json!({"path": "a.txt"}),
                }],
            },
            Message::ToolResults(vec![ToolResult {
                id: "c1".into(),
                content: "hello".into(),
                is_error: false,
            }]),
        ]
    }

    #[test]
    fn anthropic_request_shape() {
        let req = anthropic_request("m", "sys", &sample_conversation(), &sample_tools());
        assert_eq!(req["model"], "m");
        assert_eq!(req["system"], "sys");
        assert_eq!(req["tools"][0]["name"], "read_file");
        assert_eq!(req["tools"][0]["input_schema"]["type"], "object");
        let messages = req["messages"].as_array().unwrap();
        assert_eq!(
            messages[0],
            json!({"role": "user", "content": "make a page"})
        );
        assert_eq!(messages[1]["content"][0]["type"], "text");
        assert_eq!(messages[1]["content"][1]["type"], "tool_use");
        assert_eq!(messages[1]["content"][1]["input"]["path"], "a.txt");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"][0]["type"], "tool_result");
        assert_eq!(messages[2]["content"][0]["tool_use_id"], "c1");
    }

    #[test]
    fn anthropic_request_never_sends_empty_text_blocks() {
        let conv = vec![Message::Assistant {
            text: "  ".into(),
            calls: vec![ToolCall {
                id: "x".into(),
                name: "n".into(),
                input: json!({}),
            }],
        }];
        let req = anthropic_request("m", "s", &conv, &[]);
        let blocks = req["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "tool_use");
        assert!(req.get("tools").is_none());
    }

    #[test]
    fn anthropic_parses_text_and_tool_use() {
        let body = json!({
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Let me look."},
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "x"}}
            ]
        });
        let done = anthropic_parse(&body).unwrap();
        assert_eq!(done.text, "Let me look.");
        assert_eq!(
            done.calls,
            [ToolCall {
                id: "t1".into(),
                name: "read_file".into(),
                input: json!({"path": "x"})
            }]
        );
        assert!(anthropic_parse(&json!({"nope": 1})).is_err());
    }

    #[test]
    fn anthropic_flags_truncated_replies() {
        let body =
            json!({"stop_reason": "max_tokens", "content": [{"type": "text", "text": "partial"}]});
        assert!(anthropic_parse(&body).unwrap().text.contains("cut off"));
    }

    #[test]
    fn openai_request_shape() {
        let req = openai_request("m", "sys", &sample_conversation(), &sample_tools());
        let messages = req["messages"].as_array().unwrap();
        assert_eq!(messages[0], json!({"role": "system", "content": "sys"}));
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["name"],
            "read_file"
        );
        // Arguments are a JSON *string* in this protocol.
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"a.txt\"}"
        );
        assert_eq!(
            messages[3],
            json!({"role": "tool", "tool_call_id": "c1", "content": "hello"})
        );
        assert_eq!(req["tools"][0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn openai_parses_tool_calls_and_bad_arguments() {
        let body = json!({"choices": [{"message": {
            "content": null,
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"p\"}"}},
                {"id": "b", "type": "function", "function": {"name": "read_file", "arguments": "{oops"}}
            ]
        }}]});
        let done = openai_parse(&body).unwrap();
        assert_eq!(done.text, "");
        assert_eq!(done.calls[0].input, json!({"path": "p"}));
        assert_eq!(done.calls[1].input, Value::String("{oops".into()));
        assert!(openai_parse(&json!({"choices": []})).is_err());
    }

    #[test]
    fn openai_accepts_content_parts() {
        let body = json!({"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]});
        assert_eq!(openai_parse(&body).unwrap().text, "ab");
    }

    /// One recorded request: the URL and its headers.
    type Seen = (String, Vec<(String, String)>);

    struct FakeHttp {
        response: Mutex<Option<Result<HttpResponse, String>>>,
        seen: Mutex<Vec<Seen>>,
    }

    impl FakeHttp {
        fn replying(status: u16, body: &str) -> Arc<Self> {
            Arc::new(Self {
                response: Mutex::new(Some(Ok(HttpResponse {
                    status,
                    body: body.into(),
                }))),
                seen: Mutex::new(Vec::new()),
            })
        }
    }

    impl Http for FakeHttp {
        fn post_json(
            &self,
            url: &str,
            headers: &[(&str, String)],
            _: &Value,
        ) -> Result<HttpResponse, String> {
            let headers = headers
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect();
            self.seen.lock().unwrap().push((url.to_owned(), headers));
            self.response.lock().unwrap().take().expect("one call")
        }
    }

    #[test]
    fn anthropic_sends_key_and_version_headers() {
        let http = FakeHttp::replying(200, r#"{"content":[{"type":"text","text":"hi"}]}"#);
        let provider = Anthropic {
            http: http.clone(),
            api_key: "sk-test".into(),
            model: "m".into(),
            base_url: "https://example.test/".into(),
        };
        let done = provider
            .complete("s", &[Message::User("x".into())], &[])
            .unwrap();
        assert_eq!(done.text, "hi");
        let seen = http.seen.lock().unwrap();
        assert_eq!(seen[0].0, "https://example.test/v1/messages");
        assert!(seen[0].1.contains(&("x-api-key".into(), "sk-test".into())));
        assert!(seen[0].1.iter().any(|(k, _)| k == "anthropic-version"));
    }

    #[test]
    fn openai_omits_auth_header_without_a_key() {
        let http = FakeHttp::replying(200, r#"{"choices":[{"message":{"content":"ok"}}]}"#);
        let provider = OpenAiCompat {
            http: http.clone(),
            api_key: String::new(),
            model: "m".into(),
            base_url: "http://localhost:11434/v1".into(),
        };
        assert_eq!(provider.complete("s", &[], &[]).unwrap().text, "ok");
        let seen = http.seen.lock().unwrap();
        assert_eq!(seen[0].0, "http://localhost:11434/v1/chat/completions");
        assert!(seen[0].1.is_empty());
    }

    #[test]
    fn http_errors_use_the_api_message() {
        let http = FakeHttp::replying(
            401,
            r#"{"error":{"type":"authentication_error","message":"invalid x-api-key"}}"#,
        );
        let provider = Anthropic {
            http,
            api_key: "bad".into(),
            model: "m".into(),
            base_url: Anthropic::DEFAULT_BASE_URL.into(),
        };
        let err = provider.complete("s", &[], &[]).unwrap_err();
        assert_eq!(err, "HTTP 401: invalid x-api-key");
    }

    #[test]
    fn non_json_error_bodies_are_truncated() {
        let long = "x".repeat(1000);
        let err = http_error(&HttpResponse {
            status: 502,
            body: long,
        });
        assert!(err.starts_with("HTTP 502: ") && err.len() < 400);
    }
}
