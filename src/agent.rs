//! The AI agent: runs the model/tool loop on a worker thread and streams
//! progress back to the UI through a channel.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver, TryRecvError},
};

use crate::{
    llm::{Message, Provider, ToolResult},
    project::Project,
    tools::{Toolbox, describe_call},
};

/// Upper bound on model calls per user request, so a confused model cannot loop forever.
pub const MAX_STEPS: usize = 25;

const SYSTEM_PROMPT: &str = "\
You are Zn, an AI developer working inside a phone app. The user describes what they want; \
you build it by calling tools that read and write files in their project.\n\
\n\
Guidelines:\n\
- If no project is open, create one first with create_project (kind: app, game or website).\n\
- Websites are plain HTML, CSS and JavaScript that open from index.html with no build step.\n\
- Games target Godot 4 (project.godot, .tscn scenes, GDScript .gd files).\n\
- Look at existing files (list_files, read_file) before changing them, and always write complete \
file contents.\n\
- Keep replies short. Say what you did and what the user should try next. The user may be \
reading on a small screen.\n\
- Reply in the user's language.";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatEntry {
    User(String),
    Assistant(String),
    /// A tool the agent ran: a one-line summary, whether it worked, and its output.
    Tool {
        summary: String,
        ok: bool,
        detail: String,
    },
    System(String),
}

enum Event {
    Entry(ChatEntry),
    Message(Message),
    ProjectChanged(Project),
    Finished,
}

/// What happened since the last [`AgentSession::poll`].
#[derive(Default, Debug, PartialEq)]
pub struct PollOutcome {
    /// Files may have changed on disk (a tool ran).
    pub files_touched: bool,
    /// The agent created or opened a different project.
    pub project: Option<Project>,
}

pub struct AgentSession {
    pub entries: Vec<ChatEntry>,
    conversation: Vec<Message>,
    events: Option<Receiver<Event>>,
    cancel: Arc<AtomicBool>,
}

impl Default for AgentSession {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentSession {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            conversation: Vec::new(),
            events: None,
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn is_busy(&self) -> bool {
        self.events.is_some()
    }

    /// Asks the running agent to stop before its next model call.
    pub fn stop(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn clear(&mut self) {
        if !self.is_busy() {
            self.entries.clear();
            self.conversation.clear();
        }
    }

    /// Adds a note to the chat that is not sent to the model.
    pub fn note(&mut self, text: impl Into<String>) {
        self.entries.push(ChatEntry::System(text.into()));
    }

    /// Starts handling a user request. Ignored if blank or already busy.
    /// `repaint` is called from the worker thread whenever there is something new to show.
    pub fn send(
        &mut self,
        text: &str,
        mut provider: Box<dyn Provider>,
        toolbox: Toolbox,
        repaint: impl Fn() + Send + 'static,
    ) {
        let text = text.trim();
        if text.is_empty() || self.is_busy() {
            return;
        }
        self.entries.push(ChatEntry::User(text.to_owned()));
        self.conversation.push(Message::User(text.to_owned()));
        self.cancel.store(false, Ordering::Relaxed);
        provider.attach_cancel(Arc::clone(&self.cancel));

        let conversation = self.conversation.clone();
        let cancel = Arc::clone(&self.cancel);
        let (tx, rx) = mpsc::channel();
        self.events = Some(rx);
        let spawned = crate::diag::spawn_big("zn-agent", move || {
            let emit = |event: Event| {
                let _ = tx.send(event);
                repaint();
            };
            run_loop(provider.as_ref(), toolbox, conversation, &cancel, &emit);
            emit(Event::Finished);
        });
        if let Err(err) = spawned {
            self.events = None;
            self.entries.push(ChatEntry::System(format!(
                "Error: could not start the agent: {err}"
            )));
        }
    }

    /// Applies finished work to the session. Call once per frame.
    pub fn poll(&mut self) -> PollOutcome {
        let mut outcome = PollOutcome::default();
        loop {
            let event = match self.events.as_ref().map(Receiver::try_recv) {
                Some(Ok(event)) => event,
                Some(Err(TryRecvError::Empty)) | None => return outcome,
                Some(Err(TryRecvError::Disconnected)) => {
                    self.entries.push(ChatEntry::System(
                        "Error: the agent stopped unexpectedly".into(),
                    ));
                    self.events = None;
                    return outcome;
                }
            };
            match event {
                Event::Entry(entry) => {
                    outcome.files_touched |= matches!(entry, ChatEntry::Tool { .. });
                    self.entries.push(entry);
                }
                Event::Message(message) => self.conversation.push(message),
                Event::ProjectChanged(project) => outcome.project = Some(project),
                Event::Finished => {
                    self.events = None;
                    return outcome;
                }
            }
        }
    }
}

fn run_loop(
    provider: &dyn Provider,
    mut toolbox: Toolbox,
    mut conversation: Vec<Message>,
    cancel: &AtomicBool,
    emit: &dyn Fn(Event),
) {
    let say = |entry: ChatEntry| emit(Event::Entry(entry));
    let specs = toolbox.specs();

    for _ in 0..MAX_STEPS {
        if cancel.load(Ordering::Relaxed) {
            say(ChatEntry::System("Stopped.".into()));
            return;
        }
        let completion = match provider.complete(SYSTEM_PROMPT, &conversation, &specs) {
            Ok(completion) => completion,
            Err(_) if cancel.load(Ordering::Relaxed) => {
                say(ChatEntry::System("Stopped.".into()));
                return;
            }
            Err(err) => {
                say(ChatEntry::System(format!("Error: {err}")));
                return;
            }
        };
        if !completion.text.trim().is_empty() {
            say(ChatEntry::Assistant(completion.text.clone()));
        }
        if completion.calls.is_empty() {
            if completion.text.trim().is_empty() {
                say(ChatEntry::System(
                    "The model returned an empty reply.".into(),
                ));
            } else {
                emit(Event::Message(Message::Assistant {
                    text: completion.text,
                    calls: Vec::new(),
                }));
            }
            return;
        }

        let assistant = Message::Assistant {
            text: completion.text,
            calls: completion.calls.clone(),
        };
        conversation.push(assistant.clone());
        emit(Event::Message(assistant));

        // Every tool call must be answered, even if the user pressed Stop meanwhile,
        // otherwise the stored conversation would be invalid for the next request.
        let mut results: Vec<ToolResult> = Vec::new();
        for call in &completion.calls {
            let before = toolbox.project().map(|p| p.path.clone());
            let result = toolbox.execute(call);
            say(ChatEntry::Tool {
                summary: describe_call(call),
                ok: !result.is_error,
                detail: result.content.clone(),
            });
            if let Some(project) = toolbox.project()
                && Some(&project.path) != before.as_ref()
            {
                emit(Event::ProjectChanged(project.clone()));
            }
            results.push(result);
        }
        let results = Message::ToolResults(results);
        conversation.push(results.clone());
        emit(Event::Message(results));
    }
    say(ChatEntry::System(format!(
        "Stopped after {MAX_STEPS} steps. Send another message to continue."
    )));
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::Mutex,
        thread,
        time::{Duration, Instant},
    };

    use serde_json::json;

    use super::*;
    use crate::{
        llm::{Completion, ToolCall, ToolSpec},
        project::ProjectStore,
    };

    /// Replays scripted replies and records what it was sent.
    struct Script {
        replies: Mutex<VecDeque<Result<Completion, String>>>,
        seen: Arc<Mutex<Vec<Vec<Message>>>>,
    }

    impl Provider for Script {
        fn name(&self) -> String {
            "script".into()
        }
        fn complete(
            &self,
            _: &str,
            conversation: &[Message],
            _: &[ToolSpec],
        ) -> Result<Completion, String> {
            self.seen.lock().unwrap().push(conversation.to_vec());
            self.replies
                .lock()
                .unwrap()
                .pop_front()
                .expect("script ran out of replies")
        }
    }

    type Seen = Arc<Mutex<Vec<Vec<Message>>>>;

    fn script(replies: Vec<Result<Completion, String>>) -> (Box<Script>, Seen) {
        let seen: Seen = Arc::default();
        (
            Box::new(Script {
                replies: Mutex::new(replies.into()),
                seen: Arc::clone(&seen),
            }),
            seen,
        )
    }

    fn say(text: &str) -> Result<Completion, String> {
        Ok(Completion {
            text: text.into(),
            calls: Vec::new(),
        })
    }

    fn call(name: &str, input: serde_json::Value) -> Result<Completion, String> {
        Ok(Completion {
            text: String::new(),
            calls: vec![ToolCall {
                id: format!("id-{name}"),
                name: name.into(),
                input,
            }],
        })
    }

    fn toolbox() -> (tempfile::TempDir, ProjectStore, Toolbox) {
        let dir = tempfile::tempdir().unwrap();
        let store = ProjectStore::new(dir.path()).unwrap();
        let tools = Toolbox::new(store.clone(), None, false);
        (dir, store, tools)
    }

    fn run(session: &mut AgentSession) -> PollOutcome {
        let mut total = PollOutcome::default();
        let start = Instant::now();
        while session.is_busy() {
            assert!(start.elapsed() < Duration::from_secs(10), "agent timed out");
            let outcome = session.poll();
            total.files_touched |= outcome.files_touched;
            total.project = outcome.project.or(total.project);
            thread::sleep(Duration::from_millis(5));
        }
        total
    }

    #[test]
    fn plain_answer_is_recorded_in_chat_and_conversation() {
        let (_dir, _store, tools) = toolbox();
        let (provider, _) = script(vec![say("Hello!")]);
        let mut session = AgentSession::new();
        session.send("  hi ", provider, tools, || {});
        assert!(session.is_busy());
        run(&mut session);
        assert_eq!(
            session.entries,
            [
                ChatEntry::User("hi".into()),
                ChatEntry::Assistant("Hello!".into())
            ]
        );
        assert_eq!(session.conversation.len(), 2);
    }

    #[test]
    fn tool_loop_creates_a_project_and_writes_a_file() {
        let (_dir, store, tools) = toolbox();
        let (provider, seen) = script(vec![
            call("create_project", json!({"name": "site", "kind": "website"})),
            call(
                "write_file",
                json!({"path": "index.html", "content": "<h1>Hi</h1>"}),
            ),
            say("Done."),
        ]);
        let mut session = AgentSession::new();
        session.send("make a site", provider, tools, || {});
        let outcome = run(&mut session);

        assert!(outcome.files_touched);
        let project = outcome.project.expect("project change is reported");
        assert_eq!(project.name, "site");
        assert_eq!(
            store
                .read(&project, std::path::Path::new("index.html"))
                .unwrap(),
            "<h1>Hi</h1>"
        );

        let tools_run: Vec<_> = session
            .entries
            .iter()
            .filter_map(|e| match e {
                ChatEntry::Tool { summary, ok, .. } => Some((summary.as_str(), *ok)),
                _ => None,
            })
            .collect();
        assert_eq!(
            tools_run,
            [
                ("create_project site", true),
                ("write_file index.html", true)
            ]
        );
        assert_eq!(
            session.entries.last(),
            Some(&ChatEntry::Assistant("Done.".into()))
        );

        // The third request must contain the earlier tool calls and their results.
        let third = &seen.lock().unwrap()[2];
        assert!(matches!(third[1], Message::Assistant { ref calls, .. } if calls.len() == 1));
        assert!(matches!(third[2], Message::ToolResults(ref r) if r.len() == 1 && !r[0].is_error));
    }

    #[test]
    fn tool_failures_go_back_to_the_model_instead_of_ending_the_run() {
        let (_dir, _store, tools) = toolbox();
        let (provider, seen) = script(vec![call("read_file", json!({"path": "x"})), say("Sorry.")]);
        let mut session = AgentSession::new();
        session.send("go", provider, tools, || {});
        run(&mut session);
        let second = &seen.lock().unwrap()[1];
        assert!(matches!(second[2], Message::ToolResults(ref r) if r[0].is_error));
    }

    #[test]
    fn provider_errors_are_shown_and_the_conversation_stays_valid() {
        let (_dir, _store, tools) = toolbox();
        let (provider, _) = script(vec![Err("HTTP 401: bad key".into())]);
        let mut session = AgentSession::new();
        session.send("hi", provider, tools, || {});
        run(&mut session);
        assert_eq!(
            session.entries.last(),
            Some(&ChatEntry::System("Error: HTTP 401: bad key".into()))
        );
        assert!(!session.is_busy());
        // The user can simply try again.
        let (_dir2, _store2, tools2) = toolbox();
        let (provider2, _) = script(vec![say("ok")]);
        session.send("again", provider2, tools2, || {});
        run(&mut session);
        assert_eq!(
            session.entries.last(),
            Some(&ChatEntry::Assistant("ok".into()))
        );
    }

    #[test]
    fn stop_prevents_further_model_calls_but_answers_pending_tools() {
        let (_dir, _store, tools) = toolbox();
        let (provider, seen) = script(vec![call("list_files", json!({})), say("never reached")]);
        let mut session = AgentSession::new();
        session.stop(); // cleared by send(); only a stop *during* the run counts
        session.send("go", provider, tools, || {});
        session.stop();
        run(&mut session);
        // Whether the stop landed before or after the first call, no further call happens
        // and nothing is left half-answered.
        assert!(seen.lock().unwrap().len() <= 1);
        assert!(
            session
                .conversation
                .iter()
                .enumerate()
                .all(|(i, m)| match m {
                    Message::Assistant { calls, .. } if !calls.is_empty() => {
                        matches!(
                            session.conversation.get(i + 1),
                            Some(Message::ToolResults(_))
                        )
                    }
                    _ => true,
                })
        );
        assert_eq!(
            session.entries.last(),
            Some(&ChatEntry::System("Stopped.".into()))
        );
    }

    #[test]
    fn runaway_loops_are_capped() {
        let (_dir, _store, tools) = toolbox();
        let replies = (0..MAX_STEPS + 5)
            .map(|_| call("list_files", json!({})))
            .collect();
        let (provider, seen) = script(replies);
        let mut session = AgentSession::new();
        session.send("loop", provider, tools, || {});
        run(&mut session);
        assert_eq!(seen.lock().unwrap().len(), MAX_STEPS);
        assert!(
            matches!(session.entries.last(), Some(ChatEntry::System(s)) if s.contains("Stopped after"))
        );
    }

    #[test]
    fn blank_and_overlapping_messages_are_ignored() {
        let (_dir, _store, tools) = toolbox();
        let (provider, _) = script(vec![say("one")]);
        let mut session = AgentSession::new();
        session.send("   ", provider, tools, || {});
        assert!(session.entries.is_empty() && !session.is_busy());
    }
}
