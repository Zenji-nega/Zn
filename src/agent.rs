//! The AI agent layer.
//!
//! The UI talks to an [`AgentSession`], which runs a pluggable [`AgentBackend`]
//! on a worker thread so the interface never blocks. Real backends (a remote
//! API, an on-device model) implement [`AgentBackend`]; today only a
//! placeholder ships, so the plumbing can be exercised end to end.

use std::{
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, TryRecvError},
    },
    thread,
};

use crate::project::Project;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: Role,
    pub text: String,
}

pub trait AgentBackend: Send + 'static {
    fn name(&self) -> &str;

    /// Produces the assistant's next reply. May block; it runs off the UI thread.
    fn respond(
        &mut self,
        history: &[ChatMessage],
        project: Option<&Project>,
    ) -> Result<String, String>;
}

/// Stands in until a real model is connected.
pub struct PlaceholderBackend;

impl AgentBackend for PlaceholderBackend {
    fn name(&self) -> &str {
        "placeholder"
    }

    fn respond(
        &mut self,
        history: &[ChatMessage],
        project: Option<&Project>,
    ) -> Result<String, String> {
        let request = history
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map_or("", |m| m.text.as_str());
        let target = project.map_or("no project selected".to_owned(), |p| {
            format!("project \"{}\"", p.name)
        });
        Ok(format!(
            "No AI model is connected yet, so I can't act on this ({target}).\n\nYour request: {request}"
        ))
    }
}

type Reply = Result<String, String>;

pub struct AgentSession {
    pub history: Vec<ChatMessage>,
    backend: Arc<Mutex<Box<dyn AgentBackend>>>,
    pending: Option<Receiver<Reply>>,
}

impl AgentSession {
    pub fn new(backend: Box<dyn AgentBackend>) -> Self {
        Self {
            history: Vec::new(),
            backend: Arc::new(Mutex::new(backend)),
            pending: None,
        }
    }

    pub fn is_busy(&self) -> bool {
        self.pending.is_some()
    }

    pub fn backend_name(&self) -> String {
        self.backend
            .lock()
            .map_or_else(|_| "unavailable".into(), |b| b.name().to_owned())
    }

    /// Queues a user message. Ignored while a reply is pending or the text is blank.
    /// `repaint` is called from the worker thread when the reply is ready.
    pub fn send(
        &mut self,
        text: &str,
        project: Option<Project>,
        repaint: impl Fn() + Send + 'static,
    ) {
        let text = text.trim();
        if text.is_empty() || self.is_busy() {
            return;
        }
        self.history.push(ChatMessage {
            role: Role::User,
            text: text.to_owned(),
        });

        let history = self.history.clone();
        let backend = Arc::clone(&self.backend);
        let (tx, rx) = mpsc::channel();
        self.pending = Some(rx);
        thread::spawn(move || {
            let reply = match backend.lock() {
                Ok(mut b) => b.respond(&history, project.as_ref()),
                Err(_) => Err("agent backend crashed".to_owned()),
            };
            let _ = tx.send(reply);
            repaint();
        });
    }

    /// Moves a finished reply into the history. Call once per frame.
    pub fn poll(&mut self) {
        let Some(rx) = &self.pending else { return };
        let message = match rx.try_recv() {
            Ok(Ok(text)) => ChatMessage {
                role: Role::Assistant,
                text,
            },
            Ok(Err(err)) => ChatMessage {
                role: Role::System,
                text: format!("Error: {err}"),
            },
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => ChatMessage {
                role: Role::System,
                text: "Error: the agent stopped unexpectedly".to_owned(),
            },
        };
        self.history.push(message);
        self.pending = None;
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    struct Upper;
    impl AgentBackend for Upper {
        fn name(&self) -> &str {
            "upper"
        }
        fn respond(&mut self, history: &[ChatMessage], _: Option<&Project>) -> Reply {
            let last = history.last().ok_or("empty history")?;
            if last.text == "fail" {
                return Err("boom".into());
            }
            Ok(last.text.to_uppercase())
        }
    }

    fn wait(session: &mut AgentSession) {
        let start = Instant::now();
        while session.is_busy() {
            assert!(start.elapsed() < Duration::from_secs(5), "agent timed out");
            session.poll();
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn reply_arrives_via_poll() {
        let mut session = AgentSession::new(Box::new(Upper));
        session.send("  hello ", None, || {});
        assert!(session.is_busy());
        wait(&mut session);
        let texts: Vec<_> = session
            .history
            .iter()
            .map(|m| (m.role, m.text.as_str()))
            .collect();
        assert_eq!(texts, [(Role::User, "hello"), (Role::Assistant, "HELLO")]);
    }

    #[test]
    fn errors_become_system_messages_and_blank_input_is_ignored() {
        let mut session = AgentSession::new(Box::new(Upper));
        session.send("   ", None, || {});
        assert!(session.history.is_empty());
        session.send("fail", None, || {});
        wait(&mut session);
        let last = session.history.last().unwrap();
        assert_eq!(
            (last.role, last.text.as_str()),
            (Role::System, "Error: boom")
        );
    }

    #[test]
    fn second_message_is_ignored_while_busy() {
        let mut session = AgentSession::new(Box::new(Upper));
        session.send("one", None, || {});
        session.send("two", None, || {});
        wait(&mut session);
        assert_eq!(session.history.len(), 2);
    }

    #[test]
    fn placeholder_mentions_request_and_project() {
        let reply = PlaceholderBackend
            .respond(
                &[ChatMessage {
                    role: Role::User,
                    text: "make a game".into(),
                }],
                None,
            )
            .unwrap();
        assert!(reply.contains("make a game") && reply.contains("no project selected"));
    }
}
