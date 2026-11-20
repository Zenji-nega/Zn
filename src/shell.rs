//! A minimal in-app terminal: runs one command at a time via `sh -c`.
//!
//! Commands run with the app's own permissions, in the selected project's
//! directory. Output (stdout and stderr, interleaved) streams into `output`.

use std::{
    io::{BufRead, BufReader, Read},
    path::Path,
    process::{Command, Stdio},
    sync::mpsc::{self, Receiver, Sender, TryRecvError},
    thread,
};

const MAX_LINES: usize = 5000;

enum Event {
    Line(String),
    Done(Option<i32>),
}

#[derive(Default)]
pub struct ShellSession {
    pub output: Vec<String>,
    events: Option<Receiver<Event>>,
}

impl ShellSession {
    pub fn is_running(&self) -> bool {
        self.events.is_some()
    }

    pub fn clear(&mut self) {
        self.output.clear();
    }

    /// Starts `command` in `cwd`. Ignored while another command runs or if it is blank.
    pub fn run(&mut self, command: &str, cwd: &Path, repaint: impl Fn() + Send + Clone + 'static) {
        let command = command.trim();
        if command.is_empty() || self.is_running() {
            return;
        }
        self.push(format!("$ {command}"));

        let spawned = Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let mut child = match spawned {
            Ok(child) => child,
            Err(err) => {
                self.push(format!("failed to start: {err}"));
                return;
            }
        };

        let (tx, rx) = mpsc::channel();
        self.events = Some(rx);
        let readers: Vec<_> = [
            child
                .stdout
                .take()
                .map(|s| spawn_reader(s, tx.clone(), repaint.clone())),
            child
                .stderr
                .take()
                .map(|s| spawn_reader(s, tx.clone(), repaint.clone())),
        ]
        .into_iter()
        .flatten()
        .collect();
        thread::spawn(move || {
            for reader in readers {
                let _ = reader.join();
            }
            let code = child.wait().ok().and_then(|status| status.code());
            let _ = tx.send(Event::Done(code));
            repaint();
        });
    }

    /// Drains pending output. Call once per frame.
    pub fn poll(&mut self) {
        loop {
            let event = match self.events.as_ref().map(Receiver::try_recv) {
                Some(Ok(event)) => event,
                Some(Err(TryRecvError::Empty)) | None => return,
                Some(Err(TryRecvError::Disconnected)) => {
                    self.events = None;
                    return;
                }
            };
            match event {
                Event::Line(line) => self.push(line),
                Event::Done(code) => {
                    let status = code.map_or("terminated by signal".to_owned(), |c| {
                        format!("exit code {c}")
                    });
                    self.push(format!("[{status}]"));
                    self.events = None;
                    return;
                }
            }
        }
    }

    fn push(&mut self, line: String) {
        self.output.push(line);
        if self.output.len() > MAX_LINES {
            let excess = self.output.len() - MAX_LINES;
            self.output.drain(..excess);
        }
    }
}

fn spawn_reader(
    stream: impl Read + Send + 'static,
    tx: Sender<Event>,
    repaint: impl Fn() + Send + 'static,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        for line in BufReader::new(stream).lines().map_while(Result::ok) {
            if tx.send(Event::Line(line)).is_err() {
                break;
            }
            repaint();
        }
    })
}

#[cfg(all(test, unix))]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    fn run_to_end(command: &str) -> Vec<String> {
        let dir = tempfile::tempdir().unwrap();
        let mut shell = ShellSession::default();
        shell.run(command, dir.path(), || {});
        let start = Instant::now();
        while shell.is_running() {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "command timed out"
            );
            shell.poll();
            thread::sleep(Duration::from_millis(5));
        }
        shell.output
    }

    #[test]
    fn captures_stdout_and_exit_code() {
        let out = run_to_end("echo hello");
        assert_eq!(out, ["$ echo hello", "hello", "[exit code 0]"]);
    }

    #[test]
    fn captures_stderr_and_failure_code() {
        let out = run_to_end("echo oops >&2; exit 3");
        assert!(out.contains(&"oops".to_owned()));
        assert_eq!(out.last().unwrap(), "[exit code 3]");
    }

    #[test]
    fn runs_in_the_given_directory_and_ignores_blank_input() {
        let dir = tempfile::tempdir().unwrap();
        let mut shell = ShellSession::default();
        shell.run("   ", dir.path(), || {});
        assert!(shell.output.is_empty() && !shell.is_running());
        let out = run_to_end("pwd");
        assert!(out[1].starts_with('/'));
    }
}
