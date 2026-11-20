//! A minimal in-app terminal: runs one command at a time via `sh -c`.
//!
//! Commands run with the app's own permissions, in the selected project's
//! directory. Output (stdout and stderr, interleaved) streams into `output`.

use std::{
    io::{BufRead, BufReader, Read},
    path::Path,
    process::{Command, Stdio},
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, Sender, TryRecvError},
    },
    thread,
    time::{Duration, Instant},
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

/// Result of [`run_capture`].
#[derive(Debug, PartialEq, Eq)]
pub struct Captured {
    /// stdout and stderr, interleaved; only the last `max_chars` characters are kept.
    pub output: String,
    pub code: Option<i32>,
    pub timed_out: bool,
}

/// Runs `command` with `sh -c`, waits for it (killing it after `timeout`) and
/// returns what it printed. Used by the AI agent, which needs the whole result.
pub fn run_capture(
    command: &str,
    cwd: &Path,
    timeout: Duration,
    max_chars: usize,
) -> std::io::Result<Captured> {
    // Never let a runaway process buffer unbounded output in memory.
    const HARD_CAP: usize = 4 * 1024 * 1024;

    let mut child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let mut readers = 0;
    let streams: [Option<Box<dyn Read + Send>>; 2] = [
        child.stdout.take().map(|s| Box::new(s) as _),
        child.stderr.take().map(|s| Box::new(s) as _),
    ];
    for mut stream in streams.into_iter().flatten() {
        readers += 1;
        let buffer = Arc::clone(&buffer);
        let done = done_tx.clone();
        thread::spawn(move || {
            let mut chunk = [0u8; 4096];
            while let Ok(n) = stream.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                if let Ok(mut buf) = buffer.lock()
                    && buf.len() < HARD_CAP
                {
                    buf.extend_from_slice(&chunk[..n]);
                }
            }
            let _ = done.send(());
        });
    }

    let deadline = Instant::now() + timeout;
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status.code();
        }
        if Instant::now() >= deadline {
            timed_out = true;
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        thread::sleep(Duration::from_millis(15));
    };

    // Let the readers drain what is left, but never hang on a background
    // process that inherited the pipes.
    for _ in 0..readers {
        if done_rx.recv_timeout(Duration::from_millis(500)).is_err() {
            break;
        }
    }

    let bytes = buffer.lock().map(|b| b.clone()).unwrap_or_default();
    let text = String::from_utf8_lossy(&bytes);
    let total = text.chars().count();
    let output = if total > max_chars {
        let tail: String = text.chars().skip(total - max_chars).collect();
        format!("[... output truncated ...]\n{tail}")
    } else {
        text.into_owned()
    };
    Ok(Captured {
        output,
        code: status,
        timed_out,
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

    #[test]
    fn capture_returns_output_and_code() {
        let dir = tempfile::tempdir().unwrap();
        let got = run_capture(
            "echo out; echo err >&2; exit 2",
            dir.path(),
            Duration::from_secs(10),
            1000,
        )
        .unwrap();
        assert_eq!(got.code, Some(2));
        assert!(!got.timed_out);
        assert!(got.output.contains("out") && got.output.contains("err"));
    }

    #[test]
    fn capture_kills_on_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let start = Instant::now();
        let got = run_capture(
            "echo started; sleep 30",
            dir.path(),
            Duration::from_millis(300),
            1000,
        )
        .unwrap();
        assert!(got.timed_out && got.code.is_none());
        assert!(got.output.contains("started"));
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn capture_keeps_only_the_tail() {
        let dir = tempfile::tempdir().unwrap();
        let got = run_capture("seq 1 2000", dir.path(), Duration::from_secs(10), 50).unwrap();
        assert!(got.output.starts_with("[... output truncated ...]"));
        assert!(got.output.trim_end().ends_with("2000"));
    }
}
