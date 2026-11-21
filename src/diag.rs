//! Crash diagnostics for a phone where nobody can read logs.
//!
//! * A one-line "phase" breadcrumb says what the app was doing. If it is not
//!   `idle` at the next start, the app was killed or crashed during that phase.
//! * Panics and fatal signals (SIGSEGV, SIGILL, ...) append a note to `crash.txt`.
//!
//! [`init`] returns a human-readable report of the previous session, if it ended badly.

use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::OnceLock,
};

const PHASE_FILE: &str = "phase.txt";
const CRASH_FILE: &str = "crash.txt";
const IDLE: &str = "idle";

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Installs the hooks and reports how the previous session ended (`None` if it ended normally).
pub fn init(data_dir: &Path) -> Option<String> {
    let dir = data_dir.join("diag");
    let _ = fs::create_dir_all(&dir);
    let report = read_previous(&dir);
    let _ = fs::remove_file(dir.join(CRASH_FILE));
    let _ = DIR.set(dir.clone());
    set_phase(IDLE);

    let previous_hook = std::panic::take_hook();
    let panic_dir = dir.clone();
    std::panic::set_hook(Box::new(move |info| {
        append(&panic_dir.join(CRASH_FILE), &format!("panic: {info}\n"));
        previous_hook(info);
    }));
    #[cfg(unix)]
    signals::install(&dir.join(CRASH_FILE));
    report
}

fn read_previous(dir: &Path) -> Option<String> {
    let phase = fs::read_to_string(dir.join(PHASE_FILE))
        .ok()
        .map(|p| p.trim().to_owned());
    let crash = fs::read_to_string(dir.join(CRASH_FILE))
        .ok()
        .map(|c| c.trim().to_owned());
    let unfinished = phase.filter(|p| !p.is_empty() && p != IDLE);
    if unfinished.is_none() && crash.as_deref().is_none_or(str::is_empty) {
        return None;
    }
    let mut report = String::from("The app closed unexpectedly last time.");
    if let Some(phase) = unfinished {
        report.push_str(&format!("\nWhile: {phase}"));
    }
    match crash.filter(|c| !c.is_empty()) {
        Some(crash) => report.push_str(&format!("\nDetails: {crash}")),
        None => report.push_str(
            "\nNo crash note was recorded, so the system probably stopped the app (often because it ran out of memory).",
        ),
    }
    Some(report)
}

fn append(path: &Path, text: &str) {
    use std::io::Write;
    if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(text.as_bytes());
    }
}

fn set_phase(text: &str) {
    if let Some(dir) = DIR.get() {
        let _ = fs::write(dir.join(PHASE_FILE), text);
    }
}

/// Records what the app is about to do. Cheap; call at phase changes, not per token.
pub fn phase(text: impl AsRef<str>) {
    set_phase(&format!("{} [{}]", text.as_ref(), memory_summary()));
}

/// Marks the end of a risky phase.
pub fn idle() {
    set_phase(IDLE);
}

/// "RAM available 1830 MB of 5600 MB", or a note when it cannot be read.
pub fn memory_summary() -> String {
    fn kb(meminfo: &str, key: &str) -> Option<u64> {
        meminfo.lines().find_map(|line| {
            line.strip_prefix(key)?
                .trim()
                .strip_suffix("kB")?
                .trim()
                .parse()
                .ok()
        })
    }
    match fs::read_to_string("/proc/meminfo") {
        Ok(info) => match (kb(&info, "MemAvailable:"), kb(&info, "MemTotal:")) {
            (Some(avail), Some(total)) => {
                format!("RAM available {} MB of {} MB", avail / 1024, total / 1024)
            }
            _ => "RAM unknown".to_owned(),
        },
        Err(_) => "RAM unknown".to_owned(),
    }
}

/// Starts a thread with a large stack (the model code is stack-hungry), falling back to
/// smaller sizes if the system refuses, and gives it a signal stack so crashes get recorded.
pub fn spawn_big<F: FnOnce() + Send + 'static>(name: &str, work: F) -> io::Result<()> {
    use std::sync::{Arc, Mutex};
    // The closure must survive a failed attempt, so keep it in a shared slot.
    let slot = Arc::new(Mutex::new(Some(work)));
    let mut last_err = None;
    for megabytes in [128usize, 16, 2] {
        let slot = Arc::clone(&slot);
        let attempt = std::thread::Builder::new()
            .name(name.to_owned())
            .stack_size(megabytes << 20)
            .spawn(move || {
                install_thread_altstack();
                crate::local::cpu::prefer_fast_cores();
                if let Some(work) = slot.lock().ok().and_then(|mut s| s.take()) {
                    work();
                }
            });
        match attempt {
            Ok(_) => return Ok(()),
            Err(err) => last_err = Some(err),
        }
    }
    Err(last_err.unwrap_or_else(|| io::Error::other("could not start a thread")))
}

/// Gives the calling thread an alternate signal stack, so a stack overflow can still be recorded.
pub fn install_thread_altstack() {
    #[cfg(unix)]
    signals::install_altstack();
}

#[cfg(unix)]
mod signals {
    use std::{
        ffi::{c_int, c_void},
        os::unix::io::IntoRawFd,
        path::Path,
        sync::atomic::{AtomicI32, Ordering},
    };

    static FD: AtomicI32 = AtomicI32::new(-1);
    const ALTSTACK_BYTES: usize = 64 * 1024;

    pub fn install(crash_file: &Path) {
        let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(crash_file)
        else {
            return;
        };
        FD.store(file.into_raw_fd(), Ordering::Relaxed);
        install_altstack();
        for signal in [
            libc::SIGSEGV,
            libc::SIGBUS,
            libc::SIGILL,
            libc::SIGFPE,
            libc::SIGABRT,
        ] {
            // SAFETY: plain sigaction setup with a handler that only calls async-signal-safe functions.
            unsafe {
                let mut action: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = handler as *const () as usize;
                // SA_RESETHAND restores the default action first, so after we have written
                // our note the original crash proceeds as usual.
                action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_RESETHAND;
                libc::sigemptyset(&mut action.sa_mask);
                libc::sigaction(signal, &action, std::ptr::null_mut());
            }
        }
    }

    pub fn install_altstack() {
        // The buffer is leaked on purpose: it must outlive the thread's use of it.
        let stack = Box::leak(vec![0u8; ALTSTACK_BYTES].into_boxed_slice());
        let new = libc::stack_t {
            ss_sp: stack.as_mut_ptr().cast::<c_void>(),
            ss_flags: 0,
            ss_size: ALTSTACK_BYTES,
        };
        // SAFETY: `new` points at a valid, leaked buffer.
        unsafe {
            libc::sigaltstack(&new, std::ptr::null_mut());
        }
    }

    fn name(signal: c_int) -> &'static [u8] {
        match signal {
            libc::SIGSEGV => b"SIGSEGV (invalid memory access or stack overflow)",
            libc::SIGBUS => b"SIGBUS (bad memory access)",
            libc::SIGILL => b"SIGILL (CPU instruction not supported)",
            libc::SIGFPE => b"SIGFPE (arithmetic error)",
            libc::SIGABRT => b"SIGABRT (abort, e.g. memory allocation failed)",
            _ => b"unknown signal",
        }
    }

    /// Only async-signal-safe operations here: a stack buffer and `write`.
    extern "C" fn handler(signal: c_int, info: *mut libc::siginfo_t, _context: *mut c_void) {
        let mut buf = [0u8; 160];
        let mut len = 0;
        let mut push = |bytes: &[u8]| {
            let n = bytes.len().min(buf.len() - len);
            buf[len..len + n].copy_from_slice(&bytes[..n]);
            len += n;
        };
        push(b"fatal signal: ");
        push(name(signal));
        if !info.is_null() && (signal == libc::SIGSEGV || signal == libc::SIGBUS) {
            // SAFETY: the kernel passes a valid siginfo for SA_SIGINFO handlers.
            let address = unsafe { (*info).si_addr() } as usize;
            push(b", address 0x");
            let mut digits = [0u8; 16];
            let mut count = 0;
            let mut value = address;
            loop {
                digits[count] = b"0123456789abcdef"[value & 0xf];
                count += 1;
                value >>= 4;
                if value == 0 {
                    break;
                }
            }
            while count > 0 {
                count -= 1;
                push(&digits[count..=count]);
            }
        }
        push(b"\n");
        let fd = FD.load(Ordering::Relaxed);
        if fd >= 0 {
            // SAFETY: `buf` is valid for `len` bytes.
            unsafe {
                libc::write(fd, buf.as_ptr().cast::<c_void>(), len);
            }
        }
        // Returning would re-run the faulting instruction and loop forever, so hand the
        // signal back to the system: the re-run then ends the process.
        // SAFETY: `signal` is async-signal-safe.
        unsafe {
            libc::signal(signal, libc::SIG_DFL);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_previous_session_reports_nothing() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_previous(dir.path()), None);
        fs::write(dir.path().join(PHASE_FILE), IDLE).unwrap();
        assert_eq!(read_previous(dir.path()), None);
    }

    #[test]
    fn an_unfinished_phase_is_reported_with_a_hint_about_memory() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join(PHASE_FILE),
            "loading model x [RAM available 900 MB of 3000 MB]",
        )
        .unwrap();
        let report = read_previous(dir.path()).unwrap();
        assert!(report.contains("While: loading model x"));
        assert!(report.contains("out of memory"));
    }

    #[test]
    fn crash_notes_are_included() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(PHASE_FILE), "generating").unwrap();
        fs::write(
            dir.path().join(CRASH_FILE),
            "fatal signal: SIGILL (CPU instruction not supported)\n",
        )
        .unwrap();
        let report = read_previous(dir.path()).unwrap();
        assert!(report.contains("Details: fatal signal: SIGILL"));
        assert!(!report.contains("out of memory"));
    }

    #[test]
    fn memory_summary_never_panics() {
        assert!(memory_summary().starts_with("RAM"));
    }

    #[test]
    fn big_threads_run_the_work_and_survive_deep_recursion() {
        fn depth(n: u32) -> u32 {
            let pad = [n as u8; 256];
            if n == 0 {
                u32::from(pad[0])
            } else {
                depth(n - 1) + u32::from(pad[0] & 1)
            }
        }
        let (tx, rx) = std::sync::mpsc::channel();
        spawn_big("test-big", move || {
            // ~20 MB of stack: would overflow a default 2 MB thread.
            tx.send(depth(60_000)).unwrap();
        })
        .unwrap();
        assert!(rx.recv_timeout(std::time::Duration::from_secs(20)).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn a_fatal_signal_leaves_a_note_and_the_next_start_reports_it() {
        // Re-run this test as a child process that crashes on purpose.
        if let Ok(dir) = std::env::var("ZN_DIAG_CRASH_DIR") {
            init(Path::new(&dir));
            phase("pretend work");
            // SAFETY: deliberately raising a signal. The first delivery runs our handler
            // (which resets the action to default), the second one kills the process.
            unsafe {
                libc::raise(libc::SIGSEGV);
                libc::raise(libc::SIGSEGV);
            }
            return;
        }
        use std::os::unix::process::ExitStatusExt;
        let dir = tempfile::tempdir().unwrap();
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "diag::tests::a_fatal_signal_leaves_a_note_and_the_next_start_reports_it",
                "--nocapture",
            ])
            .env("ZN_DIAG_CRASH_DIR", dir.path())
            .status()
            .unwrap();
        assert_eq!(status.signal(), Some(libc::SIGSEGV));
        let report = read_previous(&dir.path().join("diag")).expect("a report");
        assert!(report.contains("While: pretend work"), "{report}");
        assert!(report.contains("fatal signal: SIGSEGV"), "{report}");
    }
}
