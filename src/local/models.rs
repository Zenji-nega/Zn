//! Model files on disk: listing, presets, and resumable downloads.
//!
//! Layout: `<data>/models/<id>/model.gguf` and `<data>/models/<id>/tokenizer.json`.

use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
    time::Duration,
};

use crate::project::sanitize_name;

const GGUF_FILE: &str = "model.gguf";
const TOKENIZER_FILE: &str = "tokenizer.json";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelEntry {
    pub id: String,
    pub gguf: PathBuf,
    pub tokenizer: PathBuf,
    pub size_bytes: u64,
}

pub struct Preset {
    pub id: &'static str,
    pub title: &'static str,
    pub approx_mb: u32,
    pub gguf_url: &'static str,
    pub tokenizer_url: &'static str,
}

/// Small instruction-tuned coding models that support tool calls.
pub const PRESETS: [Preset; 3] = [
    Preset {
        id: "qwen25-coder-05b",
        title: "Qwen2.5-Coder 0.5B (fastest, weakest)",
        approx_mb: 400,
        gguf_url: "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
        tokenizer_url: "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/resolve/main/tokenizer.json",
    },
    Preset {
        id: "qwen25-coder-15b",
        title: "Qwen2.5-Coder 1.5B (balanced)",
        approx_mb: 1100,
        gguf_url: "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        tokenizer_url: "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/tokenizer.json",
    },
    Preset {
        id: "qwen25-coder-3b",
        title: "Qwen2.5-Coder 3B (best, needs ~3 GB RAM)",
        approx_mb: 2100,
        gguf_url: "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        tokenizer_url: "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/tokenizer.json",
    },
];

pub fn models_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("models")
}

/// Fully downloaded models, sorted by name. Half-finished downloads are ignored.
pub fn list_models(data_dir: &Path) -> Vec<ModelEntry> {
    let Ok(entries) = fs::read_dir(models_dir(data_dir)) else {
        return Vec::new();
    };
    let mut models: Vec<ModelEntry> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let dir = entry.path();
            let gguf = dir.join(GGUF_FILE);
            let tokenizer = dir.join(TOKENIZER_FILE);
            let size = fs::metadata(&gguf).ok()?.len();
            tokenizer.is_file().then(|| ModelEntry {
                id: entry.file_name().to_string_lossy().into_owned(),
                gguf,
                tokenizer,
                size_bytes: size,
            })
        })
        .collect();
    models.sort_by(|a, b| a.id.cmp(&b.id));
    models
}

pub fn delete_model(data_dir: &Path, id: &str) -> std::io::Result<()> {
    let id = sanitize_name(id).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid model name")
    })?;
    fs::remove_dir_all(models_dir(data_dir).join(id))
}

/// Progress of a running download, readable from the UI thread.
pub struct Download {
    pub id: String,
    pub bytes: Arc<AtomicU64>,
    /// 0 while unknown.
    pub total: Arc<AtomicU64>,
    cancel: Arc<AtomicBool>,
    result: Receiver<Result<(), String>>,
}

impl Download {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// `Some` once the download has finished or failed.
    pub fn poll(&self) -> Option<Result<(), String>> {
        match self.result.try_recv() {
            Ok(result) => Some(result),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                Some(Err("the download stopped unexpectedly".into()))
            }
        }
    }
}

/// Downloads a model (tokenizer first, then the weights) on a background thread.
/// Interrupted downloads resume from the `.part` file next time.
pub fn start_download(
    data_dir: &Path,
    id: &str,
    gguf_url: &str,
    tokenizer_url: &str,
    repaint: impl Fn() + Send + 'static,
) -> Result<Download, String> {
    let id = sanitize_name(id).ok_or("Use letters, digits, '-' or '_' for the model name.")?;
    for url in [gguf_url, tokenizer_url] {
        if !(url.starts_with("https://") || url.starts_with("http://")) {
            return Err("Download links must start with https://".into());
        }
    }
    let dir = models_dir(data_dir).join(&id);
    fs::create_dir_all(&dir).map_err(|e| format!("could not create the model folder: {e}"))?;

    let bytes = Arc::new(AtomicU64::new(0));
    let total = Arc::new(AtomicU64::new(0));
    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let (gguf_url, tokenizer_url) = (gguf_url.to_owned(), tokenizer_url.to_owned());
    let (b, t, c) = (Arc::clone(&bytes), Arc::clone(&total), Arc::clone(&cancel));
    thread::spawn(move || {
        let result = download_model(&dir, &gguf_url, &tokenizer_url, &b, &t, &c);
        let _ = tx.send(result);
        repaint();
    });
    Ok(Download {
        id,
        bytes,
        total,
        cancel,
        result: rx,
    })
}

fn download_model(
    dir: &Path,
    gguf_url: &str,
    tokenizer_url: &str,
    bytes: &AtomicU64,
    total: &AtomicU64,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let tokenizer = dir.join(TOKENIZER_FILE);
    if !tokenizer.is_file() {
        fetch_to_file(
            tokenizer_url,
            &tokenizer,
            &AtomicU64::new(0),
            &AtomicU64::new(0),
            cancel,
        )?;
        if tokenizers::Tokenizer::from_file(&tokenizer).is_err() {
            let _ = fs::remove_file(&tokenizer);
            return Err("the tokenizer link did not return a valid tokenizer.json".into());
        }
    }
    let gguf = dir.join(GGUF_FILE);
    if !gguf.is_file() {
        fetch_to_file(gguf_url, &gguf, bytes, total, cancel)?;
        let mut magic = [0u8; 4];
        let valid = File::open(&gguf)
            .and_then(|mut f| f.read_exact(&mut magic))
            .is_ok()
            && &magic == b"GGUF";
        if !valid {
            let _ = fs::remove_file(&gguf);
            return Err("the model link did not return a GGUF file".into());
        }
    }
    Ok(())
}

/// Downloads `url` to `dest`, resuming from `dest.part` when the server allows it.
fn fetch_to_file(
    url: &str,
    dest: &Path,
    bytes: &AtomicU64,
    total: &AtomicU64,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let config = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .http_status_as_error(false)
        .build();
    let agent = ureq::Agent::new_with_config(config);
    let part = PathBuf::from(format!("{}.part", dest.display()));

    let mut attempts = 0;
    loop {
        attempts += 1;
        let have = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let mut request = agent.get(url);
        if have > 0 {
            request = request.header("Range", format!("bytes={have}-"));
        }
        let mut response = request.call().map_err(|e| format!("network error: {e}"))?;
        let status = response.status().as_u16();
        let header = |name: &str| {
            response
                .headers()
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned)
        };

        let (resume, size) = match status {
            200 => (
                false,
                header("content-length")
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(0),
            ),
            206 => {
                let size = header("content-range")
                    .and_then(|v| v.rsplit('/').next().and_then(|n| n.parse::<u64>().ok()))
                    .unwrap_or(0);
                (true, size)
            }
            416 if attempts == 1 => {
                // The partial file does not fit the remote file; start over.
                let _ = fs::remove_file(&part);
                continue;
            }
            other => return Err(format!("the server answered HTTP {other}")),
        };

        let mut file = if resume {
            OpenOptions::new().append(true).open(&part)
        } else {
            File::create(&part)
        }
        .map_err(|e| format!("could not write the file: {e}"))?;
        let mut written = if resume { have } else { 0 };
        total.store(size, Ordering::Relaxed);
        bytes.store(written, Ordering::Relaxed);

        let mut reader = response.body_mut().as_reader();
        let mut chunk = vec![0u8; 64 * 1024];
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            let n = reader
                .read(&mut chunk)
                .map_err(|e| format!("download interrupted: {e}"))?;
            if n == 0 {
                break;
            }
            file.write_all(&chunk[..n])
                .map_err(|e| format!("could not write the file: {e}"))?;
            written += n as u64;
            bytes.store(written, Ordering::Relaxed);
        }
        file.flush()
            .map_err(|e| format!("could not write the file: {e}"))?;
        if size > 0 && written != size {
            return Err(format!(
                "download incomplete ({written} of {size} bytes); try again to resume"
            ));
        }
        return fs::rename(&part, dest).map_err(|e| format!("could not finish the download: {e}"));
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{BufRead, BufReader},
        net::TcpListener,
    };

    use super::*;

    /// A one-file HTTP server: honours `Range`, or ignores it when `ignore_range` is set.
    fn serve(body: Vec<u8>, ignore_range: bool) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            for stream in listener.incoming().map_while(Result::ok) {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut range_start = None;
                let mut path = String::new();
                for (i, line) in reader.by_ref().lines().map_while(Result::ok).enumerate() {
                    if i == 0 {
                        path = line.split(' ').nth(1).unwrap_or("/").to_owned();
                    }
                    if line.is_empty() {
                        break;
                    }
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                        range_start = v.trim_end_matches('-').parse::<usize>().ok();
                    }
                }
                let mut out = stream;
                if path == "/missing" {
                    let _ = out.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    continue;
                }
                let len = body.len();
                let head = match range_start.filter(|_| !ignore_range) {
                    Some(start) if start >= len => {
                        let _ = out.write_all(b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                        continue;
                    }
                    Some(start) => {
                        let _ = write!(
                            out,
                            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{}/{len}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            len - 1,
                            len - start
                        );
                        let _ = out.write_all(&body[start..]);
                        continue;
                    }
                    None => format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n"
                    ),
                };
                let _ = out.write_all(head.as_bytes());
                let _ = out.write_all(&body);
            }
        });
        format!("http://{addr}")
    }

    fn sample(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    fn fetch(url: &str, dest: &Path) -> Result<(AtomicU64, AtomicU64), String> {
        let (bytes, total) = (AtomicU64::new(0), AtomicU64::new(0));
        fetch_to_file(url, dest, &bytes, &total, &AtomicBool::new(false))?;
        Ok((bytes, total))
    }

    #[test]
    fn downloads_a_file_and_reports_progress() {
        let body = sample(300_000);
        let url = serve(body.clone(), false);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("f.bin");
        let (bytes, total) = fetch(&url, &dest).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), body);
        assert_eq!(bytes.load(Ordering::Relaxed), 300_000);
        assert_eq!(total.load(Ordering::Relaxed), 300_000);
        assert!(!dir.path().join("f.bin.part").exists());
    }

    #[test]
    fn resumes_from_a_partial_file() {
        let body = sample(200_000);
        let url = serve(body.clone(), false);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("f.bin");
        fs::write(dir.path().join("f.bin.part"), &body[..70_000]).unwrap();
        fetch(&url, &dest).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), body);
    }

    #[test]
    fn restarts_when_the_server_ignores_range() {
        let body = sample(120_000);
        let url = serve(body.clone(), true);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("f.bin");
        fs::write(dir.path().join("f.bin.part"), &body[..50_000]).unwrap();
        fetch(&url, &dest).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), body);
    }

    #[test]
    fn a_stale_partial_file_larger_than_the_remote_is_discarded() {
        let body = sample(10_000);
        let url = serve(body.clone(), false);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("f.bin");
        fs::write(dir.path().join("f.bin.part"), sample(50_000)).unwrap();
        fetch(&url, &dest).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), body);
    }

    #[test]
    fn http_errors_are_reported() {
        let url = format!("{}/missing", serve(sample(10), false));
        let dir = tempfile::tempdir().unwrap();
        let err = fetch(&url, &dir.path().join("f.bin")).err().unwrap();
        assert!(err.contains("404"), "{err}");
        assert!(!dir.path().join("f.bin").exists());
    }

    #[test]
    fn cancelling_keeps_the_partial_file_for_later() {
        let url = serve(sample(500_000), false);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("f.bin");
        let cancel = AtomicBool::new(true);
        let err = fetch_to_file(&url, &dest, &AtomicU64::new(0), &AtomicU64::new(0), &cancel)
            .unwrap_err();
        assert_eq!(err, "cancelled");
        assert!(!dest.exists());
    }

    #[test]
    fn full_model_download_validates_both_files() {
        // A "model" that is not GGUF must be rejected and removed.
        let data = tempfile::tempdir().unwrap();
        let fake_gguf = serve(b"this is not a model".to_vec(), false);
        let tokenizer_json = serve(b"{}".to_vec(), false);
        let job = start_download(data.path(), "bad", &fake_gguf, &tokenizer_json, || {}).unwrap();
        let result = loop {
            if let Some(r) = job.poll() {
                break r;
            }
            thread::sleep(Duration::from_millis(10));
        };
        assert!(result.is_err());
        assert!(list_models(data.path()).is_empty());
    }

    #[test]
    fn rejects_bad_names_and_links() {
        let data = tempfile::tempdir().unwrap();
        assert!(start_download(data.path(), "../x", "https://a", "https://b", || {}).is_err());
        assert!(
            start_download(data.path(), "ok", "file:///etc/passwd", "https://b", || {}).is_err()
        );
    }

    #[test]
    fn lists_only_complete_models_and_deletes_them() {
        let data = tempfile::tempdir().unwrap();
        let root = models_dir(data.path());
        fs::create_dir_all(root.join("done")).unwrap();
        fs::write(root.join("done").join(GGUF_FILE), b"GGUFxxxx").unwrap();
        fs::write(root.join("done").join(TOKENIZER_FILE), b"{}").unwrap();
        fs::create_dir_all(root.join("partial")).unwrap();
        fs::write(root.join("partial").join("model.gguf.part"), b"x").unwrap();
        fs::create_dir_all(root.join("no-tokenizer")).unwrap();
        fs::write(root.join("no-tokenizer").join(GGUF_FILE), b"GGUF").unwrap();

        let models = list_models(data.path());
        assert_eq!(models.len(), 1);
        assert_eq!((models[0].id.as_str(), models[0].size_bytes), ("done", 8));

        assert!(delete_model(data.path(), "../done").is_err());
        delete_model(data.path(), "done").unwrap();
        assert!(list_models(data.path()).is_empty());
    }

    #[test]
    fn presets_have_unique_ids_and_valid_names() {
        let mut ids: Vec<_> = PRESETS.iter().map(|p| p.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), PRESETS.len());
        for preset in &PRESETS {
            assert_eq!(sanitize_name(preset.id).as_deref(), Some(preset.id));
            assert!(
                preset.gguf_url.ends_with(".gguf")
                    && preset.tokenizer_url.ends_with("tokenizer.json")
            );
        }
    }
}
