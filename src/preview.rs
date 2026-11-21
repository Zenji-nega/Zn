//! A tiny local web server so a website the agent built can be opened in the browser.
//!
//! It listens on `127.0.0.1` only, serves files from the open project's folder (never
//! outside it, never dotfiles) and tells the browser not to cache, so a refresh after
//! the agent edits a file shows the new version.

use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

pub struct PreviewServer {
    port: u16,
    root: Arc<Mutex<PathBuf>>,
}

impl PreviewServer {
    /// Starts listening on a free port. The server thread lives as long as the process.
    pub fn start(root: PathBuf) -> io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let root = Arc::new(Mutex::new(root));
        let shared = Arc::clone(&root);
        std::thread::Builder::new()
            .name("zn-preview".into())
            .spawn(move || {
                for stream in listener.incoming().flatten() {
                    let root = shared.lock().map(|r| r.clone()).unwrap_or_default();
                    let _ = std::thread::Builder::new()
                        .name("zn-preview-conn".into())
                        .spawn(move || {
                            let _ = handle(stream, &root);
                        });
                }
            })?;
        Ok(Self { port, root })
    }

    pub fn set_root(&self, root: PathBuf) {
        if let Ok(mut current) = self.root.lock() {
            *current = root;
        }
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/", self.port)
    }
}

fn percent_decode(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = text.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// Maps a request target to a file under `root`, or `None` if it would escape or hide.
pub fn resolve(root: &Path, target: &str) -> Option<PathBuf> {
    let path = target.split(['?', '#']).next()?;
    let decoded = percent_decode(path)?;
    let mut file = root.to_path_buf();
    for part in decoded.split('/').filter(|p| !p.is_empty()) {
        if part == ".." || part.starts_with('.') || part.contains(['\\', '\0']) {
            return None;
        }
        file.push(part);
    }
    if file.is_dir() {
        file.push("index.html");
    }
    // Symlinks inside the project must not lead outside it either.
    let real = file.canonicalize().ok()?;
    real.starts_with(root.canonicalize().ok()?).then_some(real)
}

pub fn content_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("html" | "htm") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js" | "mjs") => "text/javascript; charset=utf-8",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("ico") => "image/x-icon",
        Some("wasm") => "application/wasm",
        Some("txt" | "md") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

fn respond(
    stream: &mut TcpStream,
    status: &str,
    kind: &str,
    body: &[u8],
    head_only: bool,
) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    if !head_only {
        stream.write_all(body)?;
    }
    stream.flush()
}

fn handle(mut stream: TcpStream, root: &Path) -> io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    let mut reader = BufReader::new(stream.try_clone()?.take(16 * 1024));
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    // Skip the headers.
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 || line == "\r\n" || line == "\n" {
            break;
        }
    }
    let mut parts = request_line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or("/"));
    let head_only = method == "HEAD";
    if method != "GET" && !head_only {
        return respond(
            &mut stream,
            "405 Method Not Allowed",
            "text/plain",
            b"Only GET is supported",
            false,
        );
    }
    match resolve(root, target).and_then(|f| std::fs::read(&f).ok().map(|b| (f, b))) {
        Some((file, body)) => respond(&mut stream, "200 OK", content_type(&file), &body, head_only),
        None => respond(
            &mut stream,
            "404 Not Found",
            "text/plain; charset=utf-8",
            b"Not found. Ask the agent to create index.html.",
            head_only,
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::*;

    fn project() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("index.html"), "<h1>hi</h1>").unwrap();
        std::fs::create_dir(dir.path().join("css")).unwrap();
        std::fs::write(dir.path().join("css/site.css"), "body{}").unwrap();
        std::fs::write(dir.path().join(".secret"), "x").unwrap();
        dir
    }

    fn get(port: u16, request: &str) -> String {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        let mut out = String::new();
        stream.read_to_string(&mut out).unwrap();
        out
    }

    #[test]
    fn resolves_files_and_the_default_page() {
        let dir = project();
        let root = dir.path();
        assert!(resolve(root, "/").unwrap().ends_with("index.html"));
        assert!(
            resolve(root, "/css/site.css?v=2")
                .unwrap()
                .ends_with("site.css")
        );
        assert!(resolve(root, "/missing.html").is_none());
    }

    #[test]
    fn refuses_escapes_and_hidden_files() {
        let dir = project();
        let root = dir.path();
        assert!(resolve(root, "/../etc/passwd").is_none());
        assert!(resolve(root, "/%2e%2e/x").is_none());
        assert!(resolve(root, "/.secret").is_none());
        assert!(resolve(root, "/css/..%2f..%2fx").is_none());
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_out_of_the_project_is_refused() {
        let dir = project();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("a.txt"), "secret").unwrap();
        std::os::unix::fs::symlink(outside.path(), dir.path().join("link")).unwrap();
        assert!(resolve(dir.path(), "/link/a.txt").is_none());
    }

    #[test]
    fn content_types() {
        assert_eq!(
            content_type(Path::new("a.html")),
            "text/html; charset=utf-8"
        );
        assert_eq!(
            content_type(Path::new("a.JS")),
            "text/javascript; charset=utf-8"
        );
        assert_eq!(content_type(Path::new("a.bin")), "application/octet-stream");
    }

    #[test]
    fn serves_over_http_and_follows_the_root() {
        let dir = project();
        let server = PreviewServer::start(dir.path().to_path_buf()).unwrap();
        let port = server.port;
        let page = get(port, "GET / HTTP/1.1\r\nHost: x\r\n\r\n");
        assert!(page.starts_with("HTTP/1.1 200 OK"));
        assert!(page.contains("no-store"));
        assert!(page.ends_with("<h1>hi</h1>"));
        assert!(get(port, "GET /nope HTTP/1.1\r\n\r\n").starts_with("HTTP/1.1 404"));
        assert!(get(port, "POST / HTTP/1.1\r\n\r\n").starts_with("HTTP/1.1 405"));
        assert!(get(port, "GET /.secret HTTP/1.1\r\n\r\n").starts_with("HTTP/1.1 404"));

        let other = tempfile::tempdir().unwrap();
        std::fs::write(other.path().join("index.html"), "second").unwrap();
        server.set_root(other.path().to_path_buf());
        assert!(get(port, "GET / HTTP/1.1\r\n\r\n").ends_with("second"));
    }
}
