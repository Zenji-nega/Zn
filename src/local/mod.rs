//! On-device models: run open-source LLMs directly on the phone, offline.

pub mod engine;
pub mod models;
pub mod prompt;
mod qwen2;

use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use crate::llm::{Completion, Message, Provider, ToolSpec};
use engine::{Engine, EngineConfig, Progress};
use models::ModelEntry;

/// Sizes the CPU thread pool used by the model code. Call once, before any model runs.
///
/// Using every core would starve the UI thread (and the slow cores drag the fast ones),
/// so by default only half of them are used.
pub fn configure_threads(requested: usize) {
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let threads = if requested > 0 {
        requested.min(cores)
    } else {
        (cores / 2).max(2).min(cores)
    };
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .stack_size(8 << 20)
        .start_handler(|_| crate::diag::install_thread_altstack())
        .build_global();
}

/// Holds the loaded model so it survives between requests (loading takes seconds
/// and the weights take hundreds of megabytes of memory).
#[derive(Default)]
pub struct LocalHost {
    slot: Mutex<Option<(PathBuf, Engine)>>,
    pub progress: Arc<Progress>,
}

impl LocalHost {
    pub fn new() -> Arc<Self> {
        Arc::default()
    }

    /// Frees the model's memory. It is reloaded on next use.
    pub fn unload(&self) {
        if let Ok(mut slot) = self.slot.lock() {
            *slot = None;
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.slot
            .try_lock()
            .map(|slot| slot.is_some())
            .unwrap_or(true)
    }
}

pub struct LocalProvider {
    pub host: Arc<LocalHost>,
    pub model: ModelEntry,
    pub config: EngineConfig,
    cancel: Arc<AtomicBool>,
    next_call_id: AtomicUsize,
}

impl LocalProvider {
    pub fn new(host: Arc<LocalHost>, model: ModelEntry, config: EngineConfig) -> Self {
        Self {
            host,
            model,
            config,
            cancel: Arc::default(),
            next_call_id: AtomicUsize::new(0),
        }
    }
}

impl Provider for LocalProvider {
    fn name(&self) -> String {
        format!("On-device · {}", self.model.id)
    }

    fn attach_cancel(&mut self, cancel: Arc<AtomicBool>) {
        self.cancel = cancel;
    }

    fn complete(
        &self,
        system: &str,
        conversation: &[Message],
        tools: &[ToolSpec],
    ) -> Result<Completion, String> {
        let mut slot = self
            .host
            .slot
            .lock()
            .map_err(|_| "the on-device model crashed earlier; restart the app".to_owned())?;
        if !slot
            .as_ref()
            .is_some_and(|(path, _)| path == &self.model.gguf)
        {
            *slot = None; // free the old model before loading the new one
            let engine = Engine::load(
                &self.model.gguf,
                self.model.tokenizer.as_deref(),
                self.config.clone(),
                Arc::clone(&self.host.progress),
            )?;
            *slot = Some((self.model.gguf.clone(), engine));
        }
        let (_, engine) = slot.as_mut().ok_or("model not loaded")?;

        let prompt = prompt::render(engine.template(), system, conversation, tools);
        // On a stop request this returns an error; the agent loop checks its own flag
        // and reports "Stopped." instead of showing it.
        let raw = engine.generate(&prompt, &self.cancel)?;
        let mut id = self.next_call_id.load(Ordering::Relaxed);
        let completion = prompt::parse_output(&raw, &mut id);
        self.next_call_id.store(id, Ordering::Relaxed);
        Ok(completion)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::tests::sample_tools;

    fn provider(dir: &std::path::Path) -> LocalProvider {
        let (gguf, tokenizer) = engine::tests::tiny_model(dir, "qwen2");
        let model = ModelEntry {
            id: "tiny".into(),
            gguf,
            tokenizer: Some(tokenizer),
            size_bytes: 1,
            external: false,
        };
        let config = EngineConfig {
            n_ctx: 4096,
            max_new_tokens: 4,
            temperature: 0.0,
            top_p: 1.0,
            seed: 1,
        };
        LocalProvider::new(LocalHost::new(), model, config)
    }

    #[test]
    fn completes_keeps_the_model_loaded_and_can_unload() {
        let dir = tempfile::tempdir().unwrap();
        let provider = provider(dir.path());
        assert!(!provider.host.is_loaded());
        let conversation = [Message::User("hello world".into())];
        let first = provider
            .complete("system", &conversation, &sample_tools())
            .unwrap();
        assert!(provider.host.is_loaded());
        // The tiny random model writes noise, but the plumbing must hand back a completion.
        assert!(first.calls.is_empty());
        provider
            .complete("system", &conversation, &sample_tools())
            .unwrap();
        provider.host.unload();
        assert!(!provider.host.is_loaded());
        assert_eq!(provider.name(), "On-device · tiny");
    }

    #[test]
    fn a_stop_request_reaches_the_engine() {
        let dir = tempfile::tempdir().unwrap();
        let mut provider = provider(dir.path());
        let cancel = Arc::new(AtomicBool::new(true));
        provider.attach_cancel(Arc::clone(&cancel));
        let err = provider
            .complete("system", &[Message::User("hello".into())], &[])
            .unwrap_err();
        assert_eq!(err, "cancelled");
    }

    #[test]
    fn a_missing_model_file_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = provider(dir.path());
        p.model.gguf = dir.path().join("gone.gguf");
        let err = p
            .complete("s", &[Message::User("x".into())], &[])
            .unwrap_err();
        assert!(err.contains("could not open the model"), "{err}");
    }
}
