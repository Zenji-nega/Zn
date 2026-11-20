//! Runs a quantized GGUF model on the CPU with candle (pure Rust).
//!
//! Supported architectures: `qwen2` (Qwen2.5, Qwen2.5-Coder) and `qwen3`.
//! The engine keeps the attention cache between calls, so in the agent's
//! tool loop only the new part of the conversation has to be processed.

use std::{
    fs::File,
    io::BufReader,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use candle_core::{Device, Tensor, quantized::gguf_file};
use candle_transformers::{
    generation::LogitsProcessor,
    models::{quantized_qwen2, quantized_qwen3},
};
use tokenizers::Tokenizer;

use super::prompt::Template;

/// Prompt tokens processed per forward pass; bounds peak memory and keeps Stop responsive.
const PREFILL_CHUNK: usize = 256;
const CANCELLED: &str = "cancelled";

#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Longest prompt (plus reply) accepted, in tokens.
    pub n_ctx: usize,
    pub max_new_tokens: usize,
    /// 0 means greedy decoding.
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            n_ctx: 8192,
            max_new_tokens: 4096,
            temperature: 0.3,
            top_p: 0.95,
            seed: 42,
        }
    }
}

/// What the engine is doing, readable from the UI thread.
#[derive(Default)]
pub struct Progress {
    /// 0 idle, 1 loading, 2 reading the prompt, 3 writing.
    pub phase: AtomicUsize,
    pub done: AtomicUsize,
    pub total: AtomicUsize,
}

impl Progress {
    pub const IDLE: usize = 0;
    pub const LOADING: usize = 1;
    pub const READING: usize = 2;
    pub const WRITING: usize = 3;

    fn set(&self, phase: usize, done: usize, total: usize) {
        self.phase.store(phase, Ordering::Relaxed);
        self.done.store(done, Ordering::Relaxed);
        self.total.store(total, Ordering::Relaxed);
    }

    /// A short status line, or `None` when idle.
    pub fn describe(&self) -> Option<String> {
        let done = self.done.load(Ordering::Relaxed);
        let total = self.total.load(Ordering::Relaxed);
        match self.phase.load(Ordering::Relaxed) {
            Self::LOADING => Some("Loading the model…".to_owned()),
            Self::READING => Some(format!("Reading the conversation… {done}/{total}")),
            Self::WRITING => Some(format!("Writing… {done} tokens")),
            _ => None,
        }
    }
}

enum Weights {
    Qwen2(quantized_qwen2::ModelWeights),
    Qwen3(quantized_qwen3::ModelWeights),
}

impl Weights {
    fn forward(&mut self, input: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Qwen2(m) => m.forward(input, pos),
            Self::Qwen3(m) => m.forward(input, pos),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::Qwen2(m) => m.clear_kv_cache(),
            Self::Qwen3(m) => m.clear_kv_cache(),
        }
    }
}

pub struct Engine {
    weights: Weights,
    tokenizer: Tokenizer,
    template: Template,
    device: Device,
    config: EngineConfig,
    stop_ids: Vec<u32>,
    /// Exactly the tokens whose attention state sits in the model's cache.
    cached: Vec<u32>,
    last_reused: usize,
    progress: Arc<Progress>,
}

impl Engine {
    pub fn load(
        gguf: &Path,
        tokenizer: &Path,
        config: EngineConfig,
        progress: Arc<Progress>,
    ) -> Result<Self, String> {
        progress.set(Progress::LOADING, 0, 0);
        let result = Self::load_inner(gguf, tokenizer, config, Arc::clone(&progress));
        if result.is_err() {
            progress.set(Progress::IDLE, 0, 0);
        }
        result
    }

    fn load_inner(
        gguf: &Path,
        tokenizer: &Path,
        config: EngineConfig,
        progress: Arc<Progress>,
    ) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_file(tokenizer)
            .map_err(|e| format!("could not read the tokenizer: {e}"))?;
        let mut stop_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tokenizer.token_to_id(t))
            .collect();
        stop_ids.dedup();
        if tokenizer.token_to_id("<|im_end|>").is_none() {
            return Err("this model does not use the ChatML format (<|im_end|> is missing)".into());
        }

        let file = File::open(gguf).map_err(|e| format!("could not open the model: {e}"))?;
        let mut reader = BufReader::with_capacity(1 << 20, file);
        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| format!("not a valid GGUF file: {e}"))?;
        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        let device = Device::Cpu;
        let (weights, template) = match arch.as_str() {
            "qwen2" => (
                Weights::Qwen2(
                    quantized_qwen2::ModelWeights::from_gguf(content, &mut reader, &device)
                        .map_err(|e| format!("could not load the model: {e}"))?,
                ),
                Template::Qwen2,
            ),
            "qwen3" => (
                Weights::Qwen3(
                    quantized_qwen3::ModelWeights::from_gguf(content, &mut reader, &device)
                        .map_err(|e| format!("could not load the model: {e}"))?,
                ),
                Template::Qwen3,
            ),
            other => {
                return Err(format!(
                    "unsupported model architecture \"{other}\" (supported: qwen2, qwen3)"
                ));
            }
        };
        progress.set(Progress::IDLE, 0, 0);
        Ok(Self {
            weights,
            tokenizer,
            template,
            device,
            config,
            stop_ids,
            cached: Vec::new(),
            last_reused: 0,
            progress,
        })
    }

    pub fn template(&self) -> Template {
        self.template
    }

    /// How many prompt tokens the last call took from the cache instead of recomputing.
    pub fn last_reused(&self) -> usize {
        self.last_reused
    }

    /// Number of tokens currently held in the attention cache.
    pub fn cached_tokens(&self) -> usize {
        self.cached.len()
    }

    /// Continues `prompt` and returns the raw reply text (without the stop token).
    pub fn generate(&mut self, prompt: &str, cancel: &AtomicBool) -> Result<String, String> {
        let result = self.generate_inner(prompt, cancel);
        self.progress.set(Progress::IDLE, 0, 0);
        if result.is_err() {
            // The cache may be half-updated; start clean next time.
            self.cached.clear();
        }
        result
    }

    fn generate_inner(&mut self, prompt: &str, cancel: &AtomicBool) -> Result<String, String> {
        let err = |e: candle_core::Error| e.to_string();
        let tokens = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| format!("tokenizer error: {e}"))?
            .get_ids()
            .to_vec();
        if tokens.len() + 64 > self.config.n_ctx {
            return Err(format!(
                "The conversation is too long for the on-device model ({} tokens, limit {}). Start a new chat.",
                tokens.len(),
                self.config.n_ctx
            ));
        }

        // Reuse the cache only when the new prompt extends exactly what is cached.
        let reuse = !self.cached.is_empty()
            && tokens.len() > self.cached.len()
            && tokens[..self.cached.len()] == self.cached[..];
        let start = if reuse {
            self.cached.len()
        } else {
            self.weights.clear();
            self.cached.clear();
            0
        };

        self.last_reused = start;
        let mut pos = start;
        let mut logits = None;
        for chunk in tokens[start..].chunks(PREFILL_CHUNK) {
            if cancel.load(Ordering::Relaxed) {
                return Err(CANCELLED.into());
            }
            self.progress
                .set(Progress::READING, pos - start, tokens.len() - start);
            let input = Tensor::new(chunk, &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(err)?;
            logits = Some(self.weights.forward(&input, pos).map_err(err)?);
            pos += chunk.len();
            self.cached.extend_from_slice(chunk);
        }
        let mut logits = logits.ok_or("empty prompt")?;

        let temperature = (self.config.temperature > 0.0).then_some(self.config.temperature);
        let top_p = temperature.map(|_| self.config.top_p);
        let mut sampler = LogitsProcessor::new(self.config.seed, temperature, top_p);

        let mut generated: Vec<u32> = Vec::new();
        let budget = self
            .config
            .max_new_tokens
            .min(self.config.n_ctx - tokens.len());
        for step in 0..budget {
            if cancel.load(Ordering::Relaxed) {
                return Err(CANCELLED.into());
            }
            let next = sampler
                .sample(&logits.squeeze(0).map_err(err)?)
                .map_err(err)?;
            if self.stop_ids.contains(&next) {
                break;
            }
            generated.push(next);
            self.progress.set(Progress::WRITING, generated.len(), 0);
            if step + 1 == budget {
                break;
            }
            let input = Tensor::new(&[next], &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(err)?;
            logits = self.weights.forward(&input, pos).map_err(err)?;
            pos += 1;
            self.cached.push(next);
        }

        self.tokenizer
            .decode(&generated, false)
            .map_err(|e| format!("tokenizer error: {e}"))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{collections::HashMap, io::Write};

    use candle_core::quantized::{GgmlDType, QTensor, gguf_file::Value};
    use tokenizers::{
        AddedToken, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
    };

    use super::*;

    const EMBED: usize = 16;
    const HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = EMBED / HEADS;
    const FFN: usize = 32;
    const LAYERS: usize = 2;
    const WORDS: [&str; 12] = [
        "hello",
        "world",
        "user",
        "assistant",
        "system",
        "make",
        "a",
        "page",
        "tool",
        "ok",
        "yes",
        "no",
    ];

    fn vocab() -> HashMap<String, u32> {
        let mut vocab: HashMap<String, u32> =
            ["<unk>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
                .iter()
                .enumerate()
                .map(|(i, t)| ((*t).to_owned(), i as u32))
                .collect();
        for (i, word) in WORDS.iter().enumerate() {
            vocab.insert((*word).to_owned(), 4 + i as u32);
        }
        vocab
    }

    /// Writes a tiny random tokenizer + qwen2 GGUF that exercise the whole code path.
    pub(crate) fn tiny_model(dir: &Path, arch: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let vocab = vocab();
        let model = WordLevel::builder()
            .vocab(vocab.iter().map(|(k, v)| (k.clone(), *v)).collect())
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));
        let specials: Vec<AddedToken> = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
            .iter()
            .map(|t| AddedToken::from(*t, true))
            .collect();
        tokenizer.add_special_tokens(specials).unwrap();
        let tokenizer_path = dir.join("tokenizer.json");
        tokenizer.save(&tokenizer_path, false).unwrap();

        let dev = Device::Cpu;
        // Deterministic pseudo-random weights, so tests are reproducible.
        let state = std::cell::Cell::new(0x2545_F491_4F6C_DD1Du64);
        let fill = |len: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    let next = state
                        .get()
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    state.set(next);
                    ((next >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
                })
                .collect()
        };
        let quant = |values: Vec<f32>, shape: &[usize]| {
            let t = Tensor::from_vec(values, shape, &dev).unwrap();
            QTensor::quantize(&t, GgmlDType::F32).unwrap()
        };
        let rand = |rows: usize, cols: usize| quant(fill(rows * cols, 0.8), &[rows, cols]);
        let rand1 = |n: usize| quant(fill(n, 0.2), &[n]);
        let ones = |n: usize| {
            QTensor::quantize(
                &Tensor::ones(n, candle_core::DType::F32, &dev).unwrap(),
                GgmlDType::F32,
            )
            .unwrap()
        };

        let kv_dim = KV_HEADS * HEAD_DIM;
        let mut tensors: Vec<(String, QTensor)> = vec![
            ("token_embd.weight".into(), rand(vocab.len(), EMBED)),
            ("output_norm.weight".into(), ones(EMBED)),
        ];
        for i in 0..LAYERS {
            let p = format!("blk.{i}");
            tensors.push((format!("{p}.attn_q.weight"), rand(EMBED, EMBED)));
            tensors.push((format!("{p}.attn_k.weight"), rand(kv_dim, EMBED)));
            tensors.push((format!("{p}.attn_v.weight"), rand(kv_dim, EMBED)));
            tensors.push((format!("{p}.attn_q.bias"), rand1(EMBED)));
            tensors.push((format!("{p}.attn_k.bias"), rand1(kv_dim)));
            tensors.push((format!("{p}.attn_v.bias"), rand1(kv_dim)));
            tensors.push((format!("{p}.attn_output.weight"), rand(EMBED, EMBED)));
            tensors.push((format!("{p}.attn_norm.weight"), ones(EMBED)));
            tensors.push((format!("{p}.ffn_norm.weight"), ones(EMBED)));
            tensors.push((format!("{p}.ffn_gate.weight"), rand(FFN, EMBED)));
            tensors.push((format!("{p}.ffn_up.weight"), rand(FFN, EMBED)));
            tensors.push((format!("{p}.ffn_down.weight"), rand(EMBED, FFN)));
        }

        let u32v = |v: usize| Value::U32(v as u32);
        let meta: Vec<(String, Value)> = vec![
            ("general.architecture".into(), Value::String(arch.into())),
            (format!("{arch}.attention.head_count"), u32v(HEADS)),
            (format!("{arch}.attention.head_count_kv"), u32v(KV_HEADS)),
            (format!("{arch}.embedding_length"), u32v(EMBED)),
            (format!("{arch}.context_length"), u32v(256)),
            (format!("{arch}.block_count"), u32v(LAYERS)),
            (
                format!("{arch}.attention.layer_norm_rms_epsilon"),
                Value::F32(1e-6),
            ),
        ];
        let meta_refs: Vec<(&str, &Value)> = meta.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let tensor_refs: Vec<(&str, &QTensor)> =
            tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let gguf_path = dir.join("model.gguf");
        let mut file = File::create(&gguf_path).unwrap();
        gguf_file::write(&mut file, &meta_refs, &tensor_refs).unwrap();
        file.flush().unwrap();
        (gguf_path, tokenizer_path)
    }

    fn greedy(max_new_tokens: usize, n_ctx: usize) -> EngineConfig {
        EngineConfig {
            n_ctx,
            max_new_tokens,
            temperature: 0.0,
            top_p: 1.0,
            seed: 1,
        }
    }

    /// A tiny model on disk plus a way to load fresh engines from it.
    struct Fixture {
        _dir: tempfile::TempDir,
        gguf: std::path::PathBuf,
        tokenizer: std::path::PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let (gguf, tokenizer) = tiny_model(dir.path(), "qwen2");
            Self {
                _dir: dir,
                gguf,
                tokenizer,
            }
        }

        fn engine(&self, config: EngineConfig) -> Engine {
            Engine::load(&self.gguf, &self.tokenizer, config, Arc::default()).unwrap()
        }
    }

    static NEVER: AtomicBool = AtomicBool::new(false);

    #[test]
    fn loads_and_generates_deterministically() {
        let fx = Fixture::new();
        let mut a = fx.engine(greedy(6, 512));
        let prompt = "<|im_start|>user\nhello world<|im_end|>\n<|im_start|>assistant\n";
        let first = a.generate(prompt, &NEVER).unwrap();
        assert!(!first.is_empty());
        assert_eq!(
            fx.engine(greedy(6, 512)).generate(prompt, &NEVER).unwrap(),
            first
        );
        assert_eq!(a.template(), Template::Qwen2);
    }

    #[test]
    fn continuing_a_conversation_reuses_the_cache_without_changing_the_result() {
        let fx = Fixture::new();
        let mut warm = fx.engine(greedy(5, 512));
        let first = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n";
        let reply = warm.generate(first, &NEVER).unwrap();
        let after_first = warm.cached_tokens();
        assert!(after_first > 0);

        // Second prompt = first prompt + the reply the model gave + a new user turn.
        let second = format!(
            "{first}{reply}<|im_end|>\n<|im_start|>user\nworld<|im_end|>\n<|im_start|>assistant\n"
        );
        let warm_out = warm.generate(&second, &NEVER).unwrap();
        assert!(
            warm.last_reused() > 0,
            "the cache should have been reused (reply: {reply:?})"
        );

        let cold_out = fx.engine(greedy(5, 512)).generate(&second, &NEVER).unwrap();
        assert_eq!(warm_out, cold_out, "cache reuse must not change the output");
    }

    #[test]
    fn an_unrelated_prompt_resets_the_cache() {
        let fx = Fixture::new();
        let mut warm = fx.engine(greedy(4, 512));
        warm.generate(
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
            &NEVER,
        )
        .unwrap();
        let other = "<|im_start|>user\nworld yes<|im_end|>\n<|im_start|>assistant\n";
        let warm_out = warm.generate(other, &NEVER).unwrap();
        assert_eq!(warm.last_reused(), 0);
        assert_eq!(
            warm_out,
            fx.engine(greedy(4, 512)).generate(other, &NEVER).unwrap()
        );
    }

    #[test]
    fn too_long_prompts_are_rejected_with_advice() {
        let mut engine = Fixture::new().engine(greedy(4, 80));
        let long = "hello ".repeat(100);
        let err = engine.generate(&long, &NEVER).unwrap_err();
        assert!(
            err.contains("too long") && err.contains("new chat"),
            "{err}"
        );
    }

    #[test]
    fn cancel_stops_generation_and_leaves_a_usable_engine() {
        let mut engine = Fixture::new().engine(greedy(8, 512));
        let prompt = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n";
        let cancelled = AtomicBool::new(true);
        assert_eq!(engine.generate(prompt, &cancelled).unwrap_err(), CANCELLED);
        assert_eq!(engine.cached_tokens(), 0);
        assert!(engine.generate(prompt, &NEVER).is_ok());
    }

    #[test]
    fn unsupported_architectures_are_named_in_the_error() {
        let dir = tempfile::tempdir().unwrap();
        let (gguf, tok) = tiny_model(dir.path(), "llama");
        let err = Engine::load(&gguf, &tok, greedy(1, 64), Arc::default())
            .err()
            .unwrap();
        assert!(
            err.contains("unsupported model architecture \"llama\""),
            "{err}"
        );
    }

    #[test]
    fn bad_files_give_readable_errors() {
        let dir = tempfile::tempdir().unwrap();
        let (gguf, tok) = tiny_model(dir.path(), "qwen2");
        std::fs::write(dir.path().join("junk.gguf"), b"not a model").unwrap();
        let junk = Engine::load(
            &dir.path().join("junk.gguf"),
            &tok,
            greedy(1, 64),
            Arc::default(),
        )
        .err()
        .unwrap();
        assert!(junk.contains("not a valid GGUF"), "{junk}");
        let missing = Engine::load(
            &gguf,
            &dir.path().join("nope.json"),
            greedy(1, 64),
            Arc::default(),
        )
        .err()
        .unwrap();
        assert!(missing.contains("tokenizer"), "{missing}");
    }

    #[test]
    fn progress_describes_each_phase() {
        let progress = Progress::default();
        assert_eq!(progress.describe(), None);
        progress.set(Progress::READING, 10, 40);
        assert_eq!(
            progress.describe().unwrap(),
            "Reading the conversation… 10/40"
        );
        progress.set(Progress::WRITING, 7, 0);
        assert_eq!(progress.describe().unwrap(), "Writing… 7 tokens");
    }
}
