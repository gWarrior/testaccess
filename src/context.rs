//! LLM adapter: exact long-context memory for a recurrent language model.
//!
//! A recurrent model (Mamba/RWKV-style) keeps a fixed-size state and cannot
//! reproduce distant context verbatim. [`ContextMemory`] stores the token
//! stream (window of `max_tokens`, 300k by default) as overlapping chunks:
//!
//! * every chunk is written in one shot as an engram of the **lexical** SNN
//!   memory (n-gram coincidence detectors), and optionally of a **semantic**
//!   SNN memory keyed by the mean of the model's per-token attention keys;
//! * per-token keys/values are kept in a [`KvStore`] for the
//!   **cross-attention read head** ([`ContextMemory::read`]): the SNN picks a
//!   few chunks, attention runs exactly over their tokens only;
//! * [`locate`](ContextMemory::locate) / [`continuation`](ContextMemory::continuation)
//!   verify matches token by token, so what is returned is exact;
//! * answers are ternary: found (`Known`), maybe (`Unknown`), and provably
//!   not in the window (`Absent`, via an exact census of window n-grams).
//!
//! Chunks leave the index when they slide out of the window, except chunks
//! that were pinned (or consolidated): those carry their tokens in the
//! payload and stay recallable, e.g. across sessions.

use std::collections::{HashMap, VecDeque};

use crate::attention::{attend, Attended};
use crate::config::MemoryConfig;
use crate::encoder::{FlyHashEncoder, NGramEncoder};
use crate::kv::{KvPrecision, KvStore};
use crate::memory::{BatchOptions, RecallOptions, SnnMemory, Stats, Verdict};
use crate::persist::{self, Persist, SnapshotScope};
use crate::rng::mix64;
use crate::types::{Input, MemoryError, MemoryId, Tier};

/// Key/value storage and semantic indexing options.
#[derive(Clone, Debug)]
pub struct KvConfig {
    pub key_dim: usize,
    pub value_dim: usize,
    pub precision: KvPrecision,
    /// Also index chunks by their mean key (semantic retrieval).
    pub semantic_index: bool,
    /// Winners of the semantic spike encoder.
    pub dense_k: usize,
    /// Inputs sampled by each semantic encoder neuron.
    pub fan_in: usize,
}

impl KvConfig {
    pub fn new(key_dim: usize, value_dim: usize) -> Self {
        Self { key_dim, value_dim, precision: KvPrecision::F16, semantic_index: true, dense_k: 81, fan_in: 27 }
    }
}

/// Configuration of a [`ContextMemory`].
#[derive(Clone, Debug)]
pub struct ContextConfig {
    /// Tokens per chunk.
    pub chunk_size: usize,
    /// Tokens between chunk starts. Any fragment of up to
    /// `chunk_size - stride + 1` tokens lies entirely inside some chunk.
    pub stride: usize,
    /// Sliding window, in tokens.
    pub max_tokens: usize,
    /// Sizes of the n-gram coincidence detectors.
    pub ngrams: Vec<usize>,
    /// Neuron space of both SNN memories (at most one tryte, 3^9).
    pub n_neurons: u32,
    pub seed: u64,
    /// Chunks returned per query.
    pub top_k: usize,
    /// Extra strides of context added on each side of a retrieved chunk.
    pub neighbors: usize,
    pub kv: Option<KvConfig>,
    pub lexical: MemoryConfig,
    pub semantic: MemoryConfig,
}

impl Default for ContextConfig {
    fn default() -> Self {
        let base = MemoryConfig {
            dedupe_threshold: None,
            consolidate_after: None,
            max_candidates: 27,
            ..MemoryConfig::default()
        };
        Self {
            chunk_size: 27,
            stride: 9,
            max_tokens: 300_000,
            ngrams: vec![1, 2, 3],
            n_neurons: 19_683,
            seed: 0x5EED,
            top_k: 9,
            neighbors: 0,
            kv: None,
            lexical: MemoryConfig {
                max_ensemble: 81,
                recall_threshold: 4.0 / 9.0,
                reject_threshold: 1.0 / 9.0,
                ..base.clone()
            },
            semantic: MemoryConfig { max_ensemble: 81, ..base },
        }
    }
}

/// A stored chunk: its absolute start position and its tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub start: u64,
    pub tokens: Vec<u32>,
}

impl Persist for Chunk {
    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.start.to_le_bytes());
        self.tokens.write(out);
    }
    fn read(input: &mut &[u8]) -> Result<Self, MemoryError> {
        Ok(Chunk { start: persist::read_u64(input)?, tokens: Vec::<u32>::read(input)? })
    }
}

const CONTEXT_MAGIC: &[u8; 4] = b"SNNC";

/// What to look up.
#[derive(Clone, Copy, Debug)]
pub enum Probe<'a> {
    /// A token fragment (lexical memory).
    Tokens(&'a [u32]),
    /// A query key of `key_dim` (semantic memory).
    Key(&'a [f32]),
    /// Both, results merged.
    Both(&'a [u32], &'a [f32]),
}

/// A retrieved stretch of context.
#[derive(Clone, Debug)]
pub struct RetrievedSpan {
    pub start: u64,
    pub tokens: Vec<u32>,
    pub confidence: f32,
    /// Lexical/semantic memory ids of the chunks behind this span.
    pub chunks: Vec<MemoryId>,
    /// `false` for pinned chunks that already left the window.
    pub in_window: bool,
}

impl RetrievedSpan {
    pub fn end(&self) -> u64 {
        self.start + self.tokens.len() as u64
    }
}

#[derive(Clone, Debug)]
pub struct Retrieval {
    pub verdict: Verdict,
    pub spans: Vec<RetrievedSpan>,
}

/// Exact occurrences of a fragment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Located {
    /// `Known` if found, `Absent` if provably not in memory, else `Unknown`.
    pub verdict: Verdict,
    pub positions: Vec<u64>,
}

/// Retrieved key/value rows (see [`ContextMemory::retrieve_rows`]).
#[derive(Clone, Debug)]
pub struct MemoryRows {
    pub verdict: Verdict,
    pub confidence: f32,
    pub positions: Vec<u64>,
    /// `positions.len() × key_dim`, row-major.
    pub keys: Vec<f32>,
    /// `positions.len() × value_dim`, row-major.
    pub values: Vec<f32>,
}

impl Default for MemoryRows {
    fn default() -> Self {
        Self { verdict: Verdict::Unknown, confidence: 0.0, positions: Vec::new(), keys: Vec::new(), values: Vec::new() }
    }
}

/// Output of the cross-attention read head.
#[derive(Clone, Debug)]
pub struct ReadOut {
    pub verdict: Verdict,
    /// Ternary gate for the model: `confidence` if known, a third of it if
    /// unknown, `0` if absent.
    pub gate: f32,
    pub attended: Attended,
    pub spans: Vec<(u64, u64)>,
}

#[derive(Clone, Debug)]
pub struct ContextStats {
    pub window: (u64, u64),
    pub chunks: usize,
    pub unindexed_tokens: u64,
    pub lexical: Stats,
    pub semantic: Option<Stats>,
    pub token_bytes: usize,
    pub kv_bytes: usize,
    pub census_entries: usize,
}

struct ChunkRec {
    start: u64,
    lexical: MemoryId,
    semantic: Option<MemoryId>,
}

/// Exact long-context memory for a recurrent LLM.
pub struct ContextMemory {
    cfg: ContextConfig,
    lexical: SnnMemory<Chunk>,
    semantic: Option<SnnMemory<Chunk>>,
    kv: Option<KvStore>,
    tokens: Vec<u32>,
    /// Absolute position of `tokens[0]`.
    base: u64,
    next_chunk: u64,
    chunks: VecDeque<ChunkRec>,
    last: Option<(MemoryId, Option<MemoryId>)>,
    /// Counts of window n-grams (largest configured size), by 64-bit hash.
    census: HashMap<u64, u32>,
    census_from: u64,
    census_n: usize,
}

impl ContextMemory {
    pub fn new(cfg: ContextConfig) -> Result<Self, MemoryError> {
        if cfg.chunk_size == 0 || cfg.stride == 0 || cfg.stride > cfg.chunk_size {
            return Err(MemoryError::InvalidConfig("need 0 < stride <= chunk_size".into()));
        }
        if cfg.max_tokens < cfg.chunk_size {
            return Err(MemoryError::InvalidConfig("max_tokens must hold at least one chunk".into()));
        }
        let enc = NGramEncoder::new(cfg.n_neurons, &cfg.ngrams, 1, cfg.seed)?;
        let need = enc.max_code_len(cfg.chunk_size);
        if need > cfg.lexical.max_ensemble {
            return Err(MemoryError::InvalidConfig(format!(
                "a chunk needs {need} synapses, lexical max_ensemble is {}",
                cfg.lexical.max_ensemble
            )));
        }
        let census_n = *enc.ngrams().last().expect("encoder has n-gram sizes");
        let lexical = SnnMemory::new(enc, cfg.lexical.clone())?;
        let (semantic, kv) = match &cfg.kv {
            Some(k) => {
                let semantic = if k.semantic_index {
                    let enc = FlyHashEncoder::new(k.key_dim, cfg.n_neurons, k.dense_k, k.fan_in, cfg.seed ^ 0xD5)?;
                    let mcfg =
                        MemoryConfig { max_ensemble: k.dense_k.max(cfg.semantic.max_ensemble), ..cfg.semantic.clone() };
                    Some(SnnMemory::new(enc, mcfg)?)
                } else {
                    None
                };
                (semantic, Some(KvStore::new(k.key_dim, k.value_dim, k.precision)))
            }
            None => (None, None),
        };
        Ok(Self {
            cfg,
            lexical,
            semantic,
            kv,
            tokens: Vec::new(),
            base: 0,
            next_chunk: 0,
            chunks: VecDeque::new(),
            last: None,
            census: HashMap::new(),
            census_from: 0,
            census_n,
        })
    }

    pub fn config(&self) -> &ContextConfig {
        &self.cfg
    }

    /// One past the last appended position.
    pub fn position(&self) -> u64 {
        self.base + self.tokens.len() as u64
    }

    /// `[start, end)` of the sliding window.
    pub fn window(&self) -> (u64, u64) {
        let end = self.position();
        (end.saturating_sub(self.cfg.max_tokens as u64).max(self.base), end)
    }

    /// Tokens of `[start, end)` if they are still held.
    pub fn tokens(&self, start: u64, end: u64) -> Option<&[u32]> {
        if start < self.base || end > self.position() || start > end {
            return None;
        }
        Some(&self.tokens[(start - self.base) as usize..(end - self.base) as usize])
    }

    pub fn lexical(&self) -> &SnnMemory<Chunk> {
        &self.lexical
    }

    pub fn semantic(&self) -> Option<&SnnMemory<Chunk>> {
        self.semantic.as_ref()
    }

    pub fn kv(&self) -> Option<&KvStore> {
        self.kv.as_ref()
    }

    // ----- writing --------------------------------------------------------

    /// Append tokens (contexts without a K/V store). Returns new chunks.
    pub fn append(&mut self, tokens: &[u32]) -> Result<usize, MemoryError> {
        if self.kv.is_some() {
            return Err(MemoryError::InvalidConfig("this context stores K/V: use append_kv".into()));
        }
        self.append_tokens(tokens)
    }

    /// Append tokens with their attention keys (`n × key_dim`) and values
    /// (`n × value_dim`). Returns the number of chunks created.
    pub fn append_kv(&mut self, tokens: &[u32], keys: &[f32], values: &[f32]) -> Result<usize, MemoryError> {
        let kv = self.kv.as_mut().ok_or(MemoryError::InvalidConfig("no K/V store configured".into()))?;
        let (dk, dv, n) = (kv.key_dim(), kv.value_dim(), tokens.len());
        if keys.len() != n * dk {
            return Err(MemoryError::DimensionMismatch { expected: n * dk, got: keys.len() });
        }
        if values.len() != n * dv {
            return Err(MemoryError::DimensionMismatch { expected: n * dv, got: values.len() });
        }
        kv.push(keys, values);
        self.append_tokens(tokens)
    }

    fn census_hash(&self, window: &[u32]) -> u64 {
        window.iter().fold(mix64(self.cfg.seed ^ 0x00CE_4505), |h, &t| mix64(h ^ t as u64 ^ 0x9E37_79B9))
    }

    fn append_tokens(&mut self, new: &[u32]) -> Result<usize, MemoryError> {
        let old_end = self.position();
        self.tokens.extend_from_slice(new);
        let end = self.position();

        // Census of complete n-grams starting at or after `census_from`.
        let n = self.census_n as u64;
        let first = old_end.saturating_sub(n - 1).max(self.census_from);
        for s in first..(end + 1).saturating_sub(n) {
            let h = self.census_hash(self.tokens(s, s + n).expect("window n-gram is held"));
            *self.census.entry(h).or_insert(0) += 1;
        }

        let (cs, st) = (self.cfg.chunk_size as u64, self.cfg.stride as u64);
        let mut starts = Vec::new();
        while self.next_chunk + cs <= end {
            starts.push(self.next_chunk);
            self.next_chunk += st;
        }
        let created = starts.len();
        if created > 0 {
            self.write_chunks(&starts)?;
        }
        self.evict();
        Ok(created)
    }

    fn write_chunks(&mut self, starts: &[u64]) -> Result<(), MemoryError> {
        let cs = self.cfg.chunk_size as u64;
        let chunks: Vec<Chunk> = starts
            .iter()
            .map(|&s| Chunk { start: s, tokens: self.tokens(s, s + cs).expect("chunk is held").to_vec() })
            .collect();

        let inputs: Vec<Input> = chunks.iter().map(|c| Input::Tokens(&c.tokens)).collect();
        let opts = BatchOptions { link_sequence: true, after: self.last.map(|l| l.0), ..Default::default() };
        let lex_ids = self.lexical.learn_batch(&inputs, Some(chunks.clone()), &opts)?;

        let sem_ids: Vec<Option<MemoryId>> = match (&mut self.semantic, &self.kv) {
            (Some(sem), Some(kv)) => {
                let dk = kv.key_dim();
                let (mut key, mut value) = (vec![0f32; dk], vec![0f32; kv.value_dim()]);
                let pooled: Vec<Vec<f32>> = starts
                    .iter()
                    .map(|&s| {
                        let mut mean = vec![0f32; dk];
                        for p in s..s + cs {
                            kv.read(p, &mut key, &mut value);
                            mean.iter_mut().zip(&key).for_each(|(m, k)| *m += k);
                        }
                        mean
                    })
                    .collect();
                let inputs: Vec<Input> = pooled.iter().map(|m| Input::Dense(m)).collect();
                let opts =
                    BatchOptions { link_sequence: true, after: self.last.and_then(|l| l.1), ..Default::default() };
                sem.learn_batch(&inputs, Some(chunks), &opts)?.into_iter().map(Some).collect()
            }
            _ => vec![None; starts.len()],
        };

        for ((&start, lexical), semantic) in starts.iter().zip(lex_ids).zip(sem_ids) {
            self.chunks.push_back(ChunkRec { start, lexical, semantic });
            self.last = Some((lexical, semantic));
        }
        Ok(())
    }

    /// Chunk the unindexed tail now (as one chunk ending at the current
    /// position), e.g. at the end of a document.
    pub fn flush(&mut self) -> Result<usize, MemoryError> {
        let cs = self.cfg.chunk_size as u64;
        let end = self.position();
        let covered = self.chunks.back().map_or(self.base, |c| c.start + cs);
        if end <= covered || end < self.base + cs {
            return Ok(0);
        }
        self.write_chunks(&[end - cs])?;
        Ok(1)
    }

    fn evict(&mut self) {
        let (window_start, _) = self.window();
        let cs = self.cfg.chunk_size as u64;
        while let Some(front) = self.chunks.front() {
            if front.start >= window_start {
                break;
            }
            let rec = self.chunks.pop_front().expect("front exists");
            forget_if_fast(&mut self.lexical, rec.lexical);
            if let (Some(sem), Some(id)) = (&mut self.semantic, rec.semantic) {
                forget_if_fast(sem, id);
            }
        }
        // Census: drop n-grams starting before the window.
        let n = self.census_n as u64;
        while self.census_from < window_start {
            let s = self.census_from;
            if s + n <= self.position() {
                let h = self.census_hash(self.tokens(s, s + n).expect("census n-gram is held"));
                if let Some(c) = self.census.get_mut(&h) {
                    *c -= 1;
                    if *c == 0 {
                        self.census.remove(&h);
                    }
                }
            }
            self.census_from += 1;
        }
        // Drop held tokens (and K/V) in amortised batches.
        let keep_from = window_start.min(self.next_chunk);
        let slack = (self.cfg.max_tokens as u64 / 3).max(cs);
        if keep_from >= self.base + slack {
            self.tokens.drain(..(keep_from - self.base) as usize);
            self.base = keep_from;
            if let Some(kv) = &mut self.kv {
                kv.evict_before(keep_from);
            }
        }
    }

    // ----- reading --------------------------------------------------------

    /// Retrieve the chunks matching a probe, most confident first.
    pub fn retrieve(&mut self, probe: Probe<'_>, top_k: usize) -> Result<Retrieval, MemoryError> {
        let opts = RecallOptions { top_k, ..RecallOptions::default() };
        let mut verdicts = Vec::new();
        let mut hits: Vec<(f32, MemoryId, Chunk)> = Vec::new();
        let (tokens, key) = match probe {
            Probe::Tokens(t) => (Some(t), None),
            Probe::Key(k) => (None, Some(k)),
            Probe::Both(t, k) => (Some(t), Some(k)),
        };
        if let Some(t) = tokens.filter(|t| !t.is_empty()) {
            let r = self.lexical.recall(Input::Tokens(t), &opts)?;
            verdicts.push(r.verdict);
            hits.extend(r.hits.into_iter().filter_map(|h| Some((h.confidence, h.id, h.payload?))));
        }
        if let Some(k) = key {
            let sem = self.semantic.as_mut().ok_or(MemoryError::InvalidConfig("no semantic index".into()))?;
            let r = sem.recall(Input::Dense(k), &opts)?;
            verdicts.push(r.verdict);
            hits.extend(r.hits.into_iter().filter_map(|h| Some((h.confidence, h.id, h.payload?))));
        }
        let verdict = if verdicts.contains(&Verdict::Known) {
            Verdict::Known
        } else if !verdicts.is_empty() && verdicts.iter().all(|&v| v == Verdict::Absent) {
            Verdict::Absent
        } else {
            Verdict::Unknown
        };
        Ok(Retrieval { verdict, spans: self.spans_of(hits) })
    }

    /// Turn chunk hits into merged spans of context.
    fn spans_of(&self, hits: Vec<(f32, MemoryId, Chunk)>) -> Vec<RetrievedSpan> {
        let pad = (self.cfg.neighbors * self.cfg.stride) as u64;
        let (lo, hi) = (self.base, self.position());
        let mut inside: Vec<(u64, u64, f32, MemoryId)> = Vec::new();
        let mut out = Vec::new();
        for (conf, id, chunk) in hits {
            let end = chunk.start + chunk.tokens.len() as u64;
            if chunk.start >= lo && end <= hi {
                inside.push((chunk.start.saturating_sub(pad).max(lo), (end + pad).min(hi), conf, id));
            } else {
                out.push(RetrievedSpan {
                    start: chunk.start,
                    tokens: chunk.tokens,
                    confidence: conf,
                    chunks: vec![id],
                    in_window: false,
                });
            }
        }
        inside.sort_by_key(|s| s.0);
        let mut merged: Vec<(u64, u64, f32, Vec<MemoryId>)> = Vec::new();
        for (s, e, c, id) in inside {
            match merged.last_mut() {
                Some(m) if s <= m.1 => {
                    m.1 = m.1.max(e);
                    m.2 = m.2.max(c);
                    m.3.push(id);
                }
                _ => merged.push((s, e, c, vec![id])),
            }
        }
        out.extend(merged.into_iter().map(|(s, e, c, ids)| RetrievedSpan {
            start: s,
            tokens: self.tokens(s, e).expect("span is held").to_vec(),
            confidence: c,
            chunks: ids,
            in_window: true,
        }));
        out.sort_by(|a, b| b.confidence.total_cmp(&a.confidence).then(b.start.cmp(&a.start)));
        out
    }

    /// Exact positions of `fragment` (verified token by token).
    pub fn locate(&mut self, fragment: &[u32]) -> Result<Located, MemoryError> {
        let l = fragment.len();
        if l == 0 {
            return Ok(Located { verdict: Verdict::Unknown, positions: Vec::new() });
        }
        let mut positions = Vec::new();

        // 1. The unindexed tail is scanned directly (working memory).
        let cs = self.cfg.chunk_size as u64;
        let covered = self.chunks.back().map_or(self.base, |c| c.start + cs);
        let tail_from = covered.saturating_sub(cs + l as u64).max(self.base);
        find_all(self.tokens(tail_from, self.position()).unwrap_or(&[]), fragment, tail_from, &mut positions);

        // 2. The SNN proposes chunks from a probe window that always fits in
        //    one chunk; candidates are verified exactly.
        let probe_len = l.min(self.cfg.chunk_size - self.cfg.stride + 1);
        let mut snn_verdict = Verdict::Unknown;
        let offsets = if probe_len == l { vec![0] } else { vec![0, l - probe_len] };
        for offset in offsets {
            let probe = &fragment[offset..offset + probe_len];
            let r = self.retrieve(Probe::Tokens(probe), self.cfg.top_k)?;
            snn_verdict = r.verdict;
            for span in &r.spans {
                if span.in_window {
                    let from = span.start.saturating_sub(offset as u64).max(self.base);
                    let to = (span.end() + (l - offset) as u64).min(self.position());
                    find_all(self.tokens(from, to).unwrap_or(&[]), fragment, from, &mut positions);
                } else {
                    find_all(&span.tokens, fragment, span.start, &mut positions);
                }
            }
            if !positions.is_empty() || offset == l - probe_len {
                break;
            }
        }
        positions.sort_unstable();
        positions.dedup();

        let verdict = if !positions.is_empty() {
            Verdict::Known
        } else if self.census_rules_out(fragment) || snn_verdict == Verdict::Absent {
            Verdict::Absent
        } else {
            Verdict::Unknown
        };
        Ok(Located { verdict, positions })
    }

    /// `true` if some n-gram of `fragment` occurs nowhere in the window.
    fn census_rules_out(&self, fragment: &[u32]) -> bool {
        fragment.len() >= self.census_n
            && fragment.windows(self.census_n).any(|w| !self.census.contains_key(&self.census_hash(w)))
    }

    /// Ternary membership: is `fragment` in memory?
    pub fn contains(&mut self, fragment: &[u32]) -> Result<Verdict, MemoryError> {
        Ok(self.locate(fragment)?.verdict)
    }

    /// Associative recall of what followed `prefix` the last time it
    /// occurred ("induction head"): up to `n` tokens and their position.
    pub fn continuation(&mut self, prefix: &[u32], n: usize) -> Result<Option<(u64, Vec<u32>)>, MemoryError> {
        let found = self.locate(prefix)?;
        let l = prefix.len() as u64;
        for &p in found.positions.iter().rev() {
            let from = p + l;
            if from >= self.base && from < self.position() {
                let to = (from + n as u64).min(self.position());
                return Ok(Some((from, self.tokens(from, to).expect("held").to_vec())));
            }
            // Pinned chunk outside the window: continue within its tokens.
            if let Some(tokens) = self.pinned_tokens_after(from) {
                return Ok(Some((from, tokens.into_iter().take(n).collect())));
            }
        }
        Ok(None)
    }

    fn pinned_tokens_after(&self, from: u64) -> Option<Vec<u32>> {
        self.lexical.ids().into_iter().find_map(|id| {
            let c = self.lexical.payload(id)?;
            let end = c.start + c.tokens.len() as u64;
            (from >= c.start && from < end).then(|| c.tokens[(from - c.start) as usize..].to_vec())
        })
    }

    /// Cross-attention read head: the SNN selects chunks for `probe`, then
    /// `query` (key_dim) attends exactly over their tokens' keys/values.
    pub fn read(&mut self, probe: Probe<'_>, query: &[f32], top_k: usize) -> Result<ReadOut, MemoryError> {
        let dk = self.kv.as_ref().ok_or(MemoryError::InvalidConfig("no K/V store configured".into()))?.key_dim();
        if query.len() != dk {
            return Err(MemoryError::DimensionMismatch { expected: dk, got: query.len() });
        }
        let r = self.retrieve(probe, top_k)?;
        let spans: Vec<(u64, u64)> = r.spans.iter().filter(|s| s.in_window).map(|s| (s.start, s.end())).collect();
        let attended = attend(query, self.kv.as_ref().expect("checked above"), &spans);
        let best = r.spans.iter().map(|s| s.confidence).fold(0.0, f32::max);
        let gate = match r.verdict {
            Verdict::Known => best,
            Verdict::Unknown => best / 3.0,
            Verdict::Absent => 0.0,
        };
        Ok(ReadOut { verdict: r.verdict, gate, attended, spans })
    }

    /// Keys and values of the tokens the SNN retrieves for `probe`, for a
    /// model that runs its own (differentiable) attention over them. Rows
    /// come from the most confident spans first, at most `limit` rows.
    pub fn retrieve_rows(&mut self, probe: Probe<'_>, top_k: usize, limit: usize) -> Result<MemoryRows, MemoryError> {
        let kv = self.kv.as_ref().ok_or(MemoryError::InvalidConfig("no K/V store configured".into()))?;
        let (dk, dv) = (kv.key_dim(), kv.value_dim());
        let r = self.retrieve(probe, top_k)?;
        let kv = self.kv.as_ref().expect("checked above");
        let mut rows = MemoryRows { verdict: r.verdict, ..Default::default() };
        let (mut key, mut value) = (vec![0f32; dk], vec![0f32; dv]);
        'spans: for span in r.spans.iter().filter(|s| s.in_window) {
            for pos in span.start..span.end() {
                if rows.positions.len() == limit {
                    break 'spans;
                }
                if kv.read(pos, &mut key, &mut value) {
                    rows.positions.push(pos);
                    rows.keys.extend_from_slice(&key);
                    rows.values.extend_from_slice(&value);
                }
            }
        }
        rows.confidence = r.spans.iter().map(|s| s.confidence).fold(0.0, f32::max);
        Ok(rows)
    }

    // ----- lifecycle ------------------------------------------------------

    /// Pin every chunk overlapping `[start, end)`: it is consolidated and
    /// survives window eviction and resets. Returns chunks pinned.
    pub fn pin(&mut self, start: u64, end: u64) -> Result<usize, MemoryError> {
        let cs = self.cfg.chunk_size as u64;
        let mut n = 0;
        for rec in self.chunks.iter().filter(|c| c.start < end && c.start + cs > start) {
            self.lexical.pin(rec.lexical)?;
            if let (Some(sem), Some(id)) = (&mut self.semantic, rec.semantic) {
                sem.pin(id)?;
            }
            n += 1;
        }
        Ok(n)
    }

    /// Start a new conversation: forget the window, keep pinned chunks.
    /// Positions keep increasing so old pinned chunks never collide.
    pub fn reset(&mut self) {
        let end = self.position();
        self.lexical.reset_fast_memory();
        if let Some(sem) = &mut self.semantic {
            sem.reset_fast_memory();
        }
        if let Some(kv) = &mut self.kv {
            kv.reset(end);
        }
        self.tokens.clear();
        self.base = end;
        self.next_chunk = end;
        self.chunks.clear();
        self.last = None;
        self.census.clear();
        self.census_from = end;
    }

    /// Snapshot of the pinned (and consolidated) chunks: what should be
    /// carried into the next session. The window itself is not saved.
    pub fn save_pinned(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(CONTEXT_MAGIC);
        out.extend_from_slice(&persist::VERSION.to_le_bytes());
        out.extend_from_slice(&self.position().to_le_bytes());
        let lexical = self.lexical.save(SnapshotScope::LongTerm);
        out.extend_from_slice(&(lexical.len() as u64).to_le_bytes());
        out.extend_from_slice(&lexical);
        match &self.semantic {
            Some(sem) => {
                let semantic = sem.save(SnapshotScope::LongTerm);
                out.extend_from_slice(&(semantic.len() as u64).to_le_bytes());
                out.extend_from_slice(&semantic);
            }
            None => out.extend_from_slice(&0u64.to_le_bytes()),
        }
        out
    }

    /// Start a new session from [`save_pinned`](Self::save_pinned) output.
    /// Positions continue after the saved session's last position.
    pub fn restore(cfg: ContextConfig, bytes: &[u8]) -> Result<Self, MemoryError> {
        let mut input = bytes;
        if persist::take(&mut input, 4)? != CONTEXT_MAGIC {
            return Err(persist::corrupt("not a context snapshot"));
        }
        if persist::read_u32(&mut input)? != persist::VERSION {
            return Err(persist::corrupt("unsupported context snapshot version"));
        }
        let position = persist::read_u64(&mut input)?;
        let mut ctx = Self::new(cfg)?;
        let n = persist::read_u64(&mut input)? as usize;
        ctx.lexical.restore(persist::take(&mut input, n)?)?;
        let n = persist::read_u64(&mut input)? as usize;
        if n > 0 {
            let part = persist::take(&mut input, n)?;
            match &mut ctx.semantic {
                Some(sem) => {
                    sem.restore(part)?;
                }
                None => return Err(MemoryError::InvalidConfig("snapshot has a semantic index".into())),
            }
        }
        if !input.is_empty() {
            return Err(persist::corrupt("trailing bytes"));
        }
        ctx.base = position;
        ctx.next_chunk = position;
        ctx.census_from = position;
        if let Some(kv) = &mut ctx.kv {
            kv.reset(position);
        }
        Ok(ctx)
    }

    pub fn stats(&self) -> ContextStats {
        let cs = self.cfg.chunk_size as u64;
        let covered = self.chunks.back().map_or(self.base, |c| c.start + cs);
        ContextStats {
            window: self.window(),
            chunks: self.chunks.len(),
            unindexed_tokens: self.position().saturating_sub(covered),
            lexical: self.lexical.stats(),
            semantic: self.semantic.as_ref().map(SnnMemory::stats),
            token_bytes: self.tokens.capacity() * 4,
            kv_bytes: self.kv.as_ref().map_or(0, KvStore::bytes),
            census_entries: self.census.len(),
        }
    }
}

fn forget_if_fast(mem: &mut SnnMemory<Chunk>, id: MemoryId) {
    if mem.get_memory(id).is_some_and(|r| r.tier == Tier::Fast) {
        mem.forget(id);
    }
}

/// Append every start position of `needle` in `hay` (offset by `base`).
fn find_all(hay: &[u32], needle: &[u32], base: u64, out: &mut Vec<u64>) {
    if needle.is_empty() || hay.len() < needle.len() {
        return;
    }
    for (i, w) in hay.windows(needle.len()).enumerate() {
        if w == needle {
            out.push(base + i as u64);
        }
    }
}
