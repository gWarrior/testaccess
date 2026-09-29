//! Training and evaluation with the SNN memory in the loop.
//!
//! Each of the `B` parallel streams owns a [`ContextMemory`] (window 300k
//! tokens). For every window of `T` tokens:
//!
//! 1. the trunk runs and produces memory queries;
//! 2. for every block of 9 tokens, the stream's memory is probed with the
//!    last 9 tokens and the block's query vector; the retrieved keys/values
//!    (from earlier windows only) become constant tensors;
//! 3. the head attends jointly over the window and the retrieved rows; the
//!    loss is back-propagated (truncated BPTT);
//! 4. the window's tokens with their memory keys/values are appended to the
//!    stream's memory.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rayon::prelude::*;
use snn_memory::{ContextConfig, ContextMemory, KvConfig, KvPrecision, Probe, Verdict};

use crate::data::{Episodes, TaskStream};
use crate::model::{BlockRows, Config, IndCand, MemBatch, Model, State, Trunk};
use crate::optim::AdamW;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub steps: usize,
    pub lr: f64,
    pub warmup: usize,
    pub batch: usize,
    pub window: usize,
    pub memory: bool,
    pub top_k: usize,
    pub rows: usize,
    pub max_tokens: usize,
    pub clip: f64,
    pub log_every: usize,
    pub ckpt_every: usize,
    pub out: PathBuf,
    pub seed: u64,
    /// Stop after this many seconds (0 = no limit).
    pub time_limit: u64,
    /// Warm start from this checkpoint (tensors absent there keep their init).
    pub init: Option<PathBuf>,
    /// Streams jump to a random document of the whole corpus after each one.
    pub jump: bool,
    /// Probability per token of a re-reading episode and of an episode.
    pub p_reread: f64,
    pub p_episode: f64,
    /// Streams per group (gradients accumulate over groups; peak memory
    /// scales with the group).
    pub micro: usize,
    /// Weight of the auxiliary pointer loss per answer / re-read token,
    /// relative to a token's mixture loss (template answers get none).
    pub aux: f64,
    /// Extra weight of a fact answer's auxiliary loss (fact answers are
    /// ~0.7% of tokens, re-read spans ~2.5%).
    pub aux_fact: f64,
    /// Decay of the exponential moving average of the latent weights and
    /// steps, saved as `model.ema.safetensors` (0 = off).
    pub ema: f64,
    /// Validation loss (memory on, 9 streams × 27 windows) of the weights
    /// and of their average every this many steps, into val.tsv (0 = off).
    pub val_every: usize,
    /// Groups of `micro` streams computed at once (default: the CPU cores).
    pub parallel: usize,
    /// Adam's second-moment decay: 0.99 adapts within a short run (a few
    /// hundred steps), 0.999 averages over ~1000 steps.
    pub beta2: f64,
    /// Row steps of the weights: `auto` sets each row's θ to the MSE optimum
    /// of its current latents after every update (a power of two from
    /// mean |w|, BitNet style); `learned` trains θ by LSQ at lr/27.
    pub auto_steps: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 59_049,
            lr: 3e-3,
            warmup: 27,
            batch: 27,
            window: 243,
            memory: true,
            top_k: crate::model::MEM_TOP_K,
            rows: crate::model::MEM_ROWS,
            // 3^11: episodes and re-reading reach this far, close to the 300k
            // of inference.
            max_tokens: 177_147,
            clip: 1.0,
            log_every: 9,
            ckpt_every: 81,
            out: PathBuf::from("/home/user/data/run"),
            seed: 1,
            time_limit: 0,
            init: None,
            jump: true,
            p_reread: 1.0 / 2187.0,
            p_episode: 1.0 / 729.0,
            aux: 1.0,
            aux_fact: 3.0,
            micro: 3,
            // Short runs (a few thousand steps): a 243-step horizon.
            ema: 1.0 - 1.0 / 243.0,
            val_every: 729,
            parallel: std::thread::available_parallelism().map_or(1, |n| n.get()),
            beta2: 0.99,
            // A learned θ lagged behind latents that double over a run: a
            // quarter of mlp.w2 / cell.wu weights ended up clipped (analyst 7).
            auto_steps: true,
        }
    }
}

fn verdict_index(v: Verdict) -> u32 {
    match v {
        Verdict::Known => 0,
        Verdict::Unknown => 1,
        Verdict::Absent => 2,
    }
}

/// Shortest and longest hashed suffix of the induction column; a match
/// is then extended token by token up to [`IND_EXT`].
pub const IND_MIN: usize = 3;
pub const IND_MAX: usize = 8;
pub const IND_EXT: usize = 27;

/// Exact suffix index of a stream, the training-time twin of the SNN
/// memory's `continuation`: for every position, the token that followed the
/// most recent earlier occurrence of its longest suffix (8 down to 3
/// tokens, then extended up to 27), and how often that suffix occurred.
/// Keeps the last `limit` tokens.
pub struct Induction {
    tokens: Vec<u32>,
    /// 32-bit n-gram fingerprint → (last end position, occurrences). A
    /// match is verified token by token, so a fingerprint collision can only
    /// merge two counts; 32-bit keys make the index 2.7× smaller.
    index: std::collections::HashMap<u32, (u32, u32)>,
    limit: usize,
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_u32s(out: &mut Vec<u8>, v: &[u32]) {
    put_u64(out, v.len() as u64);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn take<'a>(input: &mut &'a [u8], n: usize) -> Result<&'a [u8]> {
    if input.len() < n {
        candle_core::bail!("resume state is truncated");
    }
    let (head, rest) = input.split_at(n);
    *input = rest;
    Ok(head)
}

fn get_u64(input: &mut &[u8]) -> Result<u64> {
    Ok(u64::from_le_bytes(take(input, 8)?.try_into().expect("8 bytes")))
}

fn get_u32(input: &mut &[u8]) -> Result<u32> {
    Ok(u32::from_le_bytes(take(input, 4)?.try_into().expect("4 bytes")))
}

fn get_u32s(input: &mut &[u8]) -> Result<Vec<u32>> {
    let n = get_u64(input)? as usize;
    Ok(take(input, n * 4)?.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes"))).collect())
}

pub fn ngram_key(g: &[u32]) -> u32 {
    let h =
        g.iter().fold(0xcbf2_9ce4_8422_2325u64 ^ g.len() as u64, |h, &t| (h ^ t as u64).wrapping_mul(0x100_0000_01b3));
    (h ^ (h >> 32)) as u32
}

impl Induction {
    pub fn new(limit: usize) -> Self {
        Self { tokens: Vec::new(), index: Default::default(), limit }
    }

    fn insert(&mut self, p: usize) {
        for n in IND_MIN..=IND_MAX.min(p + 1) {
            let e = self.index.entry(ngram_key(&self.tokens[p + 1 - n..=p])).or_insert((0, 0));
            *e = (p as u32, e.1 + 1);
        }
    }

    fn save(&self, out: &mut Vec<u8>) {
        put_u64(out, self.limit as u64);
        put_u32s(out, &self.tokens);
        put_u64(out, self.index.len() as u64);
        for (&k, &(p, c)) in &self.index {
            out.extend_from_slice(&k.to_le_bytes());
            out.extend_from_slice(&p.to_le_bytes());
            out.extend_from_slice(&c.to_le_bytes());
        }
    }

    fn load(input: &mut &[u8]) -> Result<Self> {
        let limit = get_u64(input)? as usize;
        let tokens = get_u32s(input)?;
        let n = get_u64(input)? as usize;
        let mut index = std::collections::HashMap::with_capacity(n);
        for _ in 0..n {
            let (k, p, c) = (get_u32(input)?, get_u32(input)?, get_u32(input)?);
            index.insert(k, (p, c));
        }
        Ok(Self { tokens, index, limit })
    }

    /// Feed a window; the induction candidate of every position.
    pub fn window(&mut self, x: &[u32]) -> Vec<IndCand> {
        // Keep about `limit` tokens (the memory's window), rebuilding the
        // index every ~limit/9 tokens.
        if self.tokens.len() + x.len() > self.limit + self.limit / 9 {
            let drop = (self.tokens.len() + x.len()).saturating_sub(self.limit).min(self.tokens.len());
            self.tokens.drain(..drop);
            self.index.clear();
            for p in 0..self.tokens.len() {
                self.insert(p);
            }
        }
        let mut out = Vec::with_capacity(x.len());
        for &tok in x {
            self.tokens.push(tok);
            let p = self.tokens.len() - 1;
            let mut best = IndCand::NONE;
            for n in (IND_MIN..=IND_MAX.min(p + 1)).rev() {
                let g = &self.tokens[p + 1 - n..=p];
                if let Some(&(q, count)) = self.index.get(&ngram_key(g)) {
                    let q = q as usize;
                    if q + 1 >= n && q < p && &self.tokens[q + 1 - n..=q] == g {
                        let mut len = n;
                        while len < IND_EXT && len <= q && self.tokens[p - len] == self.tokens[q - len] {
                            len += 1;
                        }
                        best = IndCand { tok: self.tokens[q + 1], len: len as u16, count, dist: (p - q) as u32 };
                        break;
                    }
                }
            }
            self.insert(p);
            out.push(best);
        }
        out
    }
}

/// Per-stream context memory plus the last few tokens (for probes).
pub struct StreamMemory {
    pub mem: ContextMemory,
    history: Vec<u32>,
    pub induction: Induction,
    /// Without the memory: induction over the last 243 tokens only, as the
    /// engine's ring.
    local: Induction,
}

impl StreamMemory {
    fn config(dim: usize, max_tokens: usize, precision: KvPrecision) -> ContextConfig {
        ContextConfig { max_tokens, kv: Some(KvConfig { precision, ..KvConfig::new(dim, dim) }), ..Default::default() }
    }

    pub fn new(dim: usize, max_tokens: usize, precision: KvPrecision) -> Self {
        Self {
            mem: ContextMemory::new(Self::config(dim, max_tokens, precision)).expect("valid memory config"),
            history: Vec::new(),
            induction: Induction::new(max_tokens),
            // Two windows: every position of the current window can look
            // LOCAL tokens back (a limit of one window dropped the previous
            // window entirely, so the ablation saw only the current one).
            local: Induction::new(2 * 243),
        }
    }

    /// Everything of this stream's memory, for an exact resume.
    pub fn save(&self) -> Vec<u8> {
        let mem = self.mem.save_full();
        let mut out = Vec::with_capacity(mem.len() + 8 * self.induction.tokens.len() + 64);
        put_u64(&mut out, mem.len() as u64);
        out.extend_from_slice(&mem);
        put_u32s(&mut out, &self.history);
        self.induction.save(&mut out);
        self.local.save(&mut out);
        out
    }

    pub fn load(dim: usize, max_tokens: usize, precision: KvPrecision, bytes: &[u8]) -> Result<Self> {
        let mut input = bytes;
        let n = get_u64(&mut input)? as usize;
        let mem = ContextMemory::restore_full(Self::config(dim, max_tokens, precision), take(&mut input, n)?)
            .map_err(candle_core::Error::wrap)?;
        let history = get_u32s(&mut input)?;
        let induction = Induction::load(&mut input)?;
        let local = Induction::load(&mut input)?;
        if !input.is_empty() {
            candle_core::bail!("trailing bytes in a stream memory");
        }
        Ok(Self { mem, history, induction, local })
    }

    /// Rows for every block of this window.
    fn retrieve(
        &mut self,
        x: &[u32],
        q: &[f32],
        block: usize,
        dim: usize,
        top_k: usize,
        rows: usize,
    ) -> Vec<BlockRows> {
        let ctx: Vec<u32> = self.history.iter().chain(x).copied().collect();
        let off = self.history.len();
        (0..x.len() / block)
            .map(|j| {
                let s = j * block;
                let probe_end = off + s + 1;
                let tokens = crate::model::probe_of(&ctx[probe_end.saturating_sub(crate::model::PROBE)..probe_end]);
                let key = &q[s * dim..(s + 1) * dim];
                match self.mem.retrieve_rows(Probe::Both(tokens, key), top_k, rows) {
                    Ok(r) => BlockRows {
                        next: r.positions.iter().map(|&p| self.next_token(p, x[0])).collect(),
                        verdict: verdict_index(r.verdict),
                        keys: r.keys,
                        values: r.values,
                        pos: r.positions,
                        source: r.sources,
                    },
                    Err(_) => BlockRows { verdict: 1, ..Default::default() },
                }
            })
            .collect()
    }

    /// The token that followed memory position `p`; the memory's last
    /// position is followed by the current window's first token.
    fn next_token(&self, p: u64, first: u32) -> u32 {
        if p + 1 == self.mem.position() {
            first
        } else {
            self.mem.tokens(p + 1, p + 2).map_or(u32::MAX, |t| t[0])
        }
    }

    fn write(&mut self, x: &[u32], k: &[f32], v: &[f32]) {
        self.mem.append_kv(x, k, v).expect("append");
        let keep = crate::model::PROBE - 1;
        self.history = x[x.len().saturating_sub(keep)..].to_vec();
    }
}

/// Everything needed to run the model over parallel streams.
pub struct Runner {
    pub model: Model,
    /// The last committed window's keys, values and tokens (the local
    /// window reaches back into it) and the absolute position of the next.
    prev: Option<(Tensor, Tensor, Vec<u32>)>,
    pos: u64,
    /// Drop the recurrent state between windows (ablation).
    pub reset_state: bool,
    pub state: State,
    pub memories: Vec<StreamMemory>,
    pub use_memory: bool,
    pub top_k: usize,
    pub rows: usize,
}

/// Result of one window.
pub struct WindowOut {
    pub logits: Tensor,
    /// Pointer attention `(B, T, T + M + 2)` and its null column `(B, T)`.
    pub point: Tensor,
    pub gate: Tensor,
    pub log_point: Tensor,
    /// Token after each memory row and its position `(B, nb, M)`, and `M`.
    pub far_next: Vec<u32>,
    pub far_pos: Vec<u64>,
    /// Source trit of each memory row (0 lexical, 1 semantic, 2 newest).
    pub far_src: Vec<u8>,
    /// Induction candidates `(B·T)`.
    pub ind: Vec<IndCand>,
    /// The previous window's tokens `(B·T)` (empty for the first window).
    pub prev_tokens: Vec<u32>,
    pub m: usize,
    pub trunk: Trunk,
    pub known: usize,
    pub blocks: usize,
    pub mem_rows: usize,
}

impl Runner {
    pub fn new(
        model: Model,
        batch: usize,
        use_memory: bool,
        max_tokens: usize,
        precision: KvPrecision,
        device: &Device,
    ) -> Result<Self> {
        let state = model.zero_state(batch, device)?;
        let dim = model.cfg.mem_dim;
        let memories = (0..batch).map(|_| StreamMemory::new(dim, max_tokens, precision)).collect();
        Ok(Self {
            model,
            state,
            memories,
            use_memory,
            top_k: crate::model::MEM_TOP_K,
            rows: crate::model::MEM_ROWS,
            reset_state: false,
            prev: None,
            pos: 0,
        })
    }

    /// The runner's recurrent state, previous window and stream memories
    /// into `dir` (files `{tag}.*`), for an exact resume.
    pub fn save(&self, dir: &Path, tag: &str) -> Result<()> {
        let mut t = std::collections::HashMap::new();
        for (i, h) in self.state.h.iter().enumerate() {
            t.insert(format!("h{i}"), h.clone());
        }
        for (i, s) in self.state.s.iter().enumerate() {
            t.insert(format!("s{i}"), s.clone());
        }
        if let Some((k, v, x)) = &self.prev {
            t.insert("prev_k".into(), k.clone());
            t.insert("prev_v".into(), v.clone());
            t.insert("prev_x".into(), Tensor::new(x.as_slice(), k.device())?);
        }
        t.insert("pos".into(), Tensor::new(&[self.pos as f64], &Device::Cpu)?);
        candle_core::safetensors::save(&t, dir.join(format!("{tag}.safetensors")))?;
        if self.use_memory {
            let bytes: Vec<Vec<u8>> = self.memories.par_iter().map(StreamMemory::save).collect();
            for (i, b) in bytes.iter().enumerate() {
                std::fs::write(dir.join(format!("{tag}.mem{i}")), b)?;
            }
        }
        Ok(())
    }

    /// Restore what [`save`](Self::save) wrote.
    pub fn load(&mut self, dir: &Path, tag: &str, max_tokens: usize, precision: KvPrecision) -> Result<()> {
        let t = candle_core::safetensors::load(dir.join(format!("{tag}.safetensors")), &Device::Cpu)?;
        let get = |n: &str| t.get(n).cloned().ok_or_else(|| candle_core::Error::Msg(format!("resume: no {n}")));
        for (i, h) in self.state.h.iter_mut().enumerate() {
            *h = get(&format!("h{i}"))?;
        }
        for (i, s) in self.state.s.iter_mut().enumerate() {
            *s = get(&format!("s{i}"))?;
        }
        self.prev = match t.get("prev_x") {
            Some(x) => Some((get("prev_k")?, get("prev_v")?, x.to_vec1::<u32>()?)),
            None => None,
        };
        self.pos = get("pos")?.to_vec1::<f64>()?[0] as u64;
        if self.use_memory {
            let dim = self.model.cfg.mem_dim;
            self.memories = (0..self.memories.len())
                .into_par_iter()
                .map(|i| {
                    let bytes = std::fs::read(dir.join(format!("{tag}.mem{i}")))?;
                    StreamMemory::load(dim, max_tokens, precision, &bytes)
                })
                .collect::<Result<Vec<_>>>()?;
        }
        Ok(())
    }

    /// Forward one window `x` (`B × T`, row-major) without committing it.
    pub fn forward(&mut self, x: &[u32], t: usize, device: &Device) -> Result<(WindowOut, State)> {
        let b = self.memories.len();
        let ids = Tensor::from_vec(x.to_vec(), (b, t), device)?;
        let (trunk, next) = self.model.trunk(&ids, &self.state)?;
        let (block, dim) = (self.model.cfg.block, self.model.cfg.mem_dim);
        let nb = t / block;
        let (mem, known, mem_rows) = if self.use_memory {
            let q = trunk.q.flatten_all()?.to_vec1::<f32>()?;
            let (top_k, rows) = (self.top_k, self.rows);
            let per: Vec<Vec<BlockRows>> = self
                .memories
                .par_iter_mut()
                .enumerate()
                .map(|(bi, m)| {
                    m.retrieve(&x[bi * t..(bi + 1) * t], &q[bi * t * dim..(bi + 1) * t * dim], block, dim, top_k, rows)
                })
                .collect();
            let known = per.iter().flatten().filter(|r| r.verdict == 0).count();
            let n_rows = per.iter().flatten().map(|r| r.pos.len()).sum();
            let ind: Vec<IndCand> = self
                .memories
                .par_iter_mut()
                .enumerate()
                .map(|(bi, m)| m.induction.window(&x[bi * t..(bi + 1) * t]))
                .collect::<Vec<_>>()
                .concat();
            (MemBatch::from_rows(&per, dim, device)?.with_induction(ind), known, n_rows)
        } else {
            let ind: Vec<IndCand> = self
                .memories
                .par_iter_mut()
                .enumerate()
                .map(|(bi, m)| {
                    // As the engine's ring: occurrences at most LOCAL tokens back
                    // (the most recent one is the nearest, so none nearer is lost).
                    let mut c = m.local.window(&x[bi * t..(bi + 1) * t]);
                    c.iter_mut().filter(|c| c.dist as usize > crate::model::LOCAL).for_each(|c| *c = IndCand::NONE);
                    c
                })
                .collect::<Vec<_>>()
                .concat();
            (MemBatch::empty(b, nb, dim, device)?.with_induction(ind), 0, 0)
        };
        let mem = mem.with_prev(self.prev.as_ref().map(|p| (p.0.clone(), p.1.clone())), self.pos);
        let h = self.model.head(&trunk, &mem)?;
        let out = WindowOut {
            logits: h.logits,
            point: h.point,
            gate: h.gate,
            log_point: h.log_point,
            far_src: mem.source.flatten_all()?.to_vec1::<u32>()?.into_iter().map(|s| s as u8).collect(),
            far_next: mem.next,
            far_pos: mem.pos,
            ind: mem.ind,
            prev_tokens: self.prev.as_ref().map(|p| p.2.clone()).unwrap_or_default(),
            m: mem.m,
            trunk,
            known,
            blocks: b * nb,
            mem_rows,
        };
        Ok((out, next))
    }

    /// Token each pointer column would copy: the previous window's
    /// positions, the current window's (`x[j + 1]`), the memory rows (via
    /// their block), the induction candidate.
    fn column_token(&self, out: &WindowOut, x: &[u32], t: usize, bi: usize, ti: usize, col: usize) -> u32 {
        let block = self.model.cfg.block;
        let nb = t / block;
        if col < t {
            // Previous window: its next token, or this window's first.
            match out.prev_tokens.get(bi * t + col + 1) {
                Some(&tok) if col + 1 < t => tok,
                _ if col + 1 == t && !out.prev_tokens.is_empty() => x[bi * t],
                _ => u32::MAX,
            }
        } else if col < 2 * t {
            let j = col - t;
            if j + 1 < t {
                x[bi * t + j + 1]
            } else {
                u32::MAX
            }
        } else if col < 2 * t + out.m {
            out.far_next[(bi * nb + ti / block) * out.m + (col - 2 * t)]
        } else if col == 2 * t + out.m {
            // No candidates without the memory (the column is masked).
            out.ind.get(bi * t + ti).filter(|c| c.len > 0).map_or(u32::MAX, |c| c.tok)
        } else {
            u32::MAX
        }
    }

    /// Stream position of the token column `col` copies (`None` for the
    /// null or an empty column). The window starts at `self.pos`.
    fn column_source(&self, out: &WindowOut, t: usize, bi: usize, ti: usize, col: usize) -> Option<u64> {
        let (block, nb, start) = (self.model.cfg.block, t / self.model.cfg.block, self.pos);
        if col < t {
            (!out.prev_tokens.is_empty()).then(|| start - t as u64 + col as u64 + 1)
        } else if col < 2 * t {
            Some(start + (col - t) as u64 + 1)
        } else if col < 2 * t + out.m {
            let p = out.far_pos[(bi * nb + ti / block) * out.m + (col - 2 * t)];
            (p != u64::MAX).then_some(p + 1)
        } else if col == 2 * t + out.m {
            let c = out.ind.get(bi * t + ti).filter(|c| c.len > 0)?;
            (start + ti as u64 + 1).checked_sub(c.dist as u64)
        } else {
            None
        }
    }

    /// Where column `col` copies from, as a bit: 1 the local window, 2/4/8 a
    /// memory row found lexically / semantically / from the unindexed newest
    /// tokens, 16 the induction column.
    fn column_kind(&self, out: &WindowOut, t: usize, bi: usize, ti: usize, col: usize) -> u8 {
        let nb = t / self.model.cfg.block;
        if col < 2 * t {
            1
        } else if col < 2 * t + out.m {
            let s =
                out.far_src.get((bi * nb + ti / self.model.cfg.block) * out.m + (col - 2 * t)).copied().unwrap_or(0);
            2 << s.min(2)
        } else {
            16
        }
    }

    /// Per-token loss of the mixture `a_null·p_vocab + Σ a·[next]`, `(B·T,)`.
    pub fn losses(&self, out: &WindowOut, x: &[u32], y: &[u32]) -> Result<Tensor> {
        Ok(self.losses_and_pointer(out, x, y, None)?.0)
    }

    /// The mixture loss, the pointer's own loss `−ln Σ_{j hits} a_j`
    /// (without the vocabulary), both `(B·T,)`, and whether some column
    /// hits each target. A column hits when it copies the target token; for
    /// a fact answer with a statement span in `src`, only a column copying
    /// from the statement itself hits the pointer loss — another occurrence
    /// of the same token (a syllable of an older answer) is not the fact.
    pub fn losses_and_pointer(
        &self,
        out: &WindowOut,
        x: &[u32],
        y: &[u32],
        src: Option<&[crate::data::Src]>,
    ) -> Result<(Tensor, Tensor, Vec<u8>)> {
        let (b, t, l) = out.point.dims3()?;
        let mut hit = vec![0f32; b * t * l];
        let mut own = vec![0f32; b * t * l];
        let mut found = vec![0u8; b * t];
        for bi in 0..b {
            for ti in 0..t {
                let r = bi * t + ti;
                let span = src.map_or(crate::data::NO_SRC, |s| s[r]);
                for col in 0..l {
                    if self.column_token(out, x, t, bi, ti, col) == y[r] {
                        hit[r * l + col] = 1.0;
                        let from_statement = span == crate::data::NO_SRC
                            || self.column_source(out, t, bi, ti, col).is_some_and(|p| p >= span.0 && p < span.1);
                        if from_statement {
                            own[r * l + col] = 1.0;
                            found[r] |= self.column_kind(out, t, bi, ti, col);
                        }
                    }
                }
            }
        }
        // In log space throughout: ln(copy mass) is the log-sum-exp of the
        // hit columns of ln point, ln a_null its null column.
        let miss: Vec<f32> = hit.iter().map(|&h| if h > 0.0 { 0.0 } else { -1e9 }).collect();
        let miss = Tensor::from_vec(miss, (b, t, l), out.point.device())?;
        let c = (&out.log_point + miss)?.log_sum_exp(2)?.flatten_all()?;
        let miss_own: Vec<f32> = own.iter().map(|&h| if h > 0.0 { 0.0 } else { -1e9 }).collect();
        let miss_own = Tensor::from_vec(miss_own, (b, t, l), out.point.device())?;
        let c_own = (&out.log_point + miss_own)?.log_sum_exp(2)?.flatten_all()?;
        let ce = token_losses(&out.logits, y)?;
        let log_g = out.log_point.narrow(2, l - 1, 1)?.flatten_all()?;
        // −ln(a_null·e^−ce + c), as a log-sum-exp of the two branches.
        let a = (log_g - ce)?;
        let both = Tensor::stack(&[&a, &c], 1)?;
        let mx = both.max_keepdim(1)?.detach();
        let lse = (both.broadcast_sub(&mx)?.exp()?.sum_keepdim(1)?.log()? + mx)?;
        Ok((lse.flatten_all()?.neg()?, c_own.neg()?, found))
    }

    /// Top-1 token of the mixture for every position, `(B·T)`.
    pub fn predict(&self, out: &WindowOut, x: &[u32]) -> Result<Vec<u32>> {
        let (b, t, l) = out.point.dims3()?;
        let v = out.logits.dim(2)?;
        let logits = out.logits.flatten_all()?.to_vec1::<f32>()?;
        let point = out.point.flatten_all()?.to_vec1::<f32>()?;
        let gate = out.gate.flatten_all()?.to_vec1::<f32>()?;
        let mut pred = Vec::with_capacity(b * t);
        for bi in 0..b {
            for ti in 0..t {
                let r = bi * t + ti;
                let row = &logits[r * v..(r + 1) * v];
                let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let z: f32 = row.iter().map(|&x| (x - mx).exp()).sum();
                let g = gate[r];
                let mut copy: std::collections::HashMap<u32, f32> = Default::default();
                for col in 0..l {
                    let a = point[r * l + col];
                    let tok = self.column_token(out, x, t, bi, ti, col);
                    if a > 0.0 && (tok as usize) < v {
                        *copy.entry(tok).or_default() += a;
                    }
                }
                let best_vocab = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
                let score =
                    |tok: usize| g * (row[tok] - mx).exp() / z + copy.get(&(tok as u32)).copied().unwrap_or(0.0);
                let mut best = (best_vocab, score(best_vocab));
                for &tok in copy.keys() {
                    let sc = score(tok as usize);
                    if sc > best.1 {
                        best = (tok as usize, sc);
                    }
                }
                pred.push(best.0 as u32);
            }
        }
        Ok(pred)
    }

    /// Commit a window: carry the state, keep its keys/values for the next
    /// window's local attention, and write it to the memories.
    pub fn commit(&mut self, x: &[u32], t: usize, out: &WindowOut, next: State) -> Result<()> {
        self.prev = Some((out.trunk.k.detach(), out.trunk.v.detach(), x.to_vec()));
        self.pos += t as u64;
        self.state =
            if self.reset_state { self.model.zero_state(self.memories.len(), out.logits.device())? } else { next };
        if self.use_memory {
            let dim = self.model.cfg.mem_dim;
            let k = out.trunk.k.flatten_all()?.to_vec1::<f32>()?;
            let v = out.trunk.v.flatten_all()?.to_vec1::<f32>()?;
            self.memories.par_iter_mut().enumerate().for_each(|(bi, m)| {
                m.write(
                    &x[bi * t..(bi + 1) * t],
                    &k[bi * t * dim..(bi + 1) * t * dim],
                    &v[bi * t * dim..(bi + 1) * t * dim],
                )
            });
        }
        Ok(())
    }
}

/// Resident memory of this process in GB (Linux).
fn rss_gb() -> f64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1).and_then(|p| p.parse::<f64>().ok()))
        .map_or(0.0, |pages| pages * 4096.0 / 1e9)
}

/// Per-token cross entropy `(N,)` of logits `(B, T, V)` against targets.
pub fn token_losses(logits: &Tensor, targets: &[u32]) -> Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    let op = crate::layers::CrossEntropy { targets: std::sync::Arc::new(targets.to_vec()) };
    logits.reshape((b * t, v))?.contiguous()?.apply_op1(op)
}

/// Linear warmup, then cosine decay to 40%. With a time limit the decay
/// follows elapsed time, so the schedule ends exactly when time runs out.
fn lr_at(cfg: &TrainConfig, step: usize, elapsed: f64) -> f64 {
    if step < cfg.warmup {
        return cfg.lr * (step + 1) as f64 / cfg.warmup as f64;
    }
    let p = if cfg.time_limit > 0 {
        elapsed / cfg.time_limit as f64
    } else {
        (step - cfg.warmup) as f64 / (cfg.steps - cfg.warmup).max(1) as f64
    };
    // The floor stays high: with 2-trit weights and a straight-through
    // estimator a small lr freezes the levels.
    cfg.lr * (0.4 + 0.6 * 0.5 * (1.0 + (std::f64::consts::PI * p.min(1.0)).cos()))
}

/// Scale gradients so that their global norm is at most `max`.
fn clip(grads: &mut candle_core::backprop::GradStore, vars: &[candle_core::Var], max: f64) -> Result<f64> {
    let mut total = 0f64;
    for v in vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            total += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        }
    }
    let norm = total.sqrt();
    if norm > max {
        let s = max / norm;
        for v in vars {
            if let Some(g) = grads.remove(v.as_tensor()) {
                grads.insert(v.as_tensor(), (g * s)?);
            }
        }
    }
    Ok(norm)
}

pub fn train(
    cfg: &TrainConfig,
    mcfg: Config,
    tokens: &[u16],
    val: &[u16],
    tok: &crate::tokenizer::Tokenizer,
) -> Result<()> {
    let device = Device::Cpu;
    std::fs::create_dir_all(&cfg.out)?;
    std::fs::write(
        cfg.out.join("model.cfg"),
        format!("layers={}\nmlp={}\nstate_trits={}\n", mcfg.layers, mcfg.mlp, mcfg.state_trits),
    )?;
    println!("rss at start (corpus loaded): {:.2} GB", rss_gb());
    let mut varmap = VarMap::new();
    let model = Model::new(VarBuilder::from_varmap(&varmap, DType::F32, &device), mcfg.clone())?;
    let ckpt = cfg.out.join("model.safetensors");
    // The exact resume state (see `save_resume`): weights, moving average,
    // optimizer moments, schedule clock, runners with their stream memories.
    let resume = cfg.out.join("resume");
    let resumed: Option<ResumeInfo> =
        if resume.join("state.txt").exists() { Some(ResumeInfo::read(&resume)?) } else { None };
    let mut start_step = 0;
    if let Some(r) = &resumed {
        varmap.load(resume.join("model.safetensors"))?;
        start_step = r.step;
        println!("resumed exactly from step {start_step} ({:.0}s of schedule done)", r.elapsed);
    } else if ckpt.exists() {
        varmap.load(&ckpt)?;
        start_step =
            std::fs::read_to_string(cfg.out.join("step")).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
        println!(
            "WARNING: resumed from step {start_step} without resume/ state: the optimizer moments, the \
             schedule clock and the stream memories start over"
        );
    } else if let Some(init) = &cfg.init {
        // Warm start: every tensor the old checkpoint has; new ones keep their init.
        let old = candle_core::safetensors::load(init, &device)?;
        let data = varmap.data().lock().expect("varmap lock");
        let mut n = 0;
        for (name, var) in data.iter() {
            if let Some(t) = old.get(name) {
                var.set(t)?;
                n += 1;
            }
        }
        println!("warm start from {}: {n}/{} tensors", init.display(), data.len());
        // Steps the old checkpoint lacks start at the optimum for its weights.
        for (name, q) in model.qtensors() {
            let step = format!("{name}_step");
            if !old.contains_key(&step) {
                if let Some(var) = data.get(&step) {
                    var.set(&q.optimal_theta()?)?;
                }
            }
        }
    } else {
        // A fresh run: priors for the memory head's scalar features. A
        // longer induction match is likelier right (+1 nat per length bin);
        // rows of a block the memory is sure about are likelier right.
        let data = varmap.data().lock().expect("varmap lock");
        let len: Vec<f32> = (0..crate::model::LEN_BINS).map(|i| i as f32).collect();
        data["ind_len"].set(&Tensor::from_vec(len, crate::model::LEN_BINS, &device)?)?;
        data["far_verdict"].set(&Tensor::new(&[1f32, 0.0, -1.0], &device)?)?;
    }
    let vars = varmap.all_vars();
    println!("parameters: {}", Model::n_params(&vars));
    // Moving average of the latent weights (and quantization steps): the
    // averaged latents round to a steadier ternary model than the last step.
    let ema_file = cfg.out.join("model.ema.safetensors");
    let mut ema: Vec<(String, candle_core::Var, Tensor)> = Vec::new();
    if cfg.ema > 0.0 {
        let saved = if resumed.is_some() && resume.join("model.ema.safetensors").exists() {
            candle_core::safetensors::load(resume.join("model.ema.safetensors"), &device)?
        } else if start_step > 0 && ema_file.exists() {
            candle_core::safetensors::load(&ema_file, &device)?
        } else {
            Default::default()
        };
        for (name, v) in varmap.data().lock().expect("varmap lock").iter() {
            let t = saved.get(name).cloned().unwrap_or_else(|| v.as_tensor().copy().expect("copy"));
            ema.push((name.clone(), v.clone(), t));
        }
    }
    let mut groups: Vec<Vec<(String, candle_core::Var)>> = vec![Vec::new(); OPT_GROUPS.len()];
    for (n, v) in varmap.data().lock().expect("varmap lock").iter() {
        groups[opt_group(n)].push((n.clone(), v.clone()));
    }
    // With automatic steps θ follows the latents (see `anchor_steps`).
    if cfg.auto_steps {
        groups[1].clear();
        anchor_steps(&model, &varmap)?;
    }
    let mut opts = groups
        .into_iter()
        .zip(OPT_GROUPS)
        .map(|(vars, (_, mult))| {
            let mut o = AdamW::new(vars, cfg.lr * mult)?;
            o.beta2 = cfg.beta2;
            Ok(o)
        })
        .collect::<Result<Vec<_>>>()?;
    if let Some(r) = &resumed {
        let saved = candle_core::safetensors::load(resume.join("optim.safetensors"), &device)?;
        for (o, (name, _)) in opts.iter_mut().zip(OPT_GROUPS) {
            o.load_state(name, &saved)?;
            o.t = r.t.get(name).copied().unwrap_or(0);
        }
    }

    let (b, t) = (cfg.batch, cfg.window);
    // Streams run in groups of `micro` (gradients accumulate): the autograd
    // graph of one group is freed before the next, so the peak memory of a
    // step is a third of the whole batch's.
    let micro = cfg.micro.clamp(1, b);
    // Groups running at once (their graphs are alive together: peak memory
    // grows with it).
    let parallel = cfg.parallel.max(1);
    assert!(b % micro == 0, "batch {b} is not a multiple of micro {micro}");
    // Two-trit K/V, exactly as inference stores them.
    let mut runners = (0..b / micro)
        .map(|_| {
            let mut r = Runner::new(model.clone(), micro, cfg.memory, cfg.max_tokens, KvPrecision::Trit2, &device)?;
            r.top_k = cfg.top_k;
            r.rows = cfg.rows;
            Ok(r)
        })
        .collect::<Result<Vec<_>>>()?;
    let ep = Episodes::new(tok);
    let region = tokens.len() / b;
    let mut streams: Vec<TaskStream> =
        (0..b).map(|i| TaskStream::new(i * region, region, cfg.seed * 1000 + i as u64)).collect();
    let lines = tok.newline_tokens();
    for s in &mut streams {
        s.jump = cfg.jump;
        s.lines = lines.clone();
        s.p_reread = cfg.p_reread;
        s.p_episode = cfg.p_episode;
        // Episodes and re-reading reach as far as the training memory,
        // minus two windows, so a statement is still in memory when asked.
        s.distance = (27, cfg.max_tokens.saturating_sub(2 * 243).max(243));
    }
    if resumed.is_some() {
        for (g, r) in runners.iter_mut().enumerate() {
            r.load(&resume, &format!("runner{g}"), cfg.max_tokens, KvPrecision::Trit2)?;
        }
    }
    // SIGTERM / SIGINT: finish the step, save the resume state, stop.
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    for sig in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
        signal_hook::flag::register(sig, stop.clone()).map_err(candle_core::Error::wrap)?;
    }
    // Resuming fast-forwards the streams (they are deterministic), so data
    // is neither repeated nor skipped.
    for s in &mut streams {
        for _ in 0..start_step {
            s.window(tokens, &ep, t);
        }
    }

    println!("rss after setup: {:.2} GB", rss_gb());
    let clock = Instant::now();
    // Schedule time, carried over a resume (the --hours cosine continues).
    let elapsed0 = resumed.as_ref().map_or(0.0, |r| r.elapsed);
    let elapsed = || elapsed0 + clock.elapsed().as_secs_f64();
    let (mut sum_loss, mut sum_known, mut sum_blocks, mut n_steps, mut n_logged) =
        (0f64, 0usize, 0usize, 0usize, 0usize);
    // Mean loss per token kind: plain text, fact answers, re-read spans, templates.
    let mut by_kind = [(0f64, 0usize); 4];
    // Fact answers some pointer column could copy.
    let mut found = (0usize, 0usize);
    let mut facts = FactStats::default();
    let mut timing = [0f64; 4];
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(cfg.out.join("log.tsv"))?;
    let n = (b * t) as f64;
    for step in start_step..cfg.steps {
        let lr = lr_at(cfg, step, elapsed());
        for (o, (name, mult)) in opts.iter_mut().zip(OPT_GROUPS) {
            o.lr = lr * mult;
            // The K/V step settles during warm-up and then stays: crossing a
            // rounding boundary later would double every new key's step at
            // once while the memory still holds keys on the old grid.
            if name == "kv" && step >= cfg.warmup {
                o.lr = 0.0;
            }
        }
        let mut x = Vec::with_capacity(b * t);
        let mut y = Vec::with_capacity(b * t);
        let mut kinds = Vec::with_capacity(b * t);
        let mut srcs = Vec::with_capacity(b * t);
        for s in &mut streams {
            let (w, k, src) = s.window_src(tokens, &ep, t);
            x.extend_from_slice(&w[..t]);
            y.extend_from_slice(&w[1..]);
            kinds.extend_from_slice(&k[1..]);
            srcs.extend_from_slice(&src[1..]);
        }
        let mut grads: Option<candle_core::backprop::GradStore> = None;
        let mut loss_value = 0f32;
        let mut lv = Vec::with_capacity(b * t);
        // One group: forward, losses, backward, commit. Groups are
        // independent (own streams, memories, state), so `parallel` of them
        // run at once; their gradients are summed in group order.
        let run_group = |gi: usize, runner: &mut Runner| -> Result<GroupOut> {
            let r = gi * micro * t..(gi + 1) * micro * t;
            let (xs, ys, ks, ss) = (&x[r.clone()], &y[r.clone()], &kinds[r.clone()], &srcs[r]);
            let t0 = Instant::now();
            let (out, next) = runner.forward(xs, t, &device)?;
            let (losses, pointer, findable) = runner.losses_and_pointer(&out, xs, ys, Some(ss))?;
            // Auxiliary pointer loss where the answer is in the context (fact
            // answers, re-read spans) and some column holds it — otherwise it
            // would pull the pointer towards uniform: teaches the pointer to
            // find the row instead of leaning on the vocabulary.
            let mask: Vec<f32> = ks
                .iter()
                .zip(&findable)
                .map(|(&k, &f)| match (k, f != 0) {
                    (crate::data::ANSWER, true) => cfg.aux_fact as f32,
                    (crate::data::REREAD, true) => 1.0,
                    _ => 0.0,
                })
                .collect();
            let mut found = (0, 0);
            let mut facts = FactStats::default();
            for (i, (&k, &f)) in ks.iter().zip(&findable).enumerate() {
                if k == crate::data::ANSWER {
                    found.0 += usize::from(f != 0);
                    found.1 += 1;
                    // Distance from the answer back to its statement.
                    let at = runner.pos + (i % t) as u64 + 1;
                    let d = at.saturating_sub(ss[i].0);
                    let bin = if d <= crate::model::LOCAL as u64 {
                        0
                    } else if d <= 19_683 {
                        1
                    } else {
                        2
                    };
                    facts.by_dist[bin].0 += usize::from(f != 0);
                    facts.by_dist[bin].1 += 1;
                    for (j, n) in facts.from.iter_mut().enumerate() {
                        *n += usize::from(f & (1 << j) != 0);
                    }
                }
            }
            let mask = Tensor::from_vec(mask, ks.len(), &device)?;
            let aux = ((pointer * mask)?.sum_all()? * (cfg.aux / n))?;
            let loss = ((losses.sum_all()? / n)? + aux)?;
            let t1 = Instant::now();
            let loss_value = loss.to_scalar::<f32>()?;
            let grads = loss.backward()?;
            let t2 = Instant::now();
            let lv = losses.to_vec1::<f32>()?;
            let (known, blocks) = (out.known, out.blocks);
            // Commit now: the group's graph is freed before the next group.
            runner.commit(xs, t, &out, next)?;
            Ok(GroupOut { grads, loss: loss_value, lv, known, blocks, found, facts, time: [t1 - t0, t2 - t1] })
        };
        for (ci, chunk) in runners.chunks_mut(parallel).enumerate() {
            let round = Instant::now();
            let outs: Vec<Result<GroupOut>> =
                chunk.par_iter_mut().enumerate().map(|(j, r)| run_group(ci * parallel + j, r)).collect();
            // Wall-clock of the round, split into forward and backward in the
            // proportion of the groups' own times.
            let wall = round.elapsed().as_secs_f64();
            let (mut fwd, mut bwd) = (0f64, 0f64);
            for o in outs.iter().flatten() {
                fwd += o.time[0].as_secs_f64();
                bwd += o.time[1].as_secs_f64();
            }
            timing[0] += wall * fwd / (fwd + bwd).max(1e-9);
            timing[1] += wall * bwd / (fwd + bwd).max(1e-9);
            for o in outs {
                let o = o?;
                grads = Some(match grads {
                    None => o.grads,
                    Some(mut total) => {
                        for v in &vars {
                            if let Some(gv) = o.grads.get(v.as_tensor()) {
                                let sum = match total.remove(v.as_tensor()) {
                                    Some(tv) => (tv + gv)?,
                                    None => gv.clone(),
                                };
                                total.insert(v.as_tensor(), sum);
                            }
                        }
                        total
                    }
                });
                loss_value += o.loss;
                lv.extend(o.lv);
                sum_known += o.known;
                sum_blocks += o.blocks;
                facts.add(&o.facts);
                found.0 += o.found.0;
                found.1 += o.found.1;
            }
        }
        let mut grads = grads.expect("at least one group");
        let t2 = Instant::now();
        let norm = clip(&mut grads, &vars, cfg.clip)?;
        if !loss_value.is_finite() || !norm.is_finite() {
            // Never let a non-finite update into the weights.
            println!("step {}: skipped (loss {loss_value}, grad norm {norm})", step + 1);
            // Still stop on time or a signal, with the state saved.
            if stop.load(std::sync::atomic::Ordering::Relaxed)
                || (cfg.time_limit > 0 && elapsed() >= cfg.time_limit as f64)
            {
                save_resume(&resume, &varmap, &ema, &opts, &runners, step + 1, elapsed())?;
                varmap.save(&ckpt)?;
                std::fs::write(cfg.out.join("step"), format!("{}", step + 1))?;
                println!("stopped at step {} after a skipped step; resume state saved", step + 1);
                break;
            }
            continue;
        }
        for o in &mut opts {
            o.step(&grads)?;
        }
        if cfg.auto_steps {
            anchor_steps(&model, &varmap)?;
        }
        // Early on the average follows the weights (horizon grows with the step).
        let d = cfg.ema.min((1.0 + step as f64) / (10.0 + step as f64));
        for (_, v, e) in &mut ema {
            // Detached: otherwise every average keeps the previous one alive
            // through the autograd graph (the whole history, ~64 MB a step).
            *e = ((&*e * d)? + (v.as_tensor() * (1.0 - d))?)?.detach();
        }
        timing[2] += t2.elapsed().as_secs_f64();

        sum_loss += lv.iter().map(|&l| l as f64).sum::<f64>() / lv.len() as f64;
        for (l, &k) in lv.iter().zip(&kinds) {
            by_kind[k as usize].0 += *l as f64;
            by_kind[k as usize].1 += 1;
        }
        n_steps += 1;
        n_logged += 1;

        if (step + 1) % cfg.log_every == 0 {
            let el = clock.elapsed().as_secs_f64();
            let line = format!(
                "{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.3}\t{:.0}\t{:.2e}\tfwd {:.1}s bwd {:.1}s opt {:.1}s ckpt {:.1}s",
                step + 1,
                sum_loss / n_logged as f64,
                by_kind[0].0 / by_kind[0].1.max(1) as f64,
                if by_kind[1].1 > 0 { by_kind[1].0 / by_kind[1].1 as f64 } else { f64::NAN },
                if by_kind[2].1 > 0 { by_kind[2].0 / by_kind[2].1 as f64 } else { f64::NAN },
                if by_kind[3].1 > 0 { by_kind[3].0 / by_kind[3].1 as f64 } else { f64::NAN },
                sum_known as f64 / sum_blocks.max(1) as f64,
                (n_steps * b * t) as f64 / el,
                lr_at(cfg, step, elapsed()),
                timing[0] / n_logged as f64,
                timing[1] / n_logged as f64,
                timing[2] / n_logged as f64,
                timing[3] / n_logged as f64,
            );
            println!("step {line}");
            use std::io::Write as _;
            writeln!(log, "{line}")?;
            (sum_loss, sum_known, sum_blocks, n_logged) = (0.0, 0, 0, 0);
            by_kind = [(0.0, 0); 4];
            println!("answers findable: {:.3} | rss {:.2} GB", found.0 as f64 / found.1.max(1) as f64, rss_gb());
            let d = facts.by_dist;
            println!(
                "facts found by distance: near {}/{} mid {}/{} far {}/{} | from window {} lexical {} semantic {} newest {} induction {}",
                d[0].0, d[0].1, d[1].0, d[1].1, d[2].0, d[2].1,
                facts.from[0], facts.from[1], facts.from[2], facts.from[3], facts.from[4]
            );
            facts = FactStats::default();
            found = (0, 0);
            timing = [0.0; 4];
        }
        if (step + 1) % (cfg.log_every * 9) == 0 {
            let (mut zero, mut clipped, mut err, mut n) = (0.0, 0.0, 0.0, 0.0);
            let (mut drift, mut rows) = (0usize, 0usize);
            for (_, q) in runners[0].model.qtensors() {
                let (z, c, e) = q.health()?;
                (zero, clipped, err, n) = (zero + z, clipped + c, err + e, n + 1.0);
                // Rows whose learned exponent differs from the MSE-optimal one.
                let learned = q.theta.to_vec1::<f32>()?;
                let optimal = q.optimal_theta()?.to_vec1::<f32>()?;
                drift += learned.iter().zip(&optimal).filter(|(a, b)| a.round() != b.round()).count();
                rows += learned.len();
            }
            println!("steps: {:.3} of rows differ from the MSE-optimal exponent", drift as f64 / rows.max(1) as f64);
            // Does the model use the ternary verdict and the induction features?
            let data = varmap.data().lock().expect("varmap lock");
            let show = |n: &str| {
                data.get(n).and_then(|v| v.as_tensor().to_vec1::<f32>().ok()).map_or(String::new(), |v| {
                    format!("{n} [{}]", v.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>().join(" "))
                })
            };
            println!(
                "head: {} | {} | {} | {} | {} | {} | {} | {} | {}",
                show("gate_verdict"),
                show("ind_verdict"),
                show("far_verdict"),
                show("far_source"),
                show("ind_len"),
                show("ind_count"),
                show("ind_dist"),
                show("ind_p"),
                show("kv_step")
            );
            drop(data);
            println!(
                "quant: zero {:.3} clipped {:.4} rel.err {:.3} | grad norm {norm:.3}",
                zero / n,
                clipped / n,
                err / n
            );
        }
        let save_ema = || -> Result<()> {
            if !ema.is_empty() {
                candle_core::safetensors::save(&ema_weights(&ema, &model, cfg.auto_steps)?, &ema_file)?;
            }
            Ok(())
        };
        // The last step, by count or by the time limit.
        let time_up = cfg.time_limit > 0 && elapsed() >= cfg.time_limit as f64;
        let stopped = stop.load(std::sync::atomic::Ordering::Relaxed);
        let last = step + 1 == cfg.steps || time_up;
        if !stopped && cfg.val_every > 0 && ((step + 1) % cfg.val_every == 0 || last) && !val.is_empty() {
            let t0 = Instant::now();
            let now = crate::eval::val_loss_fitted(model.clone(), val, 9, 27, cfg.memory, KvPrecision::Trit2)?;
            let avg = if ema.is_empty() {
                f64::NAN
            } else {
                let vm = VarMap::new();
                let m = Model::new(VarBuilder::from_varmap(&vm, DType::F32, &device), mcfg.clone())?;
                let data = vm.data().lock().expect("varmap lock");
                for (name, t) in ema_weights(&ema, &model, cfg.auto_steps)? {
                    if let Some(v) = data.get(&name) {
                        v.set(&t)?;
                    }
                }
                drop(data);
                crate::eval::val_loss_fitted(m, val, 9, 27, cfg.memory, KvPrecision::Trit2)?
            };
            println!("val: step {} loss {now:.4} ema {avg:.4} ({:.0}s)", step + 1, t0.elapsed().as_secs_f64());
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(cfg.out.join("val.tsv"))?;
            use std::io::Write as _;
            writeln!(f, "{}\t{now:.4}\t{avg:.4}", step + 1)?;
            // Keep every validated point: weights and average by step.
            let snaps = cfg.out.join("snapshots");
            std::fs::create_dir_all(&snaps)?;
            varmap.save(snaps.join(format!("step{}.safetensors", step + 1)))?;
            if !ema.is_empty() {
                let m = ema_weights(&ema, &model, cfg.auto_steps)?;
                candle_core::safetensors::save(&m, snaps.join(format!("step{}.ema.safetensors", step + 1)))?;
            }
        }
        if (step + 1) % cfg.ckpt_every == 0 || last || stopped {
            let t0 = Instant::now();
            save_resume(&resume, &varmap, &ema, &opts, &runners, step + 1, elapsed())?;
            save_ema()?;
            varmap.save(&ckpt)?;
            std::fs::write(cfg.out.join("step"), format!("{}", step + 1))?;
            timing[3] += t0.elapsed().as_secs_f64();
        }
        if stopped {
            println!("stopped by a signal at step {}; resume state saved", step + 1);
            break;
        }
        if time_up {
            println!("time limit reached at step {}", step + 1);
            break;
        }
    }
    Ok(())
}

/// Optimizer groups and their lr multipliers:
/// - `w`: weights;
/// - `s`: quantization steps θ (LSQ mode only) — Adam normalizes their
///   gradient and a rounding crossing rescales a whole row, so 27× slower;
/// - `kv`: the shared K/V step (one θ for many activations), 3× slower;
/// - `head`: the memory head's scalar features (null bias, induction and
///   row biases) — Adam moves a scalar at most lr a step, and they need
///   several nats (analyst 7: gate_b had not moved from 8.0 by step 891);
/// - `sign`: the Hadamard recurrence signs, 9× slower — near zero an STE
///   sign flips from step to step and reshapes the recurrence.
const OPT_GROUPS: [(&str, f64); 5] =
    [("w", 1.0), ("s", 1.0 / 27.0), ("kv", 1.0 / 3.0), ("head", 9.0), ("sign", 1.0 / 9.0)];

/// The memory head's scalar features (optimizer group `head`).
const HEAD_SCALARS: [&str; 9] =
    ["gate_b", "gate_verdict", "ind_len", "ind_count", "ind_dist", "ind_verdict", "ind_p", "far_verdict", "far_source"];

fn opt_group(name: &str) -> usize {
    if name == "kv_step" {
        2
    } else if name.ends_with("_step") {
        1
    } else if HEAD_SCALARS.contains(&name) {
        3
    } else if name.ends_with(".sign") {
        4
    } else {
        0
    }
}

/// Set every weight row's step to the MSE optimum of its latents.
///
/// The exponent is an integer and moves only when the optimum is more than
/// 0.6 away from it: with a bare rounding, rows sitting at a boundary (most
/// rows of a tensor move together) flipped their step back and forth for
/// hundreds of steps (analyst 8: 68–81% of mlp.w2 rows within ±0.05).
fn anchor_steps(model: &Model, varmap: &VarMap) -> Result<()> {
    let data = varmap.data().lock().expect("varmap lock");
    for (name, q) in model.qtensors() {
        if let Some(var) = data.get(&format!("{name}_step")) {
            let cur = var.as_tensor().to_vec1::<f32>()?;
            let opt = crate::layers::optimal_theta_of(&q.w, q.levels)?;
            let next: Vec<f32> = cur
                .iter()
                .zip(&opt)
                .map(|(&c, &o)| if c.fract() != 0.0 || (o - c).abs() > 0.6 { o.round() } else { c })
                .collect();
            var.set(&Tensor::from_vec(next, cur.len(), var.device())?)?;
        }
    }
    Ok(())
}

/// The moving average as saved and evaluated. With automatic steps the
/// averaged θ is not the step of the averaged latents: each row's exponent
/// is recomputed from the averaged latents (rounded).
fn ema_weights(
    ema: &[(String, candle_core::Var, Tensor)],
    model: &Model,
    auto_steps: bool,
) -> Result<std::collections::HashMap<String, Tensor>> {
    let mut m: std::collections::HashMap<String, Tensor> = ema.iter().map(|(n, _, t)| (n.clone(), t.clone())).collect();
    if auto_steps {
        for (name, q) in model.qtensors() {
            let step = format!("{name}_step");
            if let (Some(w), true) = (m.get(&name), m.contains_key(&step)) {
                let theta: Vec<f32> =
                    crate::layers::optimal_theta_of(w, q.levels)?.into_iter().map(f32::round).collect();
                let n = theta.len();
                m.insert(step, Tensor::from_vec(theta, n, w.device())?);
            }
        }
    }
    Ok(m)
}

/// What one group of streams returns from a step.
struct GroupOut {
    grads: candle_core::backprop::GradStore,
    loss: f32,
    lv: Vec<f32>,
    known: usize,
    blocks: usize,
    found: (usize, usize),
    facts: FactStats,
    time: [std::time::Duration; 2],
}

/// Fact answers whose statement some pointer column copies from, by the
/// distance back to the statement (local window / up to 3^9 / farther) and
/// by where the copying column looks (window, lexical row, semantic row,
/// newest row, induction).
#[derive(Default, Clone, Copy)]
struct FactStats {
    by_dist: [(usize, usize); 3],
    from: [usize; 5],
}

impl FactStats {
    fn add(&mut self, o: &FactStats) {
        for (a, b) in self.by_dist.iter_mut().zip(&o.by_dist) {
            (a.0, a.1) = (a.0 + b.0, a.1 + b.1);
        }
        for (a, b) in self.from.iter_mut().zip(&o.from) {
            *a += b;
        }
    }
}

/// What `state.txt` of a resume directory records.
struct ResumeInfo {
    step: usize,
    elapsed: f64,
    /// Updates done by each optimizer group (`t_{group}`).
    t: std::collections::HashMap<&'static str, usize>,
}

impl ResumeInfo {
    fn read(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("state.txt"))?;
        let field = |k: &str| -> Result<f64> {
            text.lines()
                .find_map(|l| l.strip_prefix(&format!("{k}=")))
                .and_then(|v| v.trim().parse().ok())
                .ok_or_else(|| candle_core::Error::Msg(format!("resume state.txt has no {k}")))
        };
        Ok(Self {
            step: field("step")? as usize,
            elapsed: field("elapsed")?,
            // A group missing from an older state starts fresh.
            t: OPT_GROUPS.iter().filter_map(|(g, _)| field(&format!("t_{g}")).ok().map(|t| (*g, t as usize))).collect(),
        })
    }
}

/// Write the exact resume state into `dir`: first into `dir.tmp`, then
/// swapped in, so a crash while saving leaves the previous state intact.
fn save_resume(
    dir: &Path,
    varmap: &VarMap,
    ema: &[(String, candle_core::Var, Tensor)],
    opts: &[AdamW],
    runners: &[Runner],
    step: usize,
    elapsed: f64,
) -> Result<()> {
    let tmp = dir.with_extension("tmp");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)?;
    varmap.save(tmp.join("model.safetensors"))?;
    if !ema.is_empty() {
        let m: std::collections::HashMap<String, Tensor> = ema.iter().map(|(n, _, t)| (n.clone(), t.clone())).collect();
        candle_core::safetensors::save(&m, tmp.join("model.ema.safetensors"))?;
    }
    let mut o = std::collections::HashMap::new();
    for (opt, (name, _)) in opts.iter().zip(OPT_GROUPS) {
        opt.state(name, &mut o);
    }
    candle_core::safetensors::save(&o, tmp.join("optim.safetensors"))?;
    for (g, r) in runners.iter().enumerate() {
        r.save(&tmp, &format!("runner{g}"))?;
    }
    std::fs::write(
        tmp.join("state.txt"),
        format!(
            "step={step}\nelapsed={elapsed}\n{}",
            opts.iter().zip(OPT_GROUPS).map(|(o, (g, _))| format!("t_{g}={}\n", o.t)).collect::<String>()
        ),
    )?;
    let old = dir.with_extension("old");
    let _ = std::fs::remove_dir_all(&old);
    if dir.exists() {
        std::fs::rename(dir, &old)?;
    }
    std::fs::rename(&tmp, dir)?;
    let _ = std::fs::remove_dir_all(&old);
    Ok(())
}

/// Load a trained model.
pub fn load_model(dir: &Path, mcfg: Config, device: &Device) -> Result<Model> {
    load_weights(&dir.join("model.safetensors"), mcfg, device)
}

/// Load a model from a weights file (`model.safetensors` or its moving
/// average `model.ema.safetensors`).
pub fn load_weights(file: &Path, mcfg: Config, device: &Device) -> Result<Model> {
    let mut varmap = VarMap::new();
    let model = Model::new(VarBuilder::from_varmap(&varmap, DType::F32, device), mcfg)?;
    varmap.load(file)?;
    Ok(model)
}

#[cfg(test)]
mod induction_tests {
    use super::*;

    #[test]
    fn induction_continues_the_longest_earlier_suffix() {
        let mut ind = Induction::new(1000);
        // 1 2 3 4 5 | 9 9 | 2 3 4 → after "2 3 4" the continuation is 5.
        let out = ind.window(&[1, 2, 3, 4, 5, 9, 9, 2, 3, 4]);
        assert_eq!(out[9], IndCand { tok: 5, len: 3, count: 1, dist: 6 });
        assert!(out[..9].iter().all(|c| c.len == 0));
        // Across windows, and preferring the longer suffix.
        let out = ind.window(&[7, 1, 2, 3, 4]);
        assert_eq!(out[4], IndCand { tok: 5, len: 4, count: 1, dist: 11 });
    }

    #[test]
    fn induction_extends_matches_beyond_the_hash_and_counts_them() {
        let mut ind = Induction::new(1000);
        let mut x: Vec<u32> = (1..=12).collect();
        x.push(99);
        x.extend(1..=12);
        let out = ind.window(&x);
        // Matched by the 8-token hash, then extended to all 12 tokens.
        assert_eq!(out[24], IndCand { tok: 99, len: 12, count: 1, dist: 13 });
        // A suffix seen twice before counts two occurrences.
        let out = ind.window(&[50, 10, 11, 12]);
        assert_eq!(out[3].count, 2);
    }

    #[test]
    fn induction_keeps_working_after_trimming() {
        let mut ind = Induction::new(12);
        ind.window(&[1, 2, 3, 4, 5, 6, 7, 8]);
        ind.window(&[10, 11, 12, 13, 14, 15, 16, 17]);
        let out = ind.window(&[11, 12, 13]);
        assert_eq!(out[2], IndCand { tok: 14, len: 3, count: 1, dist: 7 });
    }
}
