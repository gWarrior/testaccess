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
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rayon::prelude::*;
use snn_memory::{ContextConfig, ContextMemory, KvConfig, KvPrecision, Probe, Verdict};

use crate::data::{Episodes, TaskStream};
use crate::model::{BlockRows, Config, IndCand, MemBatch, Model, State, Trunk};

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
    /// Probability per token of a re-reading episode.
    pub p_reread: f64,
    /// Weight of the auxiliary pointer loss per answer / re-read token,
    /// relative to a token's mixture loss (template answers get none).
    pub aux: f64,
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
            top_k: 3,
            rows: 81,
            // Short in training so stored keys stay close to current weights.
            max_tokens: 59_049,
            clip: 1.0,
            log_every: 9,
            ckpt_every: 81,
            out: PathBuf::from("/home/user/data/run"),
            seed: 1,
            time_limit: 0,
            init: None,
            jump: true,
            p_reread: 1.0 / 729.0,
            aux: 1.0,
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
    /// n-gram hash → (last end position, occurrences).
    index: std::collections::HashMap<u64, (u32, u32)>,
    limit: usize,
}

pub fn ngram_key(g: &[u32]) -> u64 {
    g.iter().fold(0xcbf2_9ce4_8422_2325u64 ^ g.len() as u64, |h, &t| (h ^ t as u64).wrapping_mul(0x100_0000_01b3))
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
    pub fn new(dim: usize, max_tokens: usize, precision: KvPrecision) -> Self {
        let cfg = ContextConfig {
            max_tokens,
            kv: Some(KvConfig { precision, ..KvConfig::new(dim, dim) }),
            ..Default::default()
        };
        Self {
            mem: ContextMemory::new(cfg).expect("valid memory config"),
            history: Vec::new(),
            induction: Induction::new(max_tokens),
            local: Induction::new(243),
        }
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
                let tokens = &ctx[probe_end.saturating_sub(crate::model::PROBE)..probe_end];
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
    /// Token after each memory row and its position `(B, nb, M)`, and `M`.
    pub far_next: Vec<u32>,
    pub far_pos: Vec<u64>,
    /// Induction candidates `(B·T)` (empty without the memory).
    pub ind: Vec<IndCand>,
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
        Ok(Self { model, state, memories, use_memory, top_k: 3, rows: 81, reset_state: false })
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
                .map(|(bi, m)| m.local.window(&x[bi * t..(bi + 1) * t]))
                .collect::<Vec<_>>()
                .concat();
            (MemBatch::empty(b, nb, dim, device)?.with_induction(ind), 0, 0)
        };
        let h = self.model.head(&trunk, &mem)?;
        let out = WindowOut {
            logits: h.logits,
            point: h.point,
            gate: h.gate,
            far_next: mem.next,
            far_pos: mem.pos,
            ind: mem.ind,
            m: mem.m,
            trunk,
            known,
            blocks: b * nb,
            mem_rows,
        };
        Ok((out, next))
    }

    /// Token each pointer column would copy: `(B, L)` for the window part
    /// (`x[j + 1]`) and `(B, T, M)` for the memory rows via their block.
    fn column_token(&self, out: &WindowOut, x: &[u32], t: usize, bi: usize, ti: usize, col: usize) -> u32 {
        let block = self.model.cfg.block;
        let nb = t / block;
        if col < t {
            if col + 1 < t {
                x[bi * t + col + 1]
            } else {
                u32::MAX
            }
        } else if col < t + out.m {
            out.far_next[(bi * nb + ti / block) * out.m + (col - t)]
        } else if col == t + out.m {
            // No candidates without the memory (the column is masked).
            out.ind.get(bi * t + ti).filter(|c| c.len > 0).map_or(u32::MAX, |c| c.tok)
        } else {
            u32::MAX
        }
    }

    /// Per-token loss of the mixture `a_null·p_vocab + Σ a·[next]`, `(B·T,)`.
    pub fn losses(&self, out: &WindowOut, x: &[u32], y: &[u32]) -> Result<Tensor> {
        Ok(self.losses_and_pointer(out, x, y)?.0)
    }

    /// The mixture loss and the pointer's own loss `−ln Σ_{j: next_j = y} a_j`
    /// (without the vocabulary), both `(B·T,)`.
    pub fn losses_and_pointer(&self, out: &WindowOut, x: &[u32], y: &[u32]) -> Result<(Tensor, Tensor)> {
        let (b, t, l) = out.point.dims3()?;
        let mut hit = vec![0f32; b * t * l];
        for bi in 0..b {
            for ti in 0..t {
                let target = y[bi * t + ti];
                for col in 0..l {
                    if self.column_token(out, x, t, bi, ti, col) == target {
                        hit[(bi * t + ti) * l + col] = 1.0;
                    }
                }
            }
        }
        let hit = Tensor::from_vec(hit, (b, t, l), out.point.device())?;
        let copy = (&out.point * hit)?.sum(2)?.flatten_all()?;
        let ce = token_losses(&out.logits, y)?;
        let g = out.gate.flatten_all()?;
        // −ln(a_null·e^−ce + c), as a log-sum-exp of the two branches.
        let a = ((g + 1e-9)?.log()? - ce)?;
        let c = (copy + 1e-9)?.log()?;
        let both = Tensor::stack(&[&a, &c], 1)?;
        let mx = both.max_keepdim(1)?.detach();
        let lse = (both.broadcast_sub(&mx)?.exp()?.sum_keepdim(1)?.log()? + mx)?;
        Ok((lse.flatten_all()?.neg()?, c.neg()?))
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

    /// Commit a window: carry the state and write it to the memories.
    pub fn commit(&mut self, x: &[u32], t: usize, out: &WindowOut, next: State) -> Result<()> {
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

pub fn train(cfg: &TrainConfig, mcfg: Config, tokens: &[u16], tok: &crate::tokenizer::Tokenizer) -> Result<()> {
    let device = Device::Cpu;
    std::fs::create_dir_all(&cfg.out)?;
    std::fs::write(cfg.out.join("model.cfg"), format!("ternary_h={}\n", u8::from(mcfg.ternary_h)))?;
    let mut varmap = VarMap::new();
    let model = Model::new(VarBuilder::from_varmap(&varmap, DType::F32, &device), mcfg.clone())?;
    let ckpt = cfg.out.join("model.safetensors");
    let mut start_step = 0;
    if ckpt.exists() {
        varmap.load(&ckpt)?;
        start_step =
            std::fs::read_to_string(cfg.out.join("step")).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
        println!("resumed from step {start_step}");
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
    }
    let vars = varmap.all_vars();
    println!("parameters: {}", Model::n_params(&vars));
    // Quantization steps θ move 27× slower: Adam normalizes their gradient,
    // and a crossing of a rounding boundary rescales a whole row.
    let (steps, weights): (Vec<_>, Vec<_>) = {
        let data = varmap.data().lock().expect("varmap lock");
        data.iter().map(|(n, v)| (n.ends_with("_step"), v.clone())).partition(|(is_step, _)| *is_step)
    };
    let steps: Vec<candle_core::Var> = steps.into_iter().map(|(_, v)| v).collect();
    let weights: Vec<candle_core::Var> = weights.into_iter().map(|(_, v)| v).collect();
    let mut opt = AdamW::new(weights, ParamsAdamW { lr: cfg.lr, weight_decay: 0.0, ..Default::default() })?;
    let mut opt_steps = AdamW::new(steps, ParamsAdamW { lr: cfg.lr / 27.0, weight_decay: 0.0, ..Default::default() })?;

    let (b, t) = (cfg.batch, cfg.window);
    // Two-trit K/V, exactly as inference stores them.
    let mut runner = Runner::new(model, b, cfg.memory, cfg.max_tokens, KvPrecision::Trit2, &device)?;
    runner.top_k = cfg.top_k;
    runner.rows = cfg.rows;
    let ep = Episodes::new(tok);
    let region = tokens.len() / b;
    let mut streams: Vec<TaskStream> =
        (0..b).map(|i| TaskStream::new(i * region, region, cfg.seed * 1000 + i as u64)).collect();
    for s in &mut streams {
        s.jump = cfg.jump;
        s.p_reread = cfg.p_reread;
    }
    // Resuming fast-forwards the streams, so data is not repeated.
    for s in &mut streams {
        for _ in 0..start_step {
            s.window(tokens, &ep, t);
        }
    }

    let clock = Instant::now();
    let (mut sum_loss, mut sum_known, mut sum_blocks, mut n_steps, mut n_logged) =
        (0f64, 0usize, 0usize, 0usize, 0usize);
    // Mean loss per token kind: plain text, fact answers, re-read spans, templates.
    let mut by_kind = [(0f64, 0usize); 4];
    let mut timing = [0f64; 4];
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(cfg.out.join("log.tsv"))?;
    for step in start_step..cfg.steps {
        let lr = lr_at(cfg, step, clock.elapsed().as_secs_f64());
        opt.set_learning_rate(lr);
        opt_steps.set_learning_rate(lr / 27.0);
        let mut x = Vec::with_capacity(b * t);
        let mut y = Vec::with_capacity(b * t);
        let mut kinds = Vec::with_capacity(b * t);
        for s in &mut streams {
            let (w, k) = s.window(tokens, &ep, t);
            x.extend_from_slice(&w[..t]);
            y.extend_from_slice(&w[1..]);
            kinds.extend_from_slice(&k[1..]);
        }
        let t0 = Instant::now();
        let (out, next) = runner.forward(&x, t, &device)?;
        let (losses, pointer) = runner.losses_and_pointer(&out, &x, &y)?;
        // Auxiliary pointer loss where the answer is in the context (episode
        // answers, re-read spans): teaches the pointer to find the row
        // instead of leaning on the vocabulary.
        let mask: Vec<f32> =
            kinds.iter().map(|&k| f32::from(k == crate::data::ANSWER || k == crate::data::REREAD)).collect();
        let mask = Tensor::from_vec(mask, kinds.len(), &device)?;
        let aux = ((pointer * mask)?.sum_all()? * (cfg.aux / kinds.len() as f64))?;
        let loss = (losses.mean_all()? + aux)?;
        let t1 = Instant::now();
        let loss_value = loss.to_scalar::<f32>()?;
        let mut grads = loss.backward()?;
        let t2 = Instant::now();
        let norm = clip(&mut grads, &vars, cfg.clip)?;
        if !loss_value.is_finite() || !norm.is_finite() {
            // Never let a non-finite update into the weights.
            println!("step {}: skipped (loss {loss_value}, grad norm {norm})", step + 1);
            runner.commit(&x, t, &out, next)?;
            continue;
        }
        opt.step(&grads)?;
        opt_steps.step(&grads)?;
        let t3 = Instant::now();
        runner.commit(&x, t, &out, next)?;
        for (acc, d) in timing.iter_mut().zip([t1 - t0, t2 - t1, t3 - t2, t3.elapsed()]) {
            *acc += d.as_secs_f64();
        }

        let lv = losses.to_vec1::<f32>()?;
        sum_loss += lv.iter().map(|&l| l as f64).sum::<f64>() / lv.len() as f64;
        for (l, &k) in lv.iter().zip(&kinds) {
            by_kind[k as usize].0 += *l as f64;
            by_kind[k as usize].1 += 1;
        }
        sum_known += out.known;
        sum_blocks += out.blocks;
        n_steps += 1;
        n_logged += 1;

        if (step + 1) % cfg.log_every == 0 {
            let el = clock.elapsed().as_secs_f64();
            let line = format!(
                "{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.3}\t{:.0}\t{:.2e}\tfwd {:.1}s bwd {:.1}s opt {:.1}s mem {:.1}s",
                step + 1,
                sum_loss / n_logged as f64,
                by_kind[0].0 / by_kind[0].1.max(1) as f64,
                if by_kind[1].1 > 0 { by_kind[1].0 / by_kind[1].1 as f64 } else { f64::NAN },
                if by_kind[2].1 > 0 { by_kind[2].0 / by_kind[2].1 as f64 } else { f64::NAN },
                if by_kind[3].1 > 0 { by_kind[3].0 / by_kind[3].1 as f64 } else { f64::NAN },
                sum_known as f64 / sum_blocks.max(1) as f64,
                (n_steps * b * t) as f64 / el,
                lr_at(cfg, step, clock.elapsed().as_secs_f64()),
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
            timing = [0.0; 4];
        }
        if (step + 1) % (cfg.log_every * 9) == 0 {
            let (mut zero, mut clipped, mut err, mut n) = (0.0, 0.0, 0.0, 0.0);
            for (_, q) in runner.model.qtensors() {
                let (z, c, e) = q.health()?;
                (zero, clipped, err, n) = (zero + z, clipped + c, err + e, n + 1.0);
            }
            println!(
                "quant: zero {:.3} clipped {:.4} rel.err {:.3} | grad norm {norm:.3}",
                zero / n,
                clipped / n,
                err / n
            );
        }
        if (step + 1) % cfg.ckpt_every == 0 || step + 1 == cfg.steps {
            varmap.save(&ckpt)?;
            std::fs::write(cfg.out.join("step"), format!("{}", step + 1))?;
        }
        if cfg.time_limit > 0 && clock.elapsed().as_secs() >= cfg.time_limit {
            varmap.save(&ckpt)?;
            std::fs::write(cfg.out.join("step"), format!("{}", step + 1))?;
            println!("time limit reached at step {}", step + 1);
            break;
        }
    }
    Ok(())
}

/// Load a trained model.
pub fn load_model(dir: &Path, mcfg: Config, device: &Device) -> Result<Model> {
    let mut varmap = VarMap::new();
    let model = Model::new(VarBuilder::from_varmap(&varmap, DType::F32, device), mcfg)?;
    varmap.load(dir.join("model.safetensors"))?;
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
