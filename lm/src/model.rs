//! Ternary HadamRNN language model with an SNN long-context memory head.
//!
//! ```text
//! tokens → embedding (3 trits)
//!        → 4 × [ HadamRNN cell (ternary decays) | retention (swish gate) | SwiGLU ]
//!        → memory head: one softmax over
//!             · the window's own tokens (causal, differentiable), and
//!             · the rows the SNN memory retrieved from the past 300k
//!          + embedding of the memory's ternary verdict
//!        → tied output embedding
//!        → pointer (copy) mix: p = a_null·p_vocab + Σⱼ aⱼ·[token after j]
//! ```
//!
//! Memory keys and values are two-trit vectors (straight-through), in
//! training exactly as the SNN memory stores them.
//!
//! The pointer attends over the window's earlier positions, the rows the
//! SNN memory retrieved, one *induction* column and a null, and copies the
//! token that followed the attended position. The induction column is the
//! memory's exact continuation of the longest recent suffix; its logit
//! sees the match length (up to 27), a trit for how often the suffix
//! occurred, the memory's verdict, the state and how probable the
//! vocabulary finds the candidate. Memory rows get a bias per verdict and
//! per source trit (lexical / semantic / newest). The null column is the
//! gate: copying wins exactly where a match is confident.
//!
//! The forward pass is split in two: [`Model::trunk`] computes everything up
//! to the memory query; the caller then asks the SNN memory for rows
//! ([`MemBatch`]) and finishes with [`Model::head`].

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

use crate::layers::{
    quant_shared, shared_theta0, HadamCell, Mlp, QTensor, Retention, RmsNorm, TLinear, LEVELS, LEVELS3,
};

#[derive(Clone, Debug)]
pub struct Config {
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub mlp: usize,
    /// Key/value width of the memory head.
    pub mem_dim: usize,
    /// Tokens per memory read (one SNN retrieval per block). The probe is
    /// always the last [`PROBE`] tokens, whatever the block.
    pub block: usize,
    /// Trits per HadamRNN state element (0 = f32).
    pub state_trits: u8,
}

impl Default for Config {
    fn default() -> Self {
        // Four layers with a 1458 (2·3^6) wide MLP: as many parameters as
        // three layers of 2187, one more step of depth.
        Self { vocab: 6561, d: 256, layers: 4, heads: 4, mlp: 1458, mem_dim: 81, block: 3, state_trits: 0 }
    }
}

#[derive(Clone)]
struct Layer {
    n1: RmsNorm,
    cell: HadamCell,
    n2: RmsNorm,
    ret: Retention,
    n3: RmsNorm,
    mlp: Mlp,
}

/// Recurrent state carried between windows (detached).
#[derive(Clone)]
pub struct State {
    pub h: Vec<Tensor>,
    pub s: Vec<Tensor>,
}

/// Output of the trunk.
pub struct Trunk {
    /// Residual stream `(B, T, d)`.
    pub x: Tensor,
    /// Normalized residual feeding the memory head `(B, T, d)`.
    pub xn: Tensor,
    /// Memory head query `(B, T, mem_dim)` and two-trit key/value.
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

/// One block's retrieved rows.
#[derive(Clone, Debug, Default)]
pub struct BlockRows {
    /// `rows × mem_dim` each.
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    /// 0 = known, 1 = unknown, 2 = absent.
    pub verdict: u32,
    /// Token after each row (`u32::MAX` if unknown), row position, and
    /// source trit (0 lexical, 1 semantic, 2 newest).
    pub next: Vec<u32>,
    pub pos: Vec<u64>,
    pub source: Vec<u8>,
}

/// An induction candidate: the token that followed the most recent earlier
/// occurrence of the longest suffix, the matched length (0 = none) and how
/// many times that suffix occurred before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndCand {
    pub tok: u32,
    pub len: u16,
    pub count: u32,
    /// Tokens from the current position back to the occurrence's end.
    pub dist: u32,
}

impl IndCand {
    pub const NONE: Self = Self { tok: u32::MAX, len: 0, count: 0, dist: 0 };

    /// Distance trit: inside the local window (≤ 242), within 3^9 tokens,
    /// farther.
    pub fn dist_trit(&self) -> u32 {
        match self.dist {
            0..=242 => 0,
            243..=19_683 => 1,
            _ => 2,
        }
    }

    /// Length bin: 0 none, then 3, 4, 5–6, 7–8, 9–13, 14–26, 27+.
    pub fn len_bin(&self) -> u32 {
        match self.len {
            0..=2 => 0,
            3 => 1,
            4 => 2,
            5..=6 => 3,
            7..=8 => 4,
            9..=13 => 5,
            14..=26 => 6,
            _ => 7,
        }
    }

    /// Occurrence trit: once, a few times (2–3), many times (4+).
    pub fn count_trit(&self) -> u32 {
        match self.count {
            0..=1 => 0,
            2..=3 => 1,
            _ => 2,
        }
    }
}

/// Rows retrieved from the SNN memory, one set per block, and the window's
/// induction candidates.
pub struct MemBatch {
    /// `(B, nb, M, mem_dim)`.
    pub keys: Tensor,
    pub values: Tensor,
    /// Additive mask `(B, nb, M)`: `0` for real rows, `-1e9` for padding.
    pub mask: Tensor,
    /// Verdict per block `(B, nb)`: 0 = known, 1 = unknown, 2 = absent.
    pub verdict: Tensor,
    /// Source trit per row `(B, nb, M)`.
    pub source: Tensor,
    /// Token after each row and its position `(B, nb, M)`, `MAX` for padding.
    pub next: Vec<u32>,
    pub pos: Vec<u64>,
    pub m: usize,
    /// Induction candidates `(B·T)`.
    pub ind: Vec<IndCand>,
    /// The previous window's keys and values `(B, T, mem_dim)`, detached:
    /// with them every position sees the last [`LOCAL`] tokens, as the
    /// engine's ring does.
    pub prev: Option<(Tensor, Tensor)>,
    /// Absolute position of the window's first token (rows inside the
    /// local window are masked: they are already local).
    pub start: u64,
}

/// Earlier tokens every position attends to locally (the engine's ring
/// holds the current token and these).
pub const LOCAL: usize = 242;

impl MemBatch {
    /// Build from per-(stream, block) rows, padding to the longest set.
    pub fn from_rows(rows: &[Vec<BlockRows>], dim: usize, device: &Device) -> Result<Self> {
        let b = rows.len();
        let nb = rows[0].len();
        let m = rows.iter().flatten().map(|r| r.pos.len()).max().unwrap_or(0).max(1);
        let mut keys = vec![0f32; b * nb * m * dim];
        let mut values = vec![0f32; b * nb * m * dim];
        let mut mask = vec![-1e9f32; b * nb * m];
        let mut verdict = vec![1u32; b * nb];
        let mut source = vec![0u32; b * nb * m];
        let mut next = vec![u32::MAX; b * nb * m];
        let mut pos = vec![u64::MAX; b * nb * m];
        for (bi, blocks) in rows.iter().enumerate() {
            for (j, r) in blocks.iter().enumerate() {
                let n = r.pos.len();
                let base = (bi * nb + j) * m;
                keys[base * dim..(base + n) * dim].copy_from_slice(&r.keys);
                values[base * dim..(base + n) * dim].copy_from_slice(&r.values);
                mask[base..base + n].iter_mut().for_each(|x| *x = 0.0);
                verdict[bi * nb + j] = r.verdict;
                next[base..base + n].copy_from_slice(&r.next);
                pos[base..base + n].copy_from_slice(&r.pos);
                for (s, &x) in source[base..base + n].iter_mut().zip(&r.source) {
                    *s = x as u32;
                }
            }
        }
        Ok(Self {
            keys: Tensor::from_vec(keys, (b, nb, m, dim), device)?,
            values: Tensor::from_vec(values, (b, nb, m, dim), device)?,
            mask: Tensor::from_vec(mask, (b, nb, m), device)?,
            verdict: Tensor::from_vec(verdict, (b, nb), device)?,
            source: Tensor::from_vec(source, (b, nb, m), device)?,
            next,
            pos,
            m,
            ind: Vec::new(),
            prev: None,
            start: 0,
        })
    }

    /// Attach the previous window's keys/values and this window's start.
    pub fn with_prev(mut self, prev: Option<(Tensor, Tensor)>, start: u64) -> Self {
        self.prev = prev;
        self.start = start;
        self
    }

    /// Attach the window's induction candidates `(B·T)`.
    pub fn with_induction(mut self, ind: Vec<IndCand>) -> Self {
        self.ind = ind;
        self
    }

    /// No memory (ablation / first window).
    pub fn empty(b: usize, nb: usize, dim: usize, device: &Device) -> Result<Self> {
        Self::from_rows(&vec![vec![BlockRows { verdict: 1, ..Default::default() }; nb]; b], dim, device)
    }
}

/// Components switched off for ablation studies.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ablation {
    pub no_hadam: bool,
    pub no_retention: bool,
    /// All weight on the vocabulary: no copying.
    pub no_pointer: bool,
}

/// Tokens in the lexical probe of a memory read.
pub const PROBE: usize = 9;

/// The semantic probe of a memory read: the mean of the keys of the probe's
/// tokens (`keys`: the last `n` keys, row-major, `dim` wide). It lives in
/// the space of the keys the memory stores (unlike the head's query),
/// so the SNN memory's semantic index can compare them.
pub fn probe_key(keys: &[f32], dim: usize, n: usize) -> Vec<f32> {
    let rows = keys.len() / dim;
    let n = n.min(rows).max(1);
    let mut mean = vec![0f32; dim];
    for r in rows - n..rows {
        for (m, k) in mean.iter_mut().zip(&keys[r * dim..(r + 1) * dim]) {
            *m += k;
        }
    }
    mean.iter_mut().for_each(|m| *m /= n as f32);
    mean
}

/// Chunks a memory read asks for (several places, a chunk of rows each)
/// and rows it reads at most.
pub const MEM_TOP_K: usize = 9;
pub const MEM_ROWS: usize = 81;

/// The line-break token (byte 0x0A).
pub const NEWLINE: u32 = 10;

/// The lexical probe of a memory read from the last [`PROBE`] tokens: the
/// current line when at least three of its tokens are there. A question on
/// a new line ("Кто такой N -") shares few n-grams with the statement it
/// asks about, and the tail of the previous line made its coverage fall
/// below the memory's threshold; without it the statement is found.
/// Training and the engine must probe alike.
pub fn probe_of(recent: &[u32]) -> &[u32] {
    match recent.iter().rposition(|&t| t == NEWLINE) {
        Some(i) if recent.len() - i > 3 => &recent[i + 1..],
        _ => recent,
    }
}

/// Bins of the induction match length (see [`IndCand::len_bin`]).
pub const LEN_BINS: usize = 8;

/// Cloning shares every parameter (tensors are reference-counted), so
/// clones accumulate gradients into the same variables.
#[derive(Clone)]
pub struct Model {
    pub cfg: Config,
    pub ablation: Ablation,
    emb: QTensor,
    layers: Vec<Layer>,
    nm: RmsNorm,
    mq: TLinear,
    mk: TLinear,
    mv: TLinear,
    mo: TLinear,
    /// Learned steps `2^round(θ)` of the two-trit memory keys and values.
    kv_step: Tensor,
    verdict: Tensor,
    nout: RmsNorm,
    /// Pointer query (keys are the memory keys `mk`, so the SNN memory's
    /// stored rows serve both the value read and the pointer).
    pq: TLinear,
    /// Null logit: `xn·gate_w + gate_b + gate_verdict[verdict]`.
    gate_w: Tensor,
    gate_b: Tensor,
    gate_verdict: Tensor,
    /// Induction logit: `ind_len[bin] + ind_count[trit] + ind_dist[trit] + ind_verdict[v] +
    /// xn·ind_w + ind_p[0]·log p_vocab(candidate) + ind_p[1]·max log p_vocab`.
    ind_len: Tensor,
    ind_count: Tensor,
    ind_dist: Tensor,
    ind_verdict: Tensor,
    ind_w: Tensor,
    ind_p: Tensor,
    /// Memory-row biases per block verdict and per row source trit.
    far_verdict: Tensor,
    far_source: Tensor,
}

/// Output of the head.
pub struct HeadOut {
    /// Vocabulary logits `(B, T, V)`.
    pub logits: Tensor,
    /// Pointer attention `(B, T, L)` over `L = 2T + M + 2` columns: the
    /// previous window, the current window (strictly earlier positions), the
    /// block's memory rows, the induction column and the null.
    pub point: Tensor,
    /// Weight of the vocabulary distribution, the null column `(B, T)`.
    pub gate: Tensor,
    /// `ln point`, exact even where a weight underflows.
    pub log_point: Tensor,
}

/// Per-block values `(B, nb)` repeated over each block's positions `(B, T)`.
fn per_block(x: &Tensor, b: usize, nb: usize, blk: usize) -> Result<Tensor> {
    x.reshape((b, nb, 1))?.broadcast_as((b, nb, blk))?.reshape((b, nb * blk))
}

impl Model {
    pub fn new(vb: VarBuilder, cfg: Config) -> Result<Self> {
        let d = cfg.d;
        let emb = QTensor::new(&vb, "emb", cfg.vocab, d, 1.0 / (d as f64).sqrt(), LEVELS3)?;
        let layers = (0..cfg.layers)
            .map(|i| {
                let vb = vb.pp(format!("l{i}"));
                Ok(Layer {
                    n1: RmsNorm::new(vb.clone(), "n1", d)?,
                    cell: HadamCell::new(vb.pp("cell"), d, cfg.state_trits)?,
                    n2: RmsNorm::new(vb.clone(), "n2", d)?,
                    ret: Retention::new(vb.pp("ret"), d, cfg.heads)?,
                    n3: RmsNorm::new(vb.clone(), "n3", d)?,
                    mlp: Mlp::new(vb.pp("mlp"), d, cfg.mlp)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let m = cfg.mem_dim;
        let zeros = |n: usize, name: &str| vb.get_with_hints(n, name, Init::Const(0.0));
        Ok(Self {
            emb,
            layers,
            nm: RmsNorm::new(vb.clone(), "nm", d)?,
            mq: TLinear::new(vb.clone(), "mq", d, m)?,
            mk: TLinear::new(vb.clone(), "mk", d, m)?,
            mv: TLinear::new(vb.clone(), "mv", d, m)?,
            mo: TLinear::new(vb.clone(), "mo", m, d)?,
            // Keys/values start with mean |x| ≈ 0.8 (unit-variance projections).
            kv_step: vb.get_with_hints(2, "kv_step", Init::Const(shared_theta0(0.8, LEVELS)))?,
            verdict: vb.get_with_hints((3, d), "verdict", Init::Const(0.0))?,
            nout: RmsNorm::new(vb.clone(), "nout", d)?,
            pq: TLinear::new(vb.clone(), "pq", d, m)?,
            gate_w: zeros(d, "gate_w")?,
            // The vocabulary starts with nearly all the weight.
            gate_b: vb.get_with_hints(1, "gate_b", Init::Const(8.0))?,
            gate_verdict: zeros(3, "gate_verdict")?,
            ind_len: zeros(LEN_BINS, "ind_len")?,
            ind_count: zeros(3, "ind_count")?,
            ind_dist: zeros(3, "ind_dist")?,
            ind_verdict: zeros(3, "ind_verdict")?,
            ind_w: zeros(d, "ind_w")?,
            ind_p: zeros(2, "ind_p")?,
            far_verdict: zeros(3, "far_verdict")?,
            far_source: zeros(3, "far_source")?,
            ablation: Ablation::default(),
            cfg,
        })
    }

    pub fn zero_state(&self, b: usize, device: &Device) -> Result<State> {
        let (d, h) = (self.cfg.d, self.cfg.heads);
        Ok(State {
            h: (0..self.cfg.layers).map(|_| Tensor::zeros((b, d), DType::F32, device)).collect::<Result<_>>()?,
            s: (0..self.cfg.layers)
                .map(|_| Tensor::zeros((b, h, d / h, d / h), DType::F32, device))
                .collect::<Result<_>>()?,
        })
    }

    /// Everything up to the memory query. `ids`: `(B, T)` u32.
    pub fn trunk(&self, ids: &Tensor, state: &State) -> Result<(Trunk, State)> {
        let (b, t) = ids.dims2()?;
        let emb = self.emb.q()?;
        let mut x = emb.index_select(&ids.flatten_all()?, 0)?.reshape((b, t, self.cfg.d))?;
        let mut next = State { h: Vec::new(), s: Vec::new() };
        for (i, l) in self.layers.iter().enumerate() {
            let (y, h) = l.cell.forward(&l.n1.forward(&x)?, &state.h[i])?;
            if !self.ablation.no_hadam {
                x = (x + y)?;
            }
            let (y, s) = l.ret.forward(&l.n2.forward(&x)?, &state.s[i])?;
            if !self.ablation.no_retention {
                x = (x + y)?;
            }
            x = (&x + l.mlp.forward(&l.n3.forward(&x)?)?)?;
            next.h.push(h.detach());
            next.s.push(s.detach());
        }
        let xn = self.nm.forward(&x)?;
        let q = self.mq.forward(&xn)?;
        // Keys and values as the SNN memory stores them: two trits each.
        let k = quant_shared(&self.mk.forward(&xn)?, &self.kv_step.narrow(0, 0, 1)?, LEVELS)?;
        let v = quant_shared(&self.mv.forward(&xn)?, &self.kv_step.narrow(0, 1, 1)?, LEVELS)?;
        Ok((Trunk { x, xn, q, k, v }, next))
    }

    /// Memory head: output logits, pointer attention and copy gate.
    pub fn head(&self, tr: &Trunk, mem: &MemBatch) -> Result<HeadOut> {
        let (b, t, m) = tr.q.dims3()?;
        let blk = self.cfg.block;
        let nb = t / blk;
        let scale = 1.0 / (m as f64).sqrt();
        let device = tr.q.device();
        let verdict_ids = mem.verdict.flatten_all()?;

        // Local attention over the last LOCAL tokens: the previous window's
        // positions j > i (so the distance stays ≤ LOCAL) and the current
        // window up to i (the pointer: strictly before i).
        let (pk, pv, has_prev) = match &mem.prev {
            Some((k, v)) => (k.clone(), v.clone(), true),
            None => {
                (Tensor::zeros((b, t, m), DType::F32, device)?, Tensor::zeros((b, t, m), DType::F32, device)?, false)
            }
        };
        let mask = |f: &dyn Fn(usize, usize) -> bool| -> Result<Tensor> {
            let v: Vec<f32> = (0..t * t).map(|k| if f(k / t, k % t) { 0.0 } else { -1e9 }).collect();
            Tensor::from_vec(v, (t, t), device)
        };
        // Previous-window position j is i + t − j tokens back.
        let prev_mask = mask(&|i, j| has_prev && j + LOCAL >= i + t)?;
        let causal = mask(&|i, j| j <= i)?;
        let strict = mask(&|i, j| j < i)?;
        // Memory rows already inside the local window are masked per query.
        let mm = mem.m;
        let mut inside = vec![0f32; b * t * mm];
        for bi in 0..b {
            for i in 0..t {
                let limit = (mem.start + i as u64).saturating_sub(LOCAL as u64);
                let rows = &mem.pos[(bi * nb + i / blk) * mm..(bi * nb + i / blk + 1) * mm];
                for (r, &p) in rows.iter().enumerate() {
                    if p != u64::MAX && p >= limit {
                        inside[(bi * t + i) * mm + r] = -1e9;
                    }
                }
            }
        }
        let inside = Tensor::from_vec(inside, (b, t, mm), device)?;

        let prev_sc = tr.q.matmul(&pk.t()?)?.affine(scale, 0.0)?.broadcast_add(&prev_mask)?;
        let local = tr.q.matmul(&tr.k.t()?)?.affine(scale, 0.0)?.broadcast_add(&causal)?;
        let qb = tr.q.reshape((b, nb, blk, m))?;
        let far = qb
            .matmul(&mem.keys.transpose(2, 3)?.contiguous()?)?
            .affine(scale, 0.0)?
            .broadcast_add(&mem.mask.unsqueeze(2)?)?
            .reshape((b, t, mm))?;
        let far = (far + &inside)?;
        // `softmax_last_dim` has no backward in candle 0.9: use the composite.
        let att = candle_nn::ops::softmax(&Tensor::cat(&[&prev_sc, &local, &far], 2)?, D::Minus1)?;
        let o_local = (att.narrow(2, 0, t)?.matmul(&pv)? + att.narrow(2, t, t)?.matmul(&tr.v)?)?;
        let o_far = att.narrow(2, 2 * t, mm)?.reshape((b, nb, blk, mm))?.matmul(&mem.values)?.reshape((b, t, m))?;
        let o = self.mo.forward(&(o_local + o_far)?)?;

        let verdict = self.verdict.index_select(&verdict_ids, 0)?.reshape((b, nb, 1, self.cfg.d))?;
        let verdict = verdict.broadcast_as((b, nb, blk, self.cfg.d))?.reshape((b, t, self.cfg.d))?;
        let x = ((&tr.x + o)? + verdict)?;
        let emb = self.emb.q()?;
        let logits = crate::layers::linear(&self.nout.forward(&x)?, &emb)?;

        // Pointer: the local window (previous and current), memory rows,
        // induction, null.
        let pq = self.pq.forward(&tr.xn)?;
        let p_prev = pq.matmul(&pk.t()?)?.affine(scale, 0.0)?.broadcast_add(&prev_mask)?;
        let p_local = pq.matmul(&tr.k.t()?)?.affine(scale, 0.0)?.broadcast_add(&strict)?;
        let row_bias = self
            .far_source
            .index_select(&mem.source.flatten_all()?, 0)?
            .reshape((b, nb, mm))?
            .broadcast_add(&self.far_verdict.index_select(&verdict_ids, 0)?.reshape((b, nb, 1))?)?;
        let p_far = pq
            .reshape((b, nb, blk, m))?
            .matmul(&mem.keys.transpose(2, 3)?.contiguous()?)?
            .affine(scale, 0.0)?
            .broadcast_add(&(mem.mask.clone() + row_bias)?.unsqueeze(2)?)?
            .reshape((b, t, mm))?;
        let p_far = (p_far + inside)?;
        let gv = per_block(&self.gate_verdict.index_select(&verdict_ids, 0)?, b, nb, blk)?;
        let null = (tr.xn.broadcast_mul(&self.gate_w)?.sum(D::Minus1)?.broadcast_add(&self.gate_b)? + &gv)?;
        let ind = self.induction_logit(tr, mem, &logits, &verdict_ids, b, t, nb, blk)?;
        let cols = Tensor::cat(&[&p_prev, &p_local, &p_far, &ind.unsqueeze(2)?, &null.unsqueeze(2)?], 2)?;
        let log_point = if self.ablation.no_pointer {
            let l = 2 * t + mm + 2;
            let null: Vec<f32> = (0..b * t * l).map(|i| if i % l == l - 1 { 0.0 } else { -1e9 }).collect();
            Tensor::from_vec(null, (b, t, l), device)?
        } else {
            candle_nn::ops::log_softmax(&cols, D::Minus1)?
        };
        let point = log_point.exp()?;
        let gate = point.narrow(2, 2 * t + mm + 1, 1)?.squeeze(2)?;
        Ok(HeadOut { logits, point, gate, log_point })
    }

    /// The induction column's logit `(B, T)`, `-1e9` where nothing matched.
    #[allow(clippy::too_many_arguments)]
    fn induction_logit(
        &self,
        tr: &Trunk,
        mem: &MemBatch,
        logits: &Tensor,
        verdict_ids: &Tensor,
        b: usize,
        t: usize,
        nb: usize,
        blk: usize,
    ) -> Result<Tensor> {
        let device = tr.xn.device();
        let none_cands = vec![IndCand::NONE; b * t];
        let cands = if mem.ind.len() == b * t { &mem.ind } else { &none_cands };
        let u =
            |f: &dyn Fn(&IndCand) -> u32| Tensor::from_vec(cands.iter().map(f).collect::<Vec<u32>>(), b * t, device);
        let len = self.ind_len.index_select(&u(&|c| c.len_bin())?, 0)?;
        let count = self.ind_count.index_select(&u(&|c| c.count_trit())?, 0)?;
        let dist = self.ind_dist.index_select(&u(&|c| c.dist_trit())?, 0)?;
        let none = u(&|c| u32::from(c.len == 0))?.to_dtype(DType::F32)?.affine(-1e9, 0.0)?;
        let verdict = per_block(&self.ind_verdict.index_select(verdict_ids, 0)?, b, nb, blk)?.flatten_all()?;
        // How probable the vocabulary finds the candidate, and its own best
        // guess (both detached: features, not a path for gradients).
        let lg = logits.detach().flatten_to(1)?;
        let lse = lg.log_sum_exp(1)?;
        let tok = u(&|c| if c.len == 0 { 0 } else { c.tok })?;
        let lp_tok = (lg.gather(&tok.unsqueeze(1)?, 1)?.squeeze(1)? - &lse)?;
        let lp_max = (lg.max(1)? - &lse)?;
        let p0 = self.ind_p.narrow(0, 0, 1)?;
        let p1 = self.ind_p.narrow(0, 1, 1)?;
        let feats = (lp_tok.broadcast_mul(&p0)? + lp_max.broadcast_mul(&p1)?)?;
        let state = tr.xn.broadcast_mul(&self.ind_w)?.sum(D::Minus1)?.flatten_all()?;
        let logit = (((((((len + count)? + dist)? + verdict)? + state)? + feats)?) + none)?;
        logit.reshape((b, t))
    }

    /// Named quantized matrices (for packing and health checks).
    pub fn qtensors(&self) -> Vec<(String, &QTensor)> {
        let mut out = vec![("emb".to_string(), &self.emb)];
        for (i, l) in self.layers.iter().enumerate() {
            for (n, q) in l.cell.qtensors().into_iter().chain(l.ret.qtensors()).chain(l.mlp.qtensors()) {
                out.push((format!("l{i}.{n}"), q));
            }
        }
        for (n, lin) in [("mq", &self.mq), ("mk", &self.mk), ("mv", &self.mv), ("mo", &self.mo), ("pq", &self.pq)] {
            out.push((n.to_string(), &lin.w));
        }
        out
    }

    /// Number of trainable parameters.
    pub fn n_params(vars: &[candle_core::Var]) -> usize {
        vars.iter().map(|v| v.as_tensor().elem_count()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn small() -> Config {
        Config { vocab: 81, d: 16, layers: 2, heads: 2, mlp: 27, mem_dim: 9, block: 3, state_trits: 0 }
    }

    #[test]
    fn default_config_is_about_eight_million_parameters() {
        let vm = VarMap::new();
        let _ = Model::new(VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu), Config::default()).unwrap();
        let n = Model::n_params(&vm.all_vars());
        assert!((7_500_000..8_500_000).contains(&n), "{n} parameters");
    }

    #[test]
    fn memory_and_pointer_projections_receive_gradients() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let model = Model::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), small()).unwrap();
        let ids = Tensor::from_vec((0..18u32).map(|i| i % 5).collect::<Vec<_>>(), (2, 9), &dev).unwrap();
        let (tr, _) = model.trunk(&ids, &model.zero_state(2, &dev).unwrap()).unwrap();
        let mut rows = vec![vec![BlockRows { verdict: 1, ..Default::default() }; 3]; 2];
        rows[0][1] = BlockRows {
            keys: vec![0.5; 9],
            values: vec![1.0; 9],
            verdict: 0,
            next: vec![3],
            pos: vec![0],
            source: vec![0],
        };
        // Memory rows are older than the local window.
        let mem = MemBatch::from_rows(&rows, 9, &dev).unwrap().with_prev(None, 1000);
        let h = model.head(&tr, &mem).unwrap();
        let loss = (h.logits.sqr().unwrap().mean_all().unwrap() + h.point.narrow(2, 9, 9).unwrap().sum_all().unwrap())
            .unwrap();
        let grads = loss.backward().unwrap();
        for (name, var) in vm.data().lock().unwrap().iter() {
            if ["mq", "mk", "pq", "gate_w", "far_", "emb_step"].iter().any(|p| name.starts_with(p)) {
                let g =
                    grads.get(var.as_tensor()).map(|g| g.abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap());
                assert!(g.is_some_and(|g| g > 0.0), "{name}: no gradient ({g:?})");
            }
        }
    }

    #[test]
    fn memory_rows_reach_the_output_and_are_causal() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let model = Model::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), small()).unwrap();
        let ids = Tensor::from_vec((0..18u32).collect::<Vec<_>>(), (2, 9), &dev).unwrap();
        let state = model.zero_state(2, &dev).unwrap();
        let (tr, _) = model.trunk(&ids, &state).unwrap();
        let empty = MemBatch::empty(2, 3, 9, &dev).unwrap();
        let head = model.head(&tr, &empty).unwrap();
        let base = head.logits;
        assert_eq!(base.dims(), &[2, 9, 81]);

        // The pointer sees strictly earlier positions only (and the null);
        // there is no previous window yet.
        let point = head.point.to_vec3::<f32>().unwrap();
        for (t, row) in point[0].iter().enumerate() {
            assert!(row[..9].iter().all(|&a| a == 0.0), "position {t}: {row:?}");
            assert!(row[9 + t..18].iter().all(|&a| a == 0.0), "position {t}: {row:?}");
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }

        // A retrieved row for block 1 of stream 0 changes only that block.
        let mut rows = vec![vec![BlockRows { verdict: 1, ..Default::default() }; 3]; 2];
        rows[0][1] = BlockRows {
            keys: vec![3.0; 9],
            values: vec![5.0; 9],
            verdict: 0,
            next: vec![7],
            pos: vec![0],
            source: vec![1],
        };
        let mem = MemBatch::from_rows(&rows, 9, &dev).unwrap().with_prev(None, 1000);
        let with = model.head(&tr, &mem).unwrap().logits;
        let diff = (with - &base).unwrap().abs().unwrap().sum(2).unwrap().to_vec2::<f32>().unwrap();
        assert!(diff[0][..3].iter().all(|&d| d == 0.0), "{:?}", diff[0]);
        assert!(diff[0][3..6].iter().all(|&d| d > 0.0));
        assert!(diff[0][6..].iter().all(|&d| d == 0.0));
        assert!(diff[1].iter().all(|&d| d == 0.0));

        // Changing a future token does not change earlier logits.
        let mut ids2: Vec<u32> = (0..18).collect();
        ids2[8] = 80;
        let (tr2, _) = model.trunk(&Tensor::from_vec(ids2, (2, 9), &dev).unwrap(), &state).unwrap();
        let out2 = model.head(&tr2, &empty).unwrap().logits;
        let d = (out2 - &base).unwrap().abs().unwrap().sum(2).unwrap().to_vec2::<f32>().unwrap();
        assert!(d[0][..8].iter().all(|&x| x < 1e-4), "{:?}", d[0]);
    }
}

#[cfg(test)]
mod probe_tests {
    use super::*;

    #[test]
    fn the_probe_is_the_current_line_when_it_has_three_tokens() {
        assert_eq!(probe_of(&[1, 2, NEWLINE, 4, 5, 6]), &[4, 5, 6]);
        // Too short a line: the whole window.
        assert_eq!(probe_of(&[1, 2, 3, NEWLINE, 5, 6]), &[1, 2, 3, NEWLINE, 5, 6]);
        assert_eq!(probe_of(&[1, 2, 3]), &[1, 2, 3]);
    }
}
