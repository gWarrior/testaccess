//! Token-by-token inference on the packed ternary model, without `candle`.
//!
//! * **Multiplication-free linear layers.** For each output row the inputs
//!   are grouped by the weight level `±1…±4`; a row is
//!   `step · (a₁ + 2a₂ + 3a₃ + 4a₄)` with `aₖ = Σx[+k] − Σx[−k]`. The
//!   factors 2 and 4 and the power-of-two `step` only change the float
//!   exponent (exact, like a shift), 3a = 2a + a. No rounding beyond the
//!   additions themselves.
//! * **Hadamard recurrence** via the fast Walsh–Hadamard transform
//!   (additions/subtractions only) and a `1/16` exponent shift.
//! * **Retention** in its recurrent form, **memory head** over a ring of the
//!   last 243 tokens plus rows retrieved from the SNN [`ContextMemory`];
//!   tokens leave the ring into the SNN memory, exactly as in training.

use std::collections::VecDeque;

use rayon::prelude::*;
use snn_memory::{ContextConfig, ContextMemory, KvConfig, KvPrecision, Probe, Verdict};

use crate::model::Config;
use crate::model::IndCand;
use crate::pack::{Packed, PackedModel};
use crate::train::{IND_EXT, IND_MAX, IND_MIN};

/// Length of the local attention ring (the training window).
pub const RING: usize = 243;

/// Two-trit matrix applied with shifts and adds.
pub struct ShiftLinear {
    rows: usize,
    cols: usize,
    /// Largest level (4 for two trits, 13 for three).
    top: usize,
    /// Per row: `2·top + 1` offsets into `idx` delimiting the groups of
    /// levels `+1…+top, −1…−top`.
    offs: Vec<u32>,
    idx: Vec<u16>,
    steps: Vec<f32>,
    levels: Vec<i8>,
}

impl ShiftLinear {
    pub fn new(rows: usize, cols: usize, levels: &[i8], exps: &[i8]) -> Self {
        let top = levels.iter().map(|l| l.unsigned_abs() as usize).max().unwrap_or(1).max(4);
        let order: Vec<i8> = (1..=top as i8).chain((1..=top as i8).map(|l| -l)).collect();
        let mut offs = Vec::with_capacity(rows * (order.len() + 1));
        let mut idx = Vec::new();
        for r in 0..rows {
            let row = &levels[r * cols..(r + 1) * cols];
            offs.push(idx.len() as u32);
            for &lv in &order {
                idx.extend(row.iter().enumerate().filter(|(_, &l)| l == lv).map(|(c, _)| c as u16));
                offs.push(idx.len() as u32);
            }
        }
        let steps = exps.iter().map(|&e| 2f32.powi(e as i32)).collect();
        Self { rows, cols, top, offs, idx, steps, levels: levels.to_vec() }
    }

    /// `step · Σₖ k·(Σx[+k] − Σx[−k])`: additions per level, then small
    /// integer multiples; the power-of-two step is an exponent shift.
    #[inline]
    fn row(&self, r: usize, x: &[f32]) -> f32 {
        let g = 2 * self.top;
        let o = &self.offs[r * (g + 1)..(r + 1) * (g + 1)];
        let sum = |k: usize| self.idx[o[k] as usize..o[k + 1] as usize].iter().map(|&c| x[c as usize]).sum::<f32>();
        let mut acc = 0f32;
        for k in 1..=self.top {
            acc += k as f32 * (sum(k - 1) - sum(self.top + k - 1));
        }
        self.steps[r] * acc
    }

    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols);
        if self.rows >= 2187 {
            (0..self.rows).into_par_iter().map(|r| self.row(r, x)).collect()
        } else {
            (0..self.rows).map(|r| self.row(r, x)).collect()
        }
    }

    /// Dequantized row (embedding lookup).
    pub fn dense_row(&self, r: usize) -> Vec<f32> {
        self.levels[r * self.cols..(r + 1) * self.cols].iter().map(|&l| l as f32 * self.steps[r]).collect()
    }
}

fn rms_gain(x: &[f32], g: &[f32]) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + 1e-6).sqrt();
    x.iter().zip(g).map(|(v, w)| v * inv * w).collect()
}

fn rms(x: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + 1e-6).sqrt();
    x.iter_mut().for_each(|v| *v *= inv);
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// In-place fast Walsh–Hadamard transform (Sylvester order).
fn fwht(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

struct LayerW {
    n1: Vec<f32>,
    wu: ShiftLinear,
    wz: ShiftLinear,
    sign: Vec<f32>,
    /// Ternary decays `1 − 3^−k` of the Hadamard recurrence.
    decay: Vec<f32>,
    n2: Vec<f32>,
    wq: ShiftLinear,
    wk: ShiftLinear,
    wv: ShiftLinear,
    wo: ShiftLinear,
    n3: Vec<f32>,
    w1: ShiftLinear,
    w3: ShiftLinear,
    w2: ShiftLinear,
}

pub struct Engine {
    pub cfg: Config,
    emb: ShiftLinear,
    layers: Vec<LayerW>,
    nm: Vec<f32>,
    mq: ShiftLinear,
    mk: ShiftLinear,
    mv: ShiftLinear,
    mo: ShiftLinear,
    verdict: Vec<f32>,
    nout: Vec<f32>,
    decays: Vec<f32>,
    /// Pointer query and copy gate (see [`crate::model`]).
    pq: ShiftLinear,
    gate_w: Vec<f32>,
    gate_b: f32,
    gate_verdict: Vec<f32>,
    /// Ablation: all weight on the vocabulary, no copying.
    pub no_pointer: bool,
    ind_len: Vec<f32>,
    ind_count: Vec<f32>,
    ind_dist: Vec<f32>,
    ind_verdict: Vec<f32>,
    ind_w: Vec<f32>,
    ind_p: Vec<f32>,
    far_verdict: Vec<f32>,
    far_source: Vec<f32>,
}

/// Rows retrieved for the current block.
#[derive(Default)]
struct Rows {
    keys: Vec<f32>,
    values: Vec<f32>,
    next: Vec<u32>,
    source: Vec<u8>,
}

/// Recurrent state of one conversation.
pub struct Session {
    h: Vec<Vec<f32>>,
    s: Vec<Vec<f32>>,
    ring: VecDeque<(u32, Vec<f32>, Vec<f32>)>,
    pub memory: Option<ContextMemory>,
    recent: VecDeque<u32>,
    rows: Rows,
    /// The last induction candidate and the position it was computed at.
    last_ind: Option<(u64, IndCand)>,
    verdict: usize,
    pos: u64,
    /// Position where the current reply starts: the pointer and the
    /// induction column never copy from it (`u64::MAX` between replies).
    pub reply_start: u64,
}

impl Engine {
    pub fn from_packed(p: &PackedModel) -> Result<Self, String> {
        let mat = |n: &str| match p.tensors.get(n) {
            Some(Packed::Matrix { rows, cols, levels, exps, .. }) => Ok(ShiftLinear::new(*rows, *cols, levels, exps)),
            _ => Err(format!("missing matrix {n}")),
        };
        let vec = |n: &str| match p.tensors.get(n) {
            Some(Packed::F32 { data, .. }) => Ok(data.clone()),
            _ => Err(format!("missing vector {n}")),
        };
        let layers = (0..p.cfg.layers)
            .map(|i| {
                let l = |n: &str| format!("l{i}.{n}");
                Ok(LayerW {
                    n1: vec(&l("n1"))?,
                    wu: mat(&l("cell.wu"))?,
                    wz: mat(&l("cell.wz"))?,
                    sign: match p.tensors.get(&l("cell.sign")) {
                        Some(Packed::Signs { data }) => data.iter().map(|&s| s as f32).collect(),
                        _ => return Err(format!("missing {}", l("cell.sign"))),
                    },
                    decay: crate::layers::hadam_decays(p.cfg.d),
                    n2: vec(&l("n2"))?,
                    wq: mat(&l("ret.wq"))?,
                    wk: mat(&l("ret.wk"))?,
                    wv: mat(&l("ret.wv"))?,
                    wo: mat(&l("ret.wo"))?,
                    n3: vec(&l("n3"))?,
                    w1: mat(&l("mlp.w1"))?,
                    w3: mat(&l("mlp.w3"))?,
                    w2: mat(&l("mlp.w2"))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let decays = (0..p.cfg.heads).map(|h| 1.0 - 3f32.powi(-(h as i32 + 2))).collect();
        Ok(Self {
            cfg: p.cfg.clone(),
            emb: mat("emb")?,
            layers,
            nm: vec("nm")?,
            mq: mat("mq")?,
            mk: mat("mk")?,
            mv: mat("mv")?,
            mo: mat("mo")?,
            verdict: vec("verdict")?,
            nout: vec("nout")?,
            decays,
            pq: mat("pq")?,
            gate_w: vec("gate_w")?,
            gate_b: vec("gate_b")?[0],
            gate_verdict: vec("gate_verdict")?,
            no_pointer: false,
            ind_len: vec("ind_len")?,
            ind_count: vec("ind_count")?,
            ind_dist: vec("ind_dist")?,
            ind_verdict: vec("ind_verdict")?,
            ind_w: vec("ind_w")?,
            ind_p: vec("ind_p")?,
            far_verdict: vec("far_verdict")?,
            far_source: vec("far_source")?,
        })
    }

    /// A fresh session with two-trit K/V memory (the model's own keys,
    /// stored exactly); `memory_tokens = 0` disables the SNN memory.
    pub fn session(&self, memory_tokens: usize) -> Session {
        self.session_with(memory_tokens, KvPrecision::Trit2)
    }

    /// A fresh session with the given K/V precision.
    pub fn session_with(&self, memory_tokens: usize, precision: KvPrecision) -> Session {
        let (d, hd) = (self.cfg.d, self.cfg.d / self.cfg.heads);
        let memory = (memory_tokens > 0).then(|| {
            let m = self.cfg.mem_dim;
            ContextMemory::new(ContextConfig {
                max_tokens: memory_tokens,
                kv: Some(KvConfig { precision, ..KvConfig::new(m, m) }),
                ..Default::default()
            })
            .expect("valid memory config")
        });
        Session {
            h: vec![vec![0.0; d]; self.cfg.layers],
            s: vec![vec![0.0; self.cfg.heads * hd * hd]; self.cfg.layers],
            ring: VecDeque::new(),
            memory,
            recent: VecDeque::new(),
            rows: Rows::default(),
            last_ind: None,
            verdict: 1,
            pos: 0,
            reply_start: u64::MAX,
        }
    }

    /// Feed one token; returns next-token logits.
    pub fn step(&self, s: &mut Session, token: u32) -> Vec<f32> {
        self.advance(s, token, true).expect("logits requested")
    }

    /// Tokens seen so far in this session.
    pub fn position(&self, s: &Session) -> u64 {
        s.pos
    }

    /// Copy head ("induction"): find the longest recent suffix (8 down to 3
    /// tokens) seen before, in the ring or — only when the SNN memory is
    /// sure (`Known`) — in the memory, and mix the token that followed it
    /// into `logits` with weight `lambda`. Occurrences continuing at or
    /// after `limit` are skipped, so a reply never copies itself. Returns
    /// the copied token and the suffix length, or `None` ("не знаю").
    pub fn copy(&self, s: &mut Session, logits: &mut [f32], lambda: f32, limit: u64) -> Option<(u32, usize)> {
        let c = match s.last_ind {
            // Computed by the last step with the same limit.
            Some((pos, c)) if pos + 1 == s.pos && limit == s.reply_start => c,
            _ => {
                let first = s.pos - s.ring.len() as u64;
                self.continuation(s, first, limit)
            }
        };
        if c.len == 0 {
            return None;
        }
        let (token, n) = (c.tok, c.len as usize);
        // p' = (1 − λ)·p + λ·[token], written back as log-probabilities.
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = logits.iter().map(|l| (l - mx).exp()).sum();
        for (i, l) in logits.iter_mut().enumerate() {
            let p = (1.0 - lambda) * (*l - mx).exp() / z + if i == token as usize { lambda } else { 0.0 };
            *l = p.max(f32::MIN_POSITIVE).ln();
        }
        Some((token, n))
    }

    /// The induction candidate: the token that followed the most recent
    /// earlier occurrence of the longest suffix of the ring (8 down to 3
    /// tokens, then extended up to 27), in the ring or in the SNN memory
    /// when it is sure (`Known`), with the number of earlier occurrences.
    /// `first` is the position of the ring's first token; continuations at
    /// or after `limit` are skipped.
    fn continuation(&self, s: &mut Session, first: u64, limit: u64) -> IndCand {
        let ring: Vec<u32> = s.ring.iter().map(|e| e.0).collect();
        let len = ring.len();
        for n in (IND_MIN..=IND_MAX).rev().filter(|&n| n < len) {
            let suffix = &ring[len - n..];
            // Every earlier occurrence in the ring whose continuation is allowed.
            let local: Vec<usize> =
                (0..len - n).filter(|&i| &ring[i..i + n] == suffix && first + ((i + n) as u64) < limit).collect();
            let mut far: Vec<u64> = Vec::new();
            if let Some(mem) = s.memory.as_mut() {
                if let Ok(located) = mem.locate(suffix) {
                    if located.verdict == Verdict::Known {
                        let end = mem.position();
                        far = located.positions.into_iter().filter(|&p| p + (n as u64) < limit.min(end)).collect();
                    }
                }
            }
            let count = (local.len() + far.len()) as u32;
            // The most recent occurrence: the ring's last, else the memory's.
            // Distance from the ring's last token back to the occurrence's end.
            let (tok, dist, before): (u32, u32, Box<dyn Fn(usize) -> Option<u32>>) = if let Some(&i) = local.last() {
                let r = ring.clone();
                (ring[i + n], (len - i - n) as u32, Box::new(move |k: usize| (k <= i).then(|| r[i - k])))
            } else if let Some(&p) = far.iter().max() {
                let mem = s.memory.as_ref().expect("far matches come from the memory");
                let Some(t) = mem.tokens(p + n as u64, p + n as u64 + 1).map(|t| t[0]) else { continue };
                let ctx: Vec<u32> = (1..=IND_EXT as u64)
                    .map_while(|k| p.checked_sub(k).and_then(|q| mem.tokens(q, q + 1)).map(|t| t[0]))
                    .collect();
                let last = first + len as u64 - 1;
                (t, (last - (p + n as u64 - 1)) as u32, Box::new(move |k: usize| ctx.get(k - 1).copied()))
            } else {
                continue;
            };
            // Extend the match backwards, as the training index does.
            let mut m = n;
            while m < IND_EXT && m < len && before(m - n + 1) == Some(ring[len - 1 - m]) {
                m += 1;
            }
            return IndCand { tok, len: m as u16, count, dist };
        }
        IndCand::NONE
    }

    /// Feed many tokens (e.g. a document into memory); returns the logits
    /// after the last one. Skips the output projection for the others.
    pub fn feed(&self, s: &mut Session, tokens: &[u32]) -> Option<Vec<f32>> {
        let mut last = None;
        for (i, &t) in tokens.iter().enumerate() {
            last = self.advance(s, t, i + 1 == tokens.len());
        }
        last
    }

    fn advance(&self, s: &mut Session, token: u32, want_logits: bool) -> Option<Vec<f32>> {
        let (d, heads) = (self.cfg.d, self.cfg.heads);
        let hd = d / heads;
        let mut x = self.emb.dense_row(token as usize);
        for (li, l) in self.layers.iter().enumerate() {
            // HadamRNN: h = tanh(FWHT(h ⊙ sign)/√d ⊙ gain + W_u x), y = h ⊙ σ(W_z x).
            let xn = rms_gain(&x, &l.n1);
            let u = l.wu.apply(&xn);
            let z = l.wz.apply(&xn);
            let mut r: Vec<f32> = s.h[li].iter().zip(&l.sign).map(|(h, sg)| h * sg).collect();
            fwht(&mut r);
            let inv_sqrt_d = 1.0 / (d as f32).sqrt();
            for j in 0..d {
                let c = (r[j] * inv_sqrt_d * l.decay[j] + u[j]).tanh();
                s.h[li][j] = if self.cfg.ternary_h { c.round() } else { c };
                x[j] += s.h[li][j] * sigmoid(z[j]);
            }
            // Retention: S = γS + kᵀv, o = q·S.
            let xn = rms_gain(&x, &l.n2);
            let q = l.wq.apply(&xn);
            let k = l.wk.apply(&xn);
            let v = l.wv.apply(&xn);
            let qs = 1.0 / (hd as f32).sqrt();
            let mut o = vec![0f32; d];
            for h in 0..heads {
                let st = &mut s.s[li][h * hd * hd..(h + 1) * hd * hd];
                let g = self.decays[h];
                for a in 0..hd {
                    let ka = k[h * hd + a];
                    for b in 0..hd {
                        st[a * hd + b] = g * st[a * hd + b] + ka * v[h * hd + b];
                    }
                }
                let oh = &mut o[h * hd..(h + 1) * hd];
                for a in 0..hd {
                    let qa = q[h * hd + a] * qs;
                    for b in 0..hd {
                        oh[b] += qa * st[a * hd + b];
                    }
                }
                rms(oh);
            }
            for (xi, yi) in x.iter_mut().zip(l.wo.apply(&o)) {
                *xi += yi;
            }
            // SwiGLU.
            let xn = rms_gain(&x, &l.n3);
            let a = l.w1.apply(&xn);
            let b = l.w3.apply(&xn);
            let hmid: Vec<f32> = a.iter().zip(&b).map(|(a, b)| a / (1.0 + (-a).exp()) * b).collect();
            for (xi, yi) in x.iter_mut().zip(l.w2.apply(&hmid)) {
                *xi += yi;
            }
        }

        // Memory head.
        let m = self.cfg.mem_dim;
        let xn = rms_gain(&x, &self.nm);
        let q = self.mq.apply(&xn);
        // Keys and values as two-trit vectors, as in training.
        let (k, v) = (quant_act_vec(&self.mk.apply(&xn)), quant_act_vec(&self.mv.apply(&xn)));
        s.recent.push_back(token);
        if s.recent.len() > crate::model::PROBE {
            s.recent.pop_front();
        }
        if s.pos % self.cfg.block as u64 == 0 {
            s.rows = Rows::default();
            s.verdict = 1;
            if let Some(mem) = &mut s.memory {
                let probe: Vec<u32> = s.recent.iter().copied().collect();
                if let Ok(r) = mem.retrieve_rows(Probe::Both(&probe, &q), 3, 81) {
                    s.verdict = match r.verdict {
                        Verdict::Known => 0,
                        Verdict::Unknown => 1,
                        Verdict::Absent => 2,
                    };
                    // The memory's last position is followed by the ring's first token.
                    let first = s.ring.front().map_or(u32::MAX, |e| e.0);
                    let end = mem.position();
                    let next = r
                        .positions
                        .iter()
                        .map(
                            |&p| if p + 1 == end { first } else { mem.tokens(p + 1, p + 2).map_or(u32::MAX, |t| t[0]) },
                        )
                        .collect();
                    s.rows = Rows { keys: r.keys, values: r.values, next, source: r.sources };
                }
            }
        }
        s.ring.push_back((token, k, v));
        let scale = 1.0 / (m as f32).sqrt();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() * scale;
        let mut scores: Vec<f32> = s.ring.iter().map(|(_, k, _)| dot(&q, k)).collect();
        let n_rows = s.rows.next.len();
        scores.extend((0..n_rows).map(|i| dot(&q, &s.rows.keys[i * m..(i + 1) * m])));
        let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut z = 0f32;
        for sc in &mut scores {
            *sc = (*sc - mx).exp();
            z += *sc;
        }
        let mut o = vec![0f32; m];
        for (i, (_, _, vv)) in s.ring.iter().enumerate() {
            let w = scores[i] / z;
            o.iter_mut().zip(vv).for_each(|(a, b)| *a += w * b);
        }
        for i in 0..n_rows {
            let w = scores[s.ring.len() + i] / z;
            o.iter_mut().zip(&s.rows.values[i * m..(i + 1) * m]).for_each(|(a, b)| *a += w * b);
        }
        let mo = self.mo.apply(&o);
        let vr = &self.verdict[s.verdict * d..(s.verdict + 1) * d];
        for j in 0..d {
            x[j] += mo[j] + vr[j];
        }
        let logits = want_logits.then(|| self.emb.apply(&rms_gain(&x, &self.nout)));
        // Pointer over strictly earlier ring positions, the memory rows, the
        // induction column and the null: each copies the token that followed.
        let mut copy: Vec<(u32, f32)> = Vec::new();
        let mut gate = 1f32;
        if let (Some(lg), false) = (&logits, self.no_pointer) {
            let pq = self.pq.apply(&xn);
            // Ring positions: the current token (the last) is at `s.pos`; a
            // reply never copies from itself.
            let first = s.pos + 1 - s.ring.len() as u64;
            let n_ring = s.ring.len() - 1;
            let mut sc: Vec<(u32, f32)> = (0..n_ring)
                .filter(|&i| first + i as u64 + 1 < s.reply_start)
                .map(|i| (s.ring[i + 1].0, dot(&pq, &s.ring[i].1)))
                .collect();
            let fv = self.far_verdict[s.verdict];
            sc.extend((0..n_rows).map(|i| {
                let bias = fv + self.far_source[s.rows.source[i] as usize];
                (s.rows.next[i], dot(&pq, &s.rows.keys[i * m..(i + 1) * m]) + bias)
            }));
            let c = self.continuation(s, first, s.reply_start);
            s.last_ind = Some((s.pos, c));
            if c.len > 0 {
                let mx = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let lse = mx + lg.iter().map(|l| (l - mx).exp()).sum::<f32>().ln();
                let lp_tok = lg.get(c.tok as usize).map_or(0.0, |l| l - lse);
                let w: f32 = xn.iter().zip(&self.ind_w).map(|(a, b)| a * b).sum();
                let logit = self.ind_len[c.len_bin() as usize]
                    + self.ind_count[c.count_trit() as usize]
                    + self.ind_dist[c.dist_trit() as usize]
                    + self.ind_verdict[s.verdict]
                    + w
                    + self.ind_p[0] * lp_tok
                    + self.ind_p[1] * (mx - lse);
                sc.push((c.tok, logit));
            }
            // The null column's weight is the vocabulary's share.
            let null: f32 = xn.iter().zip(&self.gate_w).map(|(a, b)| a * b).sum::<f32>()
                + self.gate_b
                + self.gate_verdict[s.verdict];
            sc.push((u32::MAX, null));
            let mx = sc.iter().map(|e| e.1).fold(f32::NEG_INFINITY, f32::max);
            let z: f32 = sc.iter().map(|e| (e.1 - mx).exp()).sum();
            gate = (null - mx).exp() / z;
            copy = sc.into_iter().filter(|e| e.0 != u32::MAX).map(|(t, v)| (t, (v - mx).exp() / z)).collect();
        }
        // Tokens leave the local ring into the SNN memory nine at a time
        // (the chunk stride), so every memory token is in a chunk, as in
        // training, where whole windows are written.
        if s.ring.len() > RING {
            let (mut ts, mut ks, mut vs) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..9 {
                let (t, k, v) = s.ring.pop_front().expect("ring is not empty");
                ts.push(t);
                ks.extend(k);
                vs.extend(v);
            }
            if let Some(mem) = &mut s.memory {
                let _ = mem.append_kv(&ts, &ks, &vs);
            }
        }
        s.pos += 1;
        let logits = logits?;
        // Mixture a_null·p_vocab + Σ a·[next], returned as log-probabilities.
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = logits.iter().map(|l| (l - mx).exp()).sum();
        let mut p: Vec<f32> = logits.iter().map(|l| gate * (l - mx).exp() / z).collect();
        for (t, a) in copy {
            if let Some(pt) = p.get_mut(t as usize) {
                *pt += a;
            }
        }
        Some(p.into_iter().map(|v| v.max(f32::MIN_POSITIVE).ln()).collect())
    }
}

/// Two-trit quantization of an activation row, as [`crate::layers::quant_act`].
fn quant_act_vec(x: &[f32]) -> Vec<f32> {
    let step = crate::layers::act_step(x, crate::layers::LEVELS);
    let lv = crate::layers::LEVELS as f32;
    x.iter().map(|&v| (v / step).round().clamp(-lv, lv) * step).collect()
}

/// Sample from logits with temperature and top-k.
pub fn sample(logits: &[f32], temperature: f32, top_k: usize, rng: &mut snn_memory::rng::SplitMix64) -> u32 {
    if temperature <= 0.0 {
        return logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0);
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let k = top_k.clamp(1, logits.len());
    idx.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
    idx.truncate(k);
    let mx = idx.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<f64> = idx.iter().map(|&i| (((logits[i] - mx) / temperature) as f64).exp()).collect();
    let total: f64 = w.iter().sum();
    let mut u = rng.next_f64() * total;
    for (i, wi) in idx.iter().zip(&w) {
        if u < *wi {
            return *i as u32;
        }
        u -= wi;
    }
    idx[0] as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MemBatch, Model};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn engine_matches_the_training_model() {
        engine_matches(false);
    }

    #[test]
    fn engine_matches_the_training_model_with_ternary_state() {
        engine_matches(true);
    }

    #[test]
    fn engine_matches_the_runner_across_windows() {
        // Two windows: the second attends into the first (local window and
        // pointer), the recurrent state is carried, induction spans both.
        let dev = Device::Cpu;
        let cfg = Config { vocab: 81, d: 16, layers: 2, heads: 2, mlp: 27, mem_dim: 9, block: 3, ternary_h: false };
        let vm = VarMap::new();
        let model = Model::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), cfg.clone()).unwrap();
        for (name, var) in vm.data().lock().unwrap().iter() {
            if name.starts_with("gate") || name.starts_with("ind_") || name == "verdict" {
                var.set(&Tensor::randn(0f32, 1.0, var.as_tensor().shape(), &dev).unwrap()).unwrap();
            }
        }
        let tokens: Vec<u32> = vec![3, 10, 17, 24, 31, 38, 45, 52, 59, 3, 10, 17, 24, 5, 38, 45, 7, 8, 9];
        let packed = crate::pack::pack_model(&model, &vm).unwrap();
        let engine = Engine::from_packed(&packed).unwrap();
        let mut runner = crate::train::Runner::new(model, 1, false, 1000, KvPrecision::Trit2, &dev).unwrap();
        let mut s = engine.session(0);
        let mut logits = engine.step(&mut s, tokens[0]);
        for w in 0..2 {
            let x = &tokens[w * 9..w * 9 + 9];
            let y = &tokens[w * 9 + 1..w * 9 + 10];
            let (out, next) = runner.forward(x, 9, &dev).unwrap();
            let nll = runner.losses(&out, x, y).unwrap().to_vec1::<f32>().unwrap();
            for (i, &target) in y.iter().enumerate() {
                let diff = (-logits[target as usize] - nll[i]).abs();
                assert!(diff < 1e-3, "window {w} position {i}: loss differs by {diff}");
                logits = engine.step(&mut s, target);
            }
            runner.commit(x, 9, &out, next).unwrap();
        }
    }

    fn engine_matches(ternary_h: bool) {
        let dev = Device::Cpu;
        let cfg = Config { vocab: 81, d: 16, layers: 2, heads: 2, mlp: 27, mem_dim: 9, block: 3, ternary_h };
        let vm = VarMap::new();
        let model = Model::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), cfg.clone()).unwrap();
        // Non-trivial gains/verdict/head features so every path is exercised.
        for (name, var) in vm.data().lock().unwrap().iter() {
            if name == "verdict"
                || name.ends_with(".n1")
                || name == "nout"
                || name.starts_with("gate")
                || name.starts_with("ind_")
                || name.starts_with("far_")
            {
                var.set(&Tensor::randn(0f32, 1.0, var.as_tensor().shape(), &dev).unwrap()).unwrap();
            }
        }
        // Repeats, so the induction column fires ("3 10 17" → 24).
        let tokens: Vec<u32> = vec![3, 10, 17, 24, 31, 3, 10, 17, 5];
        let ids = Tensor::from_vec(tokens.clone(), (1, 9), &dev).unwrap();
        let (tr, _) = model.trunk(&ids, &model.zero_state(1, &dev).unwrap()).unwrap();
        let ind = crate::train::Induction::new(100).window(&tokens);
        assert_eq!(ind[7], IndCand { tok: 24, len: 3, count: 1, dist: 5 });
        let mem = MemBatch::empty(1, 3, 9, &dev).unwrap().with_induction(ind.clone());
        let head = model.head(&tr, &mem).unwrap();
        let logits = head.logits.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        let point = head.point.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        let gate = head.gate.squeeze(0).unwrap().to_vec1::<f32>().unwrap();
        // Reference mixture log-probabilities; window column j copies token j + 1.
        let reference: Vec<Vec<f32>> = (0..9)
            .map(|t| {
                let row = &logits[t];
                let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let z: f32 = row.iter().map(|l| (l - mx).exp()).sum();
                let mut p: Vec<f32> = row.iter().map(|l| gate[t] * (l - mx).exp() / z).collect();
                for j in 0..8 {
                    p[tokens[j + 1] as usize] += point[t][9 + j];
                }
                // Columns: 9 previous window (none), 9 window, 1 (empty) memory
                // row, induction, null.
                if ind[t].len > 0 {
                    p[ind[t].tok as usize] += point[t][19];
                }
                p.into_iter().map(|v| v.ln()).collect()
            })
            .collect();

        let packed = crate::pack::pack_model(&model, &vm).unwrap();
        let engine = Engine::from_packed(&packed).unwrap();
        let mut s = engine.session(0);
        for (t, &tok) in tokens.iter().enumerate() {
            let logits = engine.step(&mut s, tok);
            let diff = logits.iter().zip(&reference[t]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(diff < 1e-3, "token {t}: logits differ by {diff}");
        }
    }
}
