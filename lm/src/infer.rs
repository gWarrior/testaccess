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
use crate::pack::{Packed, PackedModel};

/// Length of the local attention ring (the training window).
pub const RING: usize = 243;

/// Two-trit matrix applied with shifts and adds.
pub struct ShiftLinear {
    rows: usize,
    cols: usize,
    /// Per row: 9 offsets into `idx` delimiting groups `+1,+2,+3,+4,-1,-2,-3,-4`.
    offs: Vec<u32>,
    idx: Vec<u16>,
    steps: Vec<f32>,
    levels: Vec<i8>,
}

impl ShiftLinear {
    pub fn new(rows: usize, cols: usize, levels: &[i8], exps: &[i8]) -> Self {
        let order = [1i8, 2, 3, 4, -1, -2, -3, -4];
        let mut offs = Vec::with_capacity(rows * 9);
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
        Self { rows, cols, offs, idx, steps, levels: levels.to_vec() }
    }

    #[inline]
    fn row(&self, r: usize, x: &[f32]) -> f32 {
        let o = &self.offs[r * 9..r * 9 + 9];
        let mut s = [0f32; 8];
        for (g, sg) in s.iter_mut().enumerate() {
            *sg = self.idx[o[g] as usize..o[g + 1] as usize].iter().map(|&c| x[c as usize]).sum();
        }
        let (a1, a2, a3, a4) = (s[0] - s[4], s[1] - s[5], s[2] - s[6], s[3] - s[7]);
        // ×2 and ×4 are exponent shifts (exact); ×3 = ×2 + ×1.
        self.steps[r] * (a1 + (a2 + a2) + (a3 + a3 + a3) + a4 * 4.0)
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
    gain: Vec<f32>,
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
}

/// Recurrent state of one conversation.
pub struct Session {
    h: Vec<Vec<f32>>,
    s: Vec<Vec<f32>>,
    ring: VecDeque<(u32, Vec<f32>, Vec<f32>)>,
    pub memory: Option<ContextMemory>,
    recent: VecDeque<u32>,
    /// Retrieved keys, values, count and the token after each row.
    rows: (Vec<f32>, Vec<f32>, usize, Vec<u32>),
    verdict: usize,
    pos: u64,
}

impl Engine {
    pub fn from_packed(p: &PackedModel) -> Result<Self, String> {
        let mat = |n: &str| match p.tensors.get(n) {
            Some(Packed::Matrix { rows, cols, levels, exps }) => Ok(ShiftLinear::new(*rows, *cols, levels, exps)),
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
                    gain: vec(&l("cell.gain"))?.into_iter().map(sigmoid).collect(),
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
        })
    }

    /// A fresh session with ternary K/V memory (~115 MB at 300k tokens);
    /// `memory_tokens = 0` disables the SNN memory.
    pub fn session(&self, memory_tokens: usize) -> Session {
        self.session_with(memory_tokens, KvPrecision::Ternary)
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
            rows: (Vec::new(), Vec::new(), 0, Vec::new()),
            verdict: 1,
            pos: 0,
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
        let ring: Vec<u32> = s.ring.iter().map(|e| e.0).collect();
        let ring_start = s.pos - ring.len() as u64;
        let found = (3..=8usize).rev().filter(|&n| n < ring.len()).find_map(|n| {
            let suffix = &ring[ring.len() - n..];
            // The ring first: exact and cheap.
            let local = (0..ring.len() - n)
                .rev()
                .find(|&i| &ring[i..i + n] == suffix && ring_start + ((i + n) as u64) < limit)
                .map(|i| ring[i + n]);
            local
                .or_else(|| {
                    let mem = s.memory.as_mut()?;
                    let located = mem.locate(suffix).ok()?;
                    if located.verdict != Verdict::Known {
                        return None;
                    }
                    let end = mem.position();
                    located.positions.iter().rev().find_map(|&p| {
                        let from = p + n as u64;
                        (from < limit && from < end).then(|| mem.tokens(from, from + 1)).flatten().map(|t| t[0])
                    })
                })
                .map(|t| (t, n))
        });
        let (token, n) = found?;
        // p' = (1 − λ)·p + λ·[token], written back as log-probabilities.
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = logits.iter().map(|l| (l - mx).exp()).sum();
        for (i, l) in logits.iter_mut().enumerate() {
            let p = (1.0 - lambda) * (*l - mx).exp() / z + if i == token as usize { lambda } else { 0.0 };
            *l = p.max(f32::MIN_POSITIVE).ln();
        }
        Some((token, n))
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
                s.h[li][j] = (r[j] * inv_sqrt_d * l.gain[j] + u[j]).tanh();
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
        let (q, k, v) = (self.mq.apply(&xn), self.mk.apply(&xn), self.mv.apply(&xn));
        s.recent.push_back(token);
        if s.recent.len() > self.cfg.block {
            s.recent.pop_front();
        }
        if s.pos % self.cfg.block as u64 == 0 {
            s.rows = (Vec::new(), Vec::new(), 0, Vec::new());
            s.verdict = 1;
            if let Some(mem) = &mut s.memory {
                let probe: Vec<u32> = s.recent.iter().copied().collect();
                if let Ok(r) = mem.retrieve_rows(Probe::Both(&probe, &q), 9, 243) {
                    s.verdict = match r.verdict {
                        Verdict::Known => 0,
                        Verdict::Unknown => 1,
                        Verdict::Absent => 2,
                    };
                    let n = r.positions.len();
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
                    s.rows = (r.keys, r.values, n, next);
                }
            }
        }
        s.ring.push_back((token, k, v));
        let scale = 1.0 / (m as f32).sqrt();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() * scale;
        let mut scores: Vec<f32> = s.ring.iter().map(|(_, k, _)| dot(&q, k)).collect();
        scores.extend((0..s.rows.2).map(|i| dot(&q, &s.rows.0[i * m..(i + 1) * m])));
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
        for i in 0..s.rows.2 {
            let w = scores[s.ring.len() + i] / z;
            o.iter_mut().zip(&s.rows.1[i * m..(i + 1) * m]).for_each(|(a, b)| *a += w * b);
        }
        let mo = self.mo.apply(&o);
        let vr = &self.verdict[s.verdict * d..(s.verdict + 1) * d];
        for j in 0..d {
            x[j] += mo[j] + vr[j];
        }
        // Pointer over strictly earlier ring positions and the memory rows:
        // each column copies the token that followed it.
        let mut copy: Vec<(u32, f32)> = Vec::new();
        let mut gate = 0f32;
        if want_logits {
            let pq = self.pq.apply(&xn);
            let n_ring = s.ring.len() - 1;
            let mut sc: Vec<(u32, f32)> = (0..n_ring).map(|i| (s.ring[i + 1].0, dot(&pq, &s.ring[i].1))).collect();
            sc.extend((0..s.rows.2).map(|i| (s.rows.3[i], dot(&pq, &s.rows.0[i * m..(i + 1) * m]))));
            sc.push((u32::MAX, 0.0));
            let mx = sc.iter().map(|e| e.1).fold(f32::NEG_INFINITY, f32::max);
            let z: f32 = sc.iter().map(|e| (e.1 - mx).exp()).sum();
            copy = sc.into_iter().filter(|e| e.0 != u32::MAX).map(|(t, v)| (t, (v - mx).exp() / z)).collect();
            let g: f32 = xn.iter().zip(&self.gate_w).map(|(a, b)| a * b).sum::<f32>()
                + self.gate_b
                + self.gate_verdict[s.verdict];
            gate = sigmoid(g);
        }
        // Tokens leave the local ring into the SNN memory.
        if s.ring.len() > RING {
            let (t, k, v) = s.ring.pop_front().expect("ring is not empty");
            if let Some(mem) = &mut s.memory {
                let _ = mem.append_kv(&[t], &k, &v);
            }
        }
        s.pos += 1;
        if !want_logits {
            return None;
        }
        let mut xo = rms_gain(&x, &self.nout);
        rms(&mut xo);
        let logits = self.emb.apply(&xo);
        // Mixture (1 − g)·p_vocab + g·p_copy, returned as log-probabilities.
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = logits.iter().map(|l| (l - mx).exp()).sum();
        let mut p: Vec<f32> = logits.iter().map(|l| (1.0 - gate) * (l - mx).exp() / z).collect();
        for (t, a) in copy {
            if let Some(pt) = p.get_mut(t as usize) {
                *pt += gate * a;
            }
        }
        Some(p.into_iter().map(|v| v.max(f32::MIN_POSITIVE).ln()).collect())
    }
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
        let dev = Device::Cpu;
        let cfg = Config { vocab: 81, d: 16, layers: 2, heads: 2, mlp: 27, mem_dim: 9, block: 3 };
        let vm = VarMap::new();
        let model = Model::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), cfg.clone()).unwrap();
        // Non-trivial gains/signs/verdict so every path is exercised.
        for (name, var) in vm.data().lock().unwrap().iter() {
            if name.ends_with("gain")
                || name == "verdict"
                || name.ends_with(".n1")
                || name == "nout"
                || name.starts_with("gate")
            {
                var.set(&Tensor::randn(0f32, 1.0, var.as_tensor().shape(), &dev).unwrap()).unwrap();
            }
        }
        let tokens: Vec<u32> = (0..9).map(|i| (i * 7 + 3) % 81).collect();
        let ids = Tensor::from_vec(tokens.clone(), (1, 9), &dev).unwrap();
        let (tr, _) = model.trunk(&ids, &model.zero_state(1, &dev).unwrap()).unwrap();
        let head = model.head(&tr, &MemBatch::empty(1, 3, 9, &dev).unwrap()).unwrap();
        let logits = head.logits.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        let point = head.point.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        let gate = head.gate.squeeze(0).unwrap().to_vec1::<f32>().unwrap();
        // Reference mixture log-probabilities; window column j copies token j + 1.
        let reference: Vec<Vec<f32>> = (0..9)
            .map(|t| {
                let row = &logits[t];
                let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let z: f32 = row.iter().map(|l| (l - mx).exp()).sum();
                let mut p: Vec<f32> = row.iter().map(|l| (1.0 - gate[t]) * (l - mx).exp() / z).collect();
                for j in 0..8 {
                    p[tokens[j + 1] as usize] += gate[t] * point[t][j];
                }
                p.into_iter().map(|v| v.ln()).collect()
            })
            .collect();

        let packed = crate::pack::pack_checkpoint(&vm, cfg).unwrap();
        let engine = Engine::from_packed(&packed).unwrap();
        let mut s = engine.session(0);
        for (t, &tok) in tokens.iter().enumerate() {
            let logits = engine.step(&mut s, tok);
            let diff = logits.iter().zip(&reference[t]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(diff < 1e-3, "token {t}: logits differ by {diff}");
        }
    }
}
