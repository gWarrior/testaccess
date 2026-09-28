//! Ternary HadamRNN language model with an SNN long-context memory head.
//!
//! ```text
//! tokens → embedding (2 trits)
//!        → 3 × [ HadamRNN cell | retention (recurrent attention) | SwiGLU ]
//!        → memory head: one softmax over
//!             · the window's own tokens (causal, differentiable), and
//!             · the tokens the SNN memory retrieved from the past 300k
//!          + embedding of the memory's ternary verdict
//!        → tied output embedding
//!        → pointer (copy) mix: p = a_null·p_vocab + Σⱼ aⱼ·[token after j]
//! ```
//!
//! The pointer attends over the window's earlier positions and the rows the
//! SNN memory retrieved, and copies the token that *followed* the attended
//! position. Its null column is the gate: its logit sees the state and the
//! memory's ternary verdict, and it competes with the matches themselves,
//! so copying wins exactly where the pointer found a confident match.
//! Unlike the value read, whose values only predict the next token, the
//! pointer copies the real one — the direct path for exact recall.
//!
//! The forward pass is split in two: [`Model::trunk`] computes everything up
//! to the memory query; the caller then asks the SNN memory for rows
//! ([`MemBatch`]) and finishes with [`Model::head`].

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

use crate::layers::{quant2, rms, HadamCell, Mlp, Retention, RmsNorm, TLinear};

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
}

impl Default for Config {
    fn default() -> Self {
        Self { vocab: 6561, d: 256, layers: 3, heads: 4, mlp: 2187, mem_dim: 81, block: 3 }
    }
}

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
    /// Memory head query/key/value `(B, T, mem_dim)`.
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

/// Rows retrieved from the SNN memory, one set per block.
pub struct MemBatch {
    /// `(B, nb, M, mem_dim)`.
    pub keys: Tensor,
    pub values: Tensor,
    /// Additive mask `(B, nb, M)`: `0` for real rows, `-1e9` for padding.
    pub mask: Tensor,
    /// Verdict per block `(B, nb)`: 0 = known, 1 = unknown, 2 = absent.
    pub verdict: Tensor,
    /// Token that followed each row `(B, nb, M)`, `u32::MAX` for padding.
    pub next: Vec<u32>,
    pub m: usize,
}

/// One retrieved block: keys, values, verdict, token after each row.
pub type BlockRows = (Vec<f32>, Vec<f32>, u32, Vec<u32>);

impl MemBatch {
    /// Build from per-(stream, block) rows, padding to the longest set.
    pub fn from_rows(rows: &[Vec<BlockRows>], dim: usize, device: &Device) -> Result<Self> {
        let b = rows.len();
        let nb = rows[0].len();
        let m = rows.iter().flatten().map(|r| r.0.len() / dim).max().unwrap_or(0).max(1);
        let mut keys = vec![0f32; b * nb * m * dim];
        let mut values = vec![0f32; b * nb * m * dim];
        let mut mask = vec![-1e9f32; b * nb * m];
        let mut verdict = vec![1u32; b * nb];
        let mut next = vec![u32::MAX; b * nb * m];
        for (bi, blocks) in rows.iter().enumerate() {
            for (j, (k, v, verd, nx)) in blocks.iter().enumerate() {
                let n = k.len() / dim;
                let base = (bi * nb + j) * m;
                keys[base * dim..(base + n) * dim].copy_from_slice(k);
                values[base * dim..(base + n) * dim].copy_from_slice(v);
                mask[base..base + n].iter_mut().for_each(|x| *x = 0.0);
                verdict[bi * nb + j] = *verd;
                next[base..base + nx.len().min(n)].copy_from_slice(&nx[..nx.len().min(n)]);
            }
        }
        Ok(Self {
            keys: Tensor::from_vec(keys, (b, nb, m, dim), device)?,
            values: Tensor::from_vec(values, (b, nb, m, dim), device)?,
            mask: Tensor::from_vec(mask, (b, nb, m), device)?,
            verdict: Tensor::from_vec(verdict, (b, nb), device)?,
            next,
            m,
        })
    }

    /// No memory (ablation / first window).
    pub fn empty(b: usize, nb: usize, dim: usize, device: &Device) -> Result<Self> {
        Self::from_rows(&vec![vec![(Vec::new(), Vec::new(), 1, Vec::new()); nb]; b], dim, device)
    }
}

/// Components switched off for ablation studies.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ablation {
    pub no_hadam: bool,
    pub no_retention: bool,
}

/// Tokens in the lexical probe of a memory read.
pub const PROBE: usize = 9;

pub struct Model {
    pub cfg: Config,
    pub ablation: Ablation,
    emb: Tensor,
    layers: Vec<Layer>,
    nm: RmsNorm,
    mq: TLinear,
    mk: TLinear,
    mv: TLinear,
    mo: TLinear,
    verdict: Tensor,
    nout: RmsNorm,
    /// Pointer query (keys are the memory keys `mk`, so the SNN memory's
    /// stored rows serve both the value read and the pointer).
    pq: TLinear,
    /// Null-column logit of the pointer: `xn·gate_w + gate_b + gate_verdict[verdict]`.
    gate_w: Tensor,
    gate_b: Tensor,
    gate_verdict: Tensor,
}

/// Output of the head.
pub struct HeadOut {
    /// Vocabulary logits `(B, T, V)`.
    pub logits: Tensor,
    /// Pointer attention `(B, T, L)` over `L = T + M + 1` columns: the
    /// window (strictly earlier positions), the block's memory rows, a null.
    pub point: Tensor,
    /// Weight of the vocabulary distribution, the null column `(B, T)`.
    pub gate: Tensor,
}

impl Model {
    pub fn new(vb: VarBuilder, cfg: Config) -> Result<Self> {
        let d = cfg.d;
        let emb =
            vb.get_with_hints((cfg.vocab, d), "emb", Init::Randn { mean: 0.0, stdev: 1.0 / (d as f64).sqrt() })?;
        let layers = (0..cfg.layers)
            .map(|i| {
                let vb = vb.pp(format!("l{i}"));
                Ok(Layer {
                    n1: RmsNorm::new(vb.clone(), "n1", d)?,
                    cell: HadamCell::new(vb.pp("cell"), d)?,
                    n2: RmsNorm::new(vb.clone(), "n2", d)?,
                    ret: Retention::new(vb.pp("ret"), d, cfg.heads)?,
                    n3: RmsNorm::new(vb.clone(), "n3", d)?,
                    mlp: Mlp::new(vb.pp("mlp"), d, cfg.mlp)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let m = cfg.mem_dim;
        Ok(Self {
            emb,
            layers,
            nm: RmsNorm::new(vb.clone(), "nm", d)?,
            mq: TLinear::new(vb.clone(), "mq", d, m)?,
            mk: TLinear::new(vb.clone(), "mk", d, m)?,
            mv: TLinear::new(vb.clone(), "mv", d, m)?,
            mo: TLinear::new(vb.clone(), "mo", m, d)?,
            verdict: vb.get_with_hints((3, d), "verdict", Init::Const(0.0))?,
            nout: RmsNorm::new(vb.clone(), "nout", d)?,
            pq: TLinear::new(vb.clone(), "pq", d, m)?,
            gate_w: vb.get_with_hints(d, "gate_w", Init::Const(0.0))?,
            gate_b: vb.get_with_hints(1, "gate_b", Init::Const(8.0))?,
            gate_verdict: vb.get_with_hints(3, "gate_verdict", Init::Const(0.0))?,
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
        let emb = quant2(&self.emb)?;
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
        let (q, k, v) = (self.mq.forward(&xn)?, self.mk.forward(&xn)?, self.mv.forward(&xn)?);
        Ok((Trunk { x, xn, q, k, v }, next))
    }

    /// Memory head: output logits, pointer attention and copy gate.
    pub fn head(&self, tr: &Trunk, mem: &MemBatch) -> Result<HeadOut> {
        let (b, t, m) = tr.q.dims3()?;
        let blk = self.cfg.block;
        let nb = t / blk;
        let scale = 1.0 / (m as f64).sqrt();
        let device = tr.q.device();

        let causal = Tensor::tril2(t, DType::F32, device)?.affine(1e9, -1e9)?;
        let local = tr.q.matmul(&tr.k.t()?)?.affine(scale, 0.0)?.broadcast_add(&causal)?;
        let qb = tr.q.reshape((b, nb, blk, m))?;
        let far = qb
            .matmul(&mem.keys.transpose(2, 3)?.contiguous()?)?
            .affine(scale, 0.0)?
            .broadcast_add(&mem.mask.unsqueeze(2)?)?;
        let mm = far.dim(D::Minus1)?;
        let far = far.reshape((b, t, mm))?;
        // `softmax_last_dim` has no backward in candle 0.9: use the composite.
        let att = candle_nn::ops::softmax(&Tensor::cat(&[&local, &far], 2)?, D::Minus1)?;
        let o_local = att.narrow(2, 0, t)?.matmul(&tr.v)?;
        let o_far = att.narrow(2, t, mm)?.reshape((b, nb, blk, mm))?.matmul(&mem.values)?.reshape((b, t, m))?;
        let o = self.mo.forward(&(o_local + o_far)?)?;

        let verdict = self.verdict.index_select(&mem.verdict.flatten_all()?, 0)?.reshape((b, nb, 1, self.cfg.d))?;
        let verdict = verdict.broadcast_as((b, nb, blk, self.cfg.d))?.reshape((b, t, self.cfg.d))?;
        let x = ((&tr.x + o)? + verdict)?;
        let emb = quant2(&self.emb)?;
        let logits = crate::layers::linear(&rms(&self.nout.forward(&x)?)?, &emb)?;

        // Pointer: strictly earlier window positions, memory rows, a null.
        let pq = self.pq.forward(&tr.xn)?;
        let strict: Vec<f32> = (0..t * t).map(|i| if i % t < i / t { 0.0 } else { -1e9 }).collect();
        let strict = Tensor::from_vec(strict, (t, t), device)?;
        let p_local = pq.matmul(&tr.k.t()?)?.affine(scale, 0.0)?.broadcast_add(&strict)?;
        let p_far = pq
            .reshape((b, nb, blk, m))?
            .matmul(&mem.keys.transpose(2, 3)?.contiguous()?)?
            .affine(scale, 0.0)?
            .broadcast_add(&mem.mask.unsqueeze(2)?)?
            .reshape((b, t, mm))?;
        let gv = self.gate_verdict.index_select(&mem.verdict.flatten_all()?, 0)?.reshape((b, nb, 1))?;
        let gv = gv.broadcast_as((b, nb, blk))?.reshape((b, t))?;
        let null = (tr.xn.broadcast_mul(&self.gate_w)?.sum(D::Minus1)?.broadcast_add(&self.gate_b)? + gv)?;
        let point = candle_nn::ops::softmax(&Tensor::cat(&[&p_local, &p_far, &null.unsqueeze(2)?], 2)?, D::Minus1)?;
        let gate = point.narrow(2, t + mm, 1)?.squeeze(2)?;
        Ok(HeadOut { logits, point, gate })
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
        Config { vocab: 81, d: 16, layers: 2, heads: 2, mlp: 27, mem_dim: 9, block: 3 }
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
        let mut rows = vec![vec![(Vec::new(), Vec::new(), 1u32, Vec::new()); 3]; 2];
        rows[0][1] = (vec![0.5; 9], vec![1.0; 9], 0, vec![3]);
        let h = model.head(&tr, &MemBatch::from_rows(&rows, 9, &dev).unwrap()).unwrap();
        let loss = (h.logits.sqr().unwrap().mean_all().unwrap() + h.point.narrow(2, 0, 9).unwrap().sum_all().unwrap())
            .unwrap();
        let grads = loss.backward().unwrap();
        for (name, var) in vm.data().lock().unwrap().iter() {
            if ["mq", "mk", "pq", "gate_w"].iter().any(|p| name.starts_with(p)) {
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

        // The pointer sees strictly earlier positions only (and the null).
        let point = head.point.to_vec3::<f32>().unwrap();
        for (t, row) in point[0].iter().enumerate() {
            assert!(row[t..9].iter().all(|&a| a == 0.0), "position {t}: {row:?}");
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }

        // A retrieved row for block 1 of stream 0 changes only that block.
        let mut rows = vec![vec![(Vec::new(), Vec::new(), 1u32, Vec::new()); 3]; 2];
        rows[0][1] = (vec![3.0; 9], vec![5.0; 9], 0, vec![7]);
        let mem = MemBatch::from_rows(&rows, 9, &dev).unwrap();
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
