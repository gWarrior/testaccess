//! Ternary building blocks.
//!
//! Every weight matrix is quantized to **two trits** per weight: nine
//! levels `{-4, …, +4}` times a per-row scale (the row's mean absolute
//! value). Training keeps latent `f32` weights and uses the straight-through
//! estimator: the forward pass sees the quantized weights, the backward pass
//! treats quantization as identity.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

/// Largest magnitude of a two-trit weight (balanced ternary `±(3 + 1)`).
pub const LEVELS: f64 = 4.0;

/// Quantize to two trits per weight (per-row scale), with STE.
pub fn quant2(w: &Tensor) -> Result<Tensor> {
    let scale = w.abs()?.mean_keepdim(D::Minus1)?.clamp(1e-8, f64::MAX)?;
    let q = w.broadcast_div(&scale)?.round()?.clamp(-LEVELS, LEVELS)?.broadcast_mul(&scale)?;
    w + (q - w)?.detach()
}

/// Integer levels and per-row scales of a weight matrix, for export.
pub fn quant2_levels(w: &Tensor) -> Result<(Vec<i8>, Vec<f32>)> {
    let scale = w.abs()?.mean_keepdim(D::Minus1)?.clamp(1e-8, f64::MAX)?;
    let q = w.broadcast_div(&scale)?.round()?.clamp(-LEVELS, LEVELS)?;
    let levels = q.flatten_all()?.to_vec1::<f32>()?.into_iter().map(|x| x as i8).collect();
    Ok((levels, scale.flatten_all()?.to_vec1::<f32>()?))
}

/// Binary sign with STE (used for the Hadamard recurrence).
pub fn ste_sign(w: &Tensor) -> Result<Tensor> {
    let s = w.ge(0.0)?.to_dtype(DType::F32)?.affine(2.0, -1.0)?;
    w + (s - w)?.detach()
}

/// Linear map `x · Wᵀ` with a two-trit weight.
pub struct TLinear {
    pub w: Tensor,
}

impl TLinear {
    pub fn new(vb: VarBuilder, name: &str, d_in: usize, d_out: usize) -> Result<Self> {
        let std = (1.0 / d_in as f64).sqrt();
        Ok(Self { w: vb.get_with_hints((d_out, d_in), name, Init::Randn { mean: 0.0, stdev: std })? })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_matmul(&quant2(&self.w)?.t()?)
    }
}

/// Root-mean-square normalisation with a learned gain.
pub struct RmsNorm {
    w: Tensor,
}

impl RmsNorm {
    pub fn new(vb: VarBuilder, name: &str, d: usize) -> Result<Self> {
        Ok(Self { w: vb.get_with_hints(d, name, Init::Const(1.0))? })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        rms(x)?.broadcast_mul(&self.w)
    }
}

/// Parameter-free RMS normalisation over the last dimension.
pub fn rms(x: &Tensor) -> Result<Tensor> {
    let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
    x.broadcast_div(&(ms + 1e-6)?.sqrt()?)
}

/// Orthonormal Sylvester–Hadamard matrix of order `n` (a power of two).
pub fn hadamard(n: usize, device: &Device) -> Result<Tensor> {
    assert!(n.is_power_of_two(), "Sylvester construction needs a power of two");
    let mut h = vec![1f32];
    let mut m = 1;
    while m < n {
        let mut next = vec![0f32; 4 * m * m];
        for i in 0..m {
            for j in 0..m {
                let v = h[i * m + j];
                next[i * 2 * m + j] = v;
                next[i * 2 * m + j + m] = v;
                next[(i + m) * 2 * m + j] = v;
                next[(i + m) * 2 * m + j + m] = -v;
            }
        }
        h = next;
        m *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    Tensor::from_vec(h.into_iter().map(|x| x * scale).collect::<Vec<_>>(), (n, n), device)
}

/// HadamRNN cell: `h_t = tanh(h_{t-1} · (diag(s) H) ⊙ g + W_u x_t)`,
/// `y_t = h_t ⊙ σ(W_z x_t)`.
///
/// `H` is the fixed orthonormal Hadamard matrix (entries `±1/√d`), `s` a
/// learned binary sign per unit and `g ∈ (0, 1)` a learned gain. With
/// `g = 1` the recurrence is exactly orthogonal, which keeps gradients from
/// exploding or vanishing over long sequences.
pub struct HadamCell {
    wu: TLinear,
    wz: TLinear,
    sign: Tensor,
    gain: Tensor,
    h: Tensor,
}

impl HadamCell {
    pub fn new(vb: VarBuilder, d: usize) -> Result<Self> {
        let device = vb.device().clone();
        Ok(Self {
            wu: TLinear::new(vb.clone(), "wu", d, d)?,
            wz: TLinear::new(vb.clone(), "wz", d, d)?,
            sign: vb.get_with_hints(d, "sign", Init::Randn { mean: 0.0, stdev: 1.0 })?,
            gain: vb.get_with_hints(d, "gain", Init::Const(2.0))?,
            h: hadamard(d, &device)?,
        })
    }

    /// `x`: `(B, T, d)`, `h0`: `(B, d)`. Returns outputs and the last state.
    pub fn forward(&self, x: &Tensor, h0: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, t, _) = x.dims3()?;
        let u = self.wu.forward(x)?;
        let z = candle_nn::ops::sigmoid(&self.wz.forward(x)?)?;
        let rec = self
            .h
            .broadcast_mul(&ste_sign(&self.sign)?.unsqueeze(1)?)?
            .broadcast_mul(&candle_nn::ops::sigmoid(&self.gain)?.unsqueeze(0)?)?;
        let mut h = h0.clone();
        let mut hs = Vec::with_capacity(t);
        for i in 0..t {
            h = (h.matmul(&rec)? + u.narrow(1, i, 1)?.squeeze(1)?)?.tanh()?;
            hs.push(h.clone());
        }
        let out = (Tensor::stack(&hs, 1)? * z)?;
        Ok((out, h))
    }
}

/// Recurrent (linear) attention with per-head exponential decay
/// (retention): `S_t = γ S_{t-1} + k_tᵀ v_t`, `o_t = q_t S_t`.
///
/// Computed in parallel inside a window (`O(T²)` with a decay mask) with
/// the state carried between windows, so the cost per token is constant.
pub struct Retention {
    wq: TLinear,
    wk: TLinear,
    wv: TLinear,
    wo: TLinear,
    heads: usize,
    decays: Vec<f64>,
}

impl Retention {
    pub fn new(vb: VarBuilder, d: usize, heads: usize) -> Result<Self> {
        // γ_h = 1 − 3^{−(h+2)}: 0.889, 0.963, 0.988, 0.996, …
        let decays = (0..heads).map(|h| 1.0 - 3f64.powi(-(h as i32 + 2))).collect();
        Ok(Self {
            wq: TLinear::new(vb.clone(), "wq", d, d)?,
            wk: TLinear::new(vb.clone(), "wk", d, d)?,
            wv: TLinear::new(vb.clone(), "wv", d, d)?,
            wo: TLinear::new(vb, "wo", d, d)?,
            heads,
            decays,
        })
    }

    fn masks(&self, t: usize, device: &Device) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let h = self.heads;
        let (mut mask, mut cross, mut upd, mut total) = (vec![0f32; h * t * t], vec![0f32; h * t], vec![0f32; h * t], vec![0f32; h]);
        for (hi, &g) in self.decays.iter().enumerate() {
            for i in 0..t {
                for j in 0..=i {
                    mask[(hi * t + i) * t + j] = g.powi((i - j) as i32) as f32;
                }
                cross[hi * t + i] = g.powi(i as i32 + 1) as f32;
                upd[hi * t + i] = g.powi((t - 1 - i) as i32) as f32;
            }
            total[hi] = g.powi(t as i32) as f32;
        }
        Ok((
            Tensor::from_vec(mask, (1, h, t, t), device)?,
            Tensor::from_vec(cross, (1, h, t, 1), device)?,
            Tensor::from_vec(upd, (1, h, t, 1), device)?,
            Tensor::from_vec(total, (1, h, 1, 1), device)?,
        ))
    }

    /// `x`: `(B, T, d)`, `s0`: `(B, H, hd, hd)`. Returns outputs and state.
    pub fn forward(&self, x: &Tensor, s0: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, t, d) = x.dims3()?;
        let (h, hd) = (self.heads, d / self.heads);
        let split = |y: Tensor| -> Result<Tensor> { y.reshape((b, t, h, hd))?.transpose(1, 2)?.contiguous() };
        let q = (split(self.wq.forward(x)?)? / (hd as f64).sqrt())?;
        let k = split(self.wk.forward(x)?)?;
        let v = split(self.wv.forward(x)?)?;
        let (mask, cross, upd, total) = self.masks(t, x.device())?;

        let scores = q.matmul(&k.t()?)?.broadcast_mul(&mask)?;
        let intra = scores.matmul(&v)?;
        let carried = q.broadcast_mul(&cross)?.matmul(s0)?;
        let o = rms(&(intra + carried)?)?;
        let s1 = (k.broadcast_mul(&upd)?.t()?.matmul(&v)? + s0.broadcast_mul(&total)?)?;
        let o = o.transpose(1, 2)?.reshape((b, t, d))?;
        Ok((self.wo.forward(&o)?, s1))
    }
}

/// Gated MLP (SwiGLU).
pub struct Mlp {
    w1: TLinear,
    w3: TLinear,
    w2: TLinear,
}

impl Mlp {
    pub fn new(vb: VarBuilder, d: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            w1: TLinear::new(vb.clone(), "w1", d, hidden)?,
            w3: TLinear::new(vb.clone(), "w3", d, hidden)?,
            w2: TLinear::new(vb, "w2", hidden, d)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.w2.forward(&(self.w1.forward(x)?.silu()? * self.w3.forward(x)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn two_trit_quantization_has_nine_levels() {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1.0, (27, 81), &dev).unwrap();
        let (levels, scales) = quant2_levels(&w).unwrap();
        assert_eq!(scales.len(), 27);
        assert!(levels.iter().all(|&l| (-4..=4).contains(&l)));
        let distinct: std::collections::HashSet<i8> = levels.iter().copied().collect();
        assert!(distinct.len() >= 7, "most of the nine levels are used: {distinct:?}");
        // Quantized weights stay close to the latent ones.
        let q = quant2(&w).unwrap();
        let err = (q - &w).unwrap().abs().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(err < 0.25, "mean quantization error {err}");
    }

    #[test]
    fn hadamard_is_orthonormal() {
        let h = hadamard(256, &Device::Cpu).unwrap();
        let eye = h.matmul(&h.t().unwrap()).unwrap();
        let diag: f32 = eye.sum_all().unwrap().to_scalar().unwrap();
        assert!((diag - 256.0).abs() < 1e-3);
        let max_off = (eye - Tensor::eye(256, DType::F32, &Device::Cpu).unwrap()).unwrap().abs().unwrap();
        assert!(max_off.max_all().unwrap().to_scalar::<f32>().unwrap() < 1e-5);
    }

    #[test]
    fn retention_parallel_matches_recurrent_form() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let r = Retention::new(vb, 12, 3).unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 6, 12), &dev).unwrap();
        let s0 = Tensor::zeros((2, 3, 4, 4), DType::F32, &dev).unwrap();
        // Whole window at once vs. two halves with carried state.
        let (full, s_full) = r.forward(&x, &s0).unwrap();
        let (a, s_a) = r.forward(&x.narrow(1, 0, 3).unwrap(), &s0).unwrap();
        let (b, s_b) = r.forward(&x.narrow(1, 3, 3).unwrap(), &s_a).unwrap();
        let halves = Tensor::cat(&[a, b], 1).unwrap();
        let diff = (full - halves).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-4, "chunked retention differs by {diff}");
        let ds = (s_full - s_b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(ds < 1e-4);
    }

    #[test]
    fn hadam_cell_gradients_flow() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let cell = HadamCell::new(vb, 16).unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 9, 16), &dev).unwrap();
        let (y, _) = cell.forward(&x, &Tensor::zeros((2, 16), DType::F32, &dev).unwrap()).unwrap();
        let grads = y.sqr().unwrap().sum_all().unwrap().backward().unwrap();
        for v in vm.all_vars() {
            let g = grads.get(v.as_tensor()).expect("every parameter gets a gradient");
            assert!(g.abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() > 0.0);
        }
    }
}
