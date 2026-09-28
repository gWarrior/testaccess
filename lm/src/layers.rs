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

/// Quantization step relative to the row's mean |w|. For Gaussian weights
/// a 9-level uniform quantizer has minimal squared error at a step of about
/// 0.55 σ = 0.69 · mean|w|: coarser steps round too many weights to zero,
/// finer ones clip too many at ±4.
const STEP: f64 = 0.69;

/// Per-row step, rounded to a power of two so dequantization is a shift.
/// Clamped far from f32 underflow.
fn row_step(w: &Tensor) -> Result<Tensor> {
    let s = w.abs()?.mean_keepdim(D::Minus1)?;
    let steps: Vec<f32> = s
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|m| 2f64.powi((m as f64 * STEP).max(1e-30).log2().round() as i32) as f32)
        .collect();
    Tensor::from_vec(steps, s.shape(), w.device())
}

/// Quantize to two trits per weight (power-of-two row step), with STE.
pub fn quant2(w: &Tensor) -> Result<Tensor> {
    w.contiguous()?.apply_op1(Quant2)
}

/// One parallel pass per row: `step = 2^round(log2(0.69·mean|w|))`,
/// `q = clamp(round(w / step), ±4) · step`; the gradient passes straight
/// through. The step is an exact power of two, as in the packed model.
struct Quant2;

impl candle_core::CustomOp1 for Quant2 {
    fn name(&self) -> &'static str {
        "quant2"
    }

    fn cpu_fwd(
        &self,
        s: &candle_core::CpuStorage,
        l: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        use rayon::prelude::*;
        let w = cpu_tensor(s, l)?.flatten_all()?.to_vec1::<f32>()?;
        let cols = *l.shape().dims().last().expect("rank >= 1");
        let mut out = vec![0f32; w.len()];
        out.par_chunks_mut(cols).zip(w.par_chunks(cols)).for_each(|(o, row)| {
            let mean = row.iter().map(|x| x.abs() as f64).sum::<f64>() / cols as f64;
            let step = 2f64.powi((mean * STEP).max(1e-30).log2().round() as i32) as f32;
            for (o, &x) in o.iter_mut().zip(row) {
                *o = (x / step).round().clamp(-LEVELS as f32, LEVELS as f32) * step;
            }
        });
        Ok((candle_core::CpuStorage::F32(out), l.shape().clone()))
    }

    fn bwd(&self, _arg: &Tensor, _res: &Tensor, grad: &Tensor) -> Result<Option<Tensor>> {
        Ok(Some(grad.clone()))
    }
}

/// Integer levels and per-row power-of-two steps of a weight matrix.
pub fn quant2_levels(w: &Tensor) -> Result<(Vec<i8>, Vec<f32>)> {
    let step = row_step(w)?;
    let q = w.broadcast_div(&step)?.round()?.clamp(-LEVELS, LEVELS)?;
    let levels = q.flatten_all()?.to_vec1::<f32>()?.into_iter().map(|x| x as i8).collect();
    Ok((levels, step.flatten_all()?.to_vec1::<f32>()?))
}

/// Fractions of weights rounded to zero and clipped at ±4, and the relative
/// quantization error ‖q − w‖ / ‖w‖.
pub fn quant2_health(w: &Tensor) -> Result<(f64, f64, f64)> {
    let step = row_step(w)?;
    let x = w.broadcast_div(&step)?;
    let n = w.elem_count() as f64;
    let zero = x.abs()?.lt(0.5)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()? as f64 / n;
    let clip = x.abs()?.gt(LEVELS + 0.5)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()? as f64 / n;
    let q = x.round()?.clamp(-LEVELS, LEVELS)?.broadcast_mul(&step)?;
    let err = (q - w)?.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    let norm = w.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    Ok((zero, clip, (err / norm.max(1e-30)).sqrt()))
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
        linear(x, &quant2(&self.w)?)
    }
}

/// `x · wᵀ` for `x` of any rank and `w: (out, in)`, as one autograd node
/// whose backward uses contiguous operands (`candle`'s generic matmul
/// backward multiplies strided transposes, which is ~10× slower on CPU).
pub fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let k = *dims.last().expect("rank >= 1");
    let y = x.reshape((x.elem_count() / k, k))?.contiguous()?.apply_op2(&w.contiguous()?, MatMulNT)?;
    let mut out = dims;
    *out.last_mut().expect("rank >= 1") = w.dim(0)?;
    y.reshape(out)
}

struct MatMulNT;

impl candle_core::CustomOp2 for MatMulNT {
    fn name(&self) -> &'static str {
        "matmul-nt"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let y = cpu_tensor(s1, l1)?.matmul(&cpu_tensor(s2, l2)?.t()?.contiguous()?)?;
        Ok((candle_core::CpuStorage::F32(y.flatten_all()?.to_vec1()?), y.shape().clone()))
    }

    fn bwd(&self, x: &Tensor, w: &Tensor, _res: &Tensor, g: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (x, w, g) = (x.detach(), w.detach(), g.detach().contiguous()?);
        let dx = g.matmul(&w)?;
        let dw = g.t()?.contiguous()?.matmul(&x)?;
        Ok((Some(dx), Some(dw)))
    }
}

/// `x · w` for `x` of any rank, as a single 2-D matrix product (a batched
/// broadcast matmul would expand `w` per batch in the backward pass).
pub fn matmul_2d(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let k = *dims.last().expect("rank >= 1");
    let rows = x.elem_count() / k;
    let y = x.reshape((rows, k))?.matmul(w)?;
    let mut out = dims;
    *out.last_mut().expect("rank >= 1") = w.dim(1)?;
    y.reshape(out)
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

    fn recurrence(&self) -> Result<Tensor> {
        self.h
            .broadcast_mul(&ste_sign(&self.sign)?.unsqueeze(1)?)?
            .broadcast_mul(&candle_nn::ops::sigmoid(&self.gain)?.unsqueeze(0)?)?
            .contiguous()
    }

    /// `x`: `(B, T, d)`, `h0`: `(B, d)`. Returns outputs and the last state.
    pub fn forward(&self, x: &Tensor, h0: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, t, _) = x.dims3()?;
        let u = self.wu.forward(x)?.contiguous()?;
        let z = candle_nn::ops::sigmoid(&self.wz.forward(x)?)?;
        let hs = u.apply_op2(&self.recurrence()?, HadamScan { h0: h0.detach() })?;
        let last = hs.narrow(1, t - 1, 1)?.squeeze(1)?;
        Ok(((hs * z)?, last))
    }
}

/// `H = scan(U, R)`: `h_t = tanh(h_{t-1} R + u_t)` over a window, as one
/// autograd node with hand-written back-propagation through time.
///
/// Expressing the loop with ordinary tensor ops makes every step's
/// backward allocate a gradient the size of the whole window; here the
/// backward pass runs one reverse loop for `dA_t = (dH_t + dA_{t+1} Rᵀ) ⊙
/// (1 − h_t²)` and gets `dU = dA` and `dR = H_prevᵀ dA` from a single
/// matrix product.
struct HadamScan {
    h0: Tensor,
}

fn cpu_tensor(s: &candle_core::CpuStorage, l: &candle_core::Layout) -> Result<Tensor> {
    let (a, b) =
        l.contiguous_offsets().ok_or_else(|| candle_core::Error::Msg("hadam-scan needs contiguous inputs".into()))?;
    Tensor::from_slice(&s.as_slice::<f32>()?[a..b], l.shape(), &Device::Cpu)
}

fn scan(u: &Tensor, r: &Tensor, h0: &Tensor) -> Result<Tensor> {
    let (_, t, _) = u.dims3()?;
    let ut = u.transpose(0, 1)?.contiguous()?;
    let mut h = h0.clone();
    let mut hs = Vec::with_capacity(t);
    for i in 0..t {
        h = (h.matmul(r)? + ut.get(i)?)?.tanh()?;
        hs.push(h.clone());
    }
    Tensor::stack(&hs, 1)
}

impl candle_core::CustomOp2 for HadamScan {
    fn name(&self) -> &'static str {
        "hadam-scan"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let hs = scan(&cpu_tensor(s1, l1)?, &cpu_tensor(s2, l2)?, &self.h0)?;
        Ok((candle_core::CpuStorage::F32(hs.flatten_all()?.to_vec1()?), hs.shape().clone()))
    }

    fn bwd(&self, _u: &Tensor, r: &Tensor, res: &Tensor, grad: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (b, t, d) = res.dims3()?;
        let (res, grad) = (res.detach(), grad.detach());
        let rt = r.detach().t()?.contiguous()?;
        let hs = res.transpose(0, 1)?.contiguous()?;
        let gs = grad.transpose(0, 1)?.contiguous()?;
        let mut da_next = Tensor::zeros((b, d), DType::F32, res.device())?;
        let mut das = Vec::with_capacity(t);
        for i in (0..t).rev() {
            let h = hs.get(i)?;
            let dh = (gs.get(i)? + da_next.matmul(&rt)?)?;
            let da = (dh * (1.0 - h.sqr()?)?)?;
            das.push(da.clone());
            da_next = da;
        }
        das.reverse();
        let da = Tensor::stack(&das, 1)?;
        let prev = Tensor::cat(&[self.h0.unsqueeze(1)?, res.narrow(1, 0, t - 1)?], 1)?;
        let dr = prev.reshape((b * t, d))?.t()?.matmul(&da.reshape((b * t, d))?)?;
        Ok((Some(da), Some(dr)))
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
        let (mut mask, mut cross, mut upd, mut total) =
            (vec![0f32; h * t * t], vec![0f32; h * t], vec![0f32; h * t], vec![0f32; h]);
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
        let (a, b) = (self.w1.forward(x)?.contiguous()?, self.w3.forward(x)?.contiguous()?);
        self.w2.forward(&a.apply_op2_no_bwd(&b, &SwiGlu)?.detach().apply_op3(&a, &b, SwiGluGrad)?)
    }
}

/// `silu(a) ⊙ b` in one parallel pass (candle's CPU element-wise ops are
/// single-threaded, and silu's backward alone takes four passes).
struct SwiGlu;

fn silu_parts(a: f32) -> (f32, f32) {
    let s = 1.0 / (1.0 + (-a).exp());
    (a * s, s * (1.0 + a * (1.0 - s)))
}

impl candle_core::CustomOp2 for SwiGlu {
    fn name(&self) -> &'static str {
        "swiglu"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        use rayon::prelude::*;
        let a = cpu_tensor(s1, l1)?.flatten_all()?.to_vec1::<f32>()?;
        let b = cpu_tensor(s2, l2)?.flatten_all()?.to_vec1::<f32>()?;
        let mut out = vec![0f32; a.len()];
        out.par_chunks_mut(1 << 14).zip(a.par_chunks(1 << 14).zip(b.par_chunks(1 << 14))).for_each(|(o, (a, b))| {
            for ((o, &a), &b) in o.iter_mut().zip(a).zip(b) {
                *o = silu_parts(a).0 * b;
            }
        });
        Ok((candle_core::CpuStorage::F32(out), l1.shape().clone()))
    }
}

/// Carries the gradient of [`SwiGlu`]: the forward returns its first
/// argument (the detached product), the backward differentiates `a`, `b`.
struct SwiGluGrad;

impl candle_core::CustomOp3 for SwiGluGrad {
    fn name(&self) -> &'static str {
        "swiglu-grad"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &candle_core::Layout,
        _s3: &candle_core::CpuStorage,
        _l3: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let y = cpu_tensor(s1, l1)?.flatten_all()?.to_vec1::<f32>()?;
        Ok((candle_core::CpuStorage::F32(y), l1.shape().clone()))
    }

    fn bwd(
        &self,
        _y: &Tensor,
        a: &Tensor,
        b: &Tensor,
        _res: &Tensor,
        g: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        use rayon::prelude::*;
        let shape = a.shape().clone();
        let av = a.detach().flatten_all()?.to_vec1::<f32>()?;
        let bv = b.detach().flatten_all()?.to_vec1::<f32>()?;
        let gv = g.detach().contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let n = av.len();
        let (mut da, mut db) = (vec![0f32; n], vec![0f32; n]);
        const C: usize = 1 << 14;
        da.par_chunks_mut(C).zip(db.par_chunks_mut(C)).enumerate().for_each(|(ci, (da, db))| {
            let o = ci * C;
            for i in 0..da.len() {
                let (silu, dsilu) = silu_parts(av[o + i]);
                da[i] = gv[o + i] * bv[o + i] * dsilu;
                db[i] = gv[o + i] * silu;
            }
        });
        let dev = a.device();
        Ok((None, Some(Tensor::from_vec(da, shape.clone(), dev)?), Some(Tensor::from_vec(db, shape, dev)?)))
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
    fn quantization_neither_zeroes_nor_clips_too_much() {
        let w = Tensor::randn(0f32, 0.05, (243, 729), &Device::Cpu).unwrap();
        let (zero, clip, err) = quant2_health(&w).unwrap();
        assert!(zero < 0.45 && clip < 0.02, "zero {zero:.3} clip {clip:.3}");
        assert!(err < 0.2, "relative error {err:.3}");
        let (_, steps) = quant2_levels(&w).unwrap();
        assert!(steps.iter().all(|s| s.log2().fract() == 0.0), "steps are powers of two");
        // Tiny weights are neither flushed to zero nor overflow.
        let tiny = Tensor::randn(0f32, 1e-20, (3, 81), &Device::Cpu).unwrap();
        let (zero, _, err) = quant2_health(&tiny).unwrap();
        assert!(zero < 0.45 && err < 0.2, "tiny weights keep precision: zero {zero} err {err}");
    }

    #[test]
    fn linear_matches_matmul_with_gradients() {
        let dev = Device::Cpu;
        let x = candle_core::Var::randn(0f32, 1.0, (2, 3, 9), &dev).unwrap();
        let w = candle_core::Var::randn(0f32, 1.0, (4, 9), &dev).unwrap();
        let a = linear(x.as_tensor(), w.as_tensor()).unwrap();
        let b = x.as_tensor().broadcast_matmul(&w.as_tensor().t().unwrap()).unwrap();
        let ga = a.sqr().unwrap().sum_all().unwrap().backward().unwrap();
        let gb = b.sqr().unwrap().sum_all().unwrap().backward().unwrap();
        let max = |t: Tensor| t.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(max((&a - &b).unwrap()) < 1e-4);
        for v in [&x, &w] {
            assert!(max((ga.get(v.as_tensor()).unwrap() - gb.get(v.as_tensor()).unwrap()).unwrap()) < 1e-3);
        }
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
    fn hadam_scan_gradients_match_autograd_loop() {
        let dev = Device::Cpu;
        let u = candle_core::Var::randn(0f32, 1.0, (2, 5, 8), &dev).unwrap();
        let r = candle_core::Var::randn(0f32, 0.3, (8, 8), &dev).unwrap();
        let h0 = Tensor::randn(0f32, 0.5, (2, 8), &dev).unwrap();
        let w = Tensor::randn(0f32, 1.0, (2, 5, 8), &dev).unwrap();

        let fast = u.as_tensor().apply_op2(r.as_tensor(), HadamScan { h0: h0.clone() }).unwrap();
        let g1 = (&fast * &w).unwrap().sum_all().unwrap().backward().unwrap();

        let mut h = h0.clone();
        let mut hs = Vec::new();
        for i in 0..5 {
            h = (h.matmul(r.as_tensor()).unwrap() + u.as_tensor().narrow(1, i, 1).unwrap().squeeze(1).unwrap())
                .unwrap()
                .tanh()
                .unwrap();
            hs.push(h.clone());
        }
        let slow = Tensor::stack(&hs, 1).unwrap();
        let g2 = (&slow * &w).unwrap().sum_all().unwrap().backward().unwrap();

        let max_diff =
            |a: &Tensor, b: &Tensor| (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(max_diff(&fast, &slow) < 1e-5);
        for v in [&u, &r] {
            let d = max_diff(g1.get(v.as_tensor()).unwrap(), g2.get(v.as_tensor()).unwrap());
            assert!(d < 1e-4, "gradient mismatch {d}");
        }
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

/// Fused softmax cross-entropy: per-row loss of `logits (N, V)` against
/// `targets`, with the analytic gradient `softmax − onehot`.
pub struct CrossEntropy {
    pub targets: std::sync::Arc<Vec<u32>>,
}

impl candle_core::CustomOp1 for CrossEntropy {
    fn name(&self) -> &'static str {
        "cross-entropy"
    }

    fn cpu_fwd(
        &self,
        s: &candle_core::CpuStorage,
        l: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        use rayon::prelude::*;
        let x = cpu_tensor(s, l)?;
        let (n, v) = x.dims2()?;
        let data = x.flatten_all()?.to_vec1::<f32>()?;
        let losses: Vec<f32> = data
            .par_chunks(v)
            .zip(self.targets.par_iter())
            .map(|(row, &t)| {
                let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let z: f32 = row.iter().map(|&a| (a - m).exp()).sum();
                m + z.ln() - row[t as usize]
            })
            .collect();
        debug_assert_eq!(losses.len(), n);
        Ok((candle_core::CpuStorage::F32(losses), candle_core::Shape::from(n)))
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad: &Tensor) -> Result<Option<Tensor>> {
        use rayon::prelude::*;
        let (n, v) = arg.dims2()?;
        let data = arg.detach().flatten_all()?.to_vec1::<f32>()?;
        let g = grad.detach().to_vec1::<f32>()?;
        let mut out = vec![0f32; n * v];
        out.par_chunks_mut(v).zip(data.par_chunks(v)).enumerate().for_each(|(i, (o, row))| {
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let z: f32 = row.iter().map(|&a| (a - m).exp()).sum();
            for (oj, &a) in o.iter_mut().zip(row) {
                *oj = (a - m).exp() / z * g[i];
            }
            o[self.targets[i] as usize] -= g[i];
        });
        Ok(Some(Tensor::from_vec(out, (n, v), arg.device())?))
    }
}

#[cfg(test)]
mod swiglu_tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn fused_swiglu_matches_autograd() {
        let dev = Device::Cpu;
        let a = candle_core::Var::from_tensor(&Tensor::randn(0f32, 2.0, (3, 5, 7), &dev).unwrap()).unwrap();
        let b = candle_core::Var::from_tensor(&Tensor::randn(0f32, 1.0, (3, 5, 7), &dev).unwrap()).unwrap();
        let w = Tensor::randn(0f32, 1.0, (3, 5, 7), &dev).unwrap();
        let fused = a.as_tensor().apply_op2_no_bwd(b.as_tensor(), &SwiGlu).unwrap().detach();
        let fused = fused.apply_op3(a.as_tensor(), b.as_tensor(), SwiGluGrad).unwrap();
        let plain = (a.as_tensor().silu().unwrap() * b.as_tensor()).unwrap();
        let diff = (&fused - &plain).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-5, "forward differs by {diff}");
        let gf = (fused * &w).unwrap().sum_all().unwrap().backward().unwrap();
        let gp = (plain * &w).unwrap().sum_all().unwrap().backward().unwrap();
        for v in [&a, &b] {
            let d = (gf.get(v.as_tensor()).unwrap() - gp.get(v.as_tensor()).unwrap())
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(d < 1e-4, "gradient differs by {d}");
        }
        let _ = VarMap::new();
    }
}

#[cfg(test)]
mod ce_tests {
    use super::*;

    #[test]
    fn fused_cross_entropy_matches_log_softmax() {
        let dev = Device::Cpu;
        let x = candle_core::Var::randn(0f32, 2.0, (5, 9), &dev).unwrap();
        let t = vec![0u32, 3, 8, 1, 1];
        let fused = x.as_tensor().apply_op1(CrossEntropy { targets: std::sync::Arc::new(t.clone()) }).unwrap();
        let g1 = fused.mean_all().unwrap().backward().unwrap();
        let lp = candle_nn::ops::log_softmax(x.as_tensor(), 1).unwrap();
        let idx = Tensor::from_vec(t, (5, 1), &dev).unwrap();
        let slow = lp.gather(&idx, 1).unwrap().squeeze(1).unwrap().neg().unwrap();
        let g2 = slow.mean_all().unwrap().backward().unwrap();
        let d = (fused - slow).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(d < 1e-5);
        let dg = (g1.get(x.as_tensor()).unwrap() - g2.get(x.as_tensor()).unwrap()).unwrap().abs().unwrap();
        assert!(dg.max_all().unwrap().to_scalar::<f32>().unwrap() < 1e-5);
    }
}
