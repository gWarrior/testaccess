//! Ternary building blocks.
//!
//! Every weight matrix is quantized to **two trits** per weight: nine
//! levels `{-4, …, +4}` times a per-row scale (the row's mean absolute
//! value). Training keeps latent `f32` weights and uses the straight-through
//! estimator: the forward pass sees the quantized weights, the backward pass
//! treats quantization as identity.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

/// Largest level of a two-trit weight (balanced ternary `±(3 + 1)`).
pub const LEVELS: i32 = 4;
/// Largest level of a three-trit weight (`±(9 + 3 + 1)`).
pub const LEVELS3: i32 = 13;

/// MSE-optimal uniform step relative to mean|w| for Gaussian weights: the
/// initial step; training then learns each row's step.
fn step_ratio(levels: i32) -> f64 {
    if levels >= LEVELS3 {
        0.272
    } else {
        0.669
    }
}

/// A weight matrix quantized to `levels` per side (4: two trits, 13: three
/// trits) with a learned per-row step that is always an exact power of two,
/// `2^round(θ)`, so dequantization is a shift in training and inference
/// alike. The forward pass sees the quantized weights; gradients pass
/// straight through inside the range (clipped outside) and reach θ by the
/// LSQ rule.
#[derive(Clone, Debug)]
pub struct QTensor {
    pub w: Tensor,
    pub theta: Tensor,
    pub levels: i32,
}

impl QTensor {
    /// `(rows, cols)` weights `N(0, std²)`, the step initialized to the optimum.
    pub fn new(vb: &VarBuilder, name: &str, rows: usize, cols: usize, std: f64, levels: i32) -> Result<Self> {
        let w = vb.get_with_hints((rows, cols), name, Init::Randn { mean: 0.0, stdev: std })?;
        let theta0 = (step_ratio(levels) * std * (2.0 / std::f64::consts::PI).sqrt()).log2();
        let theta = vb.get_with_hints(rows, &format!("{name}_step"), Init::Const(theta0))?;
        Ok(Self { w, theta, levels })
    }

    /// Every row's optimal θ for its current weights (for a warm start).
    pub fn optimal_theta(&self) -> Result<Tensor> {
        let w = self.w.to_vec2::<f32>()?;
        let theta: Vec<f32> = w
            .iter()
            .map(|row| {
                let mean = row.iter().map(|x| x.abs() as f64).sum::<f64>() / row.len().max(1) as f64;
                (mean * step_ratio(self.levels)).max(1e-30).log2() as f32
            })
            .collect();
        Tensor::from_vec(theta, self.theta.shape(), self.theta.device())
    }

    pub fn q(&self) -> Result<Tensor> {
        self.w.contiguous()?.apply_op2(&self.theta.contiguous()?, Quant { levels: self.levels })
    }

    /// Integer levels and per-row exponents (the packed form).
    pub fn levels_and_exps(&self) -> Result<(Vec<i8>, Vec<i8>)> {
        let w = self.w.to_vec2::<f32>()?;
        let theta = self.theta.to_vec1::<f32>()?;
        let mut levels = Vec::new();
        let mut exps = Vec::new();
        for (row, &t) in w.iter().zip(&theta) {
            let e = t.round().clamp(-127.0, 127.0) as i8;
            let step = 2f32.powi(e as i32);
            exps.push(e);
            levels.extend(row.iter().map(|&x| (x / step).round().clamp(-self.levels as f32, self.levels as f32) as i8));
        }
        Ok((levels, exps))
    }

    /// Fractions of weights rounded to zero and clipped, and ‖q − w‖ / ‖w‖.
    pub fn health(&self) -> Result<(f64, f64, f64)> {
        let (levels, exps) = self.levels_and_exps()?;
        let w = self.w.to_vec2::<f32>()?;
        let (mut zero, mut clip, mut err, mut norm, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let cols = w.first().map_or(0, Vec::len);
        for (r, row) in w.iter().enumerate() {
            let step = 2f64.powi(exps[r] as i32);
            for (c, &x) in row.iter().enumerate() {
                let l = levels[r * cols + c];
                zero += f64::from(l == 0);
                clip += f64::from((x as f64 / step).abs() > self.levels as f64 + 0.5);
                err += (l as f64 * step - x as f64).powi(2);
                norm += (x as f64).powi(2);
                n += 1.0;
            }
        }
        Ok((zero / n, clip / n, (err / norm.max(1e-30)).sqrt()))
    }
}

/// Quantize an activation (e.g. a memory key) to `levels` per side along
/// its last dimension, step `2^round(log2(c·mean|x|))` per row, as the SNN
/// memory stores it; the gradient passes straight through.
pub fn quant_act(x: &Tensor, levels: i32) -> Result<Tensor> {
    x.contiguous()?.apply_op1(QuantAct { levels })
}

/// The per-row step of [`quant_act`] (shared with the K/V store).
pub fn act_step(row: &[f32], levels: i32) -> f32 {
    let mean = row.iter().map(|x| x.abs() as f64).sum::<f64>() / row.len().max(1) as f64;
    2f64.powi((mean * step_ratio(levels)).max(1e-30).log2().round() as i32) as f32
}

struct QuantAct {
    levels: i32,
}

impl candle_core::CustomOp1 for QuantAct {
    fn name(&self) -> &'static str {
        "quant-act"
    }

    fn cpu_fwd(
        &self,
        s: &candle_core::CpuStorage,
        l: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        use rayon::prelude::*;
        let x = cpu_tensor(s, l)?.flatten_all()?.to_vec1::<f32>()?;
        let cols = *l.shape().dims().last().expect("rank >= 1");
        let lv = self.levels as f32;
        let mut out = vec![0f32; x.len()];
        out.par_chunks_mut(cols).zip(x.par_chunks(cols)).for_each(|(o, row)| {
            let step = act_step(row, self.levels);
            for (o, &v) in o.iter_mut().zip(row) {
                *o = (v / step).round().clamp(-lv, lv) * step;
            }
        });
        Ok((candle_core::CpuStorage::F32(out), l.shape().clone()))
    }

    fn bwd(&self, _arg: &Tensor, _res: &Tensor, grad: &Tensor) -> Result<Option<Tensor>> {
        Ok(Some(grad.clone()))
    }
}

/// Quantize rows of `w` with steps `2^round(θ)`.
struct Quant {
    levels: i32,
}

impl candle_core::CustomOp2 for Quant {
    fn name(&self) -> &'static str {
        "quant"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        use rayon::prelude::*;
        let w = cpu_tensor(s1, l1)?.flatten_all()?.to_vec1::<f32>()?;
        let theta = cpu_tensor(s2, l2)?.to_vec1::<f32>()?;
        let cols = *l1.shape().dims().last().expect("rank 2");
        let lv = self.levels as f32;
        let mut out = vec![0f32; w.len()];
        out.par_chunks_mut(cols).zip(w.par_chunks(cols)).zip(theta.par_iter()).for_each(|((o, row), &t)| {
            let step = 2f32.powi(t.round() as i32);
            for (o, &x) in o.iter_mut().zip(row) {
                *o = (x / step).round().clamp(-lv, lv) * step;
            }
        });
        Ok((candle_core::CpuStorage::F32(out), l1.shape().clone()))
    }

    fn bwd(&self, w: &Tensor, theta: &Tensor, _res: &Tensor, g: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        use rayon::prelude::*;
        let (rows, cols) = w.dims2()?;
        let wv = w.detach().flatten_all()?.to_vec1::<f32>()?;
        let tv = theta.detach().to_vec1::<f32>()?;
        let gv = g.detach().contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let lv = self.levels as f32;
        // LSQ gradient scale: 1 / sqrt(cols · levels).
        let scale = 1.0 / ((cols as f32) * lv).sqrt();
        let mut gw = vec![0f32; rows * cols];
        let gt: Vec<f32> = gw
            .par_chunks_mut(cols)
            .enumerate()
            .map(|(r, gw)| {
                let step = 2f32.powi(tv[r].round() as i32);
                let mut dt = 0f32;
                for c in 0..cols {
                    let i = r * cols + c;
                    let x = wv[i] / step;
                    let inside = x.abs() <= lv + 0.5;
                    // Clipped STE for the weight.
                    gw[c] = if inside { gv[i] } else { 0.0 };
                    // dq/ds (LSQ), then ds/dθ = s·ln2 through the rounding.
                    let dq = if inside { x.round().clamp(-lv, lv) - x } else { x.signum() * lv };
                    dt += gv[i] * dq;
                }
                dt * step * std::f32::consts::LN_2 * scale
            })
            .collect();
        let dev = w.device();
        Ok((Some(Tensor::from_vec(gw, (rows, cols), dev)?), Some(Tensor::from_vec(gt, rows, dev)?)))
    }
}

/// Binary sign with STE (used for the Hadamard recurrence).
pub fn ste_sign(w: &Tensor) -> Result<Tensor> {
    let s = w.ge(0.0)?.to_dtype(DType::F32)?.affine(2.0, -1.0)?;
    w + (s - w)?.detach()
}

/// Linear map `x · Wᵀ` with a two-trit weight.
pub struct TLinear {
    pub w: QTensor,
}

impl TLinear {
    pub fn new(vb: VarBuilder, name: &str, d_in: usize, d_out: usize) -> Result<Self> {
        Self::with_std(vb, name, d_in, d_out, (1.0 / d_in as f64).sqrt())
    }

    pub fn with_std(vb: VarBuilder, name: &str, d_in: usize, d_out: usize, std: f64) -> Result<Self> {
        Ok(Self { w: QTensor::new(&vb, name, d_out, d_in, std, LEVELS)? })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        linear(x, &self.w.q()?)
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
    decay: Tensor,
    h: Tensor,
    ternary: bool,
}

/// Ternary decays of the Hadamard recurrence: unit `j` keeps `1 − 3^−k`
/// of its state per step, `k = 1…5` in five equal groups — horizons from 3
/// to about 243 tokens, fixed rather than learned (as in retention).
pub fn hadam_decays(d: usize) -> Vec<f32> {
    (0..d).map(|j| 1.0 - 3f32.powi(-(1 + (j * 5 / d) as i32))).collect()
}

impl HadamCell {
    pub fn new(vb: VarBuilder, d: usize, ternary: bool) -> Result<Self> {
        let device = vb.device().clone();
        Ok(Self {
            // Half the usual input scale keeps tanh out of saturation.
            wu: TLinear::with_std(vb.clone(), "wu", d, d, 0.5 / (d as f64).sqrt())?,
            wz: TLinear::new(vb.clone(), "wz", d, d)?,
            sign: vb.get_with_hints(d, "sign", Init::Randn { mean: 0.0, stdev: 1.0 })?,
            decay: Tensor::from_vec(hadam_decays(d), d, &device)?,
            h: hadamard(d, &device)?,
            ternary,
        })
    }

    pub fn qtensors(&self) -> Vec<(String, &QTensor)> {
        vec![("cell.wu".into(), &self.wu.w), ("cell.wz".into(), &self.wz.w)]
    }

    fn recurrence(&self) -> Result<Tensor> {
        self.h
            .broadcast_mul(&ste_sign(&self.sign)?.unsqueeze(1)?)?
            .broadcast_mul(&self.decay.unsqueeze(0)?)?
            .contiguous()
    }

    /// `x`: `(B, T, d)`, `h0`: `(B, d)`. Returns outputs and the last state.
    pub fn forward(&self, x: &Tensor, h0: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, t, _) = x.dims3()?;
        let u = self.wu.forward(x)?.contiguous()?;
        let z = candle_nn::ops::sigmoid(&self.wz.forward(x)?)?;
        let hs = u.apply_op2(&self.recurrence()?, HadamScan { h0: h0.detach(), ternary: self.ternary })?;
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
    /// Round the state to trits `{−1, 0, +1}` after `tanh` (straight-through).
    ternary: bool,
}

fn cpu_tensor(s: &candle_core::CpuStorage, l: &candle_core::Layout) -> Result<Tensor> {
    let (a, b) =
        l.contiguous_offsets().ok_or_else(|| candle_core::Error::Msg("hadam-scan needs contiguous inputs".into()))?;
    Tensor::from_slice(&s.as_slice::<f32>()?[a..b], l.shape(), &Device::Cpu)
}

fn scan(u: &Tensor, r: &Tensor, h0: &Tensor, ternary: bool) -> Result<Tensor> {
    let (_, t, _) = u.dims3()?;
    let ut = u.transpose(0, 1)?.contiguous()?;
    let mut h = h0.clone();
    let mut hs = Vec::with_capacity(t);
    for i in 0..t {
        h = (h.matmul(r)? + ut.get(i)?)?.tanh()?;
        if ternary {
            h = h.round()?;
        }
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
        let hs = scan(&cpu_tensor(s1, l1)?, &cpu_tensor(s2, l2)?, &self.h0, self.ternary)?;
        Ok((candle_core::CpuStorage::F32(hs.flatten_all()?.to_vec1()?), hs.shape().clone()))
    }

    fn bwd(&self, u: &Tensor, r: &Tensor, res: &Tensor, grad: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (b, t, d) = res.dims3()?;
        let (res, grad) = (res.detach(), grad.detach());
        let rt = r.detach().t()?.contiguous()?;
        let hs = res.transpose(0, 1)?.contiguous()?;
        let gs = grad.transpose(0, 1)?.contiguous()?;
        let prev = Tensor::cat(&[self.h0.unsqueeze(1)?, res.narrow(1, 0, t - 1)?], 1)?;
        // With trits the state is not tanh's output: recompute tanh from
        // the stored previous state and the input (straight-through).
        let cs = if self.ternary {
            let a = (prev.reshape((b * t, d))?.matmul(&r.detach())?.reshape((b, t, d))? + u.detach())?;
            a.tanh()?.transpose(0, 1)?.contiguous()?
        } else {
            hs.clone()
        };
        let mut da_next = Tensor::zeros((b, d), DType::F32, res.device())?;
        let mut das = Vec::with_capacity(t);
        for i in (0..t).rev() {
            let c = cs.get(i)?;
            let dh = (gs.get(i)? + da_next.matmul(&rt)?)?;
            let da = (dh * (1.0 - c.sqr()?)?)?;
            das.push(da.clone());
            da_next = da;
        }
        das.reverse();
        let da = Tensor::stack(&das, 1)?;
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
    pub fn qtensors(&self) -> Vec<(String, &QTensor)> {
        vec![
            ("ret.wq".into(), &self.wq.w),
            ("ret.wk".into(), &self.wk.w),
            ("ret.wv".into(), &self.wv.w),
            ("ret.wo".into(), &self.wo.w),
        ]
    }

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
    pub fn qtensors(&self) -> Vec<(String, &QTensor)> {
        vec![("mlp.w1".into(), &self.w1.w), ("mlp.w3".into(), &self.w3.w), ("mlp.w2".into(), &self.w2.w)]
    }

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

    fn qtensor(rows: usize, cols: usize, std: f64, levels: i32) -> (VarMap, QTensor) {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let q = QTensor::new(&vb, "w", rows, cols, std, levels).unwrap();
        (vm, q)
    }

    #[test]
    fn two_and_three_trit_quantization_use_their_levels() {
        for (levels, min_distinct, max_err) in [(LEVELS, 7, 0.2), (LEVELS3, 19, 0.09)] {
            let (_, w) = qtensor(27, 243, 1.0, levels);
            let (lv, exps) = w.levels_and_exps().unwrap();
            assert_eq!(exps.len(), 27);
            assert!(lv.iter().all(|&l| (-levels..=levels).contains(&(l as i32))));
            let distinct: std::collections::HashSet<i8> = lv.iter().copied().collect();
            assert!(distinct.len() >= min_distinct, "{levels}: {distinct:?}");
            let (_, _, err) = w.health().unwrap();
            assert!(err < max_err, "{levels} levels: relative error {err:.3}");
            // The forward pass sees exactly the packed values.
            let q = w.q().unwrap().to_vec2::<f32>().unwrap();
            for (r, row) in q.iter().enumerate() {
                for (c, &x) in row.iter().enumerate() {
                    assert_eq!(x, lv[r * 243 + c] as f32 * 2f32.powi(exps[r] as i32));
                }
            }
        }
    }

    #[test]
    fn quantization_neither_zeroes_nor_clips_too_much() {
        let (_, w) = qtensor(243, 729, 0.05, LEVELS);
        let (zero, clip, err) = w.health().unwrap();
        assert!(zero < 0.45 && clip < 0.02, "zero {zero:.3} clip {clip:.3}");
        assert!(err < 0.2, "relative error {err:.3}");
        // Tiny weights are neither flushed to zero nor overflow.
        let (_, tiny) = qtensor(3, 81, 1e-20, LEVELS);
        let (zero, _, err) = tiny.health().unwrap();
        assert!(zero < 0.45 && err < 0.2, "tiny weights keep precision: zero {zero} err {err}");
    }

    #[test]
    fn step_learns_by_lsq_and_clipped_weights_get_no_gradient() {
        let (vm, w) = qtensor(3, 81, 1.0, LEVELS);
        // Push one weight far outside the range.
        let mut v = w.w.to_vec2::<f32>().unwrap();
        v[0][0] = 1e3;
        vm.data().lock().unwrap()["w"].set(&Tensor::new(v, &Device::Cpu).unwrap()).unwrap();
        let loss = w.q().unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let gw = grads.get(&w.w).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(gw[0][0], 0.0, "a clipped weight gets no gradient");
        assert!(gw[1].iter().any(|&g| g != 0.0));
        let gt = grads.get(&w.theta).unwrap().to_vec1::<f32>().unwrap();
        assert!(gt.iter().all(|g| g.is_finite()) && gt.iter().any(|&g| g != 0.0), "{gt:?}");
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

        let fast = u.as_tensor().apply_op2(r.as_tensor(), HadamScan { h0: h0.clone(), ternary: false }).unwrap();
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
        let cell = HadamCell::new(vb, 16, false).unwrap();
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
