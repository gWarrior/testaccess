//! AdamW whose state can be saved and restored, so that a resumed run
//! continues exactly (candle's `AdamW` keeps its moments private).

use std::collections::HashMap;

use candle_core::backprop::GradStore;
use candle_core::{Result, Tensor, Var};

struct Slot {
    name: String,
    var: Var,
    m: Tensor,
    v: Tensor,
}

/// AdamW without weight decay, the update of `candle_nn::AdamW`.
pub struct AdamW {
    slots: Vec<Slot>,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    /// Updates done (bias correction).
    pub t: usize,
}

impl AdamW {
    pub fn new(vars: Vec<(String, Var)>, lr: f64) -> Result<Self> {
        let slots = vars
            .into_iter()
            .map(|(name, var)| {
                let m = var.as_tensor().zeros_like()?;
                let v = var.as_tensor().zeros_like()?;
                Ok(Slot { name, var, m, v })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { slots, lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0 })
    }

    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.t += 1;
        let (b1, b2) = (self.beta1, self.beta2);
        let scale_m = 1.0 / (1.0 - b1.powi(self.t as i32));
        let scale_v = 1.0 / (1.0 - b2.powi(self.t as i32));
        for s in &mut self.slots {
            if let Some(g) = grads.get(s.var.as_tensor()) {
                s.m = ((&s.m * b1)? + (g * (1.0 - b1))?)?;
                s.v = ((&s.v * b2)? + (g.sqr()? * (1.0 - b2))?)?;
                let update = ((&s.m * scale_m)? / ((&s.v * scale_v)?.sqrt()? + self.eps)?)?;
                s.var.set(&(s.var.as_tensor() - (update * self.lr)?)?)?;
            }
        }
        Ok(())
    }

    /// Moments as `{prefix}.{name}.m` / `.v`.
    pub fn state(&self, prefix: &str, out: &mut HashMap<String, Tensor>) {
        for s in &self.slots {
            out.insert(format!("{prefix}.{}.m", s.name), s.m.clone());
            out.insert(format!("{prefix}.{}.v", s.name), s.v.clone());
        }
    }

    /// Restore the moments saved by [`state`](Self::state); a variable
    /// missing from `saved` keeps zero moments. Returns how many were found.
    pub fn load_state(&mut self, prefix: &str, saved: &HashMap<String, Tensor>) -> Result<usize> {
        let mut n = 0;
        for s in &mut self.slots {
            let (m, v) = (saved.get(&format!("{prefix}.{}.m", s.name)), saved.get(&format!("{prefix}.{}.v", s.name)));
            if let (Some(m), Some(v)) = (m, v) {
                if m.shape() != s.var.shape() || v.shape() != s.var.shape() {
                    candle_core::bail!("optimizer state for {} has shape {:?}", s.name, m.shape());
                }
                (s.m, s.v) = (m.clone(), v.clone());
                n += 1;
            }
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::Optimizer;

    #[test]
    fn matches_candle_adamw_and_resumes_exactly() {
        let dev = Device::Cpu;
        let init = Tensor::new(&[0.5f32, -1.0, 2.0], &dev).unwrap();
        let target = Tensor::new(&[1.0f32, 0.0, -1.0], &dev).unwrap();
        let loss = |w: &Var| (w.as_tensor() - &target).unwrap().sqr().unwrap().sum_all().unwrap();
        let (a, b) = (Var::from_tensor(&init).unwrap(), Var::from_tensor(&init).unwrap());
        let mut ours = AdamW::new(vec![("w".into(), a.clone())], 0.1).unwrap();
        let params = candle_nn::ParamsAdamW { lr: 0.1, weight_decay: 0.0, ..Default::default() };
        let mut theirs = candle_nn::AdamW::new(vec![b.clone()], params).unwrap();
        for _ in 0..5 {
            ours.step(&loss(&a).backward().unwrap()).unwrap();
            theirs.step(&loss(&b).backward().unwrap()).unwrap();
        }
        let diff = (a.as_tensor() - b.as_tensor()).unwrap().abs().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-6, "{diff}");
        // Save, restore into a fresh optimizer, continue: same as never stopping.
        let mut saved = HashMap::new();
        ours.state("w", &mut saved);
        let c = Var::from_tensor(a.as_tensor()).unwrap();
        let mut resumed = AdamW::new(vec![("w".into(), c.clone())], 0.1).unwrap();
        assert_eq!(resumed.load_state("w", &saved).unwrap(), 1);
        resumed.t = ours.t;
        for _ in 0..3 {
            ours.step(&loss(&a).backward().unwrap()).unwrap();
            resumed.step(&loss(&c).backward().unwrap()).unwrap();
        }
        assert_eq!(a.as_tensor().to_vec1::<f32>().unwrap(), c.as_tensor().to_vec1::<f32>().unwrap());
    }
}
