//! Bigram and trigram baselines with interpolated Kneser–Ney smoothing.
//!
//! * bigram: `P(w|v) = max(c(vw) − D, 0)/c(v) + D·N₁₊(v•)/c(v) · P_cont(w)`,
//!   `P_cont(w) ∝ N₁₊(•w) + 1` (add-one keeps unseen tokens finite);
//! * trigram: the same discounting over `c(uvw)`, interpolated with the
//!   bigram above.
//!
//! Probabilities are computed in `f64`; no probability is ever zero.

use std::collections::HashMap;

const D: f64 = 0.75;

pub struct NGram {
    vocab: usize,
    uni_cont: Vec<f64>,
    cont_total: f64,
    ctx1: Vec<f64>,
    types1: Vec<f64>,
    bi: HashMap<u32, u32>,
    ctx2: HashMap<u32, (u32, u32)>,
    tri: HashMap<u64, u32>,
}

#[inline]
fn k2(a: u16, b: u16) -> u32 {
    (a as u32) << 16 | b as u32
}

#[inline]
fn k3(a: u16, b: u16, c: u16) -> u64 {
    (a as u64) << 32 | (b as u64) << 16 | c as u64
}

impl NGram {
    pub fn train(tokens: &[u16], vocab: usize) -> Self {
        let mut bi: HashMap<u32, u32> = HashMap::new();
        let mut tri: HashMap<u64, u32> = HashMap::new();
        for w in tokens.windows(2) {
            *bi.entry(k2(w[0], w[1])).or_insert(0) += 1;
        }
        for w in tokens.windows(3) {
            *tri.entry(k3(w[0], w[1], w[2])).or_insert(0) += 1;
        }
        let (mut uni_cont, mut ctx1, mut types1) = (vec![0f64; vocab], vec![0f64; vocab], vec![0f64; vocab]);
        for (&k, &c) in &bi {
            let (v, w) = ((k >> 16) as usize, (k & 0xFFFF) as usize);
            uni_cont[w] += 1.0;
            ctx1[v] += c as f64;
            types1[v] += 1.0;
        }
        let cont_total = uni_cont.iter().sum::<f64>() + vocab as f64;
        let mut ctx2: HashMap<u32, (u32, u32)> = HashMap::new();
        for (&k, &c) in &tri {
            let e = ctx2.entry((k >> 16) as u32).or_insert((0, 0));
            e.0 += c;
            e.1 += 1;
        }
        Self { vocab, uni_cont, cont_total, ctx1, types1, bi, ctx2, tri }
    }

    fn p_cont(&self, w: u16) -> f64 {
        (self.uni_cont[w as usize] + 1.0) / self.cont_total
    }

    pub fn p_bigram(&self, v: u16, w: u16) -> f64 {
        let cv = self.ctx1[v as usize];
        if cv == 0.0 {
            return self.p_cont(w);
        }
        let c = self.bi.get(&k2(v, w)).copied().unwrap_or(0) as f64;
        (c - D).max(0.0) / cv + D * self.types1[v as usize] / cv * self.p_cont(w)
    }

    pub fn p_trigram(&self, u: u16, v: u16, w: u16) -> f64 {
        let lower = self.p_bigram(v, w);
        let Some(&(cuv, types)) = self.ctx2.get(&k2(u, v)) else { return lower };
        let c = self.tri.get(&k3(u, v, w)).copied().unwrap_or(0) as f64;
        let cuv = cuv as f64;
        (c - D).max(0.0) / cuv + D * types as f64 / cuv * lower
    }

    /// Mean cross-entropy (nats per token) of bigram and trigram on `tokens`.
    pub fn evaluate(&self, tokens: &[u16]) -> (f64, f64) {
        let (mut b, mut t, mut n) = (0f64, 0f64, 0usize);
        for w in tokens.windows(3) {
            b -= self.p_bigram(w[1], w[2]).ln();
            t -= self.p_trigram(w[0], w[1], w[2]).ln();
            n += 1;
        }
        (b / n as f64, t / n as f64)
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probabilities_sum_to_one_and_trigram_helps() {
        let text: Vec<u16> = (0..3000).map(|i| [1u16, 2, 3, 1, 2, 4][i % 6]).collect();
        let m = NGram::train(&text, 9);
        for (u, v) in [(1u16, 2u16), (7, 8), (2, 3)] {
            let sb: f64 = (0..9).map(|w| m.p_bigram(v, w)).sum();
            let st: f64 = (0..9).map(|w| m.p_trigram(u, v, w)).sum();
            assert!((sb - 1.0).abs() < 1e-9 && (st - 1.0).abs() < 1e-9, "{sb} {st}");
        }
        let (b, t) = m.evaluate(&text[..600]);
        assert!(t < b, "trigram {t} should beat bigram {b} on this pattern");
        assert!(m.p_trigram(8, 8, 8) > 0.0);
    }
}
