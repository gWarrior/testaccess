//! Spike encoders: turn inputs into sparse sets of active neurons.
//!
//! * [`FlyHashEncoder`] — dense/sparse vectors (e.g. LLM hidden states) are
//!   expanded through fixed sparse ternary (`+1/0/-1`) random connectivity
//!   into a large neuron layer, and a k-winners-take-all (lateral inhibition) step keeps only the
//!   `k` most driven neurons. This is the fruit-fly olfactory circuit
//!   (Dasgupta et al., 2017): similar inputs produce overlapping codes.
//! * [`NGramEncoder`] — token sequences drive a bank of temporal
//!   coincidence-detector neurons. Each detector fires when a specific ordered
//!   run of `n` tokens arrives (a delay-line coincidence detector), so the code
//!   preserves order. A fragment of a sequence activates a *subset* of the
//!   sequence's detectors, which is what makes partial recall exact.
//! * [`CodeEncoder`] — accepts only pre-computed spike codes.

use crate::rng::{mix64, SplitMix64};
use crate::types::{Input, MemoryError};

/// Converts inputs into a sorted, de-duplicated set of active neuron ids.
pub trait Encoder: Send + Sync {
    /// Size of the neuron space codes live in.
    fn n_neurons(&self) -> u32;

    /// Appends the active neurons for `input` to `out` (sorted, unique).
    fn encode(&self, input: Input<'_>, out: &mut Vec<u32>) -> Result<(), MemoryError>;
}

/// Encoder for pre-computed spike codes only.
#[derive(Clone, Debug)]
pub struct CodeEncoder {
    n_neurons: u32,
}

impl CodeEncoder {
    pub fn new(n_neurons: u32) -> Self {
        Self { n_neurons }
    }
}

impl Encoder for CodeEncoder {
    fn n_neurons(&self) -> u32 {
        self.n_neurons
    }

    fn encode(&self, input: Input<'_>, out: &mut Vec<u32>) -> Result<(), MemoryError> {
        match input {
            Input::Code(code) => {
                check_code(code, self.n_neurons)?;
                out.extend_from_slice(code);
                out.sort_unstable();
                out.dedup();
                Ok(())
            }
            Input::Dense(_) => Err(MemoryError::UnsupportedInput("dense")),
            Input::Sparse(_) => Err(MemoryError::UnsupportedInput("sparse")),
            Input::Tokens(_) => Err(MemoryError::UnsupportedInput("token")),
        }
    }
}

pub(crate) fn check_code(code: &[u32], n_neurons: u32) -> Result<(), MemoryError> {
    match code.iter().find(|&&n| n >= n_neurons) {
        Some(&neuron) => Err(MemoryError::NeuronOutOfRange { neuron, n_neurons }),
        None => Ok(()),
    }
}

/// Sparse random expansion + k-winners-take-all.
#[derive(Clone, Debug)]
pub struct FlyHashEncoder {
    input_dim: usize,
    n_neurons: u32,
    k: usize,
    fan_in: usize,
    /// Forward connectivity, `n_neurons * fan_in` input indices and ternary
    /// weights (`+1` / `-1`; unconnected inputs are the implicit `0`).
    fwd_idx: Vec<u32>,
    fwd_sign: Vec<i8>,
    /// Inverse connectivity (CSR by input dimension) for sparse inputs.
    inv_start: Vec<u32>,
    inv_neuron: Vec<u32>,
    inv_sign: Vec<i8>,
    /// Fixed per-neuron priority used to break activation ties.
    priority: Vec<u32>,
    center: Option<Vec<f32>>,
}

impl FlyHashEncoder {
    /// `k` winners out of `n_neurons`; each neuron samples `fan_in` inputs.
    pub fn new(
        input_dim: usize,
        n_neurons: u32,
        k: usize,
        fan_in: usize,
        seed: u64,
    ) -> Result<Self, MemoryError> {
        if input_dim == 0 || n_neurons == 0 {
            return Err(MemoryError::InvalidConfig("input_dim and n_neurons must be > 0".into()));
        }
        if k == 0 || k > n_neurons as usize {
            return Err(MemoryError::InvalidConfig(format!("k must be in 1..={n_neurons}")));
        }
        let fan_in = fan_in.clamp(1, input_dim);
        let n = n_neurons as usize;
        let mut rng = SplitMix64::new(seed ^ 0xF1A5_4A54);
        let mut fwd_idx = Vec::with_capacity(n * fan_in);
        let mut fwd_sign = Vec::with_capacity(n * fan_in);
        for _ in 0..n {
            for d in rng.sample_distinct(input_dim as u32, fan_in) {
                fwd_idx.push(d);
                fwd_sign.push(if rng.next_u64() & 1 == 0 { 1 } else { -1 });
            }
        }
        let priority = (0..n).map(|_| rng.next_u64() as u32).collect();

        let mut counts = vec![0u32; input_dim + 1];
        for &d in &fwd_idx {
            counts[d as usize + 1] += 1;
        }
        for i in 0..input_dim {
            counts[i + 1] += counts[i];
        }
        let inv_start = counts.clone();
        let mut fill = counts;
        let mut inv_neuron = vec![0u32; fwd_idx.len()];
        let mut inv_sign = vec![0i8; fwd_idx.len()];
        for (e, (&d, &s)) in fwd_idx.iter().zip(&fwd_sign).enumerate() {
            let p = fill[d as usize] as usize;
            inv_neuron[p] = (e / fan_in) as u32;
            inv_sign[p] = s;
            fill[d as usize] += 1;
        }

        Ok(Self {
            input_dim,
            n_neurons,
            k,
            fan_in,
            fwd_idx,
            fwd_sign,
            inv_start,
            inv_neuron,
            inv_sign,
            priority,
            center: None,
        })
    }

    /// Subtract `mean` from dense inputs before projection. LLM hidden states
    /// are strongly anisotropic; centering spreads codes across neurons.
    pub fn with_center(mut self, mean: Vec<f32>) -> Result<Self, MemoryError> {
        if mean.len() != self.input_dim {
            return Err(MemoryError::DimensionMismatch { expected: self.input_dim, got: mean.len() });
        }
        self.center = Some(mean);
        Ok(self)
    }

    /// Estimate the centering vector from row-major `samples`.
    pub fn fit_center(&mut self, samples: &[f32]) -> Result<(), MemoryError> {
        let d = self.input_dim;
        if samples.is_empty() || samples.len() % d != 0 {
            return Err(MemoryError::DimensionMismatch { expected: d, got: samples.len() % d });
        }
        let rows = samples.len() / d;
        let mut mean = vec![0f32; d];
        for row in samples.chunks_exact(d) {
            for (m, &x) in mean.iter_mut().zip(row) {
                *m += x;
            }
        }
        mean.iter_mut().for_each(|m| *m /= rows as f32);
        self.center = Some(mean);
        Ok(())
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn k(&self) -> usize {
        self.k
    }

    fn encode_dense(&self, x: &[f32], out: &mut Vec<u32>) -> Result<(), MemoryError> {
        if x.len() != self.input_dim {
            return Err(MemoryError::DimensionMismatch { expected: self.input_dim, got: x.len() });
        }
        let centered: Vec<f32>;
        let x = match &self.center {
            Some(mean) => {
                centered = x.iter().zip(mean).map(|(a, b)| a - b).collect();
                &centered[..]
            }
            None => x,
        };
        let n = self.n_neurons as usize;
        let mut act = vec![0f32; n];
        for (i, a) in act.iter_mut().enumerate() {
            let base = i * self.fan_in;
            let idx = &self.fwd_idx[base..base + self.fan_in];
            let sign = &self.fwd_sign[base..base + self.fan_in];
            *a = idx.iter().zip(sign).map(|(&d, &s)| if s > 0 { x[d as usize] } else { -x[d as usize] }).sum();
        }
        let mut pool: Vec<u32> = (0..self.n_neurons).collect();
        k_winners(&act, &self.priority, &mut pool, self.k, out);
        Ok(())
    }

    fn encode_sparse(&self, x: &[(u32, f32)], out: &mut Vec<u32>) -> Result<(), MemoryError> {
        let mut act = vec![0f32; self.n_neurons as usize];
        let mut touched = Vec::new();
        for &(d, v) in x {
            if d as usize >= self.input_dim {
                return Err(MemoryError::DimensionMismatch { expected: self.input_dim, got: d as usize + 1 });
            }
            let (s, e) = (self.inv_start[d as usize] as usize, self.inv_start[d as usize + 1] as usize);
            for p in s..e {
                let nrn = self.inv_neuron[p];
                if act[nrn as usize] == 0.0 {
                    touched.push(nrn);
                }
                act[nrn as usize] += self.inv_sign[p] as f32 * v;
            }
        }
        touched.sort_unstable();
        touched.dedup();
        touched.retain(|&nrn| act[nrn as usize] > 0.0);
        k_winners(&act, &self.priority, &mut touched, self.k, out);
        Ok(())
    }
}

impl Encoder for FlyHashEncoder {
    fn n_neurons(&self) -> u32 {
        self.n_neurons
    }

    fn encode(&self, input: Input<'_>, out: &mut Vec<u32>) -> Result<(), MemoryError> {
        match input {
            Input::Dense(x) => self.encode_dense(x, out),
            Input::Sparse(x) => self.encode_sparse(x, out),
            Input::Code(code) => {
                check_code(code, self.n_neurons)?;
                out.extend_from_slice(code);
                out.sort_unstable();
                out.dedup();
                Ok(())
            }
            Input::Tokens(_) => Err(MemoryError::UnsupportedInput("token")),
        }
    }
}

/// k-winners-take-all over `pool`; appends winners to `out` sorted by id.
fn k_winners(act: &[f32], priority: &[u32], pool: &mut [u32], k: usize, out: &mut Vec<u32>) {
    let k = k.min(pool.len());
    if k == 0 {
        return;
    }
    let cmp = |a: &u32, b: &u32| {
        act[*b as usize]
            .total_cmp(&act[*a as usize])
            .then(priority[*a as usize].cmp(&priority[*b as usize]))
    };
    if k < pool.len() {
        pool.select_nth_unstable_by(k - 1, cmp);
    }
    let start = out.len();
    out.extend_from_slice(&pool[..k]);
    out[start..].sort_unstable();
}

/// Temporal coincidence detectors over token n-grams.
#[derive(Clone, Debug)]
pub struct NGramEncoder {
    n_neurons: u32,
    ngrams: Vec<usize>,
    per_feature: u32,
    seed: u64,
}

impl NGramEncoder {
    /// Detectors for every n in `ngrams`, each feature driving `per_feature`
    /// neurons out of `n_neurons`.
    pub fn new(n_neurons: u32, ngrams: &[usize], per_feature: u32, seed: u64) -> Result<Self, MemoryError> {
        if n_neurons == 0 || per_feature == 0 {
            return Err(MemoryError::InvalidConfig("n_neurons and per_feature must be > 0".into()));
        }
        let mut ngrams: Vec<usize> = ngrams.iter().copied().filter(|&n| n > 0).collect();
        ngrams.sort_unstable();
        ngrams.dedup();
        if ngrams.is_empty() {
            return Err(MemoryError::InvalidConfig("at least one n-gram size is required".into()));
        }
        Ok(Self { n_neurons, ngrams, per_feature, seed })
    }

    /// Upper bound on the code size of a sequence of `n_tokens`.
    pub fn max_code_len(&self, n_tokens: usize) -> usize {
        self.ngrams
            .iter()
            .map(|&n| n_tokens.saturating_sub(n - 1) * self.per_feature as usize)
            .sum()
    }

    pub fn ngrams(&self) -> &[usize] {
        &self.ngrams
    }

    #[inline]
    fn feature_neurons(&self, h: u64, out: &mut Vec<u32>) {
        for r in 0..self.per_feature as u64 {
            let x = mix64(h.wrapping_add(r.wrapping_mul(0xD6E8_FEB8_6659_FD93)));
            out.push(((x as u128 * self.n_neurons as u128) >> 64) as u32);
        }
    }
}

impl Encoder for NGramEncoder {
    fn n_neurons(&self) -> u32 {
        self.n_neurons
    }

    fn encode(&self, input: Input<'_>, out: &mut Vec<u32>) -> Result<(), MemoryError> {
        let start = out.len();
        match input {
            Input::Tokens(tokens) => {
                for &n in &self.ngrams {
                    let salt = mix64(self.seed ^ (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    for window in tokens.windows(n) {
                        let mut h = salt;
                        for &t in window {
                            h = mix64(h ^ (t as u64).wrapping_add(0x632B_E59B_D9B4_E019));
                        }
                        self.feature_neurons(h, out);
                    }
                }
            }
            Input::Sparse(features) => {
                let salt = mix64(self.seed ^ 0x5EED_F00D);
                for &(f, v) in features {
                    if v > 0.0 {
                        self.feature_neurons(mix64(salt ^ f as u64), out);
                    }
                }
            }
            Input::Code(code) => {
                check_code(code, self.n_neurons)?;
                out.extend_from_slice(code);
            }
            Input::Dense(_) => return Err(MemoryError::UnsupportedInput("dense")),
        }
        out[start..].sort_unstable();
        let mut w = start;
        for r in start..out.len() {
            if r == start || out[r] != out[w - 1] {
                out[w] = out[r];
                w += 1;
            }
        }
        out.truncate(w);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overlap(a: &[u32], b: &[u32]) -> usize {
        a.iter().filter(|x| b.binary_search(x).is_ok()).count()
    }

    fn encode(e: &dyn Encoder, input: Input<'_>) -> Vec<u32> {
        let mut out = Vec::new();
        e.encode(input, &mut out).unwrap();
        out
    }

    #[test]
    fn flyhash_is_sparse_sorted_and_similarity_preserving() {
        let d = 243;
        let enc = FlyHashEncoder::new(d, 6561, 27, 9, 42).unwrap();
        let mut rng = SplitMix64::new(3);
        let x: Vec<f32> = (0..d).map(|_| rng.normal() as f32).collect();
        let near: Vec<f32> = x.iter().map(|v| v + 0.3 * rng.normal() as f32).collect();
        let far: Vec<f32> = (0..d).map(|_| rng.normal() as f32).collect();

        let cx = encode(&enc, Input::Dense(&x));
        assert_eq!(cx.len(), 27);
        assert!(cx.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(cx, encode(&enc, Input::Dense(&x)), "encoding must be deterministic");
        let o_near = overlap(&cx, &encode(&enc, Input::Dense(&near)));
        let o_far = overlap(&cx, &encode(&enc, Input::Dense(&far)));
        assert!(o_near > 9, "near overlap too small: {o_near}");
        assert!(o_far < 3, "far overlap too large: {o_far}");
    }

    #[test]
    fn flyhash_rejects_wrong_dimension() {
        let enc = FlyHashEncoder::new(9, 81, 9, 3, 1).unwrap();
        let err = enc.encode(Input::Dense(&[0.0; 3]), &mut Vec::new()).unwrap_err();
        assert_eq!(err, MemoryError::DimensionMismatch { expected: 9, got: 3 });
    }

    #[test]
    fn ngram_fragment_code_is_subset_of_sequence_code() {
        let enc = NGramEncoder::new(177_147, &[1, 2, 3], 1, 9).unwrap();
        let seq: Vec<u32> = (100..127).collect();
        let full = encode(&enc, Input::Tokens(&seq));
        let part = encode(&enc, Input::Tokens(&seq[9..18]));
        assert!(full.len() <= enc.max_code_len(seq.len()));
        assert_eq!(overlap(&part, &full), part.len());

        let mut reordered = seq[9..18].to_vec();
        reordered.reverse();
        let rev = encode(&enc, Input::Tokens(&reordered));
        // Unigrams still match, but bigrams/trigrams encode order.
        assert!(overlap(&rev, &full) < rev.len());
    }
}
