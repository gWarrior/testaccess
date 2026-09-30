//! Exact softmax cross-attention over a sparse set of token positions.
//!
//! This is the "fine" half of the read path: the SNN memory picks a few
//! chunks out of the whole context (coarse, sub-linear), and attention runs
//! only over the tokens of those chunks. With `top_k` chunks of `c` tokens
//! the cost is `O(top_k · c · d)` per query, independent of context length.

use crate::kv::KvStore;

/// Result of attending over retrieved positions.
#[derive(Clone, Debug, Default)]
pub struct Attended {
    /// `Σ softmax(q·k / √d) · v`, zero if nothing was attended.
    pub output: Vec<f32>,
    /// Number of attended token positions.
    pub tokens: usize,
    /// Position and weight of the most attended token.
    pub argmax: Option<(u64, f32)>,
    /// Attention entropy in nats (low = sharp, confident read).
    pub entropy: f32,
}

/// Attend with `query` (key_dim) over every position in `spans`
/// (half-open `[start, end)` ranges) that is present in `kv`.
pub fn attend(query: &[f32], kv: &KvStore, spans: &[(u64, u64)]) -> Attended {
    let (dk, dv) = (kv.key_dim(), kv.value_dim());
    assert_eq!(query.len(), dk, "query must have key_dim elements");
    let scale = 1.0 / (dk as f32).sqrt();
    let (mut key, mut value) = (vec![0f32; dk], vec![0f32; dv]);

    let mut positions = Vec::new();
    let mut scores = Vec::new();
    let mut values: Vec<f32> = Vec::new();
    for &(start, end) in spans {
        for pos in start..end {
            if kv.read(pos, &mut key, &mut value) {
                positions.push(pos);
                scores.push(query.iter().zip(&key).map(|(q, k)| q * k).sum::<f32>() * scale);
                values.extend_from_slice(&value);
            }
        }
    }
    if scores.is_empty() {
        return Attended { output: vec![0.0; dv], ..Default::default() };
    }

    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for s in &mut scores {
        *s = (*s - max).exp();
        sum += *s;
    }
    for s in &mut scores {
        *s /= sum;
    }
    let mut output = vec![0f32; dv];
    for (w, v) in scores.iter().zip(values.chunks_exact(dv)) {
        for (o, x) in output.iter_mut().zip(v) {
            *o += w * x;
        }
    }
    let entropy = -scores.iter().filter(|&&w| w > 0.0).map(|w| w * w.ln()).sum::<f32>();
    let best = (0..scores.len()).max_by(|&a, &b| scores[a].total_cmp(&scores[b])).unwrap_or(0);
    Attended { output, tokens: positions.len(), argmax: Some((positions[best], scores[best])), entropy }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::KvPrecision;

    #[test]
    fn attends_to_the_matching_key() {
        let (dk, dv) = (9, 3);
        let mut kv = KvStore::new(dk, dv, KvPrecision::F32);
        // Token p has key e_(p mod 9) * 9 and value [p, 0, 0].
        for p in 0..27u64 {
            let mut k = vec![0.0; dk];
            k[(p % 9) as usize] = 9.0;
            kv.push(&k, &[p as f32, 0.0, 0.0]);
        }
        let mut q = vec![0.0; dk];
        q[4] = 3.0;
        // Only span [9, 18) is attended: its token 13 matches.
        let a = attend(&q, &kv, &[(9, 18)]);
        assert_eq!(a.tokens, 9);
        let (pos, w) = a.argmax.unwrap();
        assert_eq!(pos, 13);
        assert!(w > 0.99);
        assert!((a.output[0] - 13.0).abs() < 0.1);
        assert!(a.entropy < 0.1);
    }

    #[test]
    fn empty_spans_give_zero_output() {
        let kv = KvStore::new(3, 2, KvPrecision::F16);
        let a = attend(&[1.0, 0.0, 0.0], &kv, &[(0, 9)]);
        assert_eq!((a.output, a.tokens, a.argmax), (vec![0.0, 0.0], 0, None));
    }
}
