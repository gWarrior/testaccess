//! Small deterministic RNG and hashing helpers (no external dependencies).
//!
//! Determinism matters: the encoder's random connectivity must be identical
//! between the moment a memory is written and the moment it is queried.

/// SplitMix64 finalizer: a fast, well-mixing 64-bit hash.
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// SplitMix64 pseudo-random generator.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.state)
    }

    /// Uniform integer in `0..n` (`n > 0`).
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }

    /// Uniform float in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal sample (Box–Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// `k` distinct values from `0..n`, in random order (`k <= n`).
    pub fn sample_distinct(&mut self, n: u32, k: usize) -> Vec<u32> {
        assert!(k <= n as usize, "cannot sample {k} distinct values from {n}");
        if k * 3 >= n as usize {
            let mut all: Vec<u32> = (0..n).collect();
            for i in 0..k {
                let j = i + self.below((n as usize - i) as u64) as usize;
                all.swap(i, j);
            }
            all.truncate(k);
            all
        } else {
            let mut out = Vec::with_capacity(k);
            while out.len() < k {
                let v = self.below(n as u64) as u32;
                if !out.contains(&v) {
                    out.push(v);
                }
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
            assert!(a.below(10) < 10);
            b.below(10);
        }
    }

    #[test]
    fn sample_distinct_has_no_duplicates() {
        let mut r = SplitMix64::new(1);
        for (n, k) in [(10, 10), (1000, 20), (100, 60)] {
            let mut s = r.sample_distinct(n, k);
            s.sort_unstable();
            s.dedup();
            assert_eq!(s.len(), k);
            assert!(s.iter().all(|&v| v < n));
        }
    }
}
