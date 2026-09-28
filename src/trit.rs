//! Ternary storage primitives.
//!
//! * A **trit** takes one of three values: `-1`, `0`, `+1` (balanced ternary).
//! * A **tryte** is 9 trits (3^9 = 19 683 states). It fits in a `u16` and is
//!   the address unit of the network: a neuron id is exactly one tryte.
//! * [`TritVec`] packs trits densely, 5 per byte (3^5 = 243 ≤ 256), i.e. 40
//!   trits per 64-bit word (3^40 ≈ 1.216·10^19 < 2^64): 1.6 bits per trit,
//!   99% of the information-theoretic density. Reading a trit is a table
//!   lookup, so there is no division by powers of three on the hot path.

/// Trits per tryte.
pub const TRITS_PER_TRYTE: usize = 9;
/// Number of distinct tryte values, 3^9.
pub const TRYTE_STATES: u32 = 19_683;
/// Trits packed into one byte.
pub const TRITS_PER_BYTE: usize = 5;
/// Trits packed into one 64-bit word.
pub const TRITS_PER_WORD: usize = 40;

const POW3: [u8; TRITS_PER_BYTE] = [1, 3, 9, 27, 81];

/// Decoding table: `DECODE[byte][i]` is trit `i` of a packed byte.
static DECODE: [[i8; TRITS_PER_BYTE]; 243] = {
    let mut t = [[0i8; TRITS_PER_BYTE]; 243];
    let mut b = 0;
    while b < 243 {
        let mut v = b;
        let mut i = 0;
        while i < TRITS_PER_BYTE {
            t[b][i] = (v % 3) as i8 - 1;
            v /= 3;
            i += 1;
        }
        b += 1;
    }
    t
};

/// The byte whose five trits are all `0`.
const ZERO_BYTE: u8 = 1 + 3 + 9 + 27 + 81;

/// Nine trits stored as an unsigned value in `0..3^9`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tryte(u16);

impl Tryte {
    pub const MIN: Tryte = Tryte(0);
    pub const MAX: Tryte = Tryte(TRYTE_STATES as u16 - 1);

    /// Tryte with unsigned value `v` (`v < 3^9`).
    pub fn new(v: u32) -> Option<Self> {
        (v < TRYTE_STATES).then_some(Self(v as u16))
    }

    /// Unsigned value in `0..3^9`.
    pub fn value(self) -> u32 {
        self.0 as u32
    }

    /// Balanced value in `-9841..=9841`.
    pub fn balanced(self) -> i32 {
        self.0 as i32 - (TRYTE_STATES as i32 - 1) / 2
    }

    /// Build from nine balanced trits, least significant first.
    pub fn from_trits(trits: [i8; TRITS_PER_TRYTE]) -> Self {
        let v = trits.iter().rev().fold(0u32, |acc, &t| acc * 3 + (t.clamp(-1, 1) + 1) as u32);
        Self(v as u16)
    }

    /// The nine balanced trits, least significant first.
    pub fn trits(self) -> [i8; TRITS_PER_TRYTE] {
        let mut v = self.0 as u32;
        let mut out = [0i8; TRITS_PER_TRYTE];
        for t in &mut out {
            *t = (v % 3) as i8 - 1;
            v /= 3;
        }
        out
    }
}

/// Densely packed vector of trits (5 per byte, 40 per 64-bit word).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TritVec {
    bytes: Vec<u8>,
    len: usize,
}

impl TritVec {
    pub fn new() -> Self {
        Self::default()
    }

    /// `len` zero trits.
    pub fn zeros(len: usize) -> Self {
        Self { bytes: vec![ZERO_BYTE; len.div_ceil(TRITS_PER_BYTE)], len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append `n` zero trits.
    pub fn extend_zeros(&mut self, n: usize) {
        self.len += n;
        self.bytes.resize(self.len.div_ceil(TRITS_PER_BYTE), ZERO_BYTE);
    }

    #[inline]
    pub fn get(&self, i: usize) -> i8 {
        debug_assert!(i < self.len);
        DECODE[self.bytes[i / TRITS_PER_BYTE] as usize][i % TRITS_PER_BYTE]
    }

    #[inline]
    pub fn set(&mut self, i: usize, t: i8) {
        debug_assert!(i < self.len && (-1..=1).contains(&t));
        let (b, k) = (i / TRITS_PER_BYTE, i % TRITS_PER_BYTE);
        let old = DECODE[self.bytes[b] as usize][k];
        let byte = self.bytes[b] as i16 + (t - old) as i16 * POW3[k] as i16;
        self.bytes[b] = byte as u8;
    }

    /// Set trits `start..end` to zero.
    pub fn clear_range(&mut self, start: usize, end: usize) {
        for i in start..end {
            self.set(i, 0);
        }
    }

    /// The packed storage as 64-bit words of 40 trits each.
    pub fn words(&self) -> Vec<u64> {
        self.bytes
            .chunks(8)
            .map(|c| c.iter().rev().fold(0u64, |acc, &b| acc * 243 + b as u64))
            .collect()
    }

    /// Rebuild from 64-bit words of 40 trits each.
    pub fn from_words(words: &[u64], len: usize) -> Option<Self> {
        let mut bytes = Vec::with_capacity(words.len() * 8);
        for &w in words {
            let mut v = w;
            for _ in 0..8 {
                bytes.push((v % 243) as u8);
                v /= 243;
            }
            if v != 0 {
                return None;
            }
        }
        let need = len.div_ceil(TRITS_PER_BYTE);
        if bytes.len() < need {
            return None;
        }
        bytes.truncate(need);
        Some(Self { bytes, len })
    }

    /// Heap bytes used.
    pub fn bytes(&self) -> usize {
        self.bytes.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    #[test]
    fn forty_trits_fit_in_a_word() {
        assert!(3u128.pow(TRITS_PER_WORD as u32) <= u64::MAX as u128 + 1);
        assert!(3u128.pow(TRITS_PER_WORD as u32 + 1) > u64::MAX as u128 + 1);
        assert_eq!(3u32.pow(TRITS_PER_TRYTE as u32), TRYTE_STATES);
        assert!(TRYTE_STATES <= u16::MAX as u32 + 1);
    }

    #[test]
    fn tryte_roundtrip() {
        for v in [0, 1, 2, 3, 9840, 9841, 19_682] {
            let t = Tryte::new(v).unwrap();
            assert_eq!(Tryte::from_trits(t.trits()), t);
        }
        assert_eq!(Tryte::new(19_683), None);
        assert_eq!(Tryte::from_trits([0; 9]).balanced(), 0);
        assert_eq!(Tryte::MAX.balanced(), 9841);
        assert_eq!(Tryte::from_trits([1, 0, 0, 0, 0, 0, 0, 0, 0]).balanced(), 1);
    }

    #[test]
    fn tritvec_get_set_and_words() {
        let mut rng = SplitMix64::new(3);
        let n = 1000;
        let mut v = TritVec::zeros(n);
        let mut model = vec![0i8; n];
        for _ in 0..5000 {
            let i = rng.below(n as u64) as usize;
            let t = rng.below(3) as i8 - 1;
            v.set(i, t);
            model[i] = t;
        }
        assert!((0..n).all(|i| v.get(i) == model[i]));
        assert_eq!(v.bytes(), n.div_ceil(5));

        let words = v.words();
        assert_eq!(words.len(), n.div_ceil(TRITS_PER_WORD));
        assert_eq!(TritVec::from_words(&words, n).unwrap(), v);

        v.extend_zeros(7);
        assert_eq!(v.get(n + 6), 0);
        v.clear_range(0, n);
        assert!((0..n + 7).all(|i| v.get(i) == 0));
    }
}
