//! Per-token key/value store for the cross-attention read head.
//!
//! The SNN index only decides *which* context chunks are relevant; exact
//! attention then runs over the keys/values of the tokens in those chunks.
//! Rows are stored with a configurable precision:
//!
//! | precision | bits / element | 300k tokens × (1024 + 1024) |
//! |-----------|---------------:|----------------------------:|
//! | `F32`     | 32             | 2.46 GB                     |
//! | `F16`     | 16             | 1.23 GB                     |
//! | `Ternary` | 1.6 (+ scale)  | 0.13 GB                     |
//!
//! `Ternary` uses absmean quantization (as in BitNet b1.58): each row is
//! stored as trits `round(clamp(x / mean|x|, -1, 1))` plus one `f32` scale.

use crate::trit::TritVec;

/// Storage precision of keys/values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KvPrecision {
    F32,
    #[default]
    F16,
    Ternary,
}

/// IEEE 754 binary16 from `f32`, round to nearest even.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1 << shift) - 1);
        let mut v = m >> shift;
        if rest > half || (rest == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1FFF;
    if rest > 0x1000 || (rest == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    sign | v as u16
}

/// `f32` from IEEE 754 binary16.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            let shift = m.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((m << shift) & 0x3FF) << 13
        }
        (0x1F, m) => sign | 0x7F80_0000 | (m << 13),
        (e, m) => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

enum Rows {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Ternary { trits: TritVec, scales: Vec<f32> },
}

/// Append-only matrix of `dim`-wide rows with front eviction.
struct RowStore {
    dim: usize,
    rows: Rows,
    len: usize,
}

impl RowStore {
    fn new(dim: usize, precision: KvPrecision) -> Self {
        let rows = match precision {
            KvPrecision::F32 => Rows::F32(Vec::new()),
            KvPrecision::F16 => Rows::F16(Vec::new()),
            KvPrecision::Ternary => Rows::Ternary { trits: TritVec::new(), scales: Vec::new() },
        };
        Self { dim, rows, len: 0 }
    }

    fn push(&mut self, row: &[f32]) {
        debug_assert_eq!(row.len(), self.dim);
        match &mut self.rows {
            Rows::F32(v) => v.extend_from_slice(row),
            Rows::F16(v) => v.extend(row.iter().map(|&x| f32_to_f16(x))),
            Rows::Ternary { trits, scales } => {
                let scale = row.iter().map(|x| x.abs()).sum::<f32>() / self.dim as f32;
                let start = trits.len();
                trits.extend_zeros(self.dim);
                if scale > 0.0 {
                    for (i, &x) in row.iter().enumerate() {
                        trits.set(start + i, (x / scale).round().clamp(-1.0, 1.0) as i8);
                    }
                }
                scales.push(scale);
            }
        }
        self.len += 1;
    }

    fn read(&self, r: usize, out: &mut [f32]) {
        let (s, d) = (r * self.dim, self.dim);
        match &self.rows {
            Rows::F32(v) => out.copy_from_slice(&v[s..s + d]),
            Rows::F16(v) => out.iter_mut().zip(&v[s..s + d]).for_each(|(o, &h)| *o = f16_to_f32(h)),
            Rows::Ternary { trits, scales } => {
                let scale = scales[r];
                out.iter_mut().enumerate().for_each(|(i, o)| *o = trits.get(s + i) as f32 * scale);
            }
        }
    }

    fn drain_front(&mut self, n: usize) {
        let n = n.min(self.len);
        let cut = n * self.dim;
        match &mut self.rows {
            Rows::F32(v) => drop(v.drain(..cut)),
            Rows::F16(v) => drop(v.drain(..cut)),
            Rows::Ternary { trits, scales } => {
                *trits = trits.slice(cut, trits.len());
                scales.drain(..n);
            }
        }
        self.len -= n;
    }

    fn clear(&mut self) {
        self.drain_front(self.len);
    }

    fn bytes(&self) -> usize {
        match &self.rows {
            Rows::F32(v) => v.capacity() * 4,
            Rows::F16(v) => v.capacity() * 2,
            Rows::Ternary { trits, scales } => trits.bytes() + scales.capacity() * 4,
        }
    }
}

/// Keys and values of consecutive token positions `[base, base + len)`.
pub struct KvStore {
    keys: RowStore,
    values: RowStore,
    base: u64,
}

impl KvStore {
    pub fn new(key_dim: usize, value_dim: usize, precision: KvPrecision) -> Self {
        Self {
            keys: RowStore::new(key_dim, precision),
            values: RowStore::new(value_dim, precision),
            base: 0,
        }
    }

    pub fn key_dim(&self) -> usize {
        self.keys.dim
    }

    pub fn value_dim(&self) -> usize {
        self.values.dim
    }

    /// First stored position.
    pub fn base(&self) -> u64 {
        self.base
    }

    /// One past the last stored position.
    pub fn end(&self) -> u64 {
        self.base + self.keys.len as u64
    }

    /// Append `n` tokens: `keys` is `n × key_dim`, `values` is `n × value_dim`.
    pub fn push(&mut self, keys: &[f32], values: &[f32]) {
        for (k, v) in keys.chunks_exact(self.keys.dim).zip(values.chunks_exact(self.values.dim)) {
            self.keys.push(k);
            self.values.push(v);
        }
    }

    /// Read the key and value of absolute position `pos`.
    pub fn read(&self, pos: u64, key: &mut [f32], value: &mut [f32]) -> bool {
        if pos < self.base || pos >= self.end() {
            return false;
        }
        let r = (pos - self.base) as usize;
        self.keys.read(r, key);
        self.values.read(r, value);
        true
    }

    /// Drop every position before `pos`.
    pub fn evict_before(&mut self, pos: u64) {
        if pos > self.base {
            let n = (pos.min(self.end()) - self.base) as usize;
            self.keys.drain_front(n);
            self.values.drain_front(n);
            self.base += n as u64;
        }
    }

    /// Drop everything and restart at position `base`.
    pub fn reset(&mut self, base: u64) {
        self.keys.clear();
        self.values.clear();
        self.base = base;
    }

    pub fn bytes(&self) -> usize {
        self.keys.bytes() + self.values.bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    #[test]
    fn f16_roundtrip() {
        for x in [0.0f32, -0.0, 1.0, -2.5, 0.333_333, 65504.0, 6.1e-5, 1e-7, 3.0e-6] {
            let y = f16_to_f32(f32_to_f16(x));
            let tol = (x.abs() * 1e-3).max(6e-8);
            assert!((x - y).abs() <= tol, "{x} -> {y}");
        }
        assert!(f16_to_f32(f32_to_f16(1e6)).is_infinite());
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    fn roundtrip(precision: KvPrecision) -> f32 {
        let mut rng = SplitMix64::new(1);
        let (dk, dv, n) = (27, 9, 81);
        let keys: Vec<f32> = (0..n * dk).map(|_| rng.normal() as f32).collect();
        let values: Vec<f32> = (0..n * dv).map(|_| rng.normal() as f32).collect();
        let mut kv = KvStore::new(dk, dv, precision);
        kv.push(&keys, &values);
        kv.evict_before(9);
        assert_eq!((kv.base(), kv.end()), (9, 81));
        let (mut k, mut v) = (vec![0.0; dk], vec![0.0; dv]);
        assert!(!kv.read(8, &mut k, &mut v));
        let mut cos = 0.0;
        for p in 9..81 {
            assert!(kv.read(p, &mut k, &mut v));
            let orig = &keys[p as usize * dk..(p as usize + 1) * dk];
            let dot: f32 = k.iter().zip(orig).map(|(a, b)| a * b).sum();
            let norm = |x: &[f32]| x.iter().map(|a| a * a).sum::<f32>().sqrt();
            cos += dot / (norm(&k) * norm(orig));
        }
        cos / 72.0
    }

    #[test]
    fn precisions_preserve_direction() {
        assert!(roundtrip(KvPrecision::F32) > 0.999_99);
        assert!(roundtrip(KvPrecision::F16) > 0.9999);
        // Ternary absmean keeps the direction of Gaussian rows well enough
        // for attention scoring.
        assert!(roundtrip(KvPrecision::Ternary) > 0.8);
    }
}
