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
//! | `Trit2`   | 3.2 (+ exp)    | 0.25 GB                     |
//!
//! `Ternary` uses absmean quantization (as in BitNet b1.58): each row is
//! stored as trits `round(clamp(x / mean|x|, -1, 1))` plus one `f32` scale.
//! `Trit2` stores two balanced trits per element (levels −4…4) and one
//! power-of-two step per row. A row that already lies on such a grid (a
//! model's two-trit keys) is stored exactly.

use crate::persist::{self, corrupt};
use crate::trit::TritVec;
use crate::types::MemoryError;

/// Storage precision of keys/values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KvPrecision {
    F32,
    #[default]
    F16,
    Ternary,
    Trit2,
}

impl KvPrecision {
    pub(crate) fn tag(self) -> u8 {
        match self {
            KvPrecision::F32 => 0,
            KvPrecision::F16 => 1,
            KvPrecision::Ternary => 2,
            KvPrecision::Trit2 => 3,
        }
    }
}

/// Power-of-two step exponent for a [`KvPrecision::Trit2`] row: the finest
/// step that keeps every element within ±4 if the row lies on that grid
/// (exact), else the MSE-optimal step `2^round(log2(0.669·mean|x|))`.
pub fn trit2_exponent(row: &[f32]) -> i32 {
    let max = row.iter().fold(0f32, |m, x| m.max(x.abs()));
    if max == 0.0 || !max.is_finite() {
        return 0;
    }
    let e = (max as f64 / 4.0).log2().ceil() as i32;
    let step = 2f64.powi(e);
    if row.iter().all(|&x| ((x as f64) / step).fract() == 0.0) {
        return e;
    }
    let mean = row.iter().map(|x| x.abs() as f64).sum::<f64>() / row.len() as f64;
    (mean * 0.669).max(1e-30).log2().round() as i32
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
    Ternary {
        trits: TritVec,
        scales: Vec<f32>,
    },
    /// Two trits per element (`L = 3·t₁ + t₀`), a step exponent per row.
    Trit2 {
        trits: TritVec,
        exps: Vec<i8>,
    },
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
            KvPrecision::Trit2 => Rows::Trit2 { trits: TritVec::new(), exps: Vec::new() },
        };
        Self { dim, rows, len: 0 }
    }

    fn push(&mut self, row: &[f32]) {
        debug_assert_eq!(row.len(), self.dim);
        match &mut self.rows {
            Rows::F32(v) => v.extend_from_slice(row),
            // Clamp to the f16 range so large values saturate instead of
            // becoming infinities.
            Rows::F16(v) => v.extend(row.iter().map(|&x| f32_to_f16(x.clamp(-65_504.0, 65_504.0)))),
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
            Rows::Trit2 { trits, exps } => {
                let e = trit2_exponent(row).clamp(-127, 127);
                let step = 2f32.powi(e);
                let start = trits.len();
                trits.extend_zeros(2 * self.dim);
                for (i, &x) in row.iter().enumerate() {
                    let l = (x / step).round().clamp(-4.0, 4.0) as i32;
                    let t0 = (l + 1).rem_euclid(3) - 1;
                    trits.set(start + 2 * i, t0 as i8);
                    trits.set(start + 2 * i + 1, ((l - t0) / 3) as i8);
                }
                exps.push(e as i8);
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
            Rows::Trit2 { trits, exps } => {
                let step = 2f32.powi(exps[r] as i32);
                let b = 2 * s;
                out.iter_mut().enumerate().for_each(|(i, o)| {
                    *o = (3 * trits.get(b + 2 * i + 1) + trits.get(b + 2 * i)) as f32 * step;
                });
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
            Rows::Trit2 { trits, exps } => {
                *trits = trits.slice(2 * cut, trits.len());
                exps.drain(..n);
            }
        }
        self.len -= n;
    }

    fn clear(&mut self) {
        self.drain_front(self.len);
    }

    /// Rows as stored: raw `f32`/`f16` words, or packed trit bytes plus
    /// per-row scales / exponents.
    fn write_image(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.len as u64).to_le_bytes());
        match &self.rows {
            Rows::F32(v) => persist::put_f32s(out, v),
            Rows::F16(v) => persist::put_u16s(out, v),
            Rows::Ternary { trits, scales } => {
                out.extend_from_slice(trits.as_bytes());
                persist::put_f32s(out, scales);
            }
            Rows::Trit2 { trits, exps } => {
                out.extend_from_slice(trits.as_bytes());
                out.extend(exps.iter().map(|&e| e as u8));
            }
        }
    }

    fn read_image(dim: usize, precision: KvPrecision, input: &mut &[u8]) -> Result<Self, MemoryError> {
        let len = usize::try_from(persist::read_u64(input)?).map_err(|_| corrupt("bad row count"))?;
        let n = len.checked_mul(dim).filter(|&n| n / 8 <= input.len()).ok_or_else(|| corrupt("bad row count"))?;
        let trits = |input: &mut &[u8], count: usize| {
            TritVec::from_bytes(persist::take(input, count.div_ceil(5))?, count).ok_or_else(|| corrupt("bad K/V trits"))
        };
        let rows = match precision {
            KvPrecision::F32 => Rows::F32(persist::read_f32s(input, n)?),
            KvPrecision::F16 => Rows::F16(persist::read_u16s(input, n)?),
            KvPrecision::Ternary => Rows::Ternary { trits: trits(input, n)?, scales: persist::read_f32s(input, len)? },
            KvPrecision::Trit2 => {
                let trits = trits(input, 2 * n)?;
                let exps = persist::take(input, len)?.iter().map(|&e| e as i8).collect();
                Rows::Trit2 { trits, exps }
            }
        };
        Ok(Self { dim, rows, len })
    }

    fn bytes(&self) -> usize {
        match &self.rows {
            Rows::F32(v) => v.capacity() * 4,
            Rows::F16(v) => v.capacity() * 2,
            Rows::Ternary { trits, scales } => trits.bytes() + scales.capacity() * 4,
            Rows::Trit2 { trits, exps } => trits.bytes() + exps.capacity(),
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
        Self { keys: RowStore::new(key_dim, precision), values: RowStore::new(value_dim, precision), base: 0 }
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

    pub fn precision(&self) -> KvPrecision {
        match self.keys.rows {
            Rows::F32(_) => KvPrecision::F32,
            Rows::F16(_) => KvPrecision::F16,
            Rows::Ternary { .. } => KvPrecision::Ternary,
            Rows::Trit2 { .. } => KvPrecision::Trit2,
        }
    }

    /// Exact image: `key_dim u32 | value_dim u32 | precision u8 | base u64`,
    /// then keys and values as `rows u64` + the stored rows.
    pub(crate) fn write_image(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.keys.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.values.dim as u32).to_le_bytes());
        out.push(self.precision().tag());
        out.extend_from_slice(&self.base.to_le_bytes());
        self.keys.write_image(out);
        self.values.write_image(out);
    }

    /// Inverse of [`write_image`](Self::write_image); the dimensions and the
    /// precision must be the configured ones.
    pub(crate) fn read_image(
        key_dim: usize,
        value_dim: usize,
        precision: KvPrecision,
        input: &mut &[u8],
    ) -> Result<Self, MemoryError> {
        let (dk, dv) = (persist::read_u32(input)? as usize, persist::read_u32(input)? as usize);
        let tag = persist::read_u8(input)?;
        if (dk, dv) != (key_dim, value_dim) || tag != precision.tag() {
            return Err(MemoryError::InvalidConfig(format!(
                "snapshot K/V is {dk}x{dv} (precision #{tag}), config is {key_dim}x{value_dim} ({precision:?})"
            )));
        }
        let base = persist::read_u64(input)?;
        let keys = RowStore::read_image(dk, precision, input)?;
        let values = RowStore::read_image(dv, precision, input)?;
        if keys.len != values.len || base.checked_add(keys.len as u64).is_none() {
            return Err(corrupt("keys and values differ in length"));
        }
        Ok(Self { keys, values, base })
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
        assert!(roundtrip(KvPrecision::Trit2) > 0.98);
    }

    #[test]
    fn trit2_stores_two_trit_rows_exactly() {
        let mut rng = SplitMix64::new(3);
        let mut kv = KvStore::new(81, 81, KvPrecision::Trit2);
        let mut rows = Vec::new();
        for e in [-9i32, -3, 0, 4] {
            let step = 2f32.powi(e);
            let row: Vec<f32> = (0..81).map(|_| (rng.below(9) as i32 - 4) as f32 * step).collect();
            kv.push(&row, &row);
            rows.push(row);
        }
        // Rows using only small levels are exact too.
        let small: Vec<f32> = (0..81).map(|i| [-1.0f32, 0.0, 1.0][i % 3] * 0.25).collect();
        kv.push(&small, &small);
        rows.push(small);
        let (mut k, mut v) = (vec![0.0; 81], vec![0.0; 81]);
        for (p, row) in rows.iter().enumerate() {
            assert!(kv.read(p as u64, &mut k, &mut v));
            assert_eq!(&k, row);
            assert_eq!(&v, row);
        }
    }
}
