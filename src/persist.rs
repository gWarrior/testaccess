//! Snapshots: carry memories across sessions.
//!
//! Format (little endian), version 1:
//!
//! ```text
//! magic "SNNM" | version u32 | encoder fingerprint u64 | next_id u64
//! contexts: count u32, then (len u32, utf-8 bytes)*
//! memories: count u64, then per memory:
//!   id u64 | context u32 | tier u8 | pinned u8 | polarity i8
//!   age f64 | remaining ttl f64 (inf = none) | strength f32 | recalls u32
//!   prev u64 | next u64
//!   synapses u16 | neuron trytes u16* | state trits: words u16, u64*
//!   payload: len u32, bytes
//! checksum u64 (FNV-1a of everything before it)
//! ```
//!
//! Neuron ids are stored as trytes; synapse states as effective trits
//! densely packed 40 per 64-bit word. Times are stored relative to the
//! moment of saving (age, remaining TTL), so a snapshot can be restored
//! under a different clock. Working memory is short-term by nature and is
//! not saved.

use crate::types::MemoryError;

pub(crate) const MAGIC: &[u8; 4] = b"SNNM";
pub(crate) const VERSION: u32 = 1;

/// Which memories a snapshot contains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotScope {
    /// Every live memory.
    All,
    /// Only consolidated / pinned memories ("what matters").
    LongTerm,
}

/// Payloads that can be written into a snapshot.
pub trait Persist: Sized {
    fn write(&self, out: &mut Vec<u8>);
    fn read(input: &mut &[u8]) -> Result<Self, MemoryError>;
}

pub(crate) fn corrupt(what: &str) -> MemoryError {
    MemoryError::Corrupt(what.to_string())
}

/// Little-endian reader over a byte slice.
pub(crate) fn take<'a>(input: &mut &'a [u8], n: usize) -> Result<&'a [u8], MemoryError> {
    if input.len() < n {
        return Err(corrupt("unexpected end of snapshot"));
    }
    let (head, rest) = input.split_at(n);
    *input = rest;
    Ok(head)
}

macro_rules! num {
    ($read:ident, $t:ty) => {
        pub(crate) fn $read(input: &mut &[u8]) -> Result<$t, MemoryError> {
            let b = take(input, std::mem::size_of::<$t>())?;
            Ok(<$t>::from_le_bytes(b.try_into().expect("length checked")))
        }
    };
}
num!(read_u8, u8);
num!(read_i8, i8);
num!(read_u16, u16);
num!(read_u32, u32);
num!(read_u64, u64);
num!(read_f32, f32);
num!(read_f64, f64);

pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xCBF2_9CE4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0100_0000_01B3))
}

/// Fast 64-bit checksum of full snapshots (four independent multiply-xor
/// lanes over 64-bit words, then a SplitMix64 finish). Any change of a
/// single word is always detected; it runs at memory speed, unlike the
/// byte-serial FNV-1a used by the (small) long-term snapshots.
pub(crate) fn checksum64(bytes: &[u8]) -> u64 {
    const P: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut h = [0x243F_6A88_85A3_08D3u64, 0x1319_8A2E_0370_7344, 0xA409_3822_299F_31D0, 0x082E_FA98_EC4E_6C89];
    let mut chunks = bytes.chunks_exact(32);
    for c in &mut chunks {
        for (i, lane) in h.iter_mut().enumerate() {
            let w = u64::from_le_bytes(c[8 * i..8 * i + 8].try_into().expect("8 bytes"));
            *lane = (*lane ^ w).wrapping_mul(P).rotate_left(29);
        }
    }
    let mut acc = crate::rng::mix64(bytes.len() as u64);
    for (i, &x) in h.iter().enumerate() {
        acc = crate::rng::mix64(acc ^ x.wrapping_add(i as u64));
    }
    for &b in chunks.remainder() {
        acc = crate::rng::mix64(acc ^ b as u64);
    }
    acc
}

/// Check the trailing [`checksum64`], magic and version of a full
/// snapshot; returns the bytes after the version.
pub(crate) fn checked_body<'a>(bytes: &'a [u8], magic: &[u8; 4], version: u32) -> Result<&'a [u8], MemoryError> {
    if bytes.len() < 16 {
        return Err(corrupt("snapshot too short"));
    }
    let (body, sum) = bytes.split_at(bytes.len() - 8);
    if checksum64(body).to_le_bytes() != sum {
        return Err(corrupt("checksum mismatch"));
    }
    let mut input = body;
    if take(&mut input, 4)? != magic {
        return Err(corrupt("wrong snapshot kind"));
    }
    let v = read_u32(&mut input)?;
    if v != version {
        return Err(corrupt(&format!("unsupported snapshot version {v}")));
    }
    Ok(input)
}

/// Bulk little-endian writers and readers for full snapshots.
pub(crate) fn put_u16s(out: &mut Vec<u8>, xs: &[u16]) {
    out.reserve(xs.len() * 2);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

pub(crate) fn put_u32s(out: &mut Vec<u8>, xs: &[u32]) {
    out.reserve(xs.len() * 4);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

pub(crate) fn put_f32s(out: &mut Vec<u8>, xs: &[f32]) {
    out.reserve(xs.len() * 4);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// `n` items of `size` bytes, checked against the remaining input before
/// anything is allocated.
fn take_items<'a>(input: &mut &'a [u8], n: usize, size: usize) -> Result<&'a [u8], MemoryError> {
    let bytes = n.checked_mul(size).ok_or_else(|| corrupt("length overflow"))?;
    take(input, bytes)
}

pub(crate) fn read_u16s(input: &mut &[u8], n: usize) -> Result<Vec<u16>, MemoryError> {
    let b = take_items(input, n, 2)?;
    Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

pub(crate) fn read_u32s(input: &mut &[u8], n: usize) -> Result<Vec<u32>, MemoryError> {
    let b = take_items(input, n, 4)?;
    Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes"))).collect())
}

pub(crate) fn read_u64s(input: &mut &[u8], n: usize) -> Result<Vec<u64>, MemoryError> {
    let b = take_items(input, n, 8)?;
    Ok(b.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes"))).collect())
}

pub(crate) fn read_f32s(input: &mut &[u8], n: usize) -> Result<Vec<f32>, MemoryError> {
    let b = take_items(input, n, 4)?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).collect())
}

/// Start a length-prefixed section; finish it with [`end_section`].
pub(crate) fn begin_section(out: &mut Vec<u8>) -> usize {
    out.extend_from_slice(&0u64.to_le_bytes());
    out.len()
}

pub(crate) fn end_section(out: &mut [u8], start: usize) {
    let len = (out.len() - start) as u64;
    out[start - 8..start].copy_from_slice(&len.to_le_bytes());
}

/// The body of a length-prefixed section.
pub(crate) fn section<'a>(input: &mut &'a [u8]) -> Result<&'a [u8], MemoryError> {
    let n = read_u64(input)?;
    let n = usize::try_from(n).map_err(|_| corrupt("section too long"))?;
    take(input, n)
}

impl Persist for () {
    fn write(&self, _: &mut Vec<u8>) {}
    fn read(_: &mut &[u8]) -> Result<Self, MemoryError> {
        Ok(())
    }
}

impl Persist for u32 {
    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
    fn read(input: &mut &[u8]) -> Result<Self, MemoryError> {
        read_u32(input)
    }
}

impl Persist for u64 {
    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
    fn read(input: &mut &[u8]) -> Result<Self, MemoryError> {
        read_u64(input)
    }
}

impl Persist for String {
    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.len() as u32).to_le_bytes());
        out.extend_from_slice(self.as_bytes());
    }
    fn read(input: &mut &[u8]) -> Result<Self, MemoryError> {
        let n = read_u32(input)? as usize;
        String::from_utf8(take(input, n)?.to_vec()).map_err(|_| corrupt("invalid utf-8"))
    }
}

impl Persist for Vec<u32> {
    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.len() as u32).to_le_bytes());
        for x in self {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    fn read(input: &mut &[u8]) -> Result<Self, MemoryError> {
        let n = read_u32(input)? as usize;
        (0..n).map(|_| read_u32(input)).collect()
    }
}

impl Persist for Vec<u8> {
    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.len() as u32).to_le_bytes());
        out.extend_from_slice(self);
    }
    fn read(input: &mut &[u8]) -> Result<Self, MemoryError> {
        let n = read_u32(input)? as usize;
        Ok(take(input, n)?.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_payloads_roundtrip() {
        let mut buf = Vec::new();
        7u32.write(&mut buf);
        "трайт".to_string().write(&mut buf);
        vec![1u32, 3, 9].write(&mut buf);
        let mut input = &buf[..];
        assert_eq!(u32::read(&mut input).unwrap(), 7);
        assert_eq!(String::read(&mut input).unwrap(), "трайт");
        assert_eq!(Vec::<u32>::read(&mut input).unwrap(), vec![1, 3, 9]);
        assert!(input.is_empty());
        assert!(u32::read(&mut input).is_err());
    }
}
