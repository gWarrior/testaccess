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
