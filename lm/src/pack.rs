//! Packed ternary model file (`.snnt`).
//!
//! Every weight matrix is stored as two balanced trits per weight — the
//! level `L ∈ {−4…4}` is exactly `3·t₁ + t₀` — packed 40 trits per 64-bit
//! word ([`TritVec`]), plus one power-of-two step per row stored as its
//! exponent. Small vectors (norm gains, recurrent gains, verdict
//! embeddings) stay `f32`; recurrent signs are single trits.
//!
//! ```text
//! "SNNT" v1 | vocab d layers heads mlp mem_dim block (u32 each) | n (u32)
//! per tensor: name (u16 len, utf-8) | kind u8 | rank u8 | dims u32* | data
//!   kind 0 (two-trit matrix): exponents i8 × rows | words u32 | u64 words
//!   kind 1 (f32):             f32 × elements
//!   kind 2 (sign trits):      words u32 | u64 words
//! ```

use std::collections::BTreeMap;
use std::io::{self, Read, Write};

use snn_memory::TritVec;

use crate::model::Config;

pub enum Packed {
    /// `levels` in `−4..=4`, row-major `(rows, cols)`, step `2^exp[row]`.
    Matrix {
        rows: usize,
        cols: usize,
        levels: Vec<i8>,
        exps: Vec<i8>,
    },
    F32 {
        dims: Vec<usize>,
        data: Vec<f32>,
    },
    Signs {
        data: Vec<i8>,
    },
}

pub struct PackedModel {
    pub cfg: Config,
    pub tensors: BTreeMap<String, Packed>,
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

fn pack_trits(trits: impl ExactSizeIterator<Item = i8>) -> Vec<u64> {
    let mut v = TritVec::zeros(trits.len());
    for (i, t) in trits.enumerate() {
        v.set(i, t);
    }
    v.words()
}

fn write_words(w: &mut impl Write, words: &[u64]) -> io::Result<()> {
    w.write_all(&(words.len() as u32).to_le_bytes())?;
    for x in words {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn read_exact<const N: usize>(r: &mut impl Read) -> io::Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_exact::<4>(r)?))
}

fn read_words(r: &mut impl Read, trits: usize) -> io::Result<TritVec> {
    let n = read_u32(r)? as usize;
    let words = (0..n).map(|_| Ok(u64::from_le_bytes(read_exact::<8>(r)?))).collect::<io::Result<Vec<_>>>()?;
    TritVec::from_words(&words, trits).ok_or_else(|| bad("bad trit words"))
}

impl PackedModel {
    pub fn save(&self, mut w: impl Write) -> io::Result<()> {
        w.write_all(b"SNNT")?;
        w.write_all(&1u32.to_le_bytes())?;
        let c = &self.cfg;
        for x in [c.vocab, c.d, c.layers, c.heads, c.mlp, c.mem_dim, c.block] {
            w.write_all(&(x as u32).to_le_bytes())?;
        }
        w.write_all(&(self.tensors.len() as u32).to_le_bytes())?;
        for (name, t) in &self.tensors {
            w.write_all(&(name.len() as u16).to_le_bytes())?;
            w.write_all(name.as_bytes())?;
            match t {
                Packed::Matrix { rows, cols, levels, exps } => {
                    w.write_all(&[0, 2])?;
                    w.write_all(&(*rows as u32).to_le_bytes())?;
                    w.write_all(&(*cols as u32).to_le_bytes())?;
                    w.write_all(&exps.iter().map(|&e| e as u8).collect::<Vec<_>>())?;
                    // L = 3·t1 + t0 with balanced trits.
                    let trits = levels.iter().flat_map(|&l| {
                        let t0 = ((l + 4).rem_euclid(3) as i8) - 1;
                        let t1 = (l - t0) / 3;
                        [t0, t1]
                    });
                    let trits: Vec<i8> = trits.collect();
                    write_words(&mut w, &pack_trits(trits.into_iter()))?;
                }
                Packed::F32 { dims, data } => {
                    w.write_all(&[1, dims.len() as u8])?;
                    for d in dims {
                        w.write_all(&(*d as u32).to_le_bytes())?;
                    }
                    for x in data {
                        w.write_all(&x.to_le_bytes())?;
                    }
                }
                Packed::Signs { data } => {
                    w.write_all(&[2, 1])?;
                    w.write_all(&(data.len() as u32).to_le_bytes())?;
                    write_words(&mut w, &pack_trits(data.iter().copied()))?;
                }
            }
        }
        Ok(())
    }

    pub fn load(mut r: impl Read) -> io::Result<Self> {
        if &read_exact::<4>(&mut r)? != b"SNNT" || read_u32(&mut r)? != 1 {
            return Err(bad("not an SNNT v1 model"));
        }
        let mut c = [0usize; 7];
        for x in &mut c {
            *x = read_u32(&mut r)? as usize;
        }
        let cfg = Config { vocab: c[0], d: c[1], layers: c[2], heads: c[3], mlp: c[4], mem_dim: c[5], block: c[6] };
        let n = read_u32(&mut r)?;
        let mut tensors = BTreeMap::new();
        for _ in 0..n {
            let len = u16::from_le_bytes(read_exact::<2>(&mut r)?) as usize;
            let mut name = vec![0u8; len];
            r.read_exact(&mut name)?;
            let name = String::from_utf8(name).map_err(|_| bad("bad name"))?;
            let [kind, rank] = read_exact::<2>(&mut r)?;
            let dims = (0..rank).map(|_| read_u32(&mut r).map(|x| x as usize)).collect::<io::Result<Vec<_>>>()?;
            let t = match kind {
                0 => {
                    let (rows, cols) = (dims[0], dims[1]);
                    let mut exps = vec![0u8; rows];
                    r.read_exact(&mut exps)?;
                    let trits = read_words(&mut r, 2 * rows * cols)?;
                    let levels = (0..rows * cols).map(|i| 3 * trits.get(2 * i + 1) + trits.get(2 * i)).collect();
                    Packed::Matrix { rows, cols, levels, exps: exps.into_iter().map(|e| e as i8).collect() }
                }
                1 => {
                    let n: usize = dims.iter().product();
                    let data =
                        (0..n).map(|_| Ok(f32::from_le_bytes(read_exact::<4>(&mut r)?))).collect::<io::Result<_>>()?;
                    Packed::F32 { dims, data }
                }
                2 => {
                    let trits = read_words(&mut r, dims[0])?;
                    Packed::Signs { data: (0..dims[0]).map(|i| trits.get(i)).collect() }
                }
                _ => return Err(bad("unknown tensor kind")),
            };
            tensors.insert(name, t);
        }
        Ok(Self { cfg, tensors })
    }
}

/// Pack a trained `candle` checkpoint.
pub fn pack_checkpoint(varmap: &candle_nn::VarMap, cfg: Config) -> candle_core::Result<PackedModel> {
    let data = varmap.data().lock().expect("varmap lock");
    let mut tensors = BTreeMap::new();
    for (name, var) in data.iter() {
        let t = var.as_tensor();
        let packed = if name.ends_with(".sign") {
            let v = t.to_vec1::<f32>()?;
            Packed::Signs { data: v.into_iter().map(|x| if x >= 0.0 { 1 } else { -1 }).collect() }
        } else if t.rank() == 2 && name != "verdict" {
            let (rows, cols) = t.dims2()?;
            let (levels, steps) = crate::layers::quant2_levels(t)?;
            let exps = steps.iter().map(|s| s.log2().round().clamp(-127.0, 127.0) as i8).collect();
            Packed::Matrix { rows, cols, levels, exps }
        } else {
            Packed::F32 { dims: t.dims().to_vec(), data: t.flatten_all()?.to_vec1::<f32>()? }
        };
        tensors.insert(name.clone(), packed);
    }
    Ok(PackedModel { cfg, tensors })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_trit_levels_roundtrip_through_the_file() {
        let levels: Vec<i8> = (0..81).map(|i| (i % 9) as i8 - 4).collect();
        let mut tensors = BTreeMap::new();
        tensors.insert("w".into(), Packed::Matrix { rows: 9, cols: 9, levels: levels.clone(), exps: vec![-5; 9] });
        tensors.insert("g".into(), Packed::F32 { dims: vec![3], data: vec![1.0, -2.5, 3e-7] });
        tensors.insert("s".into(), Packed::Signs { data: vec![1, -1, 1] });
        let m = PackedModel { cfg: Config::default(), tensors };
        let mut buf = Vec::new();
        m.save(&mut buf).unwrap();
        let back = PackedModel::load(&buf[..]).unwrap();
        match &back.tensors["w"] {
            Packed::Matrix { levels: l, exps, .. } => {
                assert_eq!(l, &levels);
                assert_eq!(exps, &vec![-5; 9]);
            }
            _ => panic!(),
        }
        assert!(matches!(&back.tensors["g"], Packed::F32 { data, .. } if data == &vec![1.0, -2.5, 3e-7]));
        assert!(matches!(&back.tensors["s"], Packed::Signs { data } if data == &vec![1, -1, 1]));
    }
}
