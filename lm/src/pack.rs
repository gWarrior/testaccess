//! Packed ternary model file (`.snnt`).
//!
//! Every weight matrix is stored as `n` balanced trits per weight — two
//! for the layers (`L = 3·t₁ + t₀ ∈ {−4…4}`), three for the embedding
//! (`L = 9·t₂ + 3·t₁ + t₀ ∈ {−13…13}`) — packed 40 trits per 64-bit word
//! ([`TritVec`]), plus one power-of-two step per row stored as its
//! exponent. Small vectors (norm gains, head features, verdict embeddings)
//! stay `f32`; recurrent signs are single trits.
//!
//! ```text
//! "SNNT" v2 | vocab d layers heads mlp mem_dim block state_trits (u32 each) | n (u32)
//! per tensor: name (u16 len, utf-8) | kind u8 | rank u8 | dims u32* | data
//!   kind 0 (trit matrix): trits u8 | exponents i8 × rows | words u32 | u64 words
//!   kind 1 (f32):             f32 × elements
//!   kind 2 (sign trits):      words u32 | u64 words
//! ```

use std::collections::BTreeMap;
use std::io::{self, Read, Write};

use snn_memory::TritVec;

use crate::model::Config;

pub enum Packed {
    /// `levels` in `±(3^trits − 1)/2`, row-major `(rows, cols)`, step
    /// `2^exp[row]`.
    Matrix {
        rows: usize,
        cols: usize,
        trits: u8,
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
        w.write_all(&2u32.to_le_bytes())?;
        let c = &self.cfg;
        for x in [c.vocab, c.d, c.layers, c.heads, c.mlp, c.mem_dim, c.block, c.state_trits as usize] {
            w.write_all(&(x as u32).to_le_bytes())?;
        }
        w.write_all(&(self.tensors.len() as u32).to_le_bytes())?;
        for (name, t) in &self.tensors {
            w.write_all(&(name.len() as u16).to_le_bytes())?;
            w.write_all(name.as_bytes())?;
            match t {
                Packed::Matrix { rows, cols, trits, levels, exps } => {
                    w.write_all(&[0, 2])?;
                    w.write_all(&(*rows as u32).to_le_bytes())?;
                    w.write_all(&(*cols as u32).to_le_bytes())?;
                    w.write_all(&[*trits])?;
                    w.write_all(&exps.iter().map(|&e| e as u8).collect::<Vec<_>>())?;
                    // L = Σ 3^i·t_i with balanced trits, lowest first.
                    let n = *trits as usize;
                    let all: Vec<i8> = levels.iter().flat_map(|&l| balanced(l as i32, n)).collect();
                    write_words(&mut w, &pack_trits(all.into_iter()))?;
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
        if &read_exact::<4>(&mut r)? != b"SNNT" || read_u32(&mut r)? != 2 {
            return Err(bad("not an SNNT v2 model"));
        }
        let mut c = [0usize; 8];
        for x in &mut c {
            *x = read_u32(&mut r)? as usize;
        }
        let cfg = Config {
            vocab: c[0],
            d: c[1],
            layers: c[2],
            heads: c[3],
            mlp: c[4],
            mem_dim: c[5],
            block: c[6],
            state_trits: c[7] as u8,
        };
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
                    let [n] = read_exact::<1>(&mut r)?;
                    let mut exps = vec![0u8; rows];
                    r.read_exact(&mut exps)?;
                    let k = n as usize;
                    let trits = read_words(&mut r, k * rows * cols)?;
                    let levels = (0..rows * cols)
                        .map(|i| (0..k).rev().fold(0i8, |acc, j| 3 * acc + trits.get(k * i + j)))
                        .collect();
                    Packed::Matrix { rows, cols, trits: n, levels, exps: exps.into_iter().map(|e| e as i8).collect() }
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

/// Balanced ternary digits of `l`, lowest first.
fn balanced(mut l: i32, n: usize) -> Vec<i8> {
    (0..n)
        .map(|_| {
            let t = (l + 1).rem_euclid(3) - 1;
            l = (l - t) / 3;
            t as i8
        })
        .collect()
}

/// Pack a trained model: its quantized matrices as trits and exponents,
/// recurrent signs as trits, every other parameter as `f32`.
pub fn pack_model(model: &crate::model::Model, varmap: &candle_nn::VarMap) -> candle_core::Result<PackedModel> {
    let mut tensors = BTreeMap::new();
    let mut steps = std::collections::HashSet::new();
    for (name, q) in model.qtensors() {
        let (rows, cols) = q.w.dims2()?;
        let (levels, exps) = q.levels_and_exps()?;
        let trits = if q.levels > crate::layers::LEVELS { 3 } else { 2 };
        steps.insert(format!("{name}_step"));
        tensors.insert(name, Packed::Matrix { rows, cols, trits, levels, exps });
    }
    let data = varmap.data().lock().expect("varmap lock");
    for (name, var) in data.iter() {
        if tensors.contains_key(name) || steps.contains(name) {
            continue;
        }
        let t = var.as_tensor();
        let packed = if name.ends_with(".sign") {
            let v = t.to_vec1::<f32>()?;
            Packed::Signs { data: v.into_iter().map(|x| if x >= 0.0 { 1 } else { -1 }).collect() }
        } else {
            Packed::F32 { dims: t.dims().to_vec(), data: t.flatten_all()?.to_vec1::<f32>()? }
        };
        tensors.insert(name.clone(), packed);
    }
    Ok(PackedModel { cfg: model.cfg.clone(), tensors })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_and_three_trit_levels_roundtrip_through_the_file() {
        let levels: Vec<i8> = (0..81).map(|i| (i % 9) as i8 - 4).collect();
        let levels3: Vec<i8> = (0..81).map(|i| (i % 27) as i8 - 13).collect();
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "w".into(),
            Packed::Matrix { rows: 9, cols: 9, trits: 2, levels: levels.clone(), exps: vec![-5; 9] },
        );
        tensors.insert(
            "e".into(),
            Packed::Matrix { rows: 3, cols: 27, trits: 3, levels: levels3.clone(), exps: vec![3; 3] },
        );
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
        assert!(matches!(&back.tensors["e"], Packed::Matrix { levels, trits: 3, .. } if levels == &levels3));
        assert!(matches!(&back.tensors["g"], Packed::F32 { data, .. } if data == &vec![1.0, -2.5, 3e-7]));
        assert!(matches!(&back.tensors["s"], Packed::Signs { data } if data == &vec![1, -1, 1]));
    }
}
