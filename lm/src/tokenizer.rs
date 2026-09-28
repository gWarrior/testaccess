//! Byte-level BPE tokenizer trained on the corpus.
//!
//! Text is pre-split into "words" (an optional leading space plus a run of
//! letters, digits or other symbols; newlines stand alone), each word is
//! a sequence of UTF-8 bytes, and the most frequent adjacent pair is merged
//! until the vocabulary is full. Byte-level means every string is encodable.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::{self, Read, Write};

/// Byte symbols `0..256`, then special tokens, then merges.
pub const N_BYTES: u32 = 256;
/// Document separator.
pub const DOC: u32 = 256;
pub const N_SPECIAL: u32 = 1;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Letter,
    Digit,
    Space,
    Newline,
    Other,
}

fn class(c: char) -> Class {
    if c == '\n' {
        Class::Newline
    } else if c.is_whitespace() {
        Class::Space
    } else if c.is_alphabetic() {
        Class::Letter
    } else if c.is_numeric() {
        Class::Digit
    } else {
        Class::Other
    }
}

/// Split text into pre-tokens. Concatenating them gives back the text.
pub fn pretokenize(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let k = class(c);
        let body = if k == Class::Space && c == ' ' {
            // A single space attaches to the following word.
            match chars.peek() {
                Some(&(_, n)) if matches!(class(n), Class::Letter | Class::Digit | Class::Other) => {
                    chars.next();
                    class(n)
                }
                _ => Class::Space,
            }
        } else {
            k
        };
        if body != Class::Newline {
            while let Some(&(_, n)) = chars.peek() {
                if class(n) == body && (body != Class::Other || n.is_ascii_punctuation() == c.is_ascii_punctuation()) {
                    chars.next();
                } else {
                    break;
                }
            }
        }
        let end = chars.peek().map_or(text.len(), |&(j, _)| j);
        out.push(&text[i.min(start.max(i))..end]);
        start = end;
    }
    out
}

/// A trained BPE vocabulary.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    merges: Vec<(u32, u32)>,
    ranks: HashMap<(u32, u32), u32>,
    pieces: Vec<Vec<u8>>,
}

impl Tokenizer {
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    fn from_merges(merges: Vec<(u32, u32)>) -> Self {
        let mut pieces: Vec<Vec<u8>> = (0..N_BYTES).map(|b| vec![b as u8]).collect();
        pieces.push(b"<|doc|>".to_vec());
        let mut ranks = HashMap::new();
        for (r, &(a, b)) in merges.iter().enumerate() {
            let mut p = pieces[a as usize].clone();
            p.extend_from_slice(&pieces[b as usize]);
            pieces.push(p);
            ranks.insert((a, b), r as u32);
        }
        Self { merges, ranks, pieces }
    }

    /// Train on `text` until the vocabulary has `vocab_size` entries.
    pub fn train(text: &str, vocab_size: usize) -> Self {
        let mut counts: HashMap<&str, u64> = HashMap::new();
        for w in pretokenize(text) {
            *counts.entry(w).or_insert(0) += 1;
        }
        let mut words: Vec<Vec<u32>> = Vec::with_capacity(counts.len());
        let mut freq: Vec<u64> = Vec::with_capacity(counts.len());
        for (w, c) in counts {
            words.push(w.bytes().map(u32::from).collect());
            freq.push(c);
        }

        let mut pair_count: HashMap<(u32, u32), i64> = HashMap::new();
        let mut where_: HashMap<(u32, u32), HashSet<u32>> = HashMap::new();
        for (wi, w) in words.iter().enumerate() {
            for p in w.windows(2) {
                let key = (p[0], p[1]);
                *pair_count.entry(key).or_insert(0) += freq[wi] as i64;
                where_.entry(key).or_default().insert(wi as u32);
            }
        }
        let mut heap: BinaryHeap<(i64, std::cmp::Reverse<(u32, u32)>)> =
            pair_count.iter().map(|(&k, &c)| (c, std::cmp::Reverse(k))).collect();

        let n_merges = vocab_size.saturating_sub((N_BYTES + N_SPECIAL) as usize);
        let mut merges = Vec::with_capacity(n_merges);
        while merges.len() < n_merges {
            let Some((c, std::cmp::Reverse(pair))) = heap.pop() else { break };
            let current = pair_count.get(&pair).copied().unwrap_or(0);
            if c != current {
                if current > 0 {
                    heap.push((current, std::cmp::Reverse(pair)));
                }
                continue;
            }
            if current <= 1 {
                break;
            }
            let new_id = N_BYTES + N_SPECIAL + merges.len() as u32;
            merges.push(pair);
            let affected: Vec<u32> = where_.remove(&pair).map(|s| s.into_iter().collect()).unwrap_or_default();
            let mut touched: HashSet<(u32, u32)> = HashSet::new();
            for wi in affected {
                let w = &mut words[wi as usize];
                let f = freq[wi as usize] as i64;
                // Remove old pair counts of this word.
                for p in w.windows(2) {
                    let key = (p[0], p[1]);
                    *pair_count.get_mut(&key).expect("counted") -= f;
                    touched.insert(key);
                }
                let mut merged = Vec::with_capacity(w.len());
                let mut i = 0;
                while i < w.len() {
                    if i + 1 < w.len() && (w[i], w[i + 1]) == pair {
                        merged.push(new_id);
                        i += 2;
                    } else {
                        merged.push(w[i]);
                        i += 1;
                    }
                }
                *w = merged;
                for p in w.windows(2) {
                    let key = (p[0], p[1]);
                    *pair_count.entry(key).or_insert(0) += f;
                    where_.entry(key).or_default().insert(wi);
                    touched.insert(key);
                }
            }
            pair_count.remove(&pair);
            for key in touched {
                if let Some(&c) = pair_count.get(&key) {
                    if c > 0 && key != pair {
                        heap.push((c, std::cmp::Reverse(key)));
                    }
                }
            }
        }
        Self::from_merges(merges)
    }

    /// Encode one pre-token.
    fn encode_word(&self, word: &str, out: &mut Vec<u32>) {
        let mut syms: Vec<u32> = word.bytes().map(u32::from).collect();
        loop {
            let best = syms
                .windows(2)
                .enumerate()
                .filter_map(|(i, p)| self.ranks.get(&(p[0], p[1])).map(|&r| (r, i)))
                .min();
            let Some((r, i)) = best else { break };
            syms[i] = N_BYTES + N_SPECIAL + r;
            syms.remove(i + 1);
        }
        out.extend(syms);
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut cache: HashMap<&str, Vec<u32>> = HashMap::new();
        let mut out = Vec::with_capacity(text.len() / 3);
        for w in pretokenize(text) {
            let ids = cache.entry(w).or_insert_with(|| {
                let mut v = Vec::new();
                self.encode_word(w, &mut v);
                v
            });
            out.extend_from_slice(ids);
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id == DOC {
                bytes.extend_from_slice(b"\n\n");
            } else if let Some(p) = self.pieces.get(id as usize) {
                bytes.extend_from_slice(p);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn save(&self, mut w: impl Write) -> io::Result<()> {
        w.write_all(b"BPE1")?;
        w.write_all(&(self.merges.len() as u32).to_le_bytes())?;
        for &(a, b) in &self.merges {
            w.write_all(&a.to_le_bytes())?;
            w.write_all(&b.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn load(mut r: impl Read) -> io::Result<Self> {
        let mut buf = Vec::new();
        r.read_to_end(&mut buf)?;
        let bad = || io::Error::new(io::ErrorKind::InvalidData, "not a BPE1 tokenizer");
        if buf.len() < 8 || &buf[..4] != b"BPE1" {
            return Err(bad());
        }
        let n = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        if buf.len() != 8 + n * 8 {
            return Err(bad());
        }
        let merges = (0..n)
            .map(|i| {
                let o = 8 + i * 8;
                (u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()), u32::from_le_bytes(buf[o + 4..o + 8].try_into().unwrap()))
            })
            .collect();
        Ok(Self::from_merges(merges))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "Жила-была кошка. Кошка любила молоко, а молоко любило кошку!\nВ 2024 году кошка ушла.\n\nКонец.";

    #[test]
    fn pretokenize_is_lossless() {
        let parts = pretokenize(TEXT);
        assert_eq!(parts.concat(), TEXT);
        assert!(parts.contains(&" кошка"));
        assert!(parts.contains(&"\n"));
    }

    #[test]
    fn train_encode_decode_roundtrip() {
        let corpus = TEXT.repeat(50);
        let tok = Tokenizer::train(&corpus, 400);
        assert!(tok.vocab_size() <= 400 && tok.vocab_size() > 300);
        let ids = tok.encode(TEXT);
        assert_eq!(tok.decode(&ids), TEXT);
        assert!(ids.len() < TEXT.len() / 3, "merges compress text: {} ids", ids.len());
        // Unseen text still round-trips (byte-level).
        let other = "Ёжик в тумане 🦔 — xyz";
        assert_eq!(tok.decode(&tok.encode(other)), other);

        let mut buf = Vec::new();
        tok.save(&mut buf).unwrap();
        let back = Tokenizer::load(&buf[..]).unwrap();
        assert_eq!(back.encode(TEXT), ids);
    }
}
