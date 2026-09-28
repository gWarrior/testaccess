//! Corpus preparation and batching.
//!
//! * `read_parquet_texts` pulls the `text` column out of parquet shards;
//! * `write_tokens` / `TokenFile` store token ids as little-endian `u16`
//!   with [`DOC`](crate::tokenizer::DOC) between documents;
//! * `Streams` cuts the token file into `batch` parallel, contiguous streams
//!   for truncated back-propagation through time: window `w` of stream `b`
//!   continues exactly where window `w - 1` ended, so recurrent state and
//!   the SNN memory carry over between windows.

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;

use crate::tokenizer::DOC;

/// All non-empty `text` values of a parquet file.
pub fn read_parquet_texts(path: &Path) -> io::Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file).map_err(io::Error::other)?;
    let mut out = Vec::new();
    for row in reader.get_row_iter(None).map_err(io::Error::other)? {
        let row = row.map_err(io::Error::other)?;
        for (name, field) in row.get_column_iter() {
            if name == "text" {
                if let Field::Str(s) = field {
                    if !s.trim().is_empty() {
                        out.push(s.clone());
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Write documents' tokens separated by `DOC`.
pub fn write_tokens(path: &Path, docs: &[Vec<u32>]) -> io::Result<usize> {
    let mut w = BufWriter::new(File::create(path)?);
    let mut n = 0;
    for d in docs {
        for &t in d.iter().chain(std::iter::once(&DOC)) {
            w.write_all(&(t as u16).to_le_bytes())?;
            n += 1;
        }
    }
    w.flush()?;
    Ok(n)
}

/// Token ids loaded from a `u16` file.
pub struct TokenFile {
    pub tokens: Vec<u16>,
}

impl TokenFile {
    pub fn load(path: &Path) -> io::Result<Self> {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        let tokens = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        Ok(Self { tokens })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// `batch` contiguous streams over a token sequence.
pub struct Streams {
    starts: Vec<usize>,
    len: usize,
    pos: usize,
}

impl Streams {
    pub fn new(n_tokens: usize, batch: usize) -> Self {
        let len = n_tokens / batch;
        Self { starts: (0..batch).map(|b| b * len).collect(), len, pos: 0 }
    }

    /// Next window: inputs and targets, `batch × window` each, or `None`
    /// once a stream is exhausted.
    pub fn next(&mut self, tokens: &[u16], window: usize) -> Option<(Vec<u32>, Vec<u32>)> {
        if self.pos + window + 1 > self.len {
            return None;
        }
        let mut x = Vec::with_capacity(self.starts.len() * window);
        let mut y = Vec::with_capacity(self.starts.len() * window);
        for &s in &self.starts {
            let a = s + self.pos;
            x.extend(tokens[a..a + window].iter().map(|&t| t as u32));
            y.extend(tokens[a + 1..a + window + 1].iter().map(|&t| t as u32));
        }
        self.pos += window;
        Some((x, y))
    }

    /// Tokens consumed per stream so far.
    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn seek(&mut self, pos: usize) {
        self.pos = pos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_contiguous() {
        let tokens: Vec<u16> = (0..100).collect();
        let mut s = Streams::new(tokens.len(), 3);
        let (x1, y1) = s.next(&tokens, 9).unwrap();
        let (x2, _) = s.next(&tokens, 9).unwrap();
        assert_eq!(&x1[..9], &(0..9).collect::<Vec<u32>>()[..]);
        assert_eq!(&y1[..9], &(1..10).collect::<Vec<u32>>()[..]);
        assert_eq!(&x1[9..18], &(33..42).collect::<Vec<u32>>()[..]);
        assert_eq!(&x2[..9], &(9..18).collect::<Vec<u32>>()[..]);
        for _ in 0..1 {
            s.next(&tokens, 9).unwrap();
        }
        assert!(s.next(&tokens, 9).is_none());
    }
}
