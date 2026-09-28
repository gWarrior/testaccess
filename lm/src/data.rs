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

/// Long-range recall episodes woven into a token stream:
/// `… Секретное слово — X. … (D tokens) … Какое было секретное слово? — X. …`
/// The answer can only be produced by remembering the key `D` tokens back.
pub struct Episodes {
    keys: Vec<Vec<u32>>,
    intros: Vec<Vec<u32>>,
    questions: Vec<Vec<u32>>,
    end: Vec<u32>,
}

const KEY_WORDS: &[&str] = &[
    "яблоко",
    "маяк",
    "комета",
    "сова",
    "якорь",
    "ландыш",
    "фонарь",
    "кит",
    "гроза",
    "ключ",
    "янтарь",
    "лиса",
    "мост",
    "ракета",
    "колокол",
    "роза",
    "пустыня",
    "дракон",
    "единорог",
    "лабиринт",
    "зеркало",
    "корона",
    "мельница",
    "черника",
    "вулкан",
    "парус",
    "подсолнух",
    "бархан",
    "фонтан",
    "метель",
    "клевер",
    "жемчуг",
    "олень",
    "сфинкс",
    "компас",
    "малахит",
    "водопад",
    "граната",
    "бумеранг",
    "сирень",
    "айсберг",
    "фламинго",
    "кедр",
    "трамвай",
    "шахматы",
    "каштан",
    "радуга",
    "полынь",
    "медведь",
    "чайник",
    "кувшин",
    "туман",
    "огурец",
    "барабан",
    "орёл",
];

impl Episodes {
    pub fn new(tok: &crate::tokenizer::Tokenizer) -> Self {
        let mut keys: Vec<Vec<u32>> = KEY_WORDS.iter().map(|w| tok.encode(&format!(" {w}"))).collect();
        keys.extend((0..81).map(|i| tok.encode(&format!(" {}", 1000 + (i * 7919) % 9000))));
        let intros =
            ["\nСекретное слово —", "\nЗапомни пароль:", "\nКод доступа:"].iter().map(|s| tok.encode(s)).collect();
        let questions = ["\nКакое было секретное слово? —", "\nКакой был пароль? —", "\nНапомни код доступа:"]
            .iter()
            .map(|s| tok.encode(s))
            .collect();
        Self { keys, intros, questions, end: tok.encode(".\n") }
    }

    /// `(intro tokens, question tokens, answer tokens)` of a random episode.
    pub fn sample(&self, rng: &mut snn_memory::rng::SplitMix64) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let key = &self.keys[rng.below(self.keys.len() as u64) as usize];
        let kind = rng.below(self.intros.len() as u64) as usize;
        let intro = [&self.intros[kind][..], key, &self.end].concat();
        (intro, self.questions[kind].clone(), [&key[..], &self.end].concat())
    }
}

/// A token stream over a region of the corpus with recall episodes.
pub struct TaskStream {
    start: usize,
    len: usize,
    pos: usize,
    rng: snn_memory::rng::SplitMix64,
    pending: std::collections::VecDeque<(u32, bool)>,
    question: Option<(usize, Vec<u32>, Vec<u32>)>,
    carry: Option<(u32, bool)>,
    /// Probability of starting an episode at a token.
    pub p_episode: f64,
    pub distance: (usize, usize),
    /// After each document, jump to a random document of the whole corpus
    /// (every topic of the corpus, not only this stream's region).
    pub jump: bool,
}

impl TaskStream {
    pub fn new(start: usize, len: usize, seed: u64) -> Self {
        Self {
            start,
            len,
            pos: 0,
            rng: snn_memory::rng::SplitMix64::new(seed),
            pending: Default::default(),
            question: None,
            carry: None,
            p_episode: 1.0 / 2187.0,
            distance: (243, 19_683),
            jump: false,
        }
    }

    /// Next token and whether it is an episode answer token.
    fn next(&mut self, tokens: &[u16], ep: &Episodes) -> (u32, bool) {
        if let Some(t) = self.pending.pop_front() {
            return t;
        }
        if let Some((left, q, a)) = &mut self.question {
            if *left == 0 {
                self.pending.extend(q.iter().map(|&t| (t, false)));
                self.pending.extend(a.iter().map(|&t| (t, true)));
                self.question = None;
                return self.pending.pop_front().expect("question is not empty");
            }
            *left -= 1;
        } else if self.p_episode > 0.0 && self.rng.next_f64() < self.p_episode {
            let (intro, q, a) = ep.sample(&mut self.rng);
            let (lo, hi) = self.distance;
            let d = lo + self.rng.below((hi - lo + 1) as u64) as usize;
            self.pending.extend(intro.into_iter().map(|t| (t, false)));
            self.question = Some((d, q, a));
            return self.pending.pop_front().expect("intro is not empty");
        }
        let t = tokens[self.start + self.pos % self.len] as u32;
        self.pos += 1;
        if self.jump && t == crate::tokenizer::DOC {
            // Land on the start of a random document.
            let mut p = self.rng.below(tokens.len() as u64) as usize;
            while p < tokens.len() && tokens[p] as u32 != crate::tokenizer::DOC {
                p += 1;
            }
            (self.start, self.len, self.pos) = (0, tokens.len(), (p + 1) % tokens.len());
        }
        (t, false)
    }

    /// `window + 1` tokens continuing the stream (the first is the last of
    /// the previous call), with answer flags.
    pub fn window(&mut self, tokens: &[u16], ep: &Episodes, window: usize) -> (Vec<u32>, Vec<bool>) {
        let first = self.carry.take().unwrap_or_else(|| self.next(tokens, ep));
        let mut out = vec![first.0];
        let mut ans = vec![first.1];
        for _ in 0..window {
            let (t, a) = self.next(tokens, ep);
            out.push(t);
            ans.push(a);
        }
        self.carry = Some((out[window], ans[window]));
        (out, ans)
    }
}

#[cfg(test)]
mod episode_tests {
    use super::*;
    use crate::tokenizer::Tokenizer;

    #[test]
    fn episodes_repeat_the_key_after_the_distance() {
        let tok = Tokenizer::train(&"Жили-были кот и пёс. Секретное слово — маяк. Код доступа 1234.\n".repeat(30), 400);
        let ep = Episodes::new(&tok);
        let filler: Vec<u16> = (0..2000).map(|i| 300 + (i % 50) as u16).collect();
        let mut s = TaskStream::new(0, filler.len(), 3);
        s.p_episode = 1.0 / 300.0;
        s.distance = (243, 243);
        let (mut all, mut ans) = (Vec::new(), Vec::new());
        let mut prev_last = None;
        for _ in 0..27 {
            let (w, a) = s.window(&filler, &ep, 81);
            if let Some(p) = prev_last {
                assert_eq!(w[0], p, "windows overlap by one token");
            }
            prev_last = Some(w[81]);
            all.extend_from_slice(&w[..81]);
            ans.extend_from_slice(&a[..81]);
        }
        let text = tok.decode(&all);
        let asked = text.matches("?").count() + text.matches("Напомни").count();
        assert!(asked >= 2, "questions were asked: {text}");
        assert!(ans.iter().any(|&a| a), "answer tokens are flagged");
        // Every flagged answer repeats tokens that appeared earlier.
        let first = ans.iter().position(|&a| a).unwrap();
        assert!(all[..first].contains(&all[first]));
    }
}
