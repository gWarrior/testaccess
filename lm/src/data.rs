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
    pub tokens: Tokens,
}

/// Token ids of a prepared file: memory-mapped (on little-endian hosts),
/// so a 440M-token corpus lives in the OS page cache, shared by every
/// process, instead of 0.9 GB of private memory per training run.
pub enum Tokens {
    Mapped(memmap2::Mmap),
    Owned(Vec<u16>),
}

impl std::ops::Deref for Tokens {
    type Target = [u16];

    fn deref(&self) -> &[u16] {
        match self {
            // SAFETY: the map is page-aligned (so aligned for u16), read-only
            // and lives as long as `self`; its length is truncated to whole
            // u16s; the file's little-endian u16s are native on this host.
            Self::Mapped(m) => unsafe { std::slice::from_raw_parts(m.as_ptr().cast::<u16>(), m.len() / 2) },
            Self::Owned(v) => v,
        }
    }
}

impl TokenFile {
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        if cfg!(target_endian = "little") {
            // SAFETY: the prepared file is not modified while mapped.
            let map = unsafe { memmap2::Mmap::map(&file)? };
            return Ok(Self { tokens: Tokens::Mapped(map) });
        }
        let mut bytes = Vec::new();
        (&file).read_to_end(&mut bytes)?;
        let tokens = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        Ok(Self { tokens: Tokens::Owned(tokens) })
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

/// Long-range recall episodes woven into a token stream. Each states
/// something once and asks for it `D` tokens later:
///
/// * a secret word: `Секретное слово - X.` … `Какое было секретное слово? - X.`
///   or, copyable, `Секретное слово - X.`;
/// * facts of an invented person — name, town, pet, sometimes a friend —
///   asked with the statement's own words (`Меня зовут` → name) or with
///   one of several paraphrased questions (`Как меня зовут? -` → name);
/// * `Кто такой N? -`: ` мой друг.` if N was named as the friend, ` не
///   знаю.` for a name never stated (the memory's "точно нет").
///
/// The answer can only be produced by remembering the statement.
pub struct Episodes {
    keys: Vec<Vec<u32>>,
    intros: Vec<Vec<u32>>,
    questions: Vec<Vec<u32>>,
    names: Vec<Vec<u32>>,
    /// Pieces of the person statement.
    me: Vec<u32>,
    town: Vec<u32>,
    friend: Vec<u32>,
    /// Per fact: the statement's own opening and paraphrased questions.
    ask_name: (Vec<u32>, Vec<Vec<u32>>),
    ask_town: (Vec<u32>, Vec<Vec<u32>>),
    ask_friend: (Vec<u32>, Vec<Vec<u32>>),
    /// Per pet: the statement line's opening (`\nМою кошку зовут`), questions.
    pets: Vec<(Vec<u32>, Vec<Vec<u32>>)>,
    who: (Vec<u32>, Vec<u32>),
    my_friend: Vec<u32>,
    dont_know: Vec<u32>,
    end: Vec<u32>,
    tok_end: Vec<u32>,
}

/// Syllables of invented names (no real names or places).
const SYLLABLES: &[&str] = &[
    "ла", "ми", "ро", "на", "ке", "ти", "ва", "со", "ре", "лу", "ни", "да", "фе", "ри", "то", "ме", "ли", "зо", "ка",
    "эн", "ор", "ай",
];

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
        let e = |t: &str| tok.encode(t);
        let many = |ts: &[&str]| ts.iter().map(|t| tok.encode(t)).collect::<Vec<_>>();
        let mut keys: Vec<Vec<u32>> = KEY_WORDS.iter().map(|w| e(&format!(" {w}"))).collect();
        // Plus 2187 invented words, so a secret cannot be guessed from a
        // small set of familiar answers.
        let mut krng = snn_memory::rng::SplitMix64::new(0x51ab);
        keys.extend((0..2187).map(|_| {
            let n = 2 + krng.below(2) as usize;
            let w: String = (0..n).map(|_| SYLLABLES[krng.below(SYLLABLES.len() as u64) as usize]).collect();
            e(&format!(" {w}"))
        }));
        // 19 683 invented names of 2–4 syllables: a fresh name is almost
        // never one the stream has already met.
        let mut rng = snn_memory::rng::SplitMix64::new(0x9e37);
        let names = (0..19_683)
            .map(|_| {
                let n = 2 + rng.below(3) as usize;
                let word: String = (0..n).map(|_| SYLLABLES[rng.below(SYLLABLES.len() as u64) as usize]).collect();
                let mut c = word.chars();
                let cap: String = c.next().into_iter().flat_map(char::to_uppercase).chain(c).collect();
                e(&format!(" {cap}"))
            })
            .collect();
        let pets = [
            ("Мою кошку", "мою кошку", "моей кошки"),
            ("Мою собаку", "мою собаку", "моей собаки"),
            ("Моего попугая", "моего попугая", "моего попугая"),
        ]
        .iter()
        .map(|(acc, acc_lower, gen)| {
            (
                e(&format!("\n{acc} зовут")),
                many(&[
                    &format!("\nКак зовут {acc_lower}? -"),
                    &format!("\nКличка {gen}? -"),
                    &format!("\nНапомни, как зовут {acc_lower}? -"),
                    // Held out: only the evaluation asks this way (another
                    // construction, not just another first word).
                    &format!("\nИмя у {gen} какое? -"),
                ]),
            )
        })
        .collect();
        let intros = many(&["\nСекретное слово -", "\nЗапомни пароль:", "\nКод доступа:"]);
        let questions = many(&["\nКакое было секретное слово? -", "\nКакой был пароль? -", "\nНапомни код доступа:"]);
        Self {
            keys,
            intros,
            questions,
            names,
            me: e("\nМеня зовут"),
            town: e("\nЯ живу в городе"),
            friend: e("\nМоего друга зовут"),
            ask_name: (
                e("\nМеня зовут"),
                many(&["\nКак меня зовут? -", "\nНапомни, как меня зовут? -", "\nМоё имя? -", "\nИмя у меня какое? -"]),
            ),
            ask_town: (
                e("\nЯ живу в городе"),
                many(&[
                    "\nВ каком городе я живу? -",
                    "\nГде я живу? -",
                    "\nНапомни, в каком городе я живу? -",
                    "\nНазови мой город? -",
                ]),
            ),
            ask_friend: (
                e("\nМоего друга зовут"),
                many(&[
                    "\nКак зовут моего друга? -",
                    "\nИмя моего друга? -",
                    "\nНапомни, как зовут моего друга? -",
                    "\nДруг мой - кто он по имени? -",
                ]),
            ),
            pets,
            who: (e("\nКто такой"), e("? -")),
            my_friend: e(" мой друг.\n"),
            dont_know: e(" не знаю.\n"),
            end: e(".\n"),
            tok_end: e("."),
        }
    }

    /// Whether an answer is a template (" мой друг." / " не знаю."), not a
    /// fact that can be found in the context.
    pub fn is_template(&self, answer: &[u32]) -> bool {
        answer == self.my_friend.as_slice() || answer == self.dont_know.as_slice()
    }

    /// Whether an answer is " не знаю." (the fact was never stated).
    pub fn is_dont_know(&self, answer: &[u32]) -> bool {
        answer == self.dont_know.as_slice()
    }

    /// Tokens of the ".\n" that ends every statement and answer.
    pub fn end_len(&self) -> usize {
        self.end.len()
    }

    /// A secret-word episode asked with a question (association, not
    /// copying): the evaluation's recall test.
    pub fn sample_secret(&self, rng: &mut snn_memory::rng::SplitMix64) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let key = &self.keys[rng.below(self.keys.len() as u64) as usize];
        let kind = rng.below(self.intros.len() as u64) as usize;
        let intro = [&self.intros[kind][..], key, &self.end].concat();
        (intro, self.questions[kind].clone(), [&key[..], &self.end].concat())
    }

    /// `(intro tokens, question tokens, answer tokens)` of a random episode.
    pub fn sample(&self, rng: &mut snn_memory::rng::SplitMix64) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        // A third asked with the statement's own words: the induction column
        // answers those by itself; paraphrases are what has to be learned.
        let same_words = rng.below(3) == 0;
        if rng.below(4) == 0 {
            let (intro, q, a) = self.sample_secret(rng);
            if same_words {
                // The question is the statement's own opening.
                let kind = self.intros.iter().position(|i| intro.starts_with(i)).unwrap_or(0);
                return (intro, self.intros[kind].clone(), a);
            }
            return (intro, q, a);
        }
        self.sample_person(rng, same_words, None)
    }

    /// A person episode: `which` = 0 name, 1 town, 2 pet, 3 friend, 4 "Кто
    /// такой N?" (random if `None`); asked with the statement's own words or
    /// with a paraphrased question.
    pub fn sample_person(
        &self,
        rng: &mut snn_memory::rng::SplitMix64,
        same_words: bool,
        which: Option<u64>,
    ) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        self.person(rng, same_words, which, false)
    }

    /// As [`Self::sample_person`], asked with the paraphrase that training
    /// never uses (an honest test of association).
    pub fn sample_person_held_out(
        &self,
        rng: &mut snn_memory::rng::SplitMix64,
        which: u64,
    ) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        self.person(rng, false, Some(which), true)
    }

    fn person(
        &self,
        rng: &mut snn_memory::rng::SplitMix64,
        same_words: bool,
        which: Option<u64>,
        held_out: bool,
    ) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let pick =
            |v: &[Vec<u32>], rng: &mut snn_memory::rng::SplitMix64| v[rng.below(v.len() as u64) as usize].clone();
        let (name, town, pet_name, friend) =
            (pick(&self.names, rng), pick(&self.names, rng), pick(&self.names, rng), pick(&self.names, rng));
        let which = which.unwrap_or_else(|| rng.below(5));
        let has_friend = which == 3 || rng.below(2) == 0;
        let pet = &self.pets[rng.below(self.pets.len() as u64) as usize];
        // One fact per line, so asking "with the same words" repeats the
        // line's opening token for token:
        // "Меня зовут N.\nЯ живу в городе T.\nМою кошку зовут P.[\nМоего друга зовут F.]"
        let mut intro =
            [&self.me[..], &name, &self.tok_end, &self.town, &town, &self.tok_end, &pet.0, &pet_name].concat();
        if has_friend {
            intro.extend([&self.tok_end[..], &self.friend, &friend].concat());
        }
        intro.extend(&self.end);
        // The last paraphrase of every list is held out for evaluation.
        let ask = |(own, paraphrases): &(Vec<u32>, Vec<Vec<u32>>), rng: &mut snn_memory::rng::SplitMix64| {
            if same_words {
                own.clone()
            } else if held_out {
                paraphrases[paraphrases.len() - 1].clone()
            } else {
                pick(&paraphrases[..paraphrases.len() - 1], rng)
            }
        };
        let answer = |a: &[u32]| [a, &self.end].concat();
        match which {
            0 => (intro, ask(&self.ask_name, rng), answer(&name)),
            1 => (intro, ask(&self.ask_town, rng), answer(&town)),
            2 => (intro, ask(pet, rng), answer(&pet_name)),
            3 if has_friend => (intro, ask(&self.ask_friend, rng), answer(&friend)),
            _ => {
                // Who is N? The friend, or a name never stated.
                let (who, a) = if has_friend && rng.below(2) == 0 {
                    (friend, self.my_friend.clone())
                } else {
                    (pick(&self.names, rng), self.dont_know.clone())
                };
                (intro, [&self.who.0[..], &who, &self.who.1].concat(), a)
            }
        }
    }
}

/// Token kinds of a stream: plain text, an episode's answer that is in the
/// context, a re-read (copyable) span, and a template answer (" не знаю.",
/// " мой друг.") that must be decided, not copied.
pub const PLAIN: u8 = 0;
pub const ANSWER: u8 = 1;
pub const REREAD: u8 = 2;
pub const TEMPLATE: u8 = 3;

/// Stream positions `[start, end)` of a fact's statement.
pub type Src = (u64, u64);

/// No statement span (plain text, templates, re-reading).
pub const NO_SRC: Src = (0, 0);

/// A token stream over a region of the corpus with recall episodes.
pub struct TaskStream {
    start: usize,
    len: usize,
    pos: usize,
    rng: snn_memory::rng::SplitMix64,
    pending: std::collections::VecDeque<(u32, u8, Src)>,
    /// Questions waiting for their distance: (tokens left, question, answer).
    /// Questions waiting: (tokens left, question, answer, statement span).
    questions: Vec<(usize, Vec<u32>, Vec<u32>, Src)>,
    carry: Option<(u32, u8, Src)>,
    /// Tokens emitted so far: the stream position of the next token.
    emitted: u64,
    /// Probability of starting an episode at a token.
    pub p_episode: f64,
    /// Distance range, sampled log-uniformly.
    pub distance: (usize, usize),
    /// After each document, jump to a random document of the whole corpus
    /// (every topic of the corpus, not only this stream's region).
    pub jump: bool,
    /// Probability of a re-reading episode: a span of 27–81 tokens seen
    /// 243–19 683 tokens ago is repeated verbatim (dense far-copy signal).
    pub p_reread: f64,
    past: std::collections::VecDeque<u32>,
    /// Tokens made only of newlines (empty: no poem filter). A jump that
    /// lands on a document with more than 1/12 of such tokens in its first
    /// 729 (poems, lists, dialogue columns) jumps again.
    pub lines: Vec<bool>,
}

impl TaskStream {
    pub fn new(start: usize, len: usize, seed: u64) -> Self {
        Self {
            start,
            len,
            pos: 0,
            rng: snn_memory::rng::SplitMix64::new(seed),
            pending: Default::default(),
            questions: Vec::new(),
            carry: None,
            emitted: 0,
            p_episode: 1.0 / 729.0,
            distance: (27, 19_683),
            jump: false,
            p_reread: 0.0,
            past: Default::default(),
            lines: Vec::new(),
        }
    }

    /// Next token, its kind ([`PLAIN`], [`ANSWER`], [`REREAD`],
    /// [`TEMPLATE`]) and, for a fact answer, the stream positions of its
    /// statement ([`NO_SRC`] otherwise).
    fn next(&mut self, tokens: &[u16], ep: &Episodes) -> (u32, u8, Src) {
        if self.pending.is_empty() && self.p_reread > 0.0 && self.past.len() > 2 * 243 {
            if self.rng.next_f64() < self.p_reread {
                let len = 27 + self.rng.below(55) as usize;
                // Log-uniform look-back from 243 tokens to all that is kept.
                let span = ((self.past.len() - len) as f64 / 243.0).ln().max(0.0);
                let back = ((243f64).ln() + self.rng.next_f64() * span).exp() as usize;
                let back = back.clamp(243 + len, self.past.len());
                let start = self.past.len() - back;
                let span: Vec<u32> = self.past.range(start..start + len).copied().collect();
                // The first token of the span cannot be predicted; the rest can be copied.
                self.pending.extend(
                    span.into_iter().enumerate().map(|(i, t)| (t, if i == 0 { PLAIN } else { REREAD }, NO_SRC)),
                );
            }
        }
        let out = self.next_raw(tokens, ep);
        self.emitted += 1;
        if self.p_reread > 0.0 {
            self.past.push_back(out.0);
            if self.past.len() > self.distance.1 {
                self.past.pop_front();
            }
        }
        out
    }

    fn next_raw(&mut self, tokens: &[u16], ep: &Episodes) -> (u32, u8, Src) {
        if let Some(t) = self.pending.pop_front() {
            return t;
        }
        for q in &mut self.questions {
            q.0 = q.0.saturating_sub(1);
        }
        if let Some(i) = self.questions.iter().position(|q| q.0 == 0) {
            let (_, q, a, src) = self.questions.swap_remove(i);
            self.pending.extend(q.iter().map(|&t| (t, PLAIN, NO_SRC)));
            let (kind, src) = if ep.is_template(&a) { (TEMPLATE, NO_SRC) } else { (ANSWER, src) };
            self.pending.extend(a.iter().map(|&t| (t, kind, src)));
            return self.pending.pop_front().expect("question is not empty");
        }
        if self.p_episode > 0.0 && self.rng.next_f64() < self.p_episode {
            let (intro, q, a) = ep.sample(&mut self.rng);
            let (lo, hi) = self.distance;
            let d = ((lo as f64).ln() + self.rng.next_f64() * ((hi as f64).ln() - (lo as f64).ln())).exp() as usize;
            // The intro starts now: the statement spans these positions.
            let src = (self.emitted, self.emitted + intro.len() as u64);
            self.pending.extend(intro.into_iter().map(|t| (t, PLAIN, NO_SRC)));
            self.questions.push((d.max(1), q, a, src));
            return self.pending.pop_front().expect("intro is not empty");
        }
        let t = tokens[self.start + self.pos % self.len] as u32;
        self.pos += 1;
        if self.jump && t == crate::tokenizer::DOC {
            // Land on the start of a random prose document.
            for _ in 0..9 {
                let mut p = self.rng.below(tokens.len() as u64) as usize;
                while p < tokens.len() && tokens[p] as u32 != crate::tokenizer::DOC {
                    p += 1;
                }
                (self.start, self.len, self.pos) = (0, tokens.len(), (p + 1) % tokens.len());
                if !self.is_poem(tokens, self.pos) {
                    break;
                }
            }
        }
        (t, PLAIN, NO_SRC)
    }

    /// Whether the document starting at `p` breaks lines more often than
    /// prose: over 1/12 newline tokens in its first 729.
    fn is_poem(&self, tokens: &[u16], p: usize) -> bool {
        if self.lines.is_empty() {
            return false;
        }
        let (mut n, mut lines) = (0usize, 0usize);
        for &t in tokens[p..].iter().take(729) {
            if t as u32 == crate::tokenizer::DOC {
                break;
            }
            n += 1;
            lines += usize::from(self.lines.get(t as usize).copied().unwrap_or(false));
        }
        lines * 12 > n
    }

    /// `window + 1` tokens continuing the stream (the first is the last of
    /// the previous call), with their kinds.
    pub fn window(&mut self, tokens: &[u16], ep: &Episodes, window: usize) -> (Vec<u32>, Vec<u8>) {
        let (out, ans, _) = self.window_src(tokens, ep, window);
        (out, ans)
    }

    /// [`window`](Self::window) with each token's statement span (the
    /// stream positions a fact answer can be copied from, [`NO_SRC`] for
    /// other tokens). Stream position `i` of window `w` is `w·window + i`.
    pub fn window_src(&mut self, tokens: &[u16], ep: &Episodes, window: usize) -> (Vec<u32>, Vec<u8>, Vec<Src>) {
        let first = self.carry.take().unwrap_or_else(|| self.next(tokens, ep));
        let (mut out, mut ans, mut src) = (vec![first.0], vec![first.1], vec![first.2]);
        for _ in 0..window {
            let (t, a, r) = self.next(tokens, ep);
            out.push(t);
            ans.push(a);
            src.push(r);
        }
        self.carry = Some((out[window], ans[window], src[window]));
        (out, ans, src)
    }
}

#[cfg(test)]
mod episode_tests {
    use super::*;
    use crate::tokenizer::Tokenizer;

    #[test]
    fn jumps_skip_documents_that_break_lines_like_poems() {
        let prose = "Она шла по улице и думала о том, что завтра будет дождь. ".repeat(9);
        let poem = "Шёл дождь\nи ветер\nпел\n".repeat(9);
        let tok = Tokenizer::train(&format!("{prose}\n{poem}"), 400);
        let doc = crate::tokenizer::DOC as u16;
        let mut tokens = Vec::new();
        for text in [&prose, &poem] {
            tokens.push(doc);
            tokens.extend(tok.encode(text).iter().map(|&t| t as u16));
        }
        let poem_at = tokens.iter().rposition(|&t| t == doc).expect("two documents") + 1;
        let mut s = TaskStream::new(0, tokens.len(), 3);
        assert!(!s.is_poem(&tokens, poem_at), "the filter is off without newline tokens");
        s.lines = tok.newline_tokens();
        assert!(s.is_poem(&tokens, poem_at));
        assert!(!s.is_poem(&tokens, 1));
    }

    #[test]
    fn every_episode_answer_is_in_its_statement() {
        let tok = Tokenizer::train(
            &"Меня зовут Лами. Я живу в городе Роке. Мою кошку зовут Нита. Секретное слово — маяк.\n".repeat(30),
            400,
        );
        let ep = Episodes::new(&tok);
        let mut rng = snn_memory::rng::SplitMix64::new(5);
        for _ in 0..200 {
            let (intro, question, answer) = ep.sample(&mut rng);
            assert!(!question.is_empty());
            if answer == ep.dont_know || answer == ep.my_friend {
                // "Кто такой N?": the answer is about N, not a copy.
                assert!(question.starts_with(&ep.who.0) && question.ends_with(&ep.who.1));
                continue;
            }
            let key = &answer[..answer.len() - ep.end.len()];
            assert!(intro.windows(key.len()).any(|w| w == key), "{}", tok.decode(&intro));
        }
        let (intro, q, a) = ep.sample(&mut snn_memory::rng::SplitMix64::new(11));
        println!("{} | {} | {}", tok.decode(&intro), tok.decode(&q), tok.decode(&a));
    }

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
        assert!(ans.iter().any(|&a| a == ANSWER), "answer tokens are flagged");
        // Some flagged answer repeats a token that appeared earlier (a fact;
        // "не знаю" and "мой друг" answers need not).
        let repeats = (0..all.len()).filter(|&i| ans[i] == ANSWER && all[..i].contains(&all[i])).count();
        assert!(repeats > 0);
    }

    #[test]
    fn fact_answers_point_at_their_statement() {
        let tok = Tokenizer::train(
            &"Меня зовут Лами. Я живу в городе Роке. Мою кошку зовут Нита. Секретное слово — маяк.\n".repeat(30),
            400,
        );
        let ep = Episodes::new(&tok);
        let filler: Vec<u16> = (0..5000).map(|i| 300 + (i % 50) as u16).collect();
        let mut s = TaskStream::new(0, filler.len(), 7);
        s.p_episode = 1.0 / 200.0;
        s.distance = (27, 729);
        let (mut all, mut kinds, mut srcs) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..81 {
            let (w, k, r) = s.window_src(&filler, &ep, 81);
            // Window w holds stream positions 81·w … 81·w + 80.
            all.extend_from_slice(&w[..81]);
            kinds.extend_from_slice(&k[..81]);
            srcs.extend_from_slice(&r[..81]);
        }
        let mut checked = 0;
        for i in 0..all.len() {
            if kinds[i] != ANSWER {
                assert_eq!(srcs[i], NO_SRC);
                continue;
            }
            let (a, b) = (srcs[i].0 as usize, srcs[i].1 as usize);
            assert!(a < b && b <= i, "the statement comes before the answer");
            // The answer's end marker need not be in the statement; its words are.
            if !ep.end.contains(&all[i]) {
                assert!(all[a..b].contains(&all[i]), "{} not in {}", tok.decode(&all[i..=i]), tok.decode(&all[a..b]));
                checked += 1;
            }
        }
        assert!(checked > 5, "{checked} answer tokens checked");
    }
}
