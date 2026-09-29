//! Evaluation: held-out loss (with and without the SNN memory) and
//! long-range recall ("secret word") accuracy at increasing distances.

use candle_core::{Device, Result};
use snn_memory::KvPrecision;

use crate::data::Episodes;
use crate::model::Model;
use crate::train::Runner;

/// Mean loss (nats/token) over `windows` windows of `batch` val streams.
pub fn val_loss(
    model: Model,
    tokens: &[u16],
    batch: usize,
    windows: usize,
    memory: bool,
    precision: KvPrecision,
) -> Result<f64> {
    Ok(val_loss_detail(model, tokens, batch, windows, memory, precision, false)?.0)
}

/// Mean loss and mean loss by position inside the window (in 9 buckets of
/// 27 positions), optionally dropping the recurrent state between windows.
pub fn val_loss_detail(
    model: Model,
    tokens: &[u16],
    batch: usize,
    windows: usize,
    memory: bool,
    precision: KvPrecision,
    reset_state: bool,
) -> Result<(f64, Vec<f64>)> {
    let device = Device::Cpu;
    let t = 243;
    let mut runner = Runner::new(model, batch, memory, 300_000, precision, &device)?;
    runner.reset_state = reset_state;
    let region = tokens.len() / batch;
    let (mut sum, mut n) = (0f64, 0usize);
    let mut by_pos = vec![(0f64, 0usize); 9];
    for w in 0..windows {
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for b in 0..batch {
            let a = b * region + w * t;
            if a + t + 1 > (b + 1) * region {
                return Ok((sum / n.max(1) as f64, by_pos.iter().map(|(s, c)| s / (*c).max(1) as f64).collect()));
            }
            x.extend(tokens[a..a + t].iter().map(|&v| v as u32));
            y.extend(tokens[a + 1..a + t + 1].iter().map(|&v| v as u32));
        }
        let (out, next) = runner.forward(&x, t, &device)?;
        let l = runner.losses(&out, &x, &y)?.to_vec1::<f32>()?;
        sum += l.iter().map(|&v| v as f64).sum::<f64>();
        n += l.len();
        for (i, &v) in l.iter().enumerate() {
            let bucket = &mut by_pos[(i % t) / 27];
            bucket.0 += v as f64;
            bucket.1 += 1;
        }
        runner.commit(&x, t, &out, next)?;
    }
    Ok((sum / n.max(1) as f64, by_pos.iter().map(|(s, c)| s / (*c).max(1) as f64).collect()))
}

/// Re-reading test: every stream reads a passage of `len` tokens, then the
/// same passage again. Returns (top-1 accuracy, loss) on the first and the
/// second reading. The second reading is far beyond the local window, so a
/// gain there measures what the model recovers from its SNN memory.
pub fn reread(model: Model, passages: &[Vec<u32>], memory: bool, precision: KvPrecision) -> Result<[(f64, f64); 2]> {
    let device = Device::Cpu;
    let t = 243;
    let b = passages.len();
    let len = passages.iter().map(Vec::len).min().unwrap_or(0) / t * t;
    let seqs: Vec<Vec<u32>> = passages.iter().map(|p| [&p[..len], &p[..len], &p[..t + 1]].concat()).collect();
    let mut runner = Runner::new(model, b, memory, 300_000, precision, &device)?;
    let mut acc = [(0f64, 0f64, 0usize); 2];
    let mut pos = 0;
    while pos + t < 2 * len + 1 {
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for s in &seqs {
            x.extend_from_slice(&s[pos..pos + t]);
            y.extend_from_slice(&s[pos + 1..pos + t + 1]);
        }
        let (out, next) = runner.forward(&x, t, &device)?;
        let losses = runner.losses(&out, &x, &y)?.to_vec1::<f32>()?;
        let argmax = runner.predict(&out, &x)?;
        let phase = usize::from(pos >= len);
        for i in 0..b * t {
            acc[phase].0 += (argmax[i] == y[i]) as u8 as f64;
            acc[phase].1 += losses[i] as f64;
            acc[phase].2 += 1;
        }
        runner.commit(&x, t, &out, next)?;
        pos += t;
    }
    Ok(acc.map(|(a, l, n)| (a / n.max(1) as f64, l / n.max(1) as f64)))
}

/// Which episodes a recall test asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpisodeSet {
    /// A secret word, asked in other words (association).
    Secret,
    /// A person's fact, asked with the statement's own words (copying).
    Same,
    /// A person's fact, asked with a paraphrase training never uses
    /// (association).
    Paraphrase,
    /// "Кто такой N?": " мой друг." if N was stated, else " не знаю.".
    Who,
}

impl EpisodeSet {
    fn sample(self, ep: &Episodes, rng: &mut snn_memory::rng::SplitMix64) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        match self {
            Self::Secret => ep.sample_secret(rng),
            Self::Same => {
                let which = rng.below(4);
                ep.sample_person(rng, true, Some(which))
            }
            Self::Paraphrase => {
                let which = rng.below(4);
                ep.sample_person_held_out(rng, which)
            }
            Self::Who => ep.sample_person(rng, false, Some(4)),
        }
    }
}

/// Recall result at one distance.
#[derive(Debug, Clone, Default)]
pub struct Recall {
    pub distance: usize,
    pub episodes: usize,
    /// Episodes whose every answer token was the top prediction, and whose
    /// first answer token was.
    pub exact: usize,
    pub first: usize,
    /// Mean loss on answer tokens.
    pub loss: f64,
    /// Retrieval: the key was among the memory rows (or in the window) at
    /// the first answer token.
    pub found: usize,
    /// Mean pointer weights at the first answer token: vocabulary (null),
    /// window, memory rows, induction column.
    pub weights: [f64; 4],
    /// For [`EpisodeSet::Who`]: exact answers and episodes for " мой друг."
    /// and for " не знаю.".
    pub who: [(usize, usize); 2],
}

/// Each stream states a secret word once and asks for it (in other words)
/// `d` tokens later, with held-out filler around. Every distance runs on
/// fresh streams and memories, one episode per stream, so no answer can be
/// copied from an earlier answer.
#[allow(clippy::too_many_arguments)]
pub fn recall(
    mut model: Model,
    filler: &[u16],
    ep: &Episodes,
    set: EpisodeSet,
    distances: &[usize],
    batch: usize,
    memory: bool,
    precision: KvPrecision,
    seed: u64,
) -> Result<Vec<Recall>> {
    let device = Device::Cpu;
    let t = 243;
    let mut out = Vec::new();
    for (di, &d) in distances.iter().enumerate() {
        let mut rng = snn_memory::rng::SplitMix64::new(seed ^ (di as u64 + 1).wrapping_mul(0x9e37_79b9));
        // (tokens, key span, answer start, answer length) per stream.
        let mut streams = Vec::new();
        for b in 0..batch {
            let (intro, question, answer) = set.sample(ep, &mut rng);
            let mut src = (b * 7919 + di * 104_729) % filler.len().max(1);
            let mut take = |n: usize, out: &mut Vec<u32>| {
                for _ in 0..n {
                    out.push(filler[src] as u32);
                    src = (src + 1) % filler.len();
                }
            };
            let mut seq = Vec::new();
            take(2 * t, &mut seq);
            // The key: where the answer stands in the statement (none for a
            // template answer, which is not in the context).
            let key_len = answer.len() - ep.end_len();
            let body = &answer[..key_len];
            let key = if ep.is_template(&answer) {
                (usize::MAX / 2, usize::MAX / 2)
            } else {
                let at = intro.windows(key_len).rposition(|w| w == body).unwrap_or(0);
                (seq.len() + at, seq.len() + at + key_len)
            };
            seq.extend(&intro);
            take((seq.len() + d).saturating_sub(seq.len() + question.len()), &mut seq);
            seq.extend(&question);
            let start = seq.len();
            seq.extend(&answer);
            take(t + 1, &mut seq);
            streams.push((seq, key, start, answer.len(), ep.is_dont_know(&answer)));
        }
        let len = streams.iter().map(|s| s.0.len()).min().unwrap_or(0);
        let mut runner = Runner::new(model, batch, memory, 300_000, precision, &device)?;
        let mut r = Recall { distance: d, episodes: batch, ..Default::default() };
        let mut hits = vec![(true, false, 0f64, 0usize); batch];
        let mut pos = 0;
        while pos + t < len {
            let (mut x, mut y) = (Vec::new(), Vec::new());
            for (seq, ..) in &streams {
                x.extend_from_slice(&seq[pos..pos + t]);
                y.extend_from_slice(&seq[pos + 1..pos + t + 1]);
            }
            let (o, next) = runner.forward(&x, t, &device)?;
            let needs = streams.iter().any(|&(_, _, s, n, _)| s < pos + t + 1 && s + n > pos + 1);
            if needs {
                let losses = runner.losses(&o, &x, &y)?.to_vec1::<f32>()?;
                let pred = runner.predict(&o, &x)?;
                let (_, _, l) = o.point.dims3()?;
                let point = o.point.flatten_all()?.to_vec1::<f32>()?;
                let blk = runner.model.cfg.block;
                let nb = t / blk;
                for (b, (seq, key, s, n, _)) in streams.iter().enumerate() {
                    for p in *s..s + n {
                        // Target at sequence position p is predicted at p − 1.
                        if p < pos + 1 || p >= pos + t + 1 {
                            continue;
                        }
                        let i = p - 1 - pos;
                        let ok = pred[b * t + i] == seq[p];
                        hits[b].0 &= ok;
                        hits[b].2 += losses[b * t + i] as f64;
                        hits[b].3 += 1;
                        if p == *s {
                            hits[b].1 = ok;
                            let row = &point[(b * t + i) * l..(b * t + i + 1) * l];
                            let m = o.m;
                            // Columns: previous and current window, rows, induction, null.
                            r.weights[0] += row[2 * t + m + 1] as f64 / batch as f64;
                            r.weights[1] += row[..2 * t].iter().map(|&a| a as f64).sum::<f64>() / batch as f64;
                            r.weights[2] += row[2 * t..2 * t + m].iter().map(|&a| a as f64).sum::<f64>() / batch as f64;
                            r.weights[3] += row[2 * t + m] as f64 / batch as f64;
                            // Found: a row (or a window position) whose next token is the key's first.
                            let (k0, k1) = (key.0 as u64, key.1 as u64);
                            let block = i / blk;
                            let rows = &o.far_pos[(b * nb + block) * m..(b * nb + block + 1) * m];
                            // A row copies the key's first token if its next is inside the key.
                            let in_rows = rows.iter().any(|&q| q != u64::MAX && q + 1 >= k0 && q + 1 < k1);
                            let in_window = key.0 + crate::model::LOCAL >= p;
                            // Templates are not in the context: nothing to find.
                            if (in_rows || in_window) && key.0 < usize::MAX / 2 {
                                r.found += 1;
                            }
                        }
                    }
                }
            }
            runner.commit(&x, t, &o, next)?;
            pos += t;
        }
        for (h, st) in hits.iter().zip(&streams) {
            let who = &mut r.who[usize::from(st.4)];
            who.0 += usize::from(h.0 && h.3 > 0);
            who.1 += 1;
            r.exact += usize::from(h.0 && h.3 > 0);
            r.first += usize::from(h.1);
            r.loss += h.2 / h.3.max(1) as f64 / batch as f64;
        }
        out.push(r);
        model = runner.model;
    }
    Ok(out)
}

/// Copy head on held-out text: mean loss without and with it, how often it
/// fired, and how often the copied token was right.
pub struct CopyLoss {
    pub tokens: usize,
    pub loss: f64,
    pub loss_copy: f64,
    pub fired: usize,
    pub fired_right: usize,
}

fn nll(logits: &[f32], target: u32) -> f64 {
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let z: f64 = logits.iter().map(|&l| ((l - mx) as f64).exp()).sum();
    z.ln() - (logits[target as usize] - mx) as f64
}

fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

pub fn copy_loss(engine: &crate::infer::Engine, tokens: &[u32], lambda: f32, precision: KvPrecision) -> CopyLoss {
    let mut s = engine.session_with(300_000, precision);
    let mut r = CopyLoss { tokens: 0, loss: 0.0, loss_copy: 0.0, fired: 0, fired_right: 0 };
    let mut logits = engine.step(&mut s, crate::tokenizer::DOC);
    for &t in tokens {
        let mut copied = logits.clone();
        if let Some((c, _)) = engine.copy(&mut s, &mut copied, lambda, u64::MAX) {
            r.fired += 1;
            r.fired_right += usize::from(c == t);
        }
        r.loss += nll(&logits, t);
        r.loss_copy += nll(&copied, t);
        r.tokens += 1;
        logits = engine.step(&mut s, t);
    }
    r.loss /= r.tokens.max(1) as f64;
    r.loss_copy /= r.tokens.max(1) as f64;
    r
}

/// Exact continuation recall at a distance, per distance:
/// (passages whose 18 continuation tokens were all top-1 without the copy
/// head, with it, number of passages).
pub fn copy_recall(
    engine: &crate::infer::Engine,
    filler: &[u32],
    passages: &[Vec<u32>],
    distances: &[usize],
    lambda: f32,
    precision: KvPrecision,
) -> Vec<(usize, usize, usize, usize)> {
    let per = passages.len() / distances.len();
    let mut s = engine.session_with(300_000, precision);
    let mut src = 0usize;
    // Filler is fed in slices: logits only for the last token of each.
    let mut fill = |s: &mut crate::infer::Session, mut n: usize| {
        while n > 0 {
            let from = src % filler.len();
            let take = n.min(filler.len() - from);
            engine.feed(s, &filler[from..from + take]);
            src += take;
            n -= take;
        }
    };
    engine.step(&mut s, crate::tokenizer::DOC);
    fill(&mut s, 2 * crate::infer::RING);
    // Every passage once, in distance order, remembering where it starts.
    let mut starts = Vec::new();
    for p in passages {
        starts.push(engine.position(&s));
        engine.feed(&mut s, p);
        fill(&mut s, 9);
    }
    let mut out = Vec::new();
    for (di, &d) in distances.iter().enumerate() {
        let (mut plain, mut copied) = (0, 0);
        for (k, p) in passages[di * per..(di + 1) * per].iter().enumerate() {
            let target = starts[di * per + k] + d as u64;
            let now = engine.position(&s);
            fill(&mut s, target.saturating_sub(now) as usize);
            // Cue with the first 9 tokens, then score the next 18.
            let mut logits = engine.feed(&mut s, &p[..9]).expect("logits");
            let (mut ok_plain, mut ok_copy) = (true, true);
            for &t in &p[9..27] {
                let mut c = logits.clone();
                engine.copy(&mut s, &mut c, lambda, u64::MAX);
                ok_plain &= argmax(&logits) == t;
                ok_copy &= argmax(&c) == t;
                logits = engine.step(&mut s, t);
            }
            plain += usize::from(ok_plain);
            copied += usize::from(ok_copy);
        }
        out.push((d, plain, copied, per));
    }
    out
}
