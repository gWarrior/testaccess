//! Evaluation: held-out loss (with and without the SNN memory) and
//! long-range recall ("secret word") accuracy at increasing distances.

use candle_core::{Device, Result};
use snn_memory::KvPrecision;

use crate::data::Episodes;
use crate::model::Model;
use crate::train::{token_losses, Runner};

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
        let l = token_losses(&out.logits, &y)?.to_vec1::<f32>()?;
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
        let losses = token_losses(&out.logits, &y)?.to_vec1::<f32>()?;
        let argmax = out.logits.argmax(2)?.flatten_all()?.to_vec1::<u32>()?;
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

/// Recall result at one distance.
#[derive(Debug, Clone)]
pub struct Recall {
    pub distance: usize,
    /// Episodes whose every answer token was the model's top prediction.
    pub exact: usize,
    pub episodes: usize,
    /// Mean loss on answer tokens.
    pub loss: f64,
}

/// Each of `batch` streams states a key once, then asks for it after each
/// distance in `distances` (ascending), with held-out filler in between.
pub fn recall(
    model: Model,
    filler: &[u16],
    ep: &Episodes,
    distances: &[usize],
    batch: usize,
    memory: bool,
    precision: KvPrecision,
    seed: u64,
) -> Result<Vec<Recall>> {
    let device = Device::Cpu;
    let t = 243;
    let mut rng = snn_memory::rng::SplitMix64::new(seed);
    // Build every stream: filler, key, then filler with questions.
    let lead = 2 * t;
    let mut streams: Vec<(Vec<u32>, Vec<(usize, usize, usize)>)> = Vec::new();
    for b in 0..batch {
        let (intro, question, answer) = ep.sample(&mut rng);
        let mut src = (b * 7919) % filler.len().max(1);
        let mut take = |n: usize, out: &mut Vec<u32>| {
            for _ in 0..n {
                out.push(filler[src] as u32);
                src = (src + 1) % filler.len();
            }
        };
        let mut seq = Vec::new();
        take(lead, &mut seq);
        seq.extend(&intro);
        let key_end = seq.len();
        let mut marks = Vec::new();
        for (di, &d) in distances.iter().enumerate() {
            let target = key_end + d;
            let n = target.saturating_sub(seq.len());
            take(n, &mut seq);
            seq.extend(&question);
            let start = seq.len();
            seq.extend(&answer);
            marks.push((di, start, answer.len()));
        }
        take(t + 1, &mut seq);
        streams.push((seq, marks));
    }
    let len = streams.iter().map(|s| s.0.len()).min().unwrap_or(0);
    let mut runner = Runner::new(model, batch, memory, 300_000, precision, &device)?;
    let mut res: Vec<Recall> =
        distances.iter().map(|&d| Recall { distance: d, exact: 0, episodes: batch, loss: 0.0 }).collect();
    let mut hits: Vec<Vec<(usize, bool, f64)>> = vec![Vec::new(); batch];
    let mut pos = 0;
    while pos + t < len {
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for (seq, _) in &streams {
            x.extend_from_slice(&seq[pos..pos + t]);
            y.extend_from_slice(&seq[pos + 1..pos + t + 1]);
        }
        let (out, next) = runner.forward(&x, t, &device)?;
        let needs = streams.iter().any(|(_, marks)| marks.iter().any(|&(_, s, n)| s < pos + t + 1 && s + n > pos + 1));
        if needs {
            let losses = token_losses(&out.logits, &y)?.to_vec1::<f32>()?;
            let argmax = out.logits.argmax(2)?.flatten_all()?.to_vec1::<u32>()?;
            for (b, (seq, marks)) in streams.iter().enumerate() {
                for &(di, s, n) in marks {
                    for p in s..s + n {
                        // Target at sequence position p is predicted at p − 1.
                        if p >= pos + 1 && p < pos + t + 1 {
                            let i = b * t + (p - 1 - pos);
                            hits[b].push((di, argmax[i] == seq[p], losses[i] as f64));
                        }
                    }
                }
            }
        }
        runner.commit(&x, t, &out, next)?;
        pos += t;
    }
    for per in &hits {
        for (di, r) in res.iter_mut().enumerate() {
            let tokens: Vec<&(usize, bool, f64)> = per.iter().filter(|h| h.0 == di).collect();
            if !tokens.is_empty() && tokens.iter().all(|h| h.1) {
                r.exact += 1;
            }
            r.loss += tokens.iter().map(|h| h.2).sum::<f64>() / tokens.len().max(1) as f64 / batch as f64;
        }
    }
    Ok(res)
}
