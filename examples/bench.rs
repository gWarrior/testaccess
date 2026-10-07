//! Benchmarks at the reference context size (300k tokens).
//!
//! `cargo run --release --example bench [lexical|dense|read|all]`

use std::time::Instant;

use snn_memory::rng::{SplitMix64, Zipf};
use snn_memory::{
    ContextConfig, ContextMemory, FlyHashEncoder, Input, KvConfig, KvPrecision, MemoryConfig, Probe, RecallOptions,
    SnnMemory, Verdict,
};

const CONTEXT: usize = 300_000;
const VOCAB: usize = 59_049;

struct Lat(Vec<f64>);

impl Lat {
    fn new() -> Self {
        Self(Vec::new())
    }
    fn time<T>(&mut self, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let r = f();
        self.0.push(t.elapsed().as_secs_f64() * 1e6);
        r
    }
    fn summary(&mut self) -> String {
        self.0.sort_by(f64::total_cmp);
        let p = |q: f64| self.0[((self.0.len() - 1) as f64 * q).round() as usize];
        format!("p50 {:>7.1} µs | p99 {:>7.1} µs | max {:>7.1} µs", p(0.5), p(0.99), p(1.0))
    }
}

fn mb(bytes: usize) -> String {
    format!("{:.1} MB", bytes as f64 / 1e6)
}

fn unit(rng: &mut SplitMix64, d: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..d).map(|_| rng.normal() as f32).collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.into_iter().map(|x| x / n).collect()
}

fn noisy(rng: &mut SplitMix64, x: &[f32], cos: f32) -> Vec<f32> {
    let e = unit(rng, x.len());
    let s = (1.0 - cos * cos).sqrt();
    x.iter().zip(e).map(|(a, b)| cos * a + s * b).collect()
}

/// 300k Zipf tokens with 729 key→value facts; exact recall at full size.
fn lexical() {
    println!("\n## Lexical context memory — {CONTEXT} tokens\n");
    let mut rng = SplitMix64::new(1);
    let zipf = Zipf::new(VOCAB, 1.1);
    let n_facts = 729;
    let filler = CONTEXT / n_facts - 10;
    let mut stream = Vec::with_capacity(CONTEXT + 1000);
    let mut facts = Vec::new();
    for _ in 0..n_facts {
        stream.extend((0..filler).map(|_| zipf.sample(&mut rng) as u32));
        let key: Vec<u32> = (0..6).map(|_| rng.below(VOCAB as u64) as u32).collect();
        let value: Vec<u32> = (0..4).map(|_| rng.below(VOCAB as u64) as u32).collect();
        stream.extend(&key);
        stream.extend(&value);
        facts.push((key, value));
    }
    while stream.len() < CONTEXT {
        stream.push(zipf.sample(&mut rng) as u32);
    }

    let mut ctx = ContextMemory::new(ContextConfig { max_tokens: CONTEXT, ..Default::default() }).unwrap();
    let t = Instant::now();
    for piece in stream.chunks(729) {
        ctx.append(piece).unwrap();
    }
    let secs = t.elapsed().as_secs_f64();
    let s = ctx.stats();
    println!("ingest            : {:.2} s, {:.0} tokens/s, {} chunks", secs, CONTEXT as f64 / secs, s.chunks);
    println!(
        "memory            : SNN {} ({} synapses), tokens {}, census {} n-grams",
        mb(s.lexical.approx_bytes),
        s.lexical.synapses,
        mb(s.token_bytes),
        s.census_entries
    );

    let (mut ok, mut lat) = (0, Lat::new());
    for (key, value) in &facts {
        let got = lat.time(|| ctx.continuation(key, 4).unwrap());
        ok += got.is_some_and(|(_, v)| &v == value) as usize;
    }
    println!("fact recall       : {ok}/{n_facts} exact | {}", lat.summary());

    let (mut ok, mut lat) = (0, Lat::new());
    for _ in 0..729 {
        let at = rng.below((CONTEXT - 9) as u64) as usize;
        let frag = &stream[at..at + 9];
        let found = lat.time(|| ctx.locate(frag).unwrap());
        ok += found.positions.contains(&(at as u64)) as usize;
    }
    println!("locate (9 tokens) : {ok}/729 found | {}", lat.summary());

    let (mut absent, mut lat) = (0, Lat::new());
    for _ in 0..729 {
        let frag: Vec<u32> = (0..9).map(|_| rng.below(VOCAB as u64) as u32).collect();
        absent += (lat.time(|| ctx.contains(&frag).unwrap()) == Verdict::Absent) as usize;
    }
    println!("absent fragments  : {absent}/729 answered 'definitely not' | {}", lat.summary());

    let mut lat = Lat::new();
    for _ in 0..729 {
        let at = rng.below((CONTEXT - 9) as u64) as usize;
        let frag = stream[at..at + 9].to_vec();
        lat.time(|| ctx.retrieve(Probe::Tokens(&frag), 9).unwrap());
    }
    println!("SNN retrieve only : {}", lat.summary());
}

/// 300k memories keyed by dense vectors (hidden-state stand-ins).
fn dense() {
    let (d, n) = (1024, CONTEXT);
    println!("\n## Dense keys — {n} memories, d = {d}\n");
    let mut rng = SplitMix64::new(2);
    let keys: Vec<Vec<f32>> = (0..n).map(|_| unit(&mut rng, d)).collect();
    for fan_in in [9usize, 27] {
        let enc = FlyHashEncoder::new(d, 19_683, 81, fan_in, 7).unwrap();
        let cfg =
            MemoryConfig { max_ensemble: 81, dedupe_threshold: None, consolidate_after: None, ..Default::default() };
        let mut mem: SnnMemory<()> = SnnMemory::new(enc, cfg).unwrap();

        let t = Instant::now();
        let mut ids = Vec::with_capacity(n);
        for batch in keys.chunks(6561) {
            let inputs: Vec<Input> = batch.iter().map(|k| Input::Dense(k)).collect();
            ids.extend(mem.learn_batch(&inputs, None, &Default::default()).unwrap());
        }
        let secs = t.elapsed().as_secs_f64();
        let s = mem.stats();
        println!(
            "fan-in {fan_in}: learn {:.2} s ({:.0}/s), {} synapses, {}",
            secs,
            n as f64 / secs,
            s.synapses,
            mb(s.approx_bytes)
        );

        let opts = RecallOptions { facilitate: false, ..Default::default() };
        for cos in [0.9f32, 0.8] {
            let (mut ok, mut wrong, mut lat) = (0, 0, Lat::new());
            for _ in 0..729 {
                let i = rng.below(n as u64) as usize;
                let q = noisy(&mut rng, &keys[i], cos);
                match lat.time(|| mem.recall(Input::Dense(&q), &opts).unwrap()).id() {
                    Some(id) if id == ids[i] => ok += 1,
                    Some(_) => wrong += 1,
                    None => {}
                }
            }
            println!("  recall cos {cos}: {ok}/729 correct, {wrong} wrong | {}", lat.summary());
        }
        let mut verdicts = [0usize; 3];
        for _ in 0..729 {
            let q = unit(&mut rng, d);
            let r = mem.recall(Input::Dense(&q), &opts).unwrap();
            verdicts[(1 - r.verdict.trit()) as usize] += 1;
        }
        println!("  random queries : known {} / unknown {} / absent {}", verdicts[0], verdicts[1], verdicts[2]);

        let t = Instant::now();
        for &id in ids.iter().step_by(3) {
            mem.forget(id);
        }
        let forget_us = t.elapsed().as_secs_f64() * 1e6 / (n / 3) as f64;
        let (mut kept, mut gone, mut n_kept, mut n_gone) = (0, 0, 0, 0);
        for j in 0..729 {
            // Every third memory was forgotten; sample both kinds evenly.
            let i = (j / 3) * (n / 243) + j % 3;
            let r = mem.recall(Input::Code(&mem.encode(Input::Dense(&keys[i])).unwrap()), &opts).unwrap();
            if i % 3 == 0 {
                gone += r.is_miss() as usize;
                n_gone += 1;
            } else {
                kept += (r.id() == Some(ids[i])) as usize;
                n_kept += 1;
            }
        }
        println!("  forget 1/3     : {forget_us:.2} µs each; forgotten silent {gone}/{n_gone}, others intact {kept}/{n_kept}");
        let t = Instant::now();
        mem.maintain();
        println!("  cleanup        : {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        mem.reset_fast_memory();
        println!("  reset fast     : {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    }
}

/// Cross-attention read head over a 300k-token K/V window.
fn read() {
    let (dk, dv) = (81, 81);
    println!("\n## Read head — {CONTEXT} tokens, d_k = d_v = {dk}\n");
    for precision in [KvPrecision::F16, KvPrecision::Ternary] {
        let mut rng = SplitMix64::new(3);
        let zipf = Zipf::new(VOCAB, 1.1);
        let cfg = ContextConfig {
            max_tokens: CONTEXT,
            kv: Some(KvConfig { precision, ..KvConfig::new(dk, dv) }),
            ..Default::default()
        };
        let mut ctx = ContextMemory::new(cfg).unwrap();
        let mut stream = Vec::with_capacity(CONTEXT);
        let mut keys = Vec::with_capacity(CONTEXT * dk);
        let t = Instant::now();
        for _ in 0..CONTEXT / 729 + 1 {
            let tokens: Vec<u32> = (0..729).map(|_| zipf.sample(&mut rng) as u32).collect();
            let k: Vec<f32> = (0..729).flat_map(|_| unit(&mut rng, dk)).collect();
            let v: Vec<f32> = (0..729 * dv).map(|_| rng.normal() as f32).collect();
            ctx.append_kv(&tokens, &k, &v).unwrap();
            stream.extend(tokens);
            keys.extend(k);
        }
        let s = ctx.stats();
        println!(
            "{precision:?}: ingest {:.2} s, K/V {}, lexical SNN {}, semantic SNN {}",
            t.elapsed().as_secs_f64(),
            mb(s.kv_bytes),
            mb(s.lexical.approx_bytes),
            mb(s.semantic.as_ref().map_or(0, |x| x.approx_bytes))
        );
        let (lo, hi) = ctx.window();
        let (mut ok, mut lat) = (0, Lat::new());
        for _ in 0..729 {
            let at = lo as usize + 9 + rng.below(hi - lo - 18) as usize;
            let probe = stream[at - 4..at + 5].to_vec();
            let q: Vec<f32> = keys[at * dk..(at + 1) * dk].iter().map(|x| x * 81.0).collect();
            let out = lat.time(|| ctx.read(Probe::Tokens(&probe), &q, 9).unwrap());
            ok += (out.attended.argmax.map(|a| a.0) == Some(at as u64)) as usize;
        }
        println!("  read (probe + attention): argmax exact {ok}/729 | {}", lat.summary());
    }
}

fn main() {
    let what = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    println!("# snn-memory benchmark ({} threads)", rayon::current_num_threads());
    if what == "lexical" || what == "all" {
        lexical();
    }
    if what == "dense" || what == "all" {
        dense();
    }
    if what == "read" || what == "all" {
        read();
    }
}
