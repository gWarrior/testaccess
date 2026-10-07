//! Small dense workload for profiling (e.g. under `valgrind --tool=callgrind`).

use std::time::Instant;

use snn_memory::rng::SplitMix64;
use snn_memory::{FlyHashEncoder, Input, MemoryConfig, RecallOptions, SnnMemory};

fn unit(rng: &mut SplitMix64, d: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..d).map(|_| rng.normal() as f32).collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.into_iter().map(|x| x / n).collect()
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(19_683);
    let d = 1024;
    let mut rng = SplitMix64::new(9);
    let keys: Vec<Vec<f32>> = (0..n).map(|_| unit(&mut rng, d)).collect();
    let enc = FlyHashEncoder::new(d, 19_683, 81, 9, 7).unwrap();
    let cfg = MemoryConfig { max_ensemble: 81, dedupe_threshold: None, consolidate_after: None, ..Default::default() };
    let mut mem: SnnMemory<()> = SnnMemory::new(enc, cfg).unwrap();

    let t = Instant::now();
    let inputs: Vec<Input> = keys.iter().map(|k| Input::Dense(k)).collect();
    for batch in inputs.chunks(6561) {
        mem.learn_batch(batch, None, &Default::default()).unwrap();
    }
    println!("learn {n}: {:.2} s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let codes: Vec<Vec<u32>> = keys.iter().take(243).map(|k| mem.encode(Input::Dense(k)).unwrap()).collect();
    println!("encode 243: {:.1} µs each", t.elapsed().as_secs_f64() * 1e6 / 243.0);

    let opts = RecallOptions { facilitate: false, ..Default::default() };
    let t = Instant::now();
    let mut ok = 0;
    for c in &codes {
        ok += mem.recall(Input::Code(c), &opts).unwrap().is_known() as usize;
    }
    println!("recall 243 (pre-encoded): {:.1} µs each, {ok} known", t.elapsed().as_secs_f64() * 1e6 / 243.0);
}
