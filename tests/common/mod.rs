#![allow(dead_code)]

use snn_memory::rng::SplitMix64;
use snn_memory::{CodeEncoder, FlyHashEncoder, ManualClock, MemoryConfig, SnnMemory};

pub const N_NEURONS: u32 = 16_384;
pub const K: usize = 48;

/// Memory over raw spike codes with a manual clock.
pub fn code_memory(cfg: MemoryConfig) -> (SnnMemory<&'static str>, ManualClock) {
    let clock = ManualClock::new(0.0);
    let mem = SnnMemory::new(CodeEncoder::new(N_NEURONS), cfg).unwrap().with_clock(clock.clone());
    (mem, clock)
}

/// Memory over dense vectors (hidden-state stand-ins).
pub fn dense_memory(dim: usize, cfg: MemoryConfig) -> SnnMemory<u32> {
    let enc = FlyHashEncoder::new(dim, N_NEURONS, K, 32, 7).unwrap();
    SnnMemory::new(enc, MemoryConfig { max_ensemble: K, ..cfg }).unwrap()
}

pub fn random_code(rng: &mut SplitMix64) -> Vec<u32> {
    let mut c = rng.sample_distinct(N_NEURONS, K);
    c.sort_unstable();
    c
}

/// Random subset with `keep` of the neurons of `code`.
pub fn partial(rng: &mut SplitMix64, code: &[u32], keep: usize) -> Vec<u32> {
    let idx = rng.sample_distinct(code.len() as u32, keep);
    let mut out: Vec<u32> = idx.into_iter().map(|i| code[i as usize]).collect();
    out.sort_unstable();
    out
}

pub fn random_vec(rng: &mut SplitMix64, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.normal() as f32).collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.into_iter().map(|x| x / n).collect()
}

/// `x` mixed with fresh noise so that cos(x, result) ≈ `cos`.
pub fn noisy(rng: &mut SplitMix64, x: &[f32], cos: f32) -> Vec<f32> {
    let noise = random_vec(rng, x.len());
    let s = (1.0 - cos * cos).sqrt();
    x.iter().zip(noise).map(|(a, b)| cos * a + s * b).collect()
}
