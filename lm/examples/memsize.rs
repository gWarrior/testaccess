//! Resident memory of one stream's training memory filled to a window.
use snn_memory::{ContextConfig, ContextMemory, KvConfig, KvPrecision};

fn rss_mb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/statm").unwrap();
    s.split_whitespace().nth(1).unwrap().parse::<f64>().unwrap() * 4096.0 / 1e6
}

fn main() {
    let window: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(177_147);
    let before = rss_mb();
    let cfg = ContextConfig {
        max_tokens: window,
        kv: Some(KvConfig { precision: KvPrecision::Trit2, ..KvConfig::new(81, 81) }),
        ..Default::default()
    };
    let mut mem = ContextMemory::new(cfg).unwrap();
    let mut rng = snn_memory::rng::SplitMix64::new(1);
    let zipf = snn_memory::rng::Zipf::new(6561, 1.1);
    let mut ind = snn_lm::train::Induction::new(window);
    let t0 = std::time::Instant::now();
    let mut xs = Vec::new();
    for _ in 0..(window + 243 * 9) / 243 {
        let x: Vec<u32> = (0..243).map(|_| zipf.sample(&mut rng) as u32).collect();
        let k: Vec<f32> = (0..243 * 81).map(|_| (rng.below(9) as f32 - 4.0) * 0.5).collect();
        mem.append_kv(&x, &k, &k).unwrap();
        xs.push(x);
    }
    let after_mem = rss_mb();
    for x in &xs {
        ind.window(x);
    }
    println!("memory {:.0} MB, induction index {:.0} MB", after_mem - before, rss_mb() - after_mem);
    let st = mem.stats();
    println!(
        "window {window}: {:.0} MB per stream ({:.1}s); stats: {st:?}",
        rss_mb() - before,
        t0.elapsed().as_secs_f64()
    );
}
