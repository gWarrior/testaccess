//! Speed of the engine's ternary matrix kernels (shift-add vs AVX-512 VNNI):
//! `cargo run --release -p snn-lm --features vnni --example kernels`.
use std::time::Instant;

use snn_lm::infer::ShiftLinear;

fn main() {
    let mut rng = snn_memory::rng::SplitMix64::new(1);
    for (name, rows, cols, top) in
        [("mlp w1 1458×256", 1458, 256, 4i8), ("mlp w2 256×1458", 256, 1458, 4), ("logits 6561×256", 6561, 256, 13)]
    {
        let levels: Vec<i8> = (0..rows * cols).map(|_| rng.below(2 * top as u64 + 1) as i8 - top).collect();
        let exps = vec![-6i8; rows];
        let m = ShiftLinear::new(rows, cols, &levels, &exps);
        let x: Vec<f32> = (0..cols).map(|_| rng.next_f64() as f32 - 0.5).collect();
        let reps = 200;
        let time = |f: &dyn Fn() -> Vec<f32>| {
            let t = Instant::now();
            let mut s = 0f32;
            for _ in 0..reps {
                s += f()[0];
            }
            (t.elapsed().as_secs_f64() / reps as f64 * 1e6, s)
        };
        let (shift, _) = time(&|| m.apply_shift_add(&x));
        #[cfg(feature = "vnni")]
        {
            if snn_lm::infer::vnni::available() {
                let (v, _) = time(&|| m.apply_vnni(&x));
                println!("{name:18} shift-add {shift:8.1} µs   vnni {v:8.1} µs   ×{:.1}", shift / v);
                continue;
            }
        }
        println!("{name:18} shift-add {shift:8.1} µs   (build with --features vnni on an AVX-512 VNNI CPU)");
    }
}
