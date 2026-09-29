//! Per-module forward/backward timing at training shapes.
use std::time::Instant;

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use snn_lm::layers::{HadamCell, Mlp, QTensor, Retention, LEVELS3};

fn time<F: FnMut() -> Tensor>(name: &str, mut f: F) {
    for _ in 0..2 {
        let t = Instant::now();
        let y = f();
        let t1 = t.elapsed();
        let _ = y.sqr().unwrap().mean_all().unwrap().backward().unwrap();
        println!(
            "{name:10} fwd {:6.0} ms  bwd {:6.0} ms",
            t1.as_secs_f64() * 1e3,
            (t.elapsed() - t1).as_secs_f64() * 1e3
        );
    }
}

fn main() {
    let dev = Device::Cpu;
    let (b, t, d) = (27, 243, 256);
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let x = Var::randn(0f32, 1.0, (b, t, d), &dev).unwrap();
    let cell = HadamCell::new(vb.pp("c"), d).unwrap();
    let ret = Retention::new(vb.pp("r"), d, 4).unwrap();
    let mlp = Mlp::new(vb.pp("m"), d, 2187).unwrap();
    let emb = Var::randn(0f32, 0.06, (6561, d), &dev).unwrap();
    let qemb = QTensor::new(&vb, "emb", 6561, d, 0.06, LEVELS3).unwrap();
    let h0 = Tensor::zeros((b, d), DType::F32, &dev).unwrap();
    let s0 = Tensor::zeros((b, 4, 64, 64), DType::F32, &dev).unwrap();
    time("hadam", || cell.forward(x.as_tensor(), &h0).unwrap().0);
    time("retention", || ret.forward(x.as_tensor(), &s0).unwrap().0);
    time("mlp", || mlp.forward(x.as_tensor()).unwrap());
    time("logits", || snn_lm::layers::matmul_2d(x.as_tensor(), &qemb.q().unwrap().t().unwrap()).unwrap());
    time("matmul", || x.as_tensor().reshape((b * t, d)).unwrap().matmul(&emb.as_tensor().t().unwrap()).unwrap());
    let ids = Tensor::from_vec((0..(b * t) as u32).map(|i| i % 6561).collect::<Vec<_>>(), b * t, &dev).unwrap();
    time("fused ce", || {
        let l = x.as_tensor().reshape((b * t, d)).unwrap().matmul(&emb.as_tensor().t().unwrap()).unwrap();
        let t = std::sync::Arc::new(ids.to_vec1::<u32>().unwrap());
        l.apply_op1(snn_lm::layers::CrossEntropy { targets: t }).unwrap()
    });
}
