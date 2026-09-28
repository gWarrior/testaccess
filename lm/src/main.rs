//! `snn-lm` command line: data preparation, training and evaluation.

use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;
use snn_lm::data::{read_parquet_texts, write_tokens};
use snn_lm::tokenizer::Tokenizer;

fn arg(args: &[String], name: &str, default: &str) -> String {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.to_string())
}

fn prepare(args: &[String]) -> std::io::Result<()> {
    let src = PathBuf::from(arg(args, "--src", "/home/user/data/taiga"));
    let out = PathBuf::from(arg(args, "--out", "/home/user/data/prepared"));
    let vocab: usize = arg(args, "--vocab", "6561").parse().expect("--vocab");
    let sample_mb: usize = arg(args, "--sample-mb", "64").parse().expect("--sample-mb");
    std::fs::create_dir_all(&out)?;

    let t = Instant::now();
    let mut shards: Vec<PathBuf> = std::fs::read_dir(&src)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "parquet"))
        .collect();
    shards.sort();
    let mut docs: Vec<String> = shards
        .par_iter()
        .map(|p| read_parquet_texts(p).expect("readable parquet"))
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect();
    // Extra user-supplied texts (plain UTF-8 files) join the corpus.
    if let Ok(extra) = std::fs::read_dir(src.join("extra")) {
        for e in extra.flatten() {
            if let Ok(text) = std::fs::read_to_string(e.path()) {
                docs.push(text);
            }
        }
    }
    let bytes: usize = docs.iter().map(String::len).sum();
    println!("read {} docs, {:.2} GB of text in {:.1}s", docs.len(), bytes as f64 / 1e9, t.elapsed().as_secs_f64());

    let t = Instant::now();
    let mut sample = String::new();
    for d in docs.iter().step_by(7) {
        if sample.len() > sample_mb << 20 {
            break;
        }
        sample.push_str(d);
        sample.push('\n');
    }
    let tok = Tokenizer::train(&sample, vocab);
    tok.save(std::fs::File::create(out.join("tokenizer.bpe"))?)?;
    println!(
        "tokenizer: {} entries trained on {} MB in {:.1}s",
        tok.vocab_size(),
        sample.len() >> 20,
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let encoded: Vec<Vec<u32>> = docs.par_iter().map(|d| tok.encode(d)).collect();
    let (val, train): (Vec<(usize, Vec<u32>)>, Vec<(usize, Vec<u32>)>) =
        encoded.into_iter().enumerate().partition(|(i, _)| i % 100 == 99);
    let strip = |v: Vec<(usize, Vec<u32>)>| v.into_iter().map(|(_, d)| d).collect::<Vec<_>>();
    let (train, val) = (strip(train), strip(val));
    let n_train = write_tokens(&out.join("train.bin"), &train)?;
    let n_val = write_tokens(&out.join("val.bin"), &val)?;
    let mut lens: Vec<usize> = train.iter().map(Vec::len).collect();
    lens.sort_unstable();
    let pct = |q: f64| lens[((lens.len() - 1) as f64 * q) as usize];
    println!(
        "tokens: train {n_train}, val {n_val} ({:.2} bytes/token) in {:.1}s",
        bytes as f64 / (n_train + n_val) as f64,
        t.elapsed().as_secs_f64()
    );
    println!("doc length in tokens: p50 {} p90 {} p99 {} max {}", pct(0.5), pct(0.9), pct(0.99), lens[lens.len() - 1]);
    Ok(())
}

fn load_tokens(dir: &std::path::Path, name: &str) -> snn_lm::data::TokenFile {
    snn_lm::data::TokenFile::load(&dir.join(name)).expect("prepared token file")
}

fn load_tokenizer(dir: &std::path::Path) -> Tokenizer {
    Tokenizer::load(std::fs::File::open(dir.join("tokenizer.bpe")).expect("tokenizer")).expect("tokenizer")
}

fn train(args: &[String]) -> std::io::Result<()> {
    use snn_lm::train::TrainConfig;
    let data = PathBuf::from(arg(args, "--data", "/home/user/data/prepared"));
    let d = TrainConfig::default();
    let cfg = TrainConfig {
        steps: arg(args, "--steps", &d.steps.to_string()).parse().expect("--steps"),
        lr: arg(args, "--lr", &d.lr.to_string()).parse().expect("--lr"),
        batch: arg(args, "--batch", &d.batch.to_string()).parse().expect("--batch"),
        memory: arg(args, "--memory", "on") == "on",
        out: PathBuf::from(arg(args, "--out", d.out.to_str().unwrap())),
        time_limit: (arg(args, "--hours", "0").parse::<f64>().expect("--hours") * 3600.0) as u64,
        log_every: arg(args, "--log-every", &d.log_every.to_string()).parse().expect("--log-every"),
        ..d
    };
    let tokens = load_tokens(&data, "train.bin");
    let tok = load_tokenizer(&data);
    println!("{cfg:?}");
    snn_lm::train::train(&cfg, snn_lm::model::Config::default(), &tokens.tokens, &tok).map_err(std::io::Error::other)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("prepare") => prepare(&args),
        Some("train") => train(&args),
        _ => {
            eprintln!("usage: snn-lm prepare [--src DIR] [--out DIR] [--vocab N] [--sample-mb N]");
            eprintln!("       snn-lm train [--steps N] [--hours H] [--memory on|off] [--out DIR] [--lr X] [--batch N]");
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
