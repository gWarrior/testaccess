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
        warmup: arg(args, "--warmup", &d.warmup.to_string()).parse().expect("--warmup"),
        init: args.iter().any(|a| a == "--init").then(|| PathBuf::from(arg(args, "--init", ""))),
        jump: arg(args, "--jump", "on") == "on",
        ..d
    };
    let tokens = load_tokens(&data, "train.bin");
    let tok = load_tokenizer(&data);
    println!("{cfg:?}");
    snn_lm::train::train(&cfg, snn_lm::model::Config::default(), &tokens.tokens, &tok).map_err(std::io::Error::other)
}

fn ngram(args: &[String]) -> std::io::Result<()> {
    let data = PathBuf::from(arg(args, "--data", "/home/user/data/prepared"));
    let n: usize = arg(args, "--tokens", "6000000").parse().expect("--tokens");
    let nv: usize = arg(args, "--val-tokens", "1000000").parse().expect("--val-tokens");
    let train = load_tokens(&data, "train.bin");
    let val = load_tokens(&data, "val.bin");
    let t = Instant::now();
    // Same data a model sees: the first `n / 27` tokens of each of 27 streams.
    let region = train.len() / 27;
    let per = n / 27;
    let mut sample = Vec::with_capacity(n);
    for b in 0..27 {
        sample.extend_from_slice(&train.tokens[b * region..b * region + per]);
    }
    let m = snn_lm::ngram::NGram::train(&sample, 6561);
    let windows: usize = arg(args, "--windows", "0").parse().expect("--windows");
    let (bi, tri) = if windows > 0 {
        // Exactly the positions `eval --windows N` scores.
        m.evaluate_streams(&val.tokens, 27, windows, 243)
    } else {
        m.evaluate(&val.tokens[..nv.min(val.len())])
    };
    println!(
        "n-gram on {} tokens ({:.1}s): bigram {:.4} nats (ppl {:.1}), trigram {:.4} nats (ppl {:.1})",
        sample.len(),
        t.elapsed().as_secs_f64(),
        bi,
        bi.exp(),
        tri,
        tri.exp()
    );
    Ok(())
}

fn kv_precision(args: &[String]) -> snn_memory::KvPrecision {
    match arg(args, "--kv", "ternary").as_str() {
        "f16" => snn_memory::KvPrecision::F16,
        "f32" => snn_memory::KvPrecision::F32,
        _ => snn_memory::KvPrecision::Ternary,
    }
}

fn load_model(dir: &std::path::Path) -> snn_lm::model::Model {
    snn_lm::train::load_model(dir, snn_lm::model::Config::default(), &candle_core::Device::Cpu).expect("checkpoint")
}

fn eval(args: &[String]) -> std::io::Result<()> {
    let data = PathBuf::from(arg(args, "--data", "/home/user/data/prepared"));
    let run = PathBuf::from(arg(args, "--run", "/home/user/data/run"));
    let windows: usize = arg(args, "--windows", "27").parse().expect("--windows");
    let val = load_tokens(&data, "val.bin");
    for memory in [true, false] {
        let t = Instant::now();
        let loss = snn_lm::eval::val_loss(load_model(&run), &val.tokens, 27, windows, memory, kv_precision(args))
            .map_err(std::io::Error::other)?;
        println!(
            "val loss (memory {}): {:.4} nats, ppl {:.1} ({:.0}s)",
            if memory { "on " } else { "off" },
            loss,
            loss.exp(),
            t.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

fn recall(args: &[String]) -> std::io::Result<()> {
    let data = PathBuf::from(arg(args, "--data", "/home/user/data/prepared"));
    let run = PathBuf::from(arg(args, "--run", "/home/user/data/run"));
    let batch: usize = arg(args, "--batch", "9").parse().expect("--batch");
    let distances: Vec<usize> = arg(args, "--distances", "243,2187,19683,177147,300000")
        .split(',')
        .map(|x| x.parse().expect("distance"))
        .collect();
    let memory = arg(args, "--memory", "on") == "on";
    let val = load_tokens(&data, "val.bin");
    let ep = snn_lm::data::Episodes::new(&load_tokenizer(&data));
    let t = Instant::now();
    let res =
        snn_lm::eval::recall(load_model(&run), &val.tokens, &ep, &distances, batch, memory, kv_precision(args), 7)
            .map_err(std::io::Error::other)?;
    for r in res {
        println!(
            "memory {} distance {:>7}: exact {}/{}  answer loss {:.3}",
            if memory { "on " } else { "off" },
            r.distance,
            r.exact,
            r.episodes,
            r.loss
        );
    }
    println!("({:.0}s)", t.elapsed().as_secs_f64());
    Ok(())
}

fn export(args: &[String]) -> std::io::Result<()> {
    let data = PathBuf::from(arg(args, "--data", "/home/user/data/prepared"));
    let run = PathBuf::from(arg(args, "--run", "/home/user/data/run"));
    let out = PathBuf::from(arg(args, "--out", "lm/model"));
    std::fs::create_dir_all(&out)?;
    let mut varmap = candle_nn::VarMap::new();
    let _ = snn_lm::model::Model::new(
        candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &candle_core::Device::Cpu),
        snn_lm::model::Config::default(),
    )
    .map_err(std::io::Error::other)?;
    varmap.load(run.join("model.safetensors")).map_err(std::io::Error::other)?;
    let packed =
        snn_lm::pack::pack_checkpoint(&varmap, snn_lm::model::Config::default()).map_err(std::io::Error::other)?;
    packed.save(std::io::BufWriter::new(std::fs::File::create(out.join("model.snnt"))?))?;
    std::fs::copy(data.join("tokenizer.bpe"), out.join("tokenizer.bpe"))?;
    println!(
        "exported {} ({} bytes)",
        out.join("model.snnt").display(),
        std::fs::metadata(out.join("model.snnt"))?.len()
    );
    Ok(())
}

fn chat(args: &[String]) -> std::io::Result<()> {
    use std::io::{BufRead, Write};
    let dir = PathBuf::from(arg(args, "--model", "lm/model"));
    let mut temperature: f32 = arg(args, "--temp", "0.8").parse().expect("--temp");
    let top_k: usize = arg(args, "--top-k", "27").parse().expect("--top-k");
    let max_new: usize = arg(args, "--max-tokens", "81").parse().expect("--max-tokens");
    let memory_tokens: usize = arg(args, "--memory", "300000").parse().expect("--memory");
    let copy: f32 = arg(args, "--copy", "0.5").parse().expect("--copy");
    let packed =
        snn_lm::pack::PackedModel::load(std::io::BufReader::new(std::fs::File::open(dir.join("model.snnt"))?))?;
    let tok = load_tokenizer(&dir);
    let mut engine = snn_lm::infer::Engine::from_packed(&packed).map_err(std::io::Error::other)?;
    engine.no_pointer = arg(args, "--pointer", "on") == "off";
    let precision = kv_precision(args);
    let mut session = engine.session_with(memory_tokens, precision);
    let seed: u64 = arg(args, "--seed", "0").parse().expect("--seed");
    let seed = if seed == 0 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64)
    } else {
        seed
    };
    let mut rng = snn_memory::rng::SplitMix64::new(seed);
    let mut logits = engine.step(&mut session, snn_lm::tokenizer::DOC);
    let context = arg(args, "--context", "");
    if !context.is_empty() {
        let limit: usize = arg(args, "--context-tokens", "300000").parse().expect("--context-tokens");
        let text = std::fs::read_to_string(&context)?;
        let mut ids = tok.encode(&text);
        ids.truncate(limit);
        let t = Instant::now();
        logits = engine.feed(&mut session, &ids).unwrap_or(logits);
        println!("(в контекст загружено {} токенов из {context} за {:.0}s)", ids.len(), t.elapsed().as_secs_f64());
    }
    println!("snn-lm: тернарная HadamRNN ~8M параметров + SNN-память на {memory_tokens} токенов.");
    println!("Команды: /reset — новый диалог, /temp X — температура, /quit — выход.\n");
    let stdin = std::io::stdin();
    loop {
        print!("вы> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim_end();
        if !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
            // Scripted session: echo the input so the transcript is complete.
            println!("{line}");
        }
        match line.split_whitespace().next() {
            Some("/quit") => break,
            Some("/reset") => {
                session = engine.session_with(memory_tokens, precision);
                logits = engine.step(&mut session, snn_lm::tokenizer::DOC);
                println!("(новый диалог)");
                continue;
            }
            Some("/temp") => {
                temperature = line[5..].trim().parse().unwrap_or(temperature);
                println!("(температура {temperature})");
                continue;
            }
            _ => {}
        }
        // A finished sentence is a turn; an unfinished one is continued in place.
        let finished = line.ends_with(['.', '!', '?', '»', '…', '"']);
        let prompt = if finished { format!("{line}\n") } else { line.to_string() };
        for t in tok.encode(&prompt) {
            logits = engine.step(&mut session, t);
        }
        print!("модель> ");
        let mut out = Vec::new();
        let reply_start = engine.position(&session);
        for _ in 0..max_new {
            if copy > 0.0 {
                engine.copy(&mut session, &mut logits, copy, reply_start);
            }
            let t = snn_lm::infer::sample(&logits, temperature, top_k, &mut rng);
            if t == snn_lm::tokenizer::DOC {
                if out.is_empty() {
                    continue;
                }
                break;
            }
            out.push(t);
            let text = tok.decode(&out);
            logits = engine.step(&mut session, t);
            // A line break ends the answer, but not before it has begun.
            if text.ends_with('\n') && !text.trim().is_empty() {
                break;
            }
        }
        // The token limit can cut a multibyte character in half.
        println!("{}", tok.decode(&out).trim().trim_end_matches('\u{FFFD}'));
        // Close the turn so the model sees a clean line break.
        if !tok.decode(&out).ends_with('\n') {
            for t in tok.encode("\n") {
                logits = engine.step(&mut session, t);
            }
        }
    }
    Ok(())
}

fn ablate(args: &[String]) -> std::io::Result<()> {
    let data = PathBuf::from(arg(args, "--data", "/home/user/data/prepared"));
    let run = PathBuf::from(arg(args, "--run", "/home/user/data/run"));
    let windows: usize = arg(args, "--windows", "27").parse().expect("--windows");
    let val = load_tokens(&data, "val.bin");
    let kv = kv_precision(args);
    let cases: [(&str, bool, bool, bool, bool, bool); 6] = [
        ("full model", true, false, false, false, false),
        ("no SNN memory", false, false, false, false, false),
        ("no pointer (copy)", true, false, false, false, true),
        ("no state between windows", true, true, false, false, false),
        ("no HadamRNN", true, false, true, false, false),
        ("no retention", true, false, false, true, false),
    ];
    for (name, memory, reset, no_hadam, no_ret, no_pointer) in cases {
        let mut model = load_model(&run);
        model.ablation = snn_lm::model::Ablation { no_hadam, no_retention: no_ret, no_pointer };
        let t = Instant::now();
        let (loss, by_pos) = snn_lm::eval::val_loss_detail(model, &val.tokens, 27, windows, memory, kv, reset)
            .map_err(std::io::Error::other)?;
        let curve: Vec<String> = by_pos.iter().map(|l| format!("{l:.2}")).collect();
        println!(
            "{name:26} loss {loss:.4} ppl {:7.1} | by position (27-token buckets): {} ({:.0}s)",
            loss.exp(),
            curve.join(" "),
            t.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

fn reread(args: &[String]) -> std::io::Result<()> {
    let data = PathBuf::from(arg(args, "--data", "/home/user/data/prepared"));
    let run = PathBuf::from(arg(args, "--run", "/home/user/data/run"));
    let file = PathBuf::from(arg(args, "--file", "/home/user/data/taiga/extra/amber.txt"));
    let len: usize = arg(args, "--len", "2187").parse().expect("--len");
    let n: usize = arg(args, "--passages", "9").parse().expect("--passages");
    let tok = load_tokenizer(&data);
    let text = std::fs::read_to_string(&file)?;
    let ids = tok.encode(&text);
    let stride = ids.len() / n;
    let passages: Vec<Vec<u32>> = (0..n).map(|i| ids[i * stride..i * stride + len].to_vec()).collect();
    for memory in [true, false] {
        let t = Instant::now();
        let r = snn_lm::eval::reread(load_model(&run), &passages, memory, kv_precision(args))
            .map_err(std::io::Error::other)?;
        println!(
            "memory {}: 1st reading acc {:.3} loss {:.3} | 2nd reading acc {:.3} loss {:.3} ({:.0}s)",
            if memory { "on " } else { "off" },
            r[0].0,
            r[0].1,
            r[1].0,
            r[1].1,
            t.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

/// Memory alone, without the model: load `--context` into the SNN memory,
/// then answer every `prefix|answer` line of `--probes` with the memory's
/// own continuation. An answer of `-` expects "точно нет" (Absent).
fn probe(args: &[String]) -> std::io::Result<()> {
    use snn_memory::{ContextConfig, ContextMemory, Verdict};
    let dir = PathBuf::from(arg(args, "--model", "lm/model"));
    let tok = load_tokenizer(&dir);
    let text = std::fs::read_to_string(arg(args, "--context", ""))?;
    let probes = std::fs::read_to_string(arg(args, "--probes", ""))?;
    let ids = tok.encode(&text);
    let mut memory = ContextMemory::new(ContextConfig { max_tokens: 300_000, ..Default::default() })
        .map_err(std::io::Error::other)?;
    let t = Instant::now();
    memory.append(&ids).map_err(std::io::Error::other)?;
    memory.flush().map_err(std::io::Error::other)?;
    println!("(в память загружено {} токенов за {:.2}s)", ids.len(), t.elapsed().as_secs_f64());
    let (mut right, mut total) = (0, 0);
    for line in probes.lines().filter(|l| !l.trim().is_empty()) {
        let (prefix, answer) = line.rsplit_once('|').unwrap_or((line, ""));
        // Mid-paragraph, BPE glues the leading space to the first word: try
        // the prefix both as a line start and as a continuation.
        let t = Instant::now();
        let mut prefix_ids = tok.encode(prefix);
        let mut located = memory.locate(&prefix_ids).map_err(std::io::Error::other)?;
        if located.positions.is_empty() {
            let spaced = tok.encode(&format!(" {prefix}"));
            let again = memory.locate(&spaced).map_err(std::io::Error::other)?;
            if !again.positions.is_empty() || again.verdict != Verdict::Absent {
                (prefix_ids, located) = (spaced, again);
            }
        }
        let cont = memory.continuation(&prefix_ids, 9).map_err(std::io::Error::other)?;
        let us = t.elapsed().as_micros();
        let (said, ok) = match (&cont, answer.trim()) {
            (_, "-") => (String::new(), located.verdict == Verdict::Absent),
            (Some((_, c)), a) => {
                let s = tok.decode(c);
                let ok = s.trim_start().starts_with(a);
                (s, ok)
            }
            (None, _) => (String::new(), false),
        };
        total += 1;
        right += usize::from(ok);
        let verdict = match located.verdict {
            Verdict::Known => "уверен",
            Verdict::Unknown => "не знаю",
            Verdict::Absent => "точно нет",
        };
        let said = said.replace('\n', "⏎");
        println!(
            "{} {prefix} → «{said}» [{verdict}, {} вхожд., {us} мкс]",
            if ok { "✓" } else { "✗" },
            located.positions.len()
        );
    }
    println!("верно {right}/{total}");
    Ok(())
}

fn copyeval(args: &[String]) -> std::io::Result<()> {
    let dir = PathBuf::from(arg(args, "--model", "lm/model"));
    let data = PathBuf::from(arg(args, "--data", "/home/user/data/prepared"));
    let lambda: f32 = arg(args, "--copy", "0.5").parse().expect("--copy");
    let n: usize = arg(args, "--tokens", "19683").parse().expect("--tokens");
    let distances: Vec<usize> = arg(args, "--distances", "243,2187,19683,177147,300000")
        .split(',')
        .map(|d| d.parse().expect("--distances"))
        .collect();
    let packed =
        snn_lm::pack::PackedModel::load(std::io::BufReader::new(std::fs::File::open(dir.join("model.snnt"))?))?;
    let mut engine = snn_lm::infer::Engine::from_packed(&packed).map_err(std::io::Error::other)?;
    // "Model alone" is the model with its own pointer head unless --pointer off.
    engine.no_pointer = arg(args, "--pointer", "on") == "off";
    println!("pointer head: {}", if engine.no_pointer { "off" } else { "on" });
    let val = snn_lm::data::TokenFile::load(&data.join("val.bin"))?;
    let val: Vec<u32> = val.tokens.iter().map(|&t| t as u32).collect();
    let precision = kv_precision(args);
    let t = Instant::now();
    let r = snn_lm::eval::copy_loss(&engine, &val[..n], lambda, precision);
    println!(
        "held-out {} tokens: loss {:.4} → with copy head {:.4}; fired {} times ({:.1}%), right {} ({:.1}%) ({:.0}s)",
        r.tokens,
        r.loss,
        r.loss_copy,
        r.fired,
        100.0 * r.fired as f64 / r.tokens.max(1) as f64,
        r.fired_right,
        100.0 * r.fired_right as f64 / r.fired.max(1) as f64,
        t.elapsed().as_secs_f64()
    );
    // Passages and filler from disjoint parts of the held-out text.
    let half = val.len() / 2;
    let passages: Vec<Vec<u32>> = (0..9 * distances.len()).map(|i| val[half + i * 2187..][..27].to_vec()).collect();
    let t = Instant::now();
    for (d, plain, copied, total) in
        snn_lm::eval::copy_recall(&engine, &val[n..half], &passages, &distances, lambda, precision)
    {
        println!(
            "distance {d:>7}: exact 18-token continuation {plain}/{total} model alone, {copied}/{total} with copy head"
        );
    }
    println!("({:.0}s)", t.elapsed().as_secs_f64());
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("prepare") => prepare(&args),
        Some("train") => train(&args),
        Some("ngram") => ngram(&args),
        Some("eval") => eval(&args),
        Some("recall") => recall(&args),
        Some("export") => export(&args),
        Some("chat") => chat(&args),
        Some("ablate") => ablate(&args),
        Some("reread") => reread(&args),
        Some("probe") => probe(&args),
        Some("copyeval") => copyeval(&args),
        _ => {
            eprintln!("usage: snn-lm prepare [--src DIR] [--out DIR] [--vocab N] [--sample-mb N]");
            eprintln!("       snn-lm train [--steps N] [--hours H] [--memory on|off] [--out DIR] [--lr X] [--batch N]");
            eprintln!(
                "       snn-lm ngram [--tokens N] | eval [--run DIR] | recall [--distances a,b,..] [--memory on|off]"
            );
            eprintln!(
                "       snn-lm export [--run DIR] [--out lm/model] | chat [--model lm/model] [--temp X] [--memory N] [--kv ternary|f16] [--copy λ]"
            );
            eprintln!("       snn-lm probe --context FILE --probes FILE [--model lm/model]  (memory alone, lines prefix|answer)");
            eprintln!("       snn-lm copyeval [--copy λ] [--pointer on|off] [--distances a,b,..] [--tokens N]");
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
