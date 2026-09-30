//! The README walkthrough as a runnable program:
//! `cargo run --release --example quickstart`

use snn_memory::{
    CodeEncoder, ContextConfig, ContextMemory, Input, KvConfig, LearnOptions, MemoryConfig, Probe, RecallOptions,
    SnapshotScope, SnnMemory, Verdict,
};

fn main() -> Result<(), snn_memory::MemoryError> {
    // --- 1. The core memory: Learn → Recall → Forget → Recall -------------
    let mut mem: SnnMemory<String> = SnnMemory::new(CodeEncoder::new(19_683), MemoryConfig::default())?;
    let apple: Vec<u32> = (0..27).map(|i| i * 729).collect(); // a sparse spike code
    let id = mem.learn(Input::Code(&apple), LearnOptions::new().payload("red apple".into()))?;

    let r = mem.recall(Input::Code(&apple[..9]), &RecallOptions::default())?; // a third of the pattern
    assert_eq!((r.verdict, r.id()), (Verdict::Known, Some(id)));
    println!("recalled {:?} from 9/27 neurons", r.best().and_then(|h| h.payload.as_deref()));

    mem.forget(id);
    let r = mem.recall(Input::Code(&apple), &RecallOptions::default())?;
    assert_eq!(r.verdict, Verdict::Absent); // "точно нет"
    println!("after forget: {:?} ({:?})", r.verdict, r.basis);

    // Negative knowledge: recalling it answers "definitely not".
    let rumour: Vec<u32> = (0..27).map(|i| i * 729 + 1).collect();
    mem.learn(Input::Code(&rumour), LearnOptions::new().negative().payload("false rumour".into()))?;
    println!("rumour: {:?}", mem.recall(Input::Code(&rumour), &RecallOptions::default())?.verdict);

    // Carry what matters into the next session.
    let snapshot = mem.save(SnapshotScope::LongTerm);
    println!("long-term snapshot: {} bytes", snapshot.len());

    // --- 2. Context memory for a recurrent LLM ---------------------------
    let (dk, dv) = (27, 27);
    let cfg = ContextConfig { kv: Some(KvConfig::new(dk, dv)), ..Default::default() };
    let mut ctx = ContextMemory::new(cfg)?;
    let tokens: Vec<u32> = (0..2_187).map(|i| (i * 7 % 6_561) as u32).collect();
    // Per-token attention keys/values as produced by the model.
    let keys: Vec<f32> = (0..tokens.len() * dk).map(|i| ((i * 31 % 97) as f32 - 48.0) / 48.0).collect();
    let values = keys.clone();
    ctx.append_kv(&tokens, &keys, &values)?;

    // Exact associative recall ("what followed this the last time?").
    let (pos, next) = ctx.continuation(&tokens[1_000..1_009], 5)?.expect("in context");
    println!("continuation at {pos}: {next:?}");
    assert_eq!(next, tokens[1_009..1_014]);

    // Ternary membership with a proof for "no".
    println!("contains known fragment: {:?}", ctx.contains(&tokens[500..509])?);
    println!("contains unseen fragment: {:?}", ctx.contains(&[1, 2, 3, 4, 5])?);

    // Cross-attention read head: the SNN picks chunks, attention reads them.
    let query = &keys[1_004 * dk..1_005 * dk];
    let out = ctx.read(Probe::Tokens(&tokens[1_000..1_009]), query, 9)?;
    println!(
        "read: verdict {:?}, gate {:.2}, attended {} tokens, output dim {}",
        out.verdict,
        out.gate,
        out.attended.tokens,
        out.attended.output.len()
    );
    Ok(())
}
