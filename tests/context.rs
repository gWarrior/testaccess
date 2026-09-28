//! ContextMemory: exact long-context memory for a recurrent LLM.

use snn_memory::rng::{SplitMix64, Zipf};
use snn_memory::{ContextConfig, ContextMemory, KvConfig, KvPrecision, MemoryError, Probe, Verdict};

const VOCAB: usize = 6561;

/// Zipf "text" over a 3^8 vocabulary.
fn text(rng: &mut SplitMix64, n: usize) -> Vec<u32> {
    let z = Zipf::new(VOCAB, 1.1);
    (0..n).map(|_| z.sample(rng) as u32).collect()
}

fn context(max_tokens: usize) -> ContextMemory {
    ContextMemory::new(ContextConfig { max_tokens, ..Default::default() }).unwrap()
}

#[test]
fn locates_fragments_exactly_anywhere_in_the_window() {
    let mut rng = SplitMix64::new(1);
    let doc = text(&mut rng, 19_683);
    let mut ctx = context(59_049);
    // Stream in uneven pieces, like generation steps.
    for piece in doc.chunks(81) {
        ctx.append(piece).unwrap();
    }
    for &(start, len) in &[(0usize, 9usize), (4_000, 5), (12_345, 19), (19_000, 27), (7_777, 40), (19_670, 13)] {
        let frag = &doc[start..start + len];
        let found = ctx.locate(frag).unwrap();
        assert_eq!(found.verdict, Verdict::Known, "fragment at {start} len {len}");
        assert!(found.positions.contains(&(start as u64)), "{start}: {:?}", found.positions);
        for &p in &found.positions {
            assert_eq!(&doc[p as usize..p as usize + len], frag, "every reported position is exact");
        }
    }
}

#[test]
fn associative_recall_of_key_value_facts() {
    let mut rng = SplitMix64::new(2);
    let mut ctx = context(300_000);
    // Facts: a random 6-token key followed by a 4-token value, buried in
    // Zipf filler text.
    let facts: Vec<(Vec<u32>, Vec<u32>)> = (0..81)
        .map(|_| {
            let key: Vec<u32> = (0..6).map(|_| rng.below(VOCAB as u64) as u32).collect();
            let value: Vec<u32> = (0..4).map(|_| rng.below(VOCAB as u64) as u32).collect();
            (key, value)
        })
        .collect();
    for (key, value) in &facts {
        ctx.append(&text(&mut rng, 729)).unwrap();
        ctx.append(key).unwrap();
        ctx.append(value).unwrap();
    }
    ctx.append(&text(&mut rng, 729)).unwrap();

    let mut correct = 0;
    for (key, value) in &facts {
        if let Some((_, next)) = ctx.continuation(key, 4).unwrap() {
            correct += (&next == value) as usize;
        }
    }
    assert_eq!(correct, facts.len(), "every fact recalled verbatim");
}

#[test]
fn ternary_contains_known_unknown_absent() {
    let mut ctx = context(59_049);
    // 1000..1100 then a stream where [7 8 9] and [8 9 5] both occur but
    // never as the contiguous run [7 8 9 5].
    let mut doc: Vec<u32> = (1000..1100).collect();
    doc.extend([7, 8, 9, 1, 2, 3, 8, 9, 5, 4, 4, 4]);
    doc.extend(2000..2100);
    ctx.append(&doc).unwrap();

    assert_eq!(ctx.contains(&[1010, 1011, 1012, 1013]).unwrap(), Verdict::Known);
    // Contains the trigram [3001 ...] that never occurred: provably absent.
    assert_eq!(ctx.contains(&[1010, 1011, 3001, 3002]).unwrap(), Verdict::Absent);
    // Every trigram of the fragment exists, but not contiguously.
    assert_eq!(ctx.contains(&[7, 8, 9, 5]).unwrap(), Verdict::Unknown);
}

#[test]
fn window_evicts_old_context_and_bounds_memory() {
    let mut rng = SplitMix64::new(3);
    let mut ctx = context(6_561);
    let early = text(&mut rng, 2_187);
    ctx.append(&early).unwrap();
    let marker = [60_001, 60_002, 60_003, 60_004, 60_005];
    ctx.append(&marker).unwrap();
    assert_eq!(ctx.contains(&marker).unwrap(), Verdict::Known);

    for _ in 0..9 {
        ctx.append(&text(&mut rng, 2_187)).unwrap();
    }
    let (start, end) = ctx.window();
    assert_eq!(end - start, 6_561);
    assert_eq!(ctx.contains(&marker).unwrap(), Verdict::Absent, "slid out of the window");
    let stats = ctx.stats();
    assert!(stats.chunks <= 6_561 / 9 + 1);
    assert!(stats.token_bytes <= 4 * 2 * 6_561 * 2, "token buffer is bounded");
    assert_eq!(stats.lexical.fast_memories, stats.chunks);
}

#[test]
fn pinned_context_survives_eviction_and_reset() {
    let mut rng = SplitMix64::new(4);
    let mut ctx = context(6_561);
    ctx.append(&text(&mut rng, 729)).unwrap();
    let important: Vec<u32> = (70_000..70_027).collect();
    let at = ctx.position();
    ctx.append(&important).unwrap();
    ctx.append(&text(&mut rng, 81)).unwrap();
    assert!(ctx.pin(at, at + 27).unwrap() >= 1);

    for _ in 0..9 {
        ctx.append(&text(&mut rng, 2_187)).unwrap();
    }
    ctx.reset();
    let found = ctx.locate(&important[3..12]).unwrap();
    assert_eq!(found.verdict, Verdict::Known);
    assert!(found.positions.contains(&(at + 3)));
    let (pos, next) = ctx.continuation(&important[0..9], 5).unwrap().expect("continues inside the pinned chunk");
    assert_eq!((pos, next), (at + 9, important[9..14].to_vec()));
}

#[test]
fn read_head_attends_over_retrieved_tokens() {
    let (dk, dv) = (27, 9);
    let cfg = ContextConfig {
        max_tokens: 59_049,
        kv: Some(KvConfig { precision: KvPrecision::F32, ..KvConfig::new(dk, dv) }),
        ..Default::default()
    };
    let mut ctx = ContextMemory::new(cfg).unwrap();
    assert!(matches!(ctx.append(&[1, 2, 3]), Err(MemoryError::InvalidConfig(_))));

    let mut rng = SplitMix64::new(5);
    let n = 6_561;
    let tokens = text(&mut rng, n);
    // Random unit keys; values carry the position in their first element.
    let mut keys = Vec::with_capacity(n * dk);
    for _ in 0..n {
        let v: Vec<f32> = (0..dk).map(|_| rng.normal() as f32).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        keys.extend(v.iter().map(|x| x / norm));
    }
    let values: Vec<f32> = (0..n).flat_map(|p| {
        let mut v = vec![0.0; dv];
        v[0] = p as f32;
        v
    }).collect();
    ctx.append_kv(&tokens, &keys, &values).unwrap();

    // Query the key of token 4000, scaled by 3^4 so attention is sharp; probe
    // the lexical index with the surrounding tokens.
    let target = 4_000usize;
    let query: Vec<f32> = keys[target * dk..(target + 1) * dk].iter().map(|x| x * 81.0).collect();
    let out = ctx.read(Probe::Tokens(&tokens[target - 4..target + 5]), &query, 9).unwrap();
    assert_eq!(out.verdict, Verdict::Known);
    assert!(out.gate > 0.9);
    let (pos, w) = out.attended.argmax.unwrap();
    assert_eq!(pos, target as u64);
    assert!(w > 0.9);
    assert!((out.attended.output[0] - target as f32).abs() < 9.0 * 27.0);
    assert!(out.attended.tokens <= 9 * 27, "attention touched only retrieved chunks");

    // Semantic probe: the mean key of a chunk retrieves that chunk.
    let chunk_start = 2_700usize;
    let mut mean = vec![0f32; dk];
    for p in chunk_start..chunk_start + 27 {
        mean.iter_mut().zip(&keys[p * dk..(p + 1) * dk]).for_each(|(m, k)| *m += k);
    }
    let r = ctx.retrieve(Probe::Key(&mean), 3).unwrap();
    assert_eq!(r.verdict, Verdict::Known);
    assert!(r.spans.iter().any(|s| s.start <= chunk_start as u64 && s.end() >= (chunk_start + 27) as u64));
}

#[test]
fn pinned_context_is_carried_into_the_next_session() {
    let mut rng = SplitMix64::new(6);
    let cfg = ContextConfig { max_tokens: 6_561, ..Default::default() };
    let mut ctx = ContextMemory::new(cfg.clone()).unwrap();
    ctx.append(&text(&mut rng, 729)).unwrap();
    let fact: Vec<u32> = (80_000..80_027).collect();
    let at = ctx.position();
    ctx.append(&fact).unwrap();
    ctx.append(&text(&mut rng, 729)).unwrap();
    ctx.pin(at, at + 27).unwrap();
    let end = ctx.position();
    let snapshot = ctx.save_pinned();
    drop(ctx);

    let mut next = ContextMemory::restore(cfg, &snapshot).unwrap();
    assert_eq!(next.position(), end, "positions continue");
    let (pos, tail) = next.continuation(&fact[..9], 9).unwrap().expect("fact survives the session");
    assert_eq!((pos, tail), (at + 9, fact[9..18].to_vec()));
    // The unpinned filler did not come along.
    assert_eq!(next.stats().lexical.fast_memories, 0);

    next.append(&text(&mut rng, 81)).unwrap();
    assert_eq!(next.position(), end + 81);
}

