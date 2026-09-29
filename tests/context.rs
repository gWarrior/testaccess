//! ContextMemory: exact long-context memory for a recurrent LLM.

use snn_memory::rng::{SplitMix64, Zipf};
use snn_memory::{
    ContextConfig, ContextMemory, ContextStats, KvConfig, KvPrecision, ManualClock, MemoryError, MemoryRows, Probe,
    Verdict,
};

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
    let values: Vec<f32> = (0..n)
        .flat_map(|p| {
            let mut v = vec![0.0; dv];
            v[0] = p as f32;
            v
        })
        .collect();
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

    // The same retrieval as raw rows, for a model's own attention.
    let rows = ctx.retrieve_rows(Probe::Tokens(&tokens[target - 4..target + 5]), 9, 243).unwrap();
    assert_eq!(rows.verdict, Verdict::Known);
    let i = rows.positions.iter().position(|&p| p == target as u64).expect("target row retrieved");
    assert_eq!(&rows.keys[i * dk..(i + 1) * dk], &keys[target * dk..(target + 1) * dk]);
    assert_eq!(rows.values.len(), rows.positions.len() * dv);
    assert!(rows.positions.len() <= 243);
    assert_eq!(rows.sources[i], 0, "found by the lexical memory");
    assert_eq!(rows.sources.len(), rows.positions.len());
    // The newest, not yet indexed tokens come first, marked as the tail.
    let end = ctx.position();
    let tail: Vec<u64> = rows.positions.iter().zip(&rows.sources).filter(|r| *r.1 == 2).map(|r| *r.0).collect();
    assert!(tail.iter().all(|&p| p + 27 > end && p < end), "{tail:?} vs {end}");

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

// ----- full snapshots --------------------------------------------------------

/// `n` K/V rows of `dim`: mostly on the two-trit grid (levels −4…4 times a
/// power of two, as a model's Trit2 keys), some Gaussian.
fn kv_rows(rng: &mut SplitMix64, n: usize, dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        if rng.below(4) == 0 {
            out.extend((0..dim).map(|_| rng.normal() as f32));
        } else {
            let step = 2f32.powi(rng.below(6) as i32 - 4);
            out.extend((0..dim).map(|_| (rng.below(9) as i32 - 4) as f32 * step));
        }
    }
    out
}

/// A test context whose SNN memories run on `clock`.
fn clocked(kv: Option<KvConfig>, max_tokens: usize, clock: &ManualClock) -> (ContextConfig, ContextMemory) {
    let cfg = ContextConfig { max_tokens, kv, ..Default::default() };
    let ctx = ContextMemory::new(cfg.clone()).unwrap().with_clock(clock.clone());
    (cfg, ctx)
}

fn assert_rows_eq(a: &MemoryRows, b: &MemoryRows, what: &str) {
    assert_eq!(a.verdict, b.verdict, "{what}: verdict");
    assert_eq!(a.confidence.to_bits(), b.confidence.to_bits(), "{what}: confidence");
    assert_eq!(a.positions, b.positions, "{what}: positions");
    assert_eq!(a.sources, b.sources, "{what}: sources");
    let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.keys), bits(&b.keys), "{what}: keys");
    assert_eq!(bits(&a.values), bits(&b.values), "{what}: values");
}

/// Everything in the statistics except allocation sizes and wall-clock
/// timings.
fn stats_key(s: &ContextStats) -> String {
    let mem = |m: &snn_memory::Stats| {
        format!(
            "{} {} {} {} {} {} {} {} {} {} {} {} {}",
            m.fast_memories,
            m.long_term_memories,
            m.pinned,
            m.pending_cleanup,
            m.free_slots,
            m.slots,
            m.contexts,
            m.synapses,
            m.max_fan_out,
            m.working_traces,
            m.learns,
            m.recalls,
            m.forgets
        )
    };
    format!(
        "{:?} {} {} {} | {} | {:?}",
        s.window,
        s.chunks,
        s.unindexed_tokens,
        s.census_entries,
        mem(&s.lexical),
        s.semantic.as_ref().map(mem)
    )
}

/// Drive `a` (original) and `b` (restored) through the same operations at
/// the same clock readings; every result must be identical.
fn lockstep(a: &mut ContextMemory, b: &mut ContextMemory, clock: &ManualClock, seed: u64, steps: usize) {
    let mut rng = SplitMix64::new(seed);
    let dims = a.kv().map(|kv| (kv.key_dim(), kv.value_dim()));
    for step in 0..steps {
        clock.advance(0.25 + rng.below(9) as f64);
        let what = format!("step {step}");
        let n = 1 + rng.below(81) as usize;
        let tokens = text(&mut rng, n);
        match dims {
            Some((dk, dv)) => {
                let (k, v) = (kv_rows(&mut rng, n, dk), kv_rows(&mut rng, n, dv));
                assert_eq!(a.append_kv(&tokens, &k, &v).unwrap(), b.append_kv(&tokens, &k, &v).unwrap(), "{what}");
                // Probe with recent tokens (lexical) and a stored key (semantic).
                let (lo, hi) = (a.window().0.max(a.position().saturating_sub(4_000)), a.position());
                let at = lo + rng.below(hi - lo - 9);
                let probe = a.tokens(at, at + 9).unwrap().to_vec();
                assert_eq!(Some(&probe[..]), b.tokens(at, at + 9));
                let key = &k[(n - 1) * dk..];
                let mut probes = vec![Probe::Tokens(&probe)];
                if a.semantic().is_some() {
                    probes.extend([Probe::Both(&probe, key), Probe::Key(key)]);
                }
                for (i, p) in probes.into_iter().enumerate() {
                    let top_k = 1 + rng.below(9) as usize;
                    let ra = a.retrieve_rows(p, top_k, 243).unwrap();
                    let rb = b.retrieve_rows(p, top_k, 243).unwrap();
                    assert_rows_eq(&ra, &rb, &format!("{what} probe {i}"));
                }
            }
            None => {
                assert_eq!(a.append(&tokens).unwrap(), b.append(&tokens).unwrap(), "{what}");
            }
        }
        // Exact lookups: a fragment from the window, and a random one.
        let (lo, hi) = a.window();
        let len = 2 + rng.below(30);
        let at = lo + rng.below(hi - lo - len);
        let frag = a.tokens(at, at + len).unwrap().to_vec();
        assert_eq!(a.locate(&frag).unwrap(), b.locate(&frag).unwrap(), "{what}: locate");
        let noise = text(&mut rng, 5);
        assert_eq!(a.locate(&noise).unwrap(), b.locate(&noise).unwrap(), "{what}: locate noise");
        assert_eq!(a.continuation(&frag[..2], 9).unwrap(), b.continuation(&frag[..2], 9).unwrap(), "{what}");
        if dims.is_none() {
            let ra = a.retrieve(Probe::Tokens(&frag), 9).unwrap();
            let rb = b.retrieve(Probe::Tokens(&frag), 9).unwrap();
            assert_eq!(ra.verdict, rb.verdict, "{what}");
            let spans = |r: &snn_memory::Retrieval| {
                r.spans
                    .iter()
                    .map(|s| (s.start, s.tokens.clone(), s.confidence.to_bits(), s.chunks.clone()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(spans(&ra), spans(&rb), "{what}: spans");
        }
        if step % 27 == 26 {
            assert_eq!(a.flush().unwrap(), b.flush().unwrap());
        }
        assert_eq!(stats_key(&a.stats()), stats_key(&b.stats()), "{what}: stats");
    }
    assert_eq!(a.position(), b.position());
    assert_eq!(a.tokens(a.window().0, a.position()), b.tokens(b.window().0, b.position()));
}

/// Fill a context: stream with K/V (if any), query, pin, evict, flush.
fn fill(ctx: &mut ContextMemory, clock: &ManualClock, seed: u64, pieces: usize) -> u64 {
    let mut rng = SplitMix64::new(seed);
    let dims = ctx.kv().map(|kv| (kv.key_dim(), kv.value_dim()));
    let mut pinned_at = 0;
    for i in 0..pieces {
        clock.advance(0.5);
        let n = 27 + rng.below(81) as usize;
        let tokens = text(&mut rng, n);
        match dims {
            Some((dk, dv)) => {
                let (k, v) = (kv_rows(&mut rng, n, dk), kv_rows(&mut rng, n, dv));
                ctx.append_kv(&tokens, &k, &v).unwrap();
                if i % 7 == 0 {
                    let probe = match ctx.semantic() {
                        Some(_) => Probe::Both(&tokens[..9], &k[..dk]),
                        None => Probe::Tokens(&tokens[..9]),
                    };
                    ctx.retrieve_rows(probe, 3, 81).unwrap();
                }
            }
            None => {
                ctx.append(&tokens).unwrap();
                if i % 7 == 0 {
                    ctx.locate(&tokens[..9]).unwrap();
                }
            }
        }
        if i == pieces / 3 {
            pinned_at = ctx.position() - 54;
            ctx.pin(pinned_at, pinned_at + 27).unwrap();
        }
        if i % 50 == 49 {
            ctx.flush().unwrap();
        }
    }
    pinned_at
}

fn full_snapshot_roundtrip(kv: Option<KvConfig>) {
    let clock = ManualClock::new(1_000.0);
    let (cfg, mut a) = clocked(kv, 6_561, &clock);
    let pinned_at = fill(&mut a, &clock, 7, 400);
    assert!(a.window().0 > pinned_at + 27, "the pinned chunk left the window");
    assert!(a.stats().lexical.long_term_memories > 0);

    let snapshot = a.save_full();
    // Restored at a later clock reading, a memory resumes at the saved moment.
    clock.advance(3.0);
    let later = ContextMemory::restore_full_with_clock(cfg.clone(), &snapshot, clock.clone()).unwrap();
    assert!(later.save_full() == snapshot, "a restored memory saves the same bytes");
    clock.advance(-3.0);

    let mut b = ContextMemory::restore_full_with_clock(cfg.clone(), &snapshot, clock.clone()).unwrap();
    assert!(b.save_full() == snapshot);
    assert_eq!(stats_key(&a.stats()), stats_key(&b.stats()));
    // The pinned chunk left the window but is recalled from both.
    let lex = a.lexical();
    let pinned = lex.ids().into_iter().filter(|&id| lex.get_memory(id).is_some_and(|m| m.pinned));
    let chunk = pinned.filter_map(|id| lex.payload(id).cloned()).min_by_key(|c| c.start).unwrap();
    assert!(chunk.start + 27 > pinned_at && chunk.start < pinned_at + 27);
    let frag = &chunk.tokens[3..12];
    let found = b.locate(frag).unwrap();
    assert_eq!(a.locate(frag).unwrap(), found);
    assert!(found.positions.contains(&(chunk.start + 3)), "{found:?}");

    lockstep(&mut a, &mut b, &clock, 11, 120);
    assert!(a.save_full() == b.save_full(), "same state after the same operations");

    a.reset();
    b.reset();
    lockstep(&mut a, &mut b, &clock, 13, 30);
    assert!(a.save_full() == b.save_full());

    // And a second generation from the restored copy.
    let again = ContextMemory::restore_full_with_clock(cfg, &b.save_full(), clock.clone()).unwrap();
    assert!(again.save_full() == b.save_full());
}

#[test]
fn full_snapshot_resumes_exactly_with_trit2_kv() {
    full_snapshot_roundtrip(Some(KvConfig { precision: KvPrecision::Trit2, ..KvConfig::new(81, 81) }));
}

#[test]
fn full_snapshot_resumes_exactly_with_f16_kv_without_semantic_index() {
    full_snapshot_roundtrip(Some(KvConfig { semantic_index: false, ..KvConfig::new(27, 9) }));
}

#[test]
fn full_snapshot_resumes_exactly_without_kv() {
    full_snapshot_roundtrip(None);
}

#[test]
fn full_snapshot_roundtrips_every_precision() {
    for precision in [KvPrecision::F32, KvPrecision::F16, KvPrecision::Ternary, KvPrecision::Trit2] {
        let clock = ManualClock::new(0.0);
        let (cfg, mut a) = clocked(Some(KvConfig { precision, ..KvConfig::new(9, 5) }), 2_187, &clock);
        fill(&mut a, &clock, 3, 60);
        let mut b = ContextMemory::restore_full_with_clock(cfg, &a.save_full(), clock.clone()).unwrap();
        let (lo, hi) = (a.kv().unwrap().base(), a.kv().unwrap().end());
        assert_eq!((lo, hi), (b.kv().unwrap().base(), b.kv().unwrap().end()));
        let (mut ka, mut va, mut kb, mut vb) = (vec![0.0; 9], vec![0.0; 5], vec![0.0; 9], vec![0.0; 5]);
        for p in lo..hi {
            assert!(a.kv().unwrap().read(p, &mut ka, &mut va) && b.kv().unwrap().read(p, &mut kb, &mut vb));
            assert_eq!((&ka, &va), (&kb, &vb), "{precision:?} row {p}");
        }
        lockstep(&mut a, &mut b, &clock, 5, 20);
        assert!(a.save_full() == b.save_full(), "{precision:?}");
    }
}

#[test]
fn full_snapshot_rejects_corruption_and_other_shapes() {
    let clock = ManualClock::new(0.0);
    let kv = KvConfig { precision: KvPrecision::Trit2, ..KvConfig::new(27, 27) };
    let (cfg, mut ctx) = clocked(Some(kv.clone()), 2_187, &clock);
    fill(&mut ctx, &clock, 9, 40);
    let snapshot = ctx.save_full();
    assert!(ContextMemory::restore_full(cfg.clone(), &snapshot).is_ok());

    // Any flipped bit, anywhere, is caught by the checksum.
    let mut rng = SplitMix64::new(1);
    for _ in 0..64 {
        let mut bad = snapshot.clone();
        let i = rng.below(bad.len() as u64) as usize;
        bad[i] ^= 1 << rng.below(8);
        match ContextMemory::restore_full(cfg.clone(), &bad) {
            Err(MemoryError::Corrupt(msg)) => assert!(msg.contains("checksum"), "{msg}"),
            other => panic!("byte {i}: expected a checksum error, got {:?}", other.map(|_| ())),
        }
    }
    for cut in [0, 7, 16, snapshot.len() / 2, snapshot.len() - 1] {
        assert!(matches!(ContextMemory::restore_full(cfg.clone(), &snapshot[..cut]), Err(MemoryError::Corrupt(_))));
    }
    // Other snapshot kinds are not full snapshots.
    assert!(ContextMemory::restore_full(cfg.clone(), &ctx.save_pinned()).is_err());

    // A config of another shape is refused.
    let other = |f: &dyn Fn(&mut ContextConfig)| {
        let mut c = cfg.clone();
        f(&mut c);
        ContextMemory::restore_full(c, &snapshot).map(|_| ())
    };
    for f in [
        &(|c: &mut ContextConfig| c.kv.as_mut().unwrap().precision = KvPrecision::F16) as &dyn Fn(&mut ContextConfig),
        &|c| c.kv.as_mut().unwrap().key_dim = 26,
        &|c| c.kv.as_mut().unwrap().value_dim = 9,
        &|c| c.kv.as_mut().unwrap().semantic_index = false,
        &|c| c.kv = None,
        &|c| c.chunk_size = 36,
        &|c| c.max_tokens = 4_374,
        &|c| c.seed ^= 1,
        &|c| c.ngrams = vec![1, 2],
        &|c| c.lexical.max_ensemble = 243,
    ] {
        assert!(matches!(other(f), Err(MemoryError::InvalidConfig(_))), "{:?}", other(f));
    }
    // Query settings and thresholds may change.
    assert!(other(&|c| c.top_k = 3).is_ok());
    assert!(other(&|c| c.lexical.recall_threshold = 0.5).is_ok());
}

/// Size and speed of a full snapshot of one training stream's memory:
/// 3^11 = 177 147 tokens, K/V 81 + 81 in Trit2, semantic index.
/// `cargo test --release -p snn-memory --test context -- --ignored --nocapture`
#[test]
#[ignore]
fn full_snapshot_size_and_speed() {
    use std::time::Instant;
    let n = 177_147;
    let cfg = ContextConfig {
        kv: Some(KvConfig { precision: KvPrecision::Trit2, ..KvConfig::new(81, 81) }),
        ..Default::default()
    };
    let mut ctx = ContextMemory::new(cfg.clone()).unwrap();
    let mut rng = SplitMix64::new(1);
    let t = Instant::now();
    for _ in 0..n / 243 {
        let tokens = text(&mut rng, 243);
        let k = kv_rows(&mut rng, 243, 81);
        let v = kv_rows(&mut rng, 243, 81);
        ctx.append_kv(&tokens, &k, &v).unwrap();
    }
    let rest = n % 243;
    let (k, v) = (kv_rows(&mut rng, rest, 81), kv_rows(&mut rng, rest, 81));
    ctx.append_kv(&text(&mut rng, rest), &k, &v).unwrap();
    println!("fill {n} tokens: {:.2?}", t.elapsed());

    let mut save = f64::MAX;
    let mut bytes = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        bytes = ctx.save_full();
        save = save.min(t.elapsed().as_secs_f64());
    }
    let mut restore = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        let r = ContextMemory::restore_full(cfg.clone(), &bytes).unwrap();
        restore = restore.min(t.elapsed().as_secs_f64());
        drop(r);
    }
    let s = ctx.stats();
    println!(
        "full snapshot: {:.2} MB (kv in memory {:.2} MB, chunks {}, synapses {} + {}), save {:.1} ms, restore {:.1} ms",
        bytes.len() as f64 / 1e6,
        s.kv_bytes as f64 / 1e6,
        s.chunks,
        s.lexical.synapses,
        s.semantic.as_ref().map_or(0, |m| m.synapses),
        save * 1e3,
        restore * 1e3
    );
    println!("pinned-only snapshot: {:.2} MB", ctx.save_pinned().len() as f64 / 1e6);
}

#[test]
fn equal_matches_go_to_the_newest_chunks() {
    // The same 9-token phrase in 40 places, each amid different text: a
    // probe with the phrase scores all ~120 chunks around them alike, more
    // than the 27 candidates the attractor takes. The candidates must be the
    // newest, not whichever a partial sort leaves first.
    let mut rng = SplitMix64::new(9);
    let phrase = text(&mut rng, 9);
    let mut ctx = context(59_049);
    let mut at = Vec::new();
    for _ in 0..40 {
        ctx.append(&text(&mut rng, 729)).unwrap();
        at.push(ctx.position());
        ctx.append(&phrase).unwrap();
    }
    ctx.append(&text(&mut rng, 243)).unwrap();
    let r = ctx.retrieve(Probe::Tokens(&phrase), 1).unwrap();
    let span = r.spans.first().expect("a span");
    // 27 candidates cover the 9 newest occurrences (3 chunks each).
    let recent = at[at.len() - 9];
    assert!(span.end() > recent, "{:?}, the 9 newest phrases start at {recent}", (span.start, span.end()));
}
