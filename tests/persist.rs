//! Snapshots: memories survive the end of a session.

mod common;

use common::*;
use snn_memory::rng::SplitMix64;
use snn_memory::{
    CodeEncoder, FlyHashEncoder, Input, LearnOptions, ManualClock, MemoryConfig, MemoryError, RecallOptions,
    SnapshotScope, SnnMemory, Tier, Verdict,
};

fn fresh(clock: &ManualClock) -> SnnMemory<String> {
    SnnMemory::new(CodeEncoder::new(N_NEURONS), MemoryConfig::default()).unwrap().with_clock(clock.clone())
}

fn restore(bytes: &[u8], clock: &ManualClock) -> SnnMemory<String> {
    let mut mem = fresh(clock);
    mem.restore(bytes).unwrap();
    mem
}

#[test]
fn full_snapshot_roundtrip() {
    let clock = ManualClock::new(100.0);
    let mut mem = fresh(&clock);
    let chat = mem.context("chat");
    let mut rng = SplitMix64::new(40);
    let (a, b, c) = (random_code(&mut rng), random_code(&mut rng), random_code(&mut rng));
    let ia = mem.learn(Input::Code(&a), LearnOptions::new().payload("fast".into()).ttl(50.0)).unwrap();
    let ib = mem.learn(Input::Code(&b), LearnOptions::new().payload("pinned".into()).pin().context(chat)).unwrap();
    let ic = mem.learn(Input::Code(&c), LearnOptions::new().negative().after(ia)).unwrap();
    let misleading: Vec<u32> = a[..16].iter().copied().chain(random_code(&mut rng).into_iter().take(11)).collect();
    mem.suppress(ia, Input::Code(&misleading)).unwrap();
    clock.advance(10.0);

    let bytes = mem.save(SnapshotScope::All);
    // A new session with a different clock.
    let clock2 = ManualClock::new(5_000.0);
    let mut back = restore(&bytes, &clock2);
    assert_eq!(back.len(), 3);
    assert_eq!(back.find_context("chat"), Some(chat));

    let ra = back.get_memory(ia).unwrap();
    assert_eq!(ra.payload.map(String::as_str), Some("fast"));
    assert_eq!(ra.expires, Some(5_040.0), "remaining TTL is preserved");
    assert_eq!(ra.next, Some(ic));
    assert_eq!(ra.states, mem.get_memory(ia).unwrap().states, "ternary synapses survive");
    let rb = back.get_memory(ib).unwrap();
    assert_eq!((rb.tier, rb.pinned, rb.context), (Tier::LongTerm, true, chat));

    let opts = RecallOptions::default();
    assert_eq!(back.recall(Input::Code(&a), &opts).unwrap().id(), Some(ia));
    assert_eq!(back.recall(Input::Code(&b), &opts).unwrap().id(), Some(ib));
    let rc = back.recall(Input::Code(&c), &opts).unwrap();
    assert_eq!((rc.verdict, rc.id()), (Verdict::Absent, Some(ic)), "negative knowledge survives");
    assert!(back.recall(Input::Code(&misleading), &opts).unwrap().is_absent(), "suppression survives");

    // New memories never reuse restored ids.
    let d = random_code(&mut rng);
    assert!(back.learn(Input::Code(&d), LearnOptions::new()).unwrap() > ic);
    // TTL keeps running in the new session.
    clock2.advance(41.0);
    assert!(!back.contains(ia));
}

#[test]
fn long_term_scope_keeps_only_what_matters() {
    let clock = ManualClock::new(0.0);
    let mut mem = fresh(&clock);
    let mut rng = SplitMix64::new(41);
    let (a, b) = (random_code(&mut rng), random_code(&mut rng));
    let ia = mem.learn(Input::Code(&a), LearnOptions::new()).unwrap();
    let ib = mem.learn(Input::Code(&b), LearnOptions::new()).unwrap();
    mem.consolidate(ib).unwrap();

    let back = restore(&mem.save(SnapshotScope::LongTerm), &clock);
    assert!(!back.contains(ia));
    assert!(back.contains(ib));
    assert!(mem.save(SnapshotScope::LongTerm).len() < mem.save(SnapshotScope::All).len());
}

#[test]
fn damaged_or_foreign_snapshots_are_rejected() {
    let clock = ManualClock::new(0.0);
    let mut mem = fresh(&clock);
    mem.learn(Input::Code(&random_code(&mut SplitMix64::new(42))), LearnOptions::new()).unwrap();
    let bytes = mem.save(SnapshotScope::All);

    let mut flipped = bytes.clone();
    flipped[20] ^= 1;
    assert!(matches!(fresh(&clock).restore(&flipped), Err(MemoryError::Corrupt(_))));
    assert!(matches!(fresh(&clock).restore(&bytes[..bytes.len() - 9]), Err(MemoryError::Corrupt(_))));

    let other: SnnMemory<String> = SnnMemory::new(CodeEncoder::new(N_NEURONS - 1), MemoryConfig::default()).unwrap();
    let mut other = other;
    assert!(matches!(other.restore(&bytes), Err(MemoryError::InvalidConfig(_))), "different encoder");

    // Restoring twice would duplicate ids.
    let mut twice = restore(&bytes, &clock);
    assert!(matches!(twice.restore(&bytes), Err(MemoryError::Corrupt(_))));
}

#[test]
fn dense_encoder_fingerprint_guards_restore() {
    let d = 81;
    let enc = |seed| FlyHashEncoder::new(d, N_NEURONS, DENSE_K, 27, seed).unwrap();
    let cfg = MemoryConfig { max_ensemble: DENSE_K, ..Default::default() };
    let mut mem: SnnMemory<()> = SnnMemory::new(enc(1), cfg.clone()).unwrap();
    let key = random_vec(&mut SplitMix64::new(43), d);
    let id = mem.learn(Input::Dense(&key), LearnOptions::new()).unwrap();
    let bytes = mem.save(SnapshotScope::All);

    let mut back = SnnMemory::<()>::load(enc(1), cfg.clone(), &bytes).unwrap();
    assert_eq!(back.recall(Input::Dense(&key), &RecallOptions::default()).unwrap().id(), Some(id));
    assert!(SnnMemory::<()>::load(enc(2), cfg, &bytes).is_err(), "another projection would give other codes");
}

/// `save_full` / `load_full`: the restored memory continues exactly,
/// including plasticity (RNG), de-duplication, working memory, TTLs and the
/// order of the synapse index after reinforcement and suppression.
#[test]
fn exact_image_resumes_learning_and_recall() {
    let cfg = MemoryConfig { default_ttl: Some(300.0), consolidate_after: Some(3), ..MemoryConfig::default() };
    let clock = ManualClock::new(10.0);
    let mut a: SnnMemory<String> =
        SnnMemory::new(CodeEncoder::new(N_NEURONS), cfg.clone()).unwrap().with_clock(clock.clone());
    let chat = a.context("chat");
    let mut rng = SplitMix64::new(77);
    let codes: Vec<Vec<u32>> = (0..400).map(|_| random_code(&mut rng)).collect();
    let noisy = |rng: &mut SplitMix64, c: &[u32]| -> Vec<u32> {
        let mut v: Vec<u32> = c.iter().copied().filter(|_| rng.below(9) != 0).collect();
        v.extend(random_code(rng).into_iter().take(2));
        v
    };
    let mut ids = Vec::new();
    for (i, c) in codes.iter().enumerate() {
        clock.advance(0.1);
        let ctx = if i % 5 == 0 { chat } else { snn_memory::ContextId::DEFAULT };
        ids.push(a.learn(Input::Code(c), LearnOptions::new().payload(format!("m{i}")).context(ctx)).unwrap());
        if i % 3 == 0 {
            // Near duplicate: reinforces (plasticity, index reordering).
            let mut n = codes[i.saturating_sub(40)].clone();
            let at = rng.below(n.len() as u64) as usize;
            n[at] = random_code(&mut rng)[0];
            n.sort_unstable();
            n.dedup();
            a.learn(Input::Code(&n), LearnOptions::new().context(ctx)).unwrap();
        }
        if i % 11 == 0 {
            let misleading: Vec<u32> =
                c[..14].iter().copied().chain(random_code(&mut rng).into_iter().take(13)).collect();
            a.suppress(ids[i], Input::Code(&misleading)).unwrap();
        }
        if i % 13 == 7 {
            a.forget(ids[i - 3]);
        }
    }
    for i in (0..400).step_by(17) {
        for _ in 0..4 {
            a.recall(Input::Code(&codes[i][..20]), &RecallOptions::default().top_k(3)).unwrap();
        }
    }
    a.pin(ids[5]).unwrap();
    a.pin(ids[34]).unwrap();
    a.unpin(ids[5]).unwrap();
    clock.advance(200.0);

    let snapshot = a.save_full();
    let mut b =
        SnnMemory::<String>::load_full(CodeEncoder::new(N_NEURONS), cfg.clone(), &snapshot, clock.clone()).unwrap();
    assert!(b.save_full() == snapshot);

    for step in 0..600 {
        clock.advance(if step % 50 == 0 { 60.0 } else { 0.3 });
        let c = &codes[rng.below(400) as usize];
        match step % 4 {
            0 => {
                let n = noisy(&mut rng, c);
                let o = LearnOptions::new().payload(format!("s{step}"));
                assert_eq!(a.learn(Input::Code(&n), o.clone()).unwrap(), b.learn(Input::Code(&n), o).unwrap());
            }
            3 if step % 40 == 3 => assert_eq!(a.maintain(), b.maintain()),
            _ => {
                let cue = noisy(&mut rng, &c[..18]);
                let opts = RecallOptions::default().top_k(3).follow(1);
                let (ra, rb) =
                    (a.recall(Input::Code(&cue), &opts).unwrap(), b.recall(Input::Code(&cue), &opts).unwrap());
                assert_eq!((ra.verdict, ra.basis, ra.candidates), (rb.verdict, rb.basis, rb.candidates), "step {step}");
                let hits = |r: &snn_memory::RecallResult<String>| {
                    r.hits
                        .iter()
                        .map(|h| (h.id, h.confidence.to_bits(), h.spikes, h.payload.clone()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(hits(&ra), hits(&rb), "step {step}");
            }
        }
    }
    assert_eq!(a.recent(), b.recent());
    assert!(a.save_full() == b.save_full(), "same state after the same operations");

    // Damage and foreign encoders are refused.
    let mut bad = snapshot.clone();
    bad[100] ^= 4;
    let load = |bytes: &[u8], n| SnnMemory::<String>::load_full(CodeEncoder::new(n), cfg.clone(), bytes, clock.clone());
    assert!(matches!(load(&bad, N_NEURONS), Err(MemoryError::Corrupt(_))));
    assert!(matches!(load(&snapshot, 6_561), Err(MemoryError::InvalidConfig(_))));
}
