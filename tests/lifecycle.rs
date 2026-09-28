//! Memory lifecycle: TTL, consolidation, pinning, resets, novelty,
//! temporal chains and short-term facilitation.

mod common;

use common::*;
use snn_memory::rng::SplitMix64;
use snn_memory::{BatchOptions, Input, LearnOptions, MemoryConfig, MemoryError, RecallOptions, Tier};

fn opts() -> RecallOptions {
    RecallOptions::default()
}

#[test]
fn ttl_expires_then_cleans_up() {
    let (mut mem, clock) = code_memory(MemoryConfig::default());
    let mut rng = SplitMix64::new(10);
    let (a, b) = (random_code(&mut rng), random_code(&mut rng));
    let ia = mem.learn(Input::Code(&a), LearnOptions::new().ttl(10.0)).unwrap();
    let ib = mem.learn(Input::Code(&b), LearnOptions::new()).unwrap();
    assert_eq!(mem.get_memory(ia).unwrap().expires, Some(10.0));

    clock.advance(9.0);
    assert_eq!(mem.recall(Input::Code(&a), &opts()).unwrap().id(), Some(ia));
    clock.advance(2.0);
    assert!(mem.recall(Input::Code(&a), &opts()).unwrap().is_miss(), "expired memory is silent");
    assert!(!mem.contains(ia));

    let report = mem.maintain();
    assert_eq!((report.expired, report.cleaned), (1, 1));
    assert_eq!(mem.recall(Input::Code(&b), &opts()).unwrap().id(), Some(ib));
}

#[test]
fn default_ttl_applies() {
    let cfg = MemoryConfig { default_ttl: Some(5.0), ..Default::default() };
    let (mut mem, clock) = code_memory(cfg);
    let a = random_code(&mut SplitMix64::new(11));
    mem.learn(Input::Code(&a), LearnOptions::new()).unwrap();
    clock.advance(6.0);
    assert!(mem.recall(Input::Code(&a), &opts()).unwrap().is_miss());
}

#[test]
fn reset_fast_memory_keeps_long_term() {
    let (mut mem, _) = code_memory(MemoryConfig::default());
    let mut rng = SplitMix64::new(12);
    let (a, b, c) = (random_code(&mut rng), random_code(&mut rng), random_code(&mut rng));
    let ia = mem.learn(Input::Code(&a), LearnOptions::new()).unwrap();
    let ib = mem.learn(Input::Code(&b), LearnOptions::new().pin()).unwrap();
    let ic = mem.learn(Input::Code(&c), LearnOptions::new()).unwrap();
    assert!(mem.consolidate(ic).unwrap());
    assert!(!mem.consolidate(ic).unwrap());

    assert_eq!(mem.reset_fast_memory(), 1);
    assert!(mem.recall(Input::Code(&a), &opts()).unwrap().is_miss());
    assert!(!mem.contains(ia));
    for (code, id) in [(&b, ib), (&c, ic)] {
        let hit = mem.recall(Input::Code(code), &opts()).unwrap();
        assert_eq!(hit.id(), Some(id));
        assert_eq!(hit.best().unwrap().tier, Tier::LongTerm);
    }

    mem.reset_all();
    assert!(mem.recall(Input::Code(&b), &opts()).unwrap().is_miss());
    assert!(mem.is_empty());
}

#[test]
fn pinned_memories_survive_ttl_and_context_reset() {
    let (mut mem, clock) = code_memory(MemoryConfig::default());
    let ctx = mem.context("chat");
    let mut rng = SplitMix64::new(13);
    let (a, b) = (random_code(&mut rng), random_code(&mut rng));
    let ia = mem.learn(Input::Code(&a), LearnOptions::new().context(ctx).ttl(1.0)).unwrap();
    let ib = mem.learn(Input::Code(&b), LearnOptions::new().context(ctx).ttl(1.0)).unwrap();
    mem.pin(ia).unwrap();
    assert!(mem.get_memory(ia).unwrap().pinned);

    clock.advance(2.0);
    assert!(!mem.contains(ib), "unpinned memory expired");
    assert_eq!(mem.reset_context(ctx, true), 0, "pinned memory is protected");
    assert_eq!(mem.recall(Input::Code(&a), &opts()).unwrap().id(), Some(ia));

    // Unpin returns it to the fast tier, where resets apply again.
    assert!(mem.unpin(ia).unwrap());
    assert_eq!(mem.get_memory(ia).unwrap().tier, Tier::Fast);
    assert_eq!(mem.reset_context(ctx, false), 1);
    assert!(!mem.contains(ia));
    assert_eq!(mem.pin(ia), Err(MemoryError::UnknownMemory(ia)));
}

#[test]
fn repeated_recall_consolidates() {
    let cfg = MemoryConfig { consolidate_after: Some(3), ..Default::default() };
    let (mut mem, _) = code_memory(cfg);
    let a = random_code(&mut SplitMix64::new(14));
    let id = mem.learn(Input::Code(&a), LearnOptions::new()).unwrap();
    for _ in 0..3 {
        mem.recall(Input::Code(&a), &opts()).unwrap();
    }
    let rec = mem.get_memory(id).unwrap();
    assert_eq!(rec.tier, Tier::LongTerm);
    assert_eq!(rec.recalls, 3);
    assert_eq!(mem.reset_fast_memory(), 0);
    assert_eq!(mem.recall(Input::Code(&a), &opts()).unwrap().id(), Some(id));
}

#[test]
fn novelty_check_reinforces_instead_of_duplicating() {
    let (mut mem, _) = code_memory(MemoryConfig::default());
    let mut rng = SplitMix64::new(15);
    let a = random_code(&mut rng);
    let id = mem.learn(Input::Code(&a), LearnOptions::new().payload("v1")).unwrap();
    let again = mem.learn(Input::Code(&a), LearnOptions::new().payload("v2")).unwrap();
    assert_eq!(id, again);
    assert_eq!(mem.len(), 1);
    let rec = mem.get_memory(id).unwrap();
    assert_eq!(rec.strength, 2.0);
    assert_eq!(rec.payload, Some(&"v2"));

    // A different context is a different memory.
    let other = mem.context("other");
    let id2 = mem.learn(Input::Code(&a), LearnOptions::new().context(other)).unwrap();
    assert_ne!(id, id2);

    // Explicitly disabled novelty check creates a new memory.
    let id3 = mem.learn(Input::Code(&a), LearnOptions::new().dedupe(false)).unwrap();
    assert_ne!(id, id3);
}

#[test]
fn reinforcement_tracks_a_drifting_pattern() {
    let cfg = MemoryConfig { dedupe_threshold: Some(0.7), ..Default::default() };
    let (mut mem, _) = code_memory(cfg);
    let mut rng = SplitMix64::new(16);
    let a = random_code(&mut rng);
    let id = mem.learn(Input::Code(&a), LearnOptions::new()).unwrap();
    // Replace 10% of the pattern with new neurons and present it repeatedly.
    let mut drifted = a[..K - 5].to_vec();
    drifted.extend(random_code(&mut rng).into_iter().take(5));
    drifted.sort_unstable();
    drifted.dedup();
    for _ in 0..6 {
        assert_eq!(mem.learn(Input::Code(&drifted), LearnOptions::new()).unwrap(), id);
    }
    let rec = mem.get_memory(id).unwrap();
    let recruited = drifted.iter().filter(|n| rec.pattern.contains(n)).count();
    assert_eq!(recruited, drifted.len(), "engram rewired to the new pattern");
}

#[test]
fn sequences_link_and_follow() {
    let (mut mem, _) = code_memory(MemoryConfig::default());
    let mut rng = SplitMix64::new(17);
    let codes: Vec<Vec<u32>> = (0..5).map(|_| random_code(&mut rng)).collect();
    let inputs: Vec<Input> = codes.iter().map(|c| Input::Code(c)).collect();
    let ids = mem
        .learn_batch(&inputs, Some(vec!["a", "b", "c", "d", "e"]), &BatchOptions { link_sequence: true, ..Default::default() })
        .unwrap();

    let r = mem.recall(Input::Code(&codes[1]), &opts().follow(10)).unwrap();
    assert_eq!(r.best().unwrap().sequence, ids[2..].to_vec());
    assert_eq!(mem.get_memory(ids[2]).unwrap().prev, Some(ids[1]));

    // Forgetting a link breaks the chain at that point.
    mem.forget(ids[3]);
    let r = mem.recall(Input::Code(&codes[1]), &opts().follow(10)).unwrap();
    assert_eq!(r.best().unwrap().sequence, vec![ids[2]]);
}

#[test]
fn facilitation_prefers_recent_duplicate() {
    let cfg = MemoryConfig { dedupe_threshold: None, ..Default::default() };
    let (mut mem, clock) = code_memory(cfg);
    let a = random_code(&mut SplitMix64::new(18));
    let old = mem.learn(Input::Code(&a), LearnOptions::new()).unwrap();
    clock.advance(1000.0);
    let new = mem.learn(Input::Code(&a), LearnOptions::new()).unwrap();
    let r = mem.recall(Input::Code(&a), &opts().top_k(2)).unwrap();
    let ids: Vec<u64> = r.hits.iter().map(|h| h.id).collect();
    assert_eq!(ids, vec![new, old], "both are recalled, the recent one first");
    assert_eq!(mem.recent()[0], new);
    assert!(!mem.working_state().is_empty());
}

#[test]
fn errors_are_reported() {
    let (mut mem, _) = code_memory(MemoryConfig::default());
    let big: Vec<u32> = (0..200).collect();
    assert!(matches!(
        mem.learn(Input::Code(&big), LearnOptions::new()),
        Err(MemoryError::EnsembleTooLarge { len: 200, .. })
    ));
    assert!(matches!(
        mem.learn(Input::Code(&[N_NEURONS]), LearnOptions::new()),
        Err(MemoryError::NeuronOutOfRange { .. })
    ));
    assert_eq!(mem.learn(Input::Code(&[]), LearnOptions::new()), Err(MemoryError::EmptyCode));
    assert!(matches!(
        mem.learn(Input::Dense(&[1.0]), LearnOptions::new()),
        Err(MemoryError::UnsupportedInput(_))
    ));
    let bogus = snn_memory::ContextId(99);
    assert_eq!(
        mem.learn(Input::Code(&[1, 2]), LearnOptions::new().context(bogus)),
        Err(MemoryError::UnknownContext(bogus))
    );
}

#[test]
fn slots_are_recycled_after_cleanup() {
    let cfg = MemoryConfig { auto_cleanup: 8, dedupe_threshold: None, ..Default::default() };
    let (mut mem, _) = code_memory(cfg);
    let mut rng = SplitMix64::new(19);
    for _ in 0..100 {
        let id = mem.learn(Input::Code(&random_code(&mut rng)), LearnOptions::new()).unwrap();
        mem.forget(id);
    }
    let s = mem.stats();
    assert!(s.slots <= 9, "slots grow unbounded: {}", s.slots);
    assert_eq!(s.fast_memories, 0);
}
