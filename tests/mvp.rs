//! The critical MVP experiments from `docs/concept.md` §20.

mod common;

use common::*;
use snn_memory::rng::SplitMix64;
use snn_memory::{Input, LearnOptions, MemoryConfig, RecallOptions};

fn opts() -> RecallOptions {
    RecallOptions::default()
}

#[test]
fn test1_one_shot() {
    let (mut mem, _) = code_memory(MemoryConfig::default());
    let mut rng = SplitMix64::new(1);
    let a = random_code(&mut rng);
    let id = mem.learn(Input::Code(&a), LearnOptions::new().payload("A")).unwrap();

    let r = mem.recall(Input::Code(&a), &opts()).unwrap();
    let hit = r.best().expect("A must be recalled after one presentation");
    assert_eq!(hit.id, id);
    assert_eq!(hit.payload, Some("A"));
    assert_eq!(hit.pattern, a);
    assert!(hit.confidence > 0.99);
}

#[test]
fn test2_partial_recall() {
    let (mut mem, _) = code_memory(MemoryConfig::default());
    let mut rng = SplitMix64::new(2);
    let a = random_code(&mut rng);
    let id = mem.learn(Input::Code(&a), LearnOptions::new()).unwrap();

    for keep in [24, 12, 6] {
        let cue = partial(&mut rng, &a, keep);
        let r = mem.recall(Input::Code(&cue), &opts()).unwrap();
        let hit = r.best().unwrap_or_else(|| panic!("partial cue of {keep}/{K} failed"));
        assert_eq!(hit.id, id);
        assert_eq!(hit.pattern, a, "pattern must be completed");
        assert!((hit.completeness - keep as f32 / K as f32).abs() < 1e-5);
    }
}

#[test]
fn test2b_partial_recall_dense_hidden_states() {
    let dim = 512;
    let mut mem = dense_memory(dim, MemoryConfig::default());
    let mut rng = SplitMix64::new(22);
    let keys: Vec<Vec<f32>> = (0..200).map(|_| random_vec(&mut rng, dim)).collect();
    let ids: Vec<u64> = keys.iter().map(|k| mem.learn(Input::Dense(k), LearnOptions::new()).unwrap()).collect();

    let mut correct = 0;
    for (k, &id) in keys.iter().zip(&ids) {
        let q = noisy(&mut rng, k, 0.9);
        if mem.recall(Input::Dense(&q), &opts()).unwrap().id() == Some(id) {
            correct += 1;
        }
    }
    assert!(correct >= 195, "noisy dense recall {correct}/200");
}

#[test]
fn test3_online_learning() {
    let (mut mem, _) = code_memory(MemoryConfig::default());
    let mut rng = SplitMix64::new(3);
    let codes: Vec<Vec<u32>> = (0..3).map(|_| random_code(&mut rng)).collect();
    let ids: Vec<u64> = codes.iter().map(|c| mem.learn(Input::Code(c), LearnOptions::new()).unwrap()).collect();
    for (c, id) in codes.iter().zip(ids) {
        assert_eq!(mem.recall(Input::Code(c), &opts()).unwrap().id(), Some(id));
    }
}

#[test]
fn test4_selective_forgetting() {
    let (mut mem, _) = code_memory(MemoryConfig::default());
    let mut rng = SplitMix64::new(4);
    let (a, b, c) = (random_code(&mut rng), random_code(&mut rng), random_code(&mut rng));
    let ia = mem.learn(Input::Code(&a), LearnOptions::new()).unwrap();
    let ib = mem.learn(Input::Code(&b), LearnOptions::new()).unwrap();
    let ic = mem.learn(Input::Code(&c), LearnOptions::new()).unwrap();

    assert!(mem.forget(ib));
    assert!(!mem.forget(ib), "second forget is a no-op");

    assert_eq!(mem.recall(Input::Code(&a), &opts()).unwrap().id(), Some(ia));
    assert!(mem.recall(Input::Code(&b), &opts()).unwrap().is_miss(), "B must be UNKNOWN");
    assert_eq!(mem.recall(Input::Code(&c), &opts()).unwrap().id(), Some(ic));

    // Physical cleanup must not change the answers either.
    mem.maintain();
    assert_eq!(mem.stats().pending_cleanup, 0);
    assert_eq!(mem.recall(Input::Code(&a), &opts()).unwrap().id(), Some(ia));
    assert!(mem.recall(Input::Code(&b), &opts()).unwrap().is_miss());
    assert_eq!(mem.recall(Input::Code(&c), &opts()).unwrap().id(), Some(ic));
}

#[test]
fn test5_context_reset() {
    let (mut mem, _) = code_memory(MemoryConfig::default());
    let (c1, c2) = (mem.context("conversation_1"), mem.context("conversation_2"));
    let mut rng = SplitMix64::new(5);
    let (a, b, c) = (random_code(&mut rng), random_code(&mut rng), random_code(&mut rng));
    mem.learn(Input::Code(&a), LearnOptions::new().context(c1)).unwrap();
    mem.learn(Input::Code(&b), LearnOptions::new().context(c1)).unwrap();
    let ic = mem.learn(Input::Code(&c), LearnOptions::new().context(c2)).unwrap();

    assert_eq!(mem.reset_context(c1, false), 2);

    let in1 = RecallOptions::in_context(c1);
    assert!(mem.recall(Input::Code(&a), &in1).unwrap().is_miss());
    assert!(mem.recall(Input::Code(&b), &in1).unwrap().is_miss());
    assert_eq!(mem.recall(Input::Code(&c), &RecallOptions::in_context(c2)).unwrap().id(), Some(ic));
    // Context isolation: C is not visible from context 1.
    assert!(mem.recall(Input::Code(&c), &in1).unwrap().is_miss());
}

/// Many similar patterns: families of variants sharing 60% of a prototype.
#[test]
fn test6_interference() {
    let cfg = MemoryConfig { dedupe_threshold: None, ..Default::default() };
    let (mut mem, _) = code_memory(cfg);
    let mut rng = SplitMix64::new(6);
    let shared = K * 6 / 10;
    let mut stored = Vec::new();
    for _ in 0..100 {
        let proto = random_code(&mut rng);
        for _ in 0..20 {
            let mut v = partial(&mut rng, &proto, shared);
            while v.len() < K {
                let n = rng.below(N_NEURONS as u64) as u32;
                if !v.contains(&n) && !proto.contains(&n) {
                    v.push(n);
                }
            }
            v.sort_unstable();
            let id = mem.learn(Input::Code(&v), LearnOptions::new()).unwrap();
            stored.push((v, id));
        }
    }

    // Cue: 75% of a variant. The variant explains all of it; its siblings
    // explain at most the shared part.
    let (mut correct, mut wrong) = (0, 0);
    for (v, id) in &stored {
        let cue = partial(&mut rng, v, K * 3 / 4);
        match mem.recall(Input::Code(&cue), &opts()).unwrap().id() {
            Some(got) if got == *id => correct += 1,
            Some(_) => wrong += 1,
            None => {}
        }
    }
    let n = stored.len();
    assert!(correct as f32 / n as f32 > 0.97, "accuracy {correct}/{n}");
    assert!((wrong as f32) < 0.02 * n as f32, "false recall {wrong}/{n}");

    // Unseen random patterns must not be recalled.
    let false_hits = (0..500)
        .filter(|_| !mem.recall(Input::Code(&random_code(&mut rng)), &opts()).unwrap().is_miss())
        .count();
    assert_eq!(false_hits, 0);

    // The first memories are still intact after 2000 later writes.
    for (v, id) in stored.iter().take(20) {
        assert_eq!(mem.recall(Input::Code(v), &opts()).unwrap().id(), Some(*id));
    }
}
