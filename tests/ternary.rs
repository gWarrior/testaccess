//! Ternary answers: +1 "уверен" (Known), 0 "не знаю" (Unknown),
//! -1 "точно нет" (Absent).

mod common;

use common::*;
use snn_memory::rng::SplitMix64;
use snn_memory::{Basis, Input, LearnOptions, MemoryConfig, RecallOptions, Verdict};

fn opts() -> RecallOptions {
    RecallOptions::default()
}

/// `take` neurons of `code` plus fresh random neurons up to `K`.
fn mixed(rng: &mut SplitMix64, code: &[u32], take: usize) -> Vec<u32> {
    let mut cue = partial(rng, code, take);
    while cue.len() < K {
        let n = rng.below(N_NEURONS as u64) as u32;
        if !code.contains(&n) && !cue.contains(&n) {
            cue.push(n);
        }
    }
    cue.sort_unstable();
    cue
}

#[test]
fn known_unknown_absent() {
    let (mut mem, _) = code_memory(MemoryConfig::default());
    let mut rng = SplitMix64::new(30);
    let a = random_code(&mut rng);
    let id = mem.learn(Input::Code(&a), LearnOptions::new()).unwrap();

    let r = mem.recall(Input::Code(&a), &opts()).unwrap();
    assert_eq!((r.verdict, r.basis, r.id()), (Verdict::Known, Basis::Match, Some(id)));
    assert_eq!(r.verdict.trit(), 1);

    // Nothing in memory overlaps this cue at all.
    let r = mem.recall(Input::Code(&random_code(&mut rng)), &opts()).unwrap();
    assert_eq!((r.verdict, r.basis), (Verdict::Absent, Basis::NoEvidence));
    assert_eq!(r.verdict.trit(), -1);

    // A cue weakly explained by A: evidence between the two thresholds
    // (reject 1/9, recall 2/9).
    let cue = mixed(&mut rng, &a, 6);
    let r = mem.recall(Input::Code(&cue), &opts()).unwrap();
    assert!(r.evidence > 1.0 / 9.0 && r.evidence < 2.0 / 9.0, "evidence {}", r.evidence);
    assert_eq!((r.verdict, r.basis), (Verdict::Unknown, Basis::Partial));
    assert_eq!(r.verdict.trit(), 0);
}

#[test]
fn forgotten_memory_is_definitely_absent() {
    let (mut mem, _) = code_memory(MemoryConfig::default());
    let mut rng = SplitMix64::new(31);
    let (a, b) = (random_code(&mut rng), random_code(&mut rng));
    mem.learn(Input::Code(&a), LearnOptions::new()).unwrap();
    let ib = mem.learn(Input::Code(&b), LearnOptions::new()).unwrap();
    mem.forget(ib);
    let r = mem.recall(Input::Code(&b), &opts()).unwrap();
    assert_eq!((r.verdict, r.basis), (Verdict::Absent, Basis::NoEvidence));
}

#[test]
fn negative_knowledge_answers_definitely_not() {
    let (mut mem, _) = code_memory(MemoryConfig::default());
    let mut rng = SplitMix64::new(32);
    let fact = random_code(&mut rng);
    let id = mem.learn(Input::Code(&fact), LearnOptions::new().negative().payload("false claim")).unwrap();

    let r = mem.recall(Input::Code(&partial(&mut rng, &fact, 18)), &opts()).unwrap();
    assert_eq!((r.verdict, r.basis), (Verdict::Absent, Basis::NegativeKnowledge));
    let hit = r.best().expect("the negative memory is the evidence");
    assert_eq!((hit.id, hit.polarity, hit.payload), (id, -1, Some("false claim")));
    assert_eq!(mem.get_memory(id).unwrap().polarity, -1);
}

#[test]
fn belief_revision_flips_polarity() {
    let (mut mem, _) = code_memory(MemoryConfig::default());
    let fact = random_code(&mut SplitMix64::new(33));
    let id = mem.learn(Input::Code(&fact), LearnOptions::new()).unwrap();
    assert!(mem.recall(Input::Code(&fact), &opts()).unwrap().is_known());

    assert_eq!(mem.learn(Input::Code(&fact), LearnOptions::new().negative()).unwrap(), id);
    let r = mem.recall(Input::Code(&fact), &opts()).unwrap();
    assert_eq!((r.verdict, r.id()), (Verdict::Absent, Some(id)));

    assert_eq!(mem.learn(Input::Code(&fact), LearnOptions::new()).unwrap(), id);
    assert!(mem.recall(Input::Code(&fact), &opts()).unwrap().is_known());
}

#[test]
fn suppression_vetoes_a_misleading_cue_only() {
    let (mut mem, _) = code_memory(MemoryConfig::default());
    let mut rng = SplitMix64::new(34);
    let a = random_code(&mut rng);
    let id = mem.learn(Input::Code(&a), LearnOptions::new()).unwrap();

    // 16 of A's 27 neurons plus foreign ones: recalled as A before correction.
    let misleading = mixed(&mut rng, &a, 16);
    assert_eq!(mem.recall(Input::Code(&misleading), &opts()).unwrap().id(), Some(id));

    // Every foreign cue neuron becomes an inhibitory input of A.
    let n = mem.suppress(id, Input::Code(&misleading)).unwrap();
    assert_eq!(n, K - 16);
    let states = mem.get_memory(id).unwrap().states;
    assert_eq!(states.iter().filter(|&&s| s == -1).count(), n);

    let r = mem.recall(Input::Code(&misleading), &opts()).unwrap();
    assert_eq!((r.verdict, r.basis), (Verdict::Absent, Basis::Inhibited));
    // A is still recalled from its own pattern and from clean partial cues.
    assert_eq!(mem.recall(Input::Code(&a), &opts()).unwrap().id(), Some(id));
    assert_eq!(mem.recall(Input::Code(&partial(&mut rng, &a, 12)), &opts()).unwrap().id(), Some(id));
}

#[test]
fn suppression_in_long_term_memory_is_fast_and_resettable() {
    let (mut mem, _) = code_memory(MemoryConfig::default());
    let mut rng = SplitMix64::new(35);
    let a = random_code(&mut rng);
    let id = mem.learn(Input::Code(&a), LearnOptions::new().pin()).unwrap();
    let misleading = mixed(&mut rng, &a, 16);
    mem.suppress(id, Input::Code(&misleading)).unwrap();
    assert!(mem.recall(Input::Code(&misleading), &opts()).unwrap().is_absent());

    // The correction lives in the fast synaptic component.
    mem.reset_fast_memory();
    assert_eq!(mem.recall(Input::Code(&misleading), &opts()).unwrap().id(), Some(id));
}
