//! Recall dynamics: a small spiking network assembled on the fly from the
//! cue and the candidate engrams returned by the index.
//!
//! * **Ensemble layer** — binary spiking neurons. Cue neurons are clamped
//!   on; other ensemble neurons turn on when an active engram reinstates
//!   them (top-down feedback = pattern completion, concept §8).
//! * **Engram layer** — leaky integrate-and-fire neurons, one per candidate.
//!   The drive of engram `j` is the gain-weighted fraction of the currently
//!   active ensemble explained by its synapses (divisive normalisation):
//!
//!   `x_j = Σ_{i ∈ A ∩ E_j} g_i w_ij / Σ_{i ∈ A} g_i`
//!
//!   `g_i` is the homeostatic gain of neuron `i` and `w_ij ∈ {-1, 0, +1}`
//!   the ternary synapse: excitatory inputs are evidence for the memory,
//!   inhibitory inputs are evidence against it.
//! * **Inhibitory interneuron** — driven by the strongest engram that is in
//!   its attractor (fired recently), it inhibits all engrams equally
//!   (winner competition, concept §9). With inhibition `h`, a competitor
//!   survives only if its drive exceeds `h · x_winner + threshold`, so
//!   near-duplicates of the winner are still reported while weaker partial
//!   matches are silenced.
//!
//! When an engram fires, its ensemble is reinstated, its own drive rises
//! towards 1, and competitors' drives fall because the active set now
//! contains neurons they do not explain. The network settles into the
//! attractor of the best-matching memory. An engram that never reaches its
//! threshold is not recalled, so an unfamiliar cue yields UNKNOWN.

use crate::config::DynamicsConfig;

/// A candidate engram handed to the dynamics.
pub(crate) struct Cand {
    pub code: Vec<u32>,
    /// Ternary synaptic states as weights, aligned with `code`.
    pub w: Vec<f32>,
    /// Short-term facilitation gain (>= 1).
    pub gain: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct Outcome {
    /// Feed-forward similarity at `t = 0`: signed fraction of the cue
    /// explained (inhibitory synapses count against).
    pub similarity: f32,
    /// Same, counting excitatory synapses only.
    pub excitatory: f32,
    /// Fraction of the engram's excitatory synapses present in the cue.
    pub completeness: f32,
    pub spikes: u32,
    pub first_spike: Option<u32>,
}

pub(crate) struct Settled {
    pub outcomes: Vec<Outcome>,
    /// Last step at which the network state changed.
    pub settle_step: usize,
    /// Active ensemble neurons after settling (cue + completion).
    pub active: Vec<u32>,
}

/// Feed-forward match of one engram against a cue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Match {
    pub coverage: f32,
    pub excitatory: f32,
    pub completeness: f32,
}

/// Match of one engram (`code`, ternary `w`) against a sorted cue.
pub(crate) fn similarity(cue: &[u32], gains: &[f32], code: &[u32], w: &[f32]) -> Match {
    let gsum: f32 = gains.iter().sum();
    let wsum: f32 = w.iter().map(|&w| w.max(0.0)).sum();
    let (mut signed, mut exc, mut hit) = (0.0f32, 0.0f32, 0.0f32);
    for (n, &wi) in code.iter().zip(w) {
        if let Ok(p) = cue.binary_search(n) {
            signed += gains[p] * wi;
            exc += gains[p] * wi.max(0.0);
            hit += wi.max(0.0);
        }
    }
    let frac = |x: f32| if gsum > 0.0 { (x / gsum).clamp(-1.0, 1.0) } else { 0.0 };
    Match {
        coverage: frac(signed),
        excitatory: frac(exc),
        completeness: if wsum > 0.0 { hit / wsum } else { 0.0 },
    }
}

/// Firing threshold such that a constant drive of exactly `threshold`
/// reaches it on the last simulation step.
fn membrane_threshold(threshold: f32, cfg: &DynamicsConfig) -> f32 {
    let beta = (-1.0 / cfg.tau_m).exp();
    threshold.max(1e-4) * (1.0 - beta.powi(cfg.steps as i32)) / (1.0 - beta)
}

pub(crate) fn settle(
    cue: &[u32],
    cands: &[Cand],
    gain: &dyn Fn(u32) -> f32,
    threshold: f32,
    cfg: &DynamicsConfig,
) -> Settled {
    let mut local: Vec<u32> = cue.to_vec();
    for c in cands {
        local.extend_from_slice(&c.code);
    }
    local.sort_unstable();
    local.dedup();
    let g: Vec<f32> = local.iter().map(|&n| gain(n)).collect();
    let to_local = |n: &u32| local.binary_search(n).expect("neuron is in the local set") as u32;
    let cl: Vec<Vec<u32>> = cands.iter().map(|c| c.code.iter().map(to_local).collect()).collect();

    let mut cue_active = vec![false; local.len()];
    for n in cue {
        cue_active[to_local(n) as usize] = true;
    }
    let mut active = cue_active.clone();

    let drive = |active: &[bool]| -> Vec<f32> {
        let denom: f32 = g.iter().zip(active).filter(|(_, &a)| a).map(|(g, _)| g).sum();
        cl.iter()
            .zip(cands)
            .map(|(idx, c)| {
                let num: f32 = idx
                    .iter()
                    .zip(&c.w)
                    .filter(|(&i, _)| active[i as usize])
                    .map(|(&i, &w)| g[i as usize] * w)
                    .sum();
                if denom > 0.0 { (num / denom).clamp(-1.0, 1.0) } else { 0.0 }
            })
            .collect()
    };

    let sim0 = drive(&active);
    let cue_gains: Vec<f32> = cue.iter().map(|n| g[to_local(n) as usize]).collect();
    let mut outcomes: Vec<Outcome> = cands
        .iter()
        .zip(&sim0)
        .map(|(c, &similarity)| {
            let m = self::similarity(cue, &cue_gains, &c.code, &c.w);
            Outcome {
                similarity,
                excitatory: m.excitatory,
                completeness: m.completeness,
                spikes: 0,
                first_spike: None,
            }
        })
        .collect();

    let collect_active = |active: &[bool]| -> Vec<u32> {
        local.iter().zip(active).filter(|(_, &a)| a).map(|(&n, _)| n).collect()
    };

    if !cfg.enabled {
        for o in &mut outcomes {
            if o.similarity >= threshold {
                o.spikes = 1;
                o.first_spike = Some(0);
            }
        }
        return Settled { outcomes, settle_step: 0, active: collect_active(&active) };
    }

    let beta = (-1.0 / cfg.tau_m).exp();
    let theta = membrane_threshold(threshold, cfg);
    let m = cands.len();
    let (mut v, mut z) = (vec![0f32; m], vec![0f32; m]);
    let mut inh = 0f32;
    let mut x = sim0;
    let mut settle_step = 0;

    for t in 0..cfg.steps {
        if t > 0 {
            x = drive(&active);
        }
        let mut changed = false;
        for j in 0..m {
            v[j] = beta * v[j] + cands[j].gain * x[j] - inh;
            let spike = v[j] >= theta;
            if spike {
                v[j] = 0.0;
                outcomes[j].spikes += 1;
                if outcomes[j].first_spike.is_none() {
                    outcomes[j].first_spike = Some(t as u32);
                    changed = true;
                }
            }
            z[j] = cfg.rate_decay * z[j] + (1.0 - cfg.rate_decay) * spike as u8 as f32;
        }
        // The interneuron tracks the strongest engram that is currently in
        // its attractor (recently fired) and inhibits every engram equally.
        inh = cfg.inhibition
            * z.iter()
                .zip(&x)
                .filter(|(&z, _)| z >= cfg.completion_min_rate)
                .map(|(_, &x)| x)
                .fold(0.0, f32::max);

        if cfg.completion {
            let mut next = cue_active.clone();
            for j in (0..m).filter(|&j| z[j] >= cfg.completion_min_rate) {
                for (&i, &w) in cl[j].iter().zip(&cands[j].w) {
                    if w >= cfg.completion_min_weight {
                        next[i as usize] = true;
                    }
                }
            }
            if next != active {
                active = next;
                changed = true;
            }
        }
        if changed {
            settle_step = t;
        }
    }

    Settled { outcomes, settle_step, active: collect_active(&active) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(code: &[u32]) -> Cand {
        Cand { code: code.to_vec(), w: vec![1.0; code.len()], gain: 1.0 }
    }

    fn run(cue: &[u32], cands: &[Cand]) -> Settled {
        settle(cue, cands, &|_| 1.0, 0.3, &DynamicsConfig::default())
    }

    #[test]
    fn partial_cue_is_completed_and_wins() {
        let a: Vec<u32> = (0..20).collect();
        let b: Vec<u32> = (10..30).collect();
        // Cue: half of A, overlapping B on 5 neurons.
        let cue: Vec<u32> = (5..15).collect();
        let s = run(&cue, &[cand(&a), cand(&b)]);
        assert_eq!(s.outcomes[0].similarity, 1.0);
        assert!((s.outcomes[0].completeness - 0.5).abs() < 1e-6);
        assert!(s.outcomes[0].spikes > 0);
        assert!(s.outcomes[0].spikes > s.outcomes[1].spikes);
        for n in &a {
            assert!(s.active.contains(n), "A must be fully reinstated");
        }
    }

    #[test]
    fn weak_match_stays_silent() {
        let a: Vec<u32> = (0..20).collect();
        let cue: Vec<u32> = (18..40).collect(); // 2/22 of the cue explained
        let s = run(&cue, &[cand(&a)]);
        assert_eq!(s.outcomes[0].spikes, 0);
    }

    #[test]
    fn competition_suppresses_weaker_candidate() {
        let a: Vec<u32> = (0..20).collect();
        let b: Vec<u32> = (8..28).collect();
        let cue: Vec<u32> = (0..16).collect(); // A explains 100%, B 50%
        let s = run(&cue, &[cand(&a), cand(&b)]);
        assert!(s.outcomes[1].similarity >= 0.3, "B alone would be recalled");
        assert!(s.outcomes[0].spikes > 0);
        assert_eq!(s.outcomes[1].spikes, 0, "inhibition silences B");
    }

    #[test]
    fn similarity_matches_dynamics_initial_drive() {
        let a: Vec<u32> = vec![1, 3, 5, 7];
        let cue = vec![1, 2, 3];
        let gains = vec![2.0, 1.0, 1.0];
        let m = similarity(&cue, &gains, &a, &[1.0; 4]);
        assert!((m.coverage - 0.75).abs() < 1e-6);
        assert!((m.completeness - 0.5).abs() < 1e-6);
        let g = |n: u32| if n == 1 { 2.0 } else { 1.0 };
        let s = settle(&cue, &[cand(&a)], &g, 0.3, &DynamicsConfig::default());
        assert!((s.outcomes[0].similarity - m.coverage).abs() < 1e-6);
    }

    #[test]
    fn inhibitory_synapses_veto_a_match() {
        // A explains the whole cue, but two cue neurons are inhibitory
        // inputs of A: the evidence against it cancels the evidence for it.
        let a = Cand { code: vec![1, 2, 3, 4, 5, 6], w: vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0], gain: 1.0 };
        let cue = vec![1, 2, 5, 6];
        let m = similarity(&cue, &[1.0; 4], &a.code, &a.w);
        assert_eq!((m.coverage, m.excitatory), (0.0, 0.5));
        let s = settle(&cue, &[a], &|_| 1.0, 0.3, &DynamicsConfig::default());
        assert_eq!(s.outcomes[0].spikes, 0);
        assert_eq!(s.outcomes[0].excitatory, 0.5);
    }
}
