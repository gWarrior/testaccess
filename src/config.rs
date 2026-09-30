//! Tunable parameters of the memory.

use crate::types::MemoryError;

/// LIF attractor dynamics used during recall (concept §8–9).
#[derive(Clone, Debug)]
pub struct DynamicsConfig {
    /// Run spiking dynamics. When `false`, recall is a single feed-forward
    /// pass (index scoring + threshold); useful as a baseline.
    pub enabled: bool,
    /// Simulation steps per recall.
    pub steps: usize,
    /// Membrane time constant of engram neurons, in steps.
    pub tau_m: f32,
    /// Strength of the shared inhibitory interneuron (winner competition).
    pub inhibition: f32,
    /// Decay of the per-engram firing-rate trace, in `(0, 1)`.
    pub rate_decay: f32,
    /// Let active engrams reinstate their full ensemble (pattern completion).
    pub completion: bool,
    /// Rate trace above which an engram counts as active: it drives
    /// completion and the inhibitory interneuron.
    pub completion_min_rate: f32,
    /// Synapses below this state do not complete (`0.5` = excitatory only).
    pub completion_min_weight: f32,
}

impl Default for DynamicsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            steps: 9,
            tau_m: 3.0,
            inhibition: 0.5,
            rate_decay: 0.8,
            completion: true,
            completion_min_rate: 0.05,
            completion_min_weight: 0.5,
        }
    }
}

/// Discrete plasticity of ternary synapses, applied when an existing memory
/// is reinforced. Transitions are probabilistic (bounded-synapse model), so
/// a single atypical presentation cannot rewrite a memory.
#[derive(Clone, Debug)]
pub struct PlasticityConfig {
    /// Probability that a co-active synapse steps up (`-1 → 0 → +1`) and that
    /// a newly active input gets wired in.
    pub p_potentiate: f32,
    /// Probability that an excitatory synapse whose input stayed silent
    /// becomes silent (`+1 → 0`).
    pub p_depress: f32,
    /// Maximum synapses created per reinforcement.
    pub max_new_synapses: usize,
}

impl Default for PlasticityConfig {
    fn default() -> Self {
        Self { p_potentiate: 1.0, p_depress: 0.35, max_new_synapses: 9 }
    }
}

/// Short-term facilitation (working-memory recency bias).
#[derive(Clone, Debug)]
pub struct StpConfig {
    /// Utilization increment per activation, in `(0, 1]`.
    pub utilization: f32,
    /// Facilitation decay time constant, seconds.
    pub tau: f64,
    /// Maximum relative gain added to a facilitated engram.
    pub max_gain: f32,
    /// Number of engrams tracked by working memory.
    pub capacity: usize,
}

impl Default for StpConfig {
    fn default() -> Self {
        Self { utilization: 0.3, tau: 27.0, max_gain: 0.15, capacity: 6561 }
    }
}

/// Top-level configuration.
#[derive(Clone, Debug)]
pub struct MemoryConfig {
    /// Maximum synapses of one engram (at most 243 = 3^5).
    pub max_ensemble: usize,
    /// Candidates passed from the index to the spiking stage.
    pub max_candidates: usize,
    /// Minimal similarity (fraction of the cue explained) to recall anything;
    /// below it the answer is UNKNOWN.
    pub recall_threshold: f32,
    /// Evidence below which the memory answers "definitely not": no stored
    /// memory explains even this fraction of the cue. Between
    /// `reject_threshold` and `recall_threshold` the answer is "unknown".
    pub reject_threshold: f32,
    /// Index-stage pre-filter, as a fraction of `recall_threshold`.
    pub prefilter: f32,
    /// Synapses the index stage may scan per query. Cue neurons are scanned
    /// rarest first (most informative first) until the budget is spent; the
    /// rest only enter the exact similarity of the candidates found.
    pub scan_budget: usize,
    /// Homeostatic scaling: frequently used neurons get lower gain.
    pub homeostasis: bool,
    /// Novelty check: a new input this similar to an existing memory (in
    /// the same context) reinforces it instead of creating a new one.
    /// `None` disables de-duplication.
    pub dedupe_threshold: Option<f32>,
    /// TTL for fast memories, seconds (`None` = no expiry).
    pub default_ttl: Option<f64>,
    /// Consolidate a fast memory after this many successful recalls.
    pub consolidate_after: Option<u32>,
    /// Run physical cleanup once this many deleted engrams are pending, or a
    /// ninth of the live memories if that is more (`0` = only on explicit
    /// `maintain`). Cleanup cost grows with the total number of synapses, so
    /// the proportional floor keeps its amortised cost per forget constant.
    pub auto_cleanup: usize,
    pub dynamics: DynamicsConfig,
    pub plasticity: PlasticityConfig,
    pub stp: StpConfig,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_ensemble: 81,
            max_candidates: 27,
            recall_threshold: 2.0 / 9.0,
            reject_threshold: 1.0 / 9.0,
            prefilter: 0.5,
            scan_budget: 19_683,
            homeostasis: true,
            dedupe_threshold: Some(0.9),
            default_ttl: None,
            consolidate_after: Some(9),
            auto_cleanup: 729,
            dynamics: DynamicsConfig::default(),
            plasticity: PlasticityConfig::default(),
            stp: StpConfig::default(),
        }
    }
}

impl MemoryConfig {
    pub fn validate(&self) -> Result<(), MemoryError> {
        let bad = |m: &str| Err(MemoryError::InvalidConfig(m.to_string()));
        if self.max_ensemble == 0 || self.max_ensemble > crate::index::MAX_WIDTH {
            return bad("max_ensemble must be in 1..=243");
        }
        if self.max_candidates == 0 {
            return bad("max_candidates must be > 0");
        }
        if !(self.recall_threshold > 0.0 && self.recall_threshold <= 1.0) {
            return bad("recall_threshold must be in (0, 1]");
        }
        if !(0.0..=self.recall_threshold).contains(&self.reject_threshold) {
            return bad("reject_threshold must be in [0, recall_threshold]");
        }
        if !(0.0..=1.0).contains(&self.prefilter) {
            return bad("prefilter must be in [0, 1]");
        }
        if let Some(t) = self.dedupe_threshold {
            if !(t > 0.0 && t <= 1.0) {
                return bad("dedupe_threshold must be in (0, 1]");
            }
        }
        if let Some(t) = self.default_ttl {
            if t.is_nan() || t <= 0.0 {
                return bad("default_ttl must be > 0");
            }
        }
        let d = &self.dynamics;
        if d.steps == 0 || d.tau_m <= 0.0 || !(0.0..1.0).contains(&d.rate_decay) || d.inhibition < 0.0 {
            return bad("dynamics: steps > 0, tau_m > 0, rate_decay in [0,1), inhibition >= 0");
        }
        let p = &self.plasticity;
        if !(0.0..=1.0).contains(&p.p_potentiate) || !(0.0..=1.0).contains(&p.p_depress) {
            return bad("plasticity: probabilities must be in [0, 1]");
        }
        if self.stp.tau <= 0.0 || !(0.0..=1.0).contains(&self.stp.utilization) {
            return bad("stp: tau > 0, utilization in [0,1]");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_valid_and_bad_values_rejected() {
        assert!(MemoryConfig::default().validate().is_ok());
        let cfg = MemoryConfig { recall_threshold: 0.0, ..Default::default() };
        assert!(cfg.validate().is_err());
        let cfg = MemoryConfig { default_ttl: Some(-1.0), ..Default::default() };
        assert!(cfg.validate().is_err());
    }
}
