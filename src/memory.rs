//! The memory manager: addressing, lifecycle and the public API
//! (concept §19): `learn`, `recall`, `forget`, `reset_context`,
//! `reset_fast_memory`, `reset_all`, `pin`, `unpin`, `get_memory`, `stats`.

use std::collections::HashMap;
use std::time::Instant;

use rayon::prelude::*;

use crate::bank::{Engram, EngramBank, SlotMeta, SlotStatus, NO_ID};
use crate::clock::{Clock, SystemClock};
use crate::config::MemoryConfig;
use crate::dynamics::{self, Cand};
use crate::encoder::{check_code, Encoder};
use crate::trit::TRYTE_STATES;
use crate::types::{ContextId, Input, MemoryError, MemoryId, Tier};
use crate::working::WorkingMemory;

const TIERS: [Tier; 2] = [Tier::Fast, Tier::LongTerm];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Loc {
    tier: Tier,
    slot: u32,
}

/// Options for [`SnnMemory::learn`].
#[derive(Clone, Debug)]
pub struct LearnOptions<P> {
    pub context: ContextId,
    pub payload: Option<P>,
    /// Lifetime in seconds; `None` uses `MemoryConfig::default_ttl`.
    pub ttl: Option<f64>,
    /// Store directly in long-term memory, protected from resets and TTL.
    pub pin: bool,
    /// Negative knowledge: recalling this memory answers "definitely not".
    pub negative: bool,
    /// Override the novelty check (`None` = enabled iff configured).
    pub dedupe: Option<bool>,
    /// Link this memory as the successor of `after` (temporal association).
    pub after: Option<MemoryId>,
}

impl<P> Default for LearnOptions<P> {
    fn default() -> Self {
        Self {
            context: ContextId::DEFAULT,
            payload: None,
            ttl: None,
            pin: false,
            negative: false,
            dedupe: None,
            after: None,
        }
    }
}

impl<P> LearnOptions<P> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn context(mut self, ctx: ContextId) -> Self {
        self.context = ctx;
        self
    }
    pub fn payload(mut self, p: P) -> Self {
        self.payload = Some(p);
        self
    }
    pub fn ttl(mut self, seconds: f64) -> Self {
        self.ttl = Some(seconds);
        self
    }
    pub fn pin(mut self) -> Self {
        self.pin = true;
        self
    }
    /// Mark the memory as negative knowledge ("this is false / absent").
    pub fn negative(mut self) -> Self {
        self.negative = true;
        self
    }
    pub fn dedupe(mut self, on: bool) -> Self {
        self.dedupe = Some(on);
        self
    }
    pub fn after(mut self, id: MemoryId) -> Self {
        self.after = Some(id);
        self
    }
}

/// Options for [`SnnMemory::learn_batch`].
#[derive(Clone, Debug, Default)]
pub struct BatchOptions {
    pub context: ContextId,
    pub ttl: Option<f64>,
    /// Chain the batch as a sequence (each item follows the previous one).
    pub link_sequence: bool,
    /// Predecessor of the first item when `link_sequence` is set.
    pub after: Option<MemoryId>,
    pub dedupe: Option<bool>,
}

/// Options for [`SnnMemory::recall`].
#[derive(Clone, Debug)]
pub struct RecallOptions {
    /// Restrict recall to one context (`None` = all contexts).
    pub context: Option<ContextId>,
    /// Maximum number of recalled memories.
    pub top_k: usize,
    /// Drop hits whose confidence is below this value.
    pub min_confidence: f32,
    /// Also return up to this many sequence successors of each hit.
    pub follow: usize,
    /// Update working memory and recall statistics.
    pub facilitate: bool,
}

impl Default for RecallOptions {
    fn default() -> Self {
        Self { context: None, top_k: 1, min_confidence: 0.0, follow: 0, facilitate: true }
    }
}

impl RecallOptions {
    pub fn in_context(ctx: ContextId) -> Self {
        Self { context: Some(ctx), ..Self::default() }
    }
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }
    pub fn follow(mut self, n: usize) -> Self {
        self.follow = n;
        self
    }
}

/// One recalled memory.
#[derive(Clone, Debug)]
pub struct Hit<P> {
    pub id: MemoryId,
    /// `similarity` scaled by the engram's firing relative to the winner.
    pub confidence: f32,
    /// Fraction of the cue explained by this memory.
    pub similarity: f32,
    /// Fraction of this memory present in the cue.
    pub completeness: f32,
    /// Spikes emitted by the engram during recall.
    pub spikes: u32,
    /// Simulation step of the first spike (recall latency in steps).
    pub first_spike: u32,
    pub context: ContextId,
    pub tier: Tier,
    pub created: f64,
    pub strength: f32,
    /// `+1` ordinary memory, `-1` negative knowledge.
    pub polarity: i8,
    /// The completed pattern (the engram's full ensemble).
    pub pattern: Vec<u32>,
    pub payload: Option<P>,
    /// Successors along the temporal chain (see `RecallOptions::follow`).
    pub sequence: Vec<MemoryId>,
}

/// Ternary answer of the memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Verdict {
    /// `+1` — "уверен": a memory settled into its attractor.
    Known,
    /// `0` — "не знаю": partial evidence, nothing recalled.
    Unknown,
    /// `-1` — "точно нет": evidence of absence.
    Absent,
}

impl Verdict {
    /// The verdict as a trit: `+1`, `0` or `-1`.
    pub fn trit(self) -> i8 {
        match self {
            Verdict::Known => 1,
            Verdict::Unknown => 0,
            Verdict::Absent => -1,
        }
    }
}

/// Why the memory answered the way it did.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Basis {
    /// A stored memory matched the cue (`Known`).
    Match,
    /// The recalled memory is negative knowledge (`Absent`).
    NegativeKnowledge,
    /// No stored memory explains even `reject_threshold` of the cue
    /// (`Absent`). For n-gram codes this proves the fragment is not stored.
    NoEvidence,
    /// A memory would match, but its inhibitory synapses veto this cue
    /// (`Absent`), see [`SnnMemory::suppress`].
    Inhibited,
    /// Some memories partially match, none convincingly (`Unknown`).
    Partial,
}

/// Result of a recall.
#[derive(Clone, Debug)]
pub struct RecallResult<P> {
    pub verdict: Verdict,
    pub basis: Basis,
    /// Upper bound on the fraction of the cue any stored memory explains.
    pub evidence: f32,
    pub hits: Vec<Hit<P>>,
    /// Candidates the index passed to the spiking stage.
    pub candidates: usize,
    /// Step at which the network settled.
    pub settle_step: usize,
    pub cue_size: usize,
}

impl<P> RecallResult<P> {
    fn miss(cue_size: usize, verdict: Verdict, basis: Basis, evidence: f32) -> Self {
        Self { verdict, basis, evidence, hits: Vec::new(), candidates: 0, settle_step: 0, cue_size }
    }
    pub fn best(&self) -> Option<&Hit<P>> {
        self.hits.first()
    }
    pub fn id(&self) -> Option<MemoryId> {
        self.best().map(|h| h.id)
    }
    /// Nothing was recalled (the verdict is `Unknown` or `Absent`).
    pub fn is_miss(&self) -> bool {
        self.hits.is_empty()
    }
    pub fn is_known(&self) -> bool {
        self.verdict == Verdict::Known
    }
    pub fn is_absent(&self) -> bool {
        self.verdict == Verdict::Absent
    }
}

/// A read-only view of a stored memory.
#[derive(Debug)]
pub struct MemoryRecord<'a, P> {
    pub id: MemoryId,
    pub context: ContextId,
    pub tier: Tier,
    pub created: f64,
    /// Absolute expiry time, if any.
    pub expires: Option<f64>,
    pub strength: f32,
    pub recalls: u32,
    pub pinned: bool,
    /// `+1` ordinary memory, `-1` negative knowledge.
    pub polarity: i8,
    pub prev: Option<MemoryId>,
    pub next: Option<MemoryId>,
    /// Presynaptic neurons (each id is one tryte).
    pub pattern: Vec<u32>,
    /// Ternary synapse states (`+1` excitatory, `0` silent, `-1` inhibitory),
    /// aligned with `pattern`.
    pub states: Vec<i8>,
    pub payload: Option<&'a P>,
}

/// Aggregate statistics (concept §21).
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub fast_memories: usize,
    pub long_term_memories: usize,
    pub pinned: usize,
    pub pending_cleanup: usize,
    pub free_slots: usize,
    pub slots: usize,
    pub contexts: usize,
    pub synapses: usize,
    pub max_fan_out: usize,
    pub working_traces: usize,
    pub learns: u64,
    pub recalls: u64,
    pub forgets: u64,
    pub avg_learn_us: f64,
    pub avg_recall_us: f64,
    pub approx_bytes: usize,
}

/// Outcome of [`SnnMemory::maintain`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MaintenanceReport {
    pub expired: usize,
    pub cleaned: usize,
}

#[derive(Default)]
struct Counters {
    learns: u64,
    recalls: u64,
    forgets: u64,
    learn_ns: u128,
    recall_ns: u128,
}

/// Fast SNN memory with an explicit memory lifecycle.
pub struct SnnMemory<P = ()> {
    cfg: MemoryConfig,
    encoder: Box<dyn Encoder>,
    n_neurons: u32,
    clock: Box<dyn Clock>,
    fast: EngramBank<P>,
    long: EngramBank<P>,
    loc: HashMap<MemoryId, Loc>,
    next_id: MemoryId,
    contexts: Vec<String>,
    context_ids: HashMap<String, ContextId>,
    working: WorkingMemory,
    counters: Counters,
}

fn encode_with(enc: &dyn Encoder, n_neurons: u32, input: Input<'_>) -> Result<Vec<u32>, MemoryError> {
    let mut out = Vec::new();
    match input {
        Input::Code(code) => {
            check_code(code, n_neurons)?;
            out.extend_from_slice(code);
            out.sort_unstable();
            out.dedup();
        }
        other => enc.encode(other, &mut out)?,
    }
    if out.is_empty() {
        Err(MemoryError::EmptyCode)
    } else {
        Ok(out)
    }
}

impl<P: Clone> SnnMemory<P> {
    pub fn new<E: Encoder + 'static>(encoder: E, cfg: MemoryConfig) -> Result<Self, MemoryError> {
        cfg.validate()?;
        let n = encoder.n_neurons();
        if n == 0 || n > TRYTE_STATES {
            return Err(MemoryError::InvalidConfig(format!(
                "neuron ids are one tryte: n_neurons must be in 1..={TRYTE_STATES}"
            )));
        }
        Ok(Self {
            fast: EngramBank::new(Tier::Fast, n, cfg.max_ensemble),
            long: EngramBank::new(Tier::LongTerm, n, cfg.max_ensemble),
            working: WorkingMemory::new(cfg.stp.clone()),
            encoder: Box::new(encoder),
            n_neurons: n,
            clock: Box::new(SystemClock::new()),
            loc: HashMap::new(),
            next_id: 1,
            contexts: vec!["default".to_string()],
            context_ids: HashMap::from([("default".to_string(), ContextId::DEFAULT)]),
            counters: Counters::default(),
            cfg,
        })
    }

    /// Replace the time source (e.g. with a [`ManualClock`](crate::ManualClock)).
    pub fn with_clock(mut self, clock: impl Clock + 'static) -> Self {
        self.clock = Box::new(clock);
        self
    }

    pub fn config(&self) -> &MemoryConfig {
        &self.cfg
    }

    pub fn n_neurons(&self) -> u32 {
        self.n_neurons
    }

    pub fn now(&self) -> f64 {
        self.clock.now()
    }

    // ----- contexts -------------------------------------------------------

    /// Intern a context name, creating it on first use.
    pub fn context(&mut self, name: &str) -> ContextId {
        if let Some(&id) = self.context_ids.get(name) {
            return id;
        }
        let id = ContextId(self.contexts.len() as u32);
        self.contexts.push(name.to_string());
        self.context_ids.insert(name.to_string(), id);
        id
    }

    pub fn find_context(&self, name: &str) -> Option<ContextId> {
        self.context_ids.get(name).copied()
    }

    pub fn context_name(&self, id: ContextId) -> Option<&str> {
        self.contexts.get(id.0 as usize).map(String::as_str)
    }

    fn check_context(&self, ctx: ContextId) -> Result<(), MemoryError> {
        if (ctx.0 as usize) < self.contexts.len() {
            Ok(())
        } else {
            Err(MemoryError::UnknownContext(ctx))
        }
    }

    // ----- encoding -------------------------------------------------------

    /// Encode an input into its spike code (sorted active neurons).
    pub fn encode(&self, input: Input<'_>) -> Result<Vec<u32>, MemoryError> {
        encode_with(&*self.encoder, self.n_neurons, input)
    }

    // ----- helpers --------------------------------------------------------

    fn bank(&self, tier: Tier) -> &EngramBank<P> {
        match tier {
            Tier::Fast => &self.fast,
            Tier::LongTerm => &self.long,
        }
    }

    fn bank_mut(&mut self, tier: Tier) -> &mut EngramBank<P> {
        match tier {
            Tier::Fast => &mut self.fast,
            Tier::LongTerm => &mut self.long,
        }
    }

    fn meta(&self, l: Loc) -> &SlotMeta {
        &self.bank(l.tier).meta[l.slot as usize]
    }

    fn meta_mut(&mut self, l: Loc) -> &mut SlotMeta {
        &mut self.bank_mut(l.tier).meta[l.slot as usize]
    }

    fn live_loc(&self, id: MemoryId, now: f64) -> Option<Loc> {
        self.loc.get(&id).copied().filter(|l| self.bank(l.tier).is_live(l.slot, now))
    }

    /// Homeostatic gain of a neuron: the more engrams it already projects
    /// to, the less evidence its spike carries (intrinsic plasticity; the
    /// neural analogue of inverse document frequency).
    fn gain(&self, neuron: u32) -> f32 {
        if !self.cfg.homeostasis {
            return 1.0;
        }
        let fan_out = self.fast.fan_out(neuron) + self.long.fan_out(neuron);
        let total = self.fast.n_active() + self.long.n_active();
        (1.0 + (total as f32 + 1.0) / (fan_out as f32 + 1.0)).ln()
    }

    fn gains(&self, cue: &[u32]) -> Vec<f32> {
        cue.iter().map(|&n| self.gain(n)).collect()
    }

    /// Rarest-first selection of the cue neurons the index stage scans.
    /// Returns the selected neurons, their gains and the total gain of the
    /// neurons left out (possible evidence the index did not look at).
    fn scan_plan(&self, cue: &[u32], gains: &[f32]) -> (Vec<u32>, Vec<f32>, f32) {
        let fan_out = |n: u32| self.fast.fan_out(n) + self.long.fan_out(n);
        let mut order: Vec<usize> = (0..cue.len()).collect();
        order.sort_by_key(|&i| fan_out(cue[i]));
        let (mut neurons, mut sel_gains, mut unscanned, mut spent) = (Vec::new(), Vec::new(), 0f32, 0usize);
        for i in order {
            let f = fan_out(cue[i]);
            if neurons.is_empty() || spent + f <= self.cfg.scan_budget {
                spent += f;
                neurons.push(cue[i]);
                sel_gains.push(gains[i]);
            } else {
                unscanned += gains[i];
            }
        }
        (neurons, sel_gains, unscanned)
    }

    fn deadline(&self, ttl: Option<f64>, now: f64) -> f64 {
        match ttl.or(self.cfg.default_ttl) {
            Some(t) => now + t,
            None => f64::INFINITY,
        }
    }

    fn link(&mut self, prev: MemoryId, next: MemoryId, now: f64) {
        let (Some(a), Some(b)) = (self.live_loc(prev, now), self.live_loc(next, now)) else {
            return;
        };
        self.meta_mut(a).next = next;
        self.meta_mut(b).prev = prev;
    }

    fn maybe_cleanup(&mut self) {
        let limit = self.cfg.auto_cleanup;
        if limit > 0 && self.fast.pending() + self.long.pending() >= limit {
            self.fast.cleanup();
            self.long.cleanup();
        }
    }

    fn remove(&mut self, id: MemoryId, l: Loc) {
        self.bank_mut(l.tier).mark_deleted(l.slot);
        self.loc.remove(&id);
        self.working.forget(id);
    }

    // ----- learn ----------------------------------------------------------

    /// Write a memory in one shot. It is recallable immediately.
    ///
    /// If de-duplication is on and a sufficiently similar memory already
    /// exists in the same context, that memory is reinforced instead and its
    /// id is returned.
    pub fn learn(&mut self, input: Input<'_>, opts: LearnOptions<P>) -> Result<MemoryId, MemoryError> {
        let start = Instant::now();
        self.check_context(opts.context)?;
        let code = self.encode(input)?;
        let id = self.learn_code(code, opts)?;
        self.counters.learns += 1;
        self.counters.learn_ns += start.elapsed().as_nanos();
        Ok(id)
    }

    /// Write many memories. Encoding runs in parallel.
    pub fn learn_batch(
        &mut self,
        inputs: &[Input<'_>],
        payloads: Option<Vec<P>>,
        opts: &BatchOptions,
    ) -> Result<Vec<MemoryId>, MemoryError> {
        let start = Instant::now();
        self.check_context(opts.context)?;
        if let Some(p) = &payloads {
            if p.len() != inputs.len() {
                return Err(MemoryError::InvalidConfig("payloads and inputs differ in length".into()));
            }
        }
        let (enc, n) = (&*self.encoder, self.n_neurons);
        let codes: Vec<Vec<u32>> =
            inputs.par_iter().map(|&i| encode_with(enc, n, i)).collect::<Result<_, _>>()?;
        if let Some(c) = codes.iter().find(|c| c.len() > self.cfg.max_ensemble) {
            return Err(MemoryError::EnsembleTooLarge { len: c.len(), max: self.cfg.max_ensemble });
        }

        let mut payloads = payloads.map(Vec::into_iter);
        let mut prev = if opts.link_sequence { opts.after } else { None };
        let mut ids = Vec::with_capacity(codes.len());
        for code in codes {
            let lo = LearnOptions {
                context: opts.context,
                payload: payloads.as_mut().and_then(Iterator::next),
                ttl: opts.ttl,
                pin: false,
                negative: false,
                dedupe: opts.dedupe,
                after: prev,
            };
            let id = self.learn_code(code, lo)?;
            if opts.link_sequence {
                prev = Some(id);
            }
            ids.push(id);
        }
        self.counters.learns += ids.len() as u64;
        self.counters.learn_ns += start.elapsed().as_nanos();
        Ok(ids)
    }

    fn learn_code(&mut self, code: Vec<u32>, opts: LearnOptions<P>) -> Result<MemoryId, MemoryError> {
        if code.len() > self.cfg.max_ensemble {
            return Err(MemoryError::EnsembleTooLarge { len: code.len(), max: self.cfg.max_ensemble });
        }
        let now = self.clock.now();
        let dedupe = opts.dedupe.unwrap_or(self.cfg.dedupe_threshold.is_some());
        if dedupe {
            let thr = self.cfg.dedupe_threshold.unwrap_or(0.9);
            if let Some(l) = self.find_similar(&code, opts.context, now, thr) {
                return self.reinforce(l, &code, opts, now);
            }
        }

        let id = self.next_id;
        self.next_id += 1;
        let tier = if opts.pin { Tier::LongTerm } else { Tier::Fast };
        let expires = if opts.pin { f64::INFINITY } else { self.deadline(opts.ttl, now) };
        let meta = SlotMeta {
            id,
            ctx: opts.context,
            status: SlotStatus::Active,
            created: now,
            expires,
            strength: 1.0,
            recalls: 0,
            pinned: opts.pin,
            polarity: if opts.negative { -1 } else { 1 },
            prev: NO_ID,
            next: NO_ID,
            len: code.len() as u32,
        };
        if self.bank(tier).is_full() {
            return Err(MemoryError::CapacityExceeded);
        }
        let states = vec![1; code.len()];
        let slot = self.bank_mut(tier).insert(Engram { code, states, meta, payload: opts.payload });
        self.loc.insert(id, Loc { tier, slot });
        if let Some(prev) = opts.after {
            self.link(prev, id, now);
        }
        self.working.facilitate(id, now);
        self.maybe_cleanup();
        Ok(id)
    }

    /// Novelty check: the most similar live memory in `ctx` whose symmetric
    /// similarity `min(coverage, completeness)` reaches `thr`.
    fn find_similar(&mut self, code: &[u32], ctx: ContextId, now: f64, thr: f32) -> Option<Loc> {
        let gains = self.gains(code);
        let gsum: f32 = gains.iter().sum();
        let (scan, scan_gains, unscanned) = self.scan_plan(code, &gains);
        let mut pool: Vec<(f32, Loc)> = Vec::new();
        let mut found = Vec::new();
        for tier in TIERS {
            found.clear();
            self.bank_mut(tier).gather(&scan, &scan_gains, &mut found);
            pool.extend(
                found
                    .iter()
                    .filter(|g| g.score + unscanned >= 0.5 * thr * gsum)
                    .map(|g| (g.score, Loc { tier, slot: g.slot })),
            );
        }
        let mut best: Option<(f32, Loc)> = None;
        for (_, l) in self.select_live(&mut pool, now, Some(ctx), self.cfg.max_candidates) {
            let bank = self.bank(l.tier);
            let m = dynamics::similarity(code, &gains, &bank.code(l.slot), &bank.weights(l.slot));
            let sym = m.coverage.min(m.completeness);
            if sym >= thr && best.map_or(true, |(s, _)| sym > s) {
                best = Some((sym, l));
            }
        }
        best.map(|(_, l)| l)
    }

    /// Up to `limit` live candidates (in `ctx`, if given) with the highest
    /// index scores. Liveness is only checked for the best-scored slots,
    /// widening the search window by 3x while too few are live.
    fn select_live(&self, pool: &mut [(f32, Loc)], now: f64, ctx: Option<ContextId>, limit: usize) -> Vec<(f32, Loc)> {
        let mut out = Vec::with_capacity(limit);
        let (mut start, mut window) = (0, 3 * limit.max(1));
        while out.len() < limit && start < pool.len() {
            let rest = &mut pool[start..];
            let w = window.min(rest.len());
            if w < rest.len() {
                rest.select_nth_unstable_by(w - 1, |a, b| b.0.total_cmp(&a.0));
            }
            rest[..w].sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(score, l) in &rest[..w] {
                if out.len() == limit {
                    break;
                }
                if self.bank(l.tier).accepts(l.slot, now, ctx) {
                    out.push((score, l));
                }
            }
            start += w;
            window *= 3;
        }
        out
    }

    fn reinforce(
        &mut self,
        l: Loc,
        code: &[u32],
        opts: LearnOptions<P>,
        now: f64,
    ) -> Result<MemoryId, MemoryError> {
        let pcfg = self.cfg.plasticity.clone();
        let deadline = self.deadline(opts.ttl, now);
        let bank = self.bank_mut(l.tier);
        bank.reinforce(l.slot, code, &pcfg);
        if let Some(p) = opts.payload {
            bank.set_payload(l.slot, p);
        }
        let m = &mut bank.meta[l.slot as usize];
        let polarity = if opts.negative { -1 } else { 1 };
        if m.polarity != polarity {
            // Belief revision: the same pattern now carries the opposite
            // polarity; the newest statement wins and starts over.
            m.polarity = polarity;
            m.strength = 0.0;
        }
        m.strength += 1.0;
        if l.tier == Tier::Fast {
            m.expires = deadline;
        }
        let id = m.id;
        if let Some(prev) = opts.after {
            self.link(prev, id, now);
        }
        if opts.pin {
            self.pin(id)?;
        }
        self.working.facilitate(id, now);
        Ok(id)
    }

    // ----- recall ---------------------------------------------------------

    /// Associative recall from a full or partial cue.
    pub fn recall(&mut self, input: Input<'_>, opts: &RecallOptions) -> Result<RecallResult<P>, MemoryError> {
        let start = Instant::now();
        let cue = self.encode(input)?;
        let result = self.recall_code(&cue, opts);
        self.counters.recalls += 1;
        self.counters.recall_ns += start.elapsed().as_nanos();
        Ok(result)
    }

    fn recall_code(&mut self, cue: &[u32], opts: &RecallOptions) -> RecallResult<P> {
        let now = self.clock.now();
        let (thr, reject) = (self.cfg.recall_threshold, self.cfg.reject_threshold);
        if opts.context.is_some_and(|c| self.check_context(c).is_err()) {
            return RecallResult::miss(cue.len(), Verdict::Absent, Basis::NoEvidence, 0.0);
        }
        if opts.top_k == 0 {
            return RecallResult::miss(cue.len(), Verdict::Unknown, Basis::Partial, 0.0);
        }

        // Stage 1: index. Gain-weighted input through the cue's synapses.
        let gains = self.gains(cue);
        let gsum: f32 = gains.iter().sum();
        // Neurons beyond the scan budget count as possible evidence, so that
        // "definitely not" stays sound.
        let (scan, scan_gains, unscanned) = self.scan_plan(cue, &gains);
        let mut pool: Vec<(f32, Loc)> = Vec::new();
        let mut found = Vec::new();
        let (mut best_exc, mut vetoed) = (0f32, false);
        for tier in TIERS {
            found.clear();
            self.bank_mut(tier).gather(&scan, &scan_gains, &mut found);
            let bank = self.bank(tier);
            for f in &found {
                // Only a running maximum is checked for liveness, so this
                // costs O(log touched) metadata reads on average.
                if f.excitatory > best_exc && bank.accepts(f.slot, now, opts.context) {
                    best_exc = f.excitatory;
                }
                vetoed |= f.excitatory >= thr * gsum
                    && f.score < thr * gsum
                    && bank.accepts(f.slot, now, opts.context);
                pool.push((f.score, Loc { tier, slot: f.slot }));
            }
        }
        let evidence = if gsum > 0.0 { ((best_exc + unscanned) / gsum).min(1.0) } else { 0.0 };
        let judge_miss = |vetoed: bool| {
            if evidence < reject {
                (Verdict::Absent, Basis::NoEvidence)
            } else if vetoed {
                (Verdict::Absent, Basis::Inhibited)
            } else {
                (Verdict::Unknown, Basis::Partial)
            }
        };

        let min_score = gsum * thr * self.cfg.prefilter - unscanned;
        pool.retain(|c| c.0 >= min_score);
        let pool = self.select_live(&mut pool, now, opts.context, self.cfg.max_candidates);
        if pool.is_empty() {
            let (verdict, basis) = judge_miss(vetoed);
            return RecallResult::miss(cue.len(), verdict, basis, evidence);
        }

        // Stage 2: spiking dynamics over the candidates.
        let cands: Vec<Cand> = pool
            .iter()
            .map(|&(_, l)| {
                let bank = self.bank(l.tier);
                Cand {
                    code: bank.code(l.slot),
                    w: bank.weights(l.slot),
                    gain: self.working.gain(self.meta(l).id, now),
                }
            })
            .collect();
        let settled = {
            let gain = |n: u32| self.gain(n);
            dynamics::settle(cue, &cands, &gain, thr, &self.cfg.dynamics)
        };

        // Competition outcome → hits.
        let max_spikes = settled.outcomes.iter().map(|o| o.spikes).max().unwrap_or(0);
        let mut ranked: Vec<(f32, usize)> = settled
            .outcomes
            .iter()
            .enumerate()
            .filter(|(_, o)| o.spikes > 0 && o.similarity >= thr)
            .map(|(j, o)| (o.similarity * o.spikes as f32 / max_spikes as f32, j))
            .filter(|&(c, _)| c >= opts.min_confidence)
            .collect();
        ranked.sort_by(|a, b| {
            let (oa, ob) = (&settled.outcomes[a.1], &settled.outcomes[b.1]);
            b.0.total_cmp(&a.0)
                .then(ob.completeness.total_cmp(&oa.completeness))
                .then(self.meta(pool[b.1].1).created.total_cmp(&self.meta(pool[a.1].1).created))
        });
        ranked.truncate(opts.top_k);

        let hits: Vec<Hit<P>> = ranked
            .iter()
            .map(|&(confidence, j)| {
                let (o, l) = (&settled.outcomes[j], pool[j].1);
                let m = self.meta(l);
                Hit {
                    id: m.id,
                    confidence,
                    similarity: o.similarity,
                    completeness: o.completeness,
                    spikes: o.spikes,
                    first_spike: o.first_spike.unwrap_or(0),
                    context: m.ctx,
                    tier: l.tier,
                    created: m.created,
                    strength: m.strength,
                    polarity: m.polarity,
                    pattern: cands[j].code.clone(),
                    payload: self.bank(l.tier).payload(l.slot).cloned(),
                    sequence: self.successors(m.id, opts.follow, now),
                }
            })
            .collect();

        if opts.facilitate {
            self.working.last_state = settled.active;
            // Best hit last, so it is the most recent working-memory item.
            for h in hits.iter().rev() {
                self.working.facilitate(h.id, now);
            }
            if let Some(best) = hits.first() {
                self.on_recalled(best.id);
            }
        }
        let (verdict, basis) = match hits.first() {
            Some(h) if h.polarity < 0 => (Verdict::Absent, Basis::NegativeKnowledge),
            Some(_) => (Verdict::Known, Basis::Match),
            None => judge_miss(vetoed),
        };
        RecallResult {
            verdict,
            basis,
            evidence,
            hits,
            candidates: pool.len(),
            settle_step: settled.settle_step,
            cue_size: cue.len(),
        }
    }

    fn on_recalled(&mut self, id: MemoryId) {
        let Some(&l) = self.loc.get(&id) else { return };
        let m = self.meta_mut(l);
        m.recalls += 1;
        let recalls = m.recalls;
        if l.tier == Tier::Fast && self.cfg.consolidate_after.is_some_and(|n| recalls >= n) {
            let _ = self.consolidate(id);
        }
    }

    /// Follow the temporal chain from `id` for up to `n` live successors.
    pub fn successors(&self, id: MemoryId, n: usize, now: f64) -> Vec<MemoryId> {
        let mut out = Vec::new();
        let mut cur = id;
        while out.len() < n {
            let Some(l) = self.live_loc(cur, now) else { break };
            let next = self.meta(l).next;
            if next == NO_ID || self.live_loc(next, now).is_none() {
                break;
            }
            out.push(next);
            cur = next;
        }
        out
    }

    // ----- forgetting -----------------------------------------------------

    /// Forget one memory. Logical deletion is immediate (O(1)); synapses are
    /// reclaimed by cleanup. Other memories are not affected.
    pub fn forget(&mut self, id: MemoryId) -> bool {
        let Some(&l) = self.loc.get(&id) else { return false };
        self.remove(id, l);
        self.counters.forgets += 1;
        self.maybe_cleanup();
        true
    }

    /// Anti-Hebbian correction ("this cue is *not* that memory"): cue
    /// neurons that are not excitatory inputs of `id` get inhibitory
    /// synapses, so this cue no longer recalls `id` and is answered with
    /// `Absent` / `Inhibited` instead. The memory itself stays recallable
    /// from its own pattern. Returns the number of inhibitory synapses.
    pub fn suppress(&mut self, id: MemoryId, cue: Input<'_>) -> Result<usize, MemoryError> {
        let code = self.encode(cue)?;
        let now = self.clock.now();
        let l = self.live_loc(id, now).ok_or(MemoryError::UnknownMemory(id))?;
        let max_new = self.cfg.plasticity.max_new_synapses.max(code.len());
        Ok(self.bank_mut(l.tier).inhibit(l.slot, &code, max_new))
    }

    /// Forget every memory of a context. Long-term memories are kept unless
    /// `include_long_term`; pinned memories are always kept. Returns the
    /// number of live memories removed (expired ones are swept silently).
    pub fn reset_context(&mut self, ctx: ContextId, include_long_term: bool) -> usize {
        let now = self.clock.now();
        let mut live = 0;
        let mut victims = Vec::new();
        for tier in TIERS {
            if tier == Tier::LongTerm && !include_long_term {
                continue;
            }
            let bank = self.bank(tier);
            for slot in bank.active_slots() {
                let m = &bank.meta[slot as usize];
                if m.ctx == ctx && !m.pinned {
                    live += (m.expires > now) as usize;
                    victims.push((m.id, Loc { tier, slot }));
                }
            }
        }
        for &(id, l) in &victims {
            self.remove(id, l);
        }
        self.counters.forgets += live as u64;
        self.maybe_cleanup();
        live
    }

    /// Clear all fast memory and working memory and the fast component of
    /// long-term synapses. Consolidated knowledge survives.
    pub fn reset_fast_memory(&mut self) -> usize {
        let ids: Vec<MemoryId> = self.fast.active_slots().map(|s| self.fast.meta[s as usize].id).collect();
        for id in &ids {
            self.loc.remove(id);
        }
        self.fast.clear();
        self.long.reset_fast_weights();
        self.working.clear();
        ids.len()
    }

    /// Clear every memory (contexts stay registered, ids are not reused).
    pub fn reset_all(&mut self) {
        self.fast.clear();
        self.long.clear();
        self.loc.clear();
        self.working.clear();
    }

    /// Expire memories past their TTL and physically clean up deleted ones.
    pub fn maintain(&mut self) -> MaintenanceReport {
        let now = self.clock.now();
        let mut expired = Vec::new();
        for tier in TIERS {
            let bank = self.bank(tier);
            for slot in bank.active_slots() {
                let m = &bank.meta[slot as usize];
                if m.expires <= now {
                    expired.push((m.id, Loc { tier, slot }));
                }
            }
        }
        for &(id, l) in &expired {
            self.remove(id, l);
        }
        let cleaned = self.fast.cleanup() + self.long.cleanup();
        MaintenanceReport { expired: expired.len(), cleaned }
    }

    // ----- consolidation --------------------------------------------------

    /// Move a fast memory into long-term memory. Returns `false` if it was
    /// already long-term.
    pub fn consolidate(&mut self, id: MemoryId) -> Result<bool, MemoryError> {
        let now = self.clock.now();
        let l = self.live_loc(id, now).ok_or(MemoryError::UnknownMemory(id))?;
        if l.tier == Tier::LongTerm {
            return Ok(false);
        }
        let mut e = self.fast.take(l.slot);
        e.meta.expires = f64::INFINITY;
        let slot = self.long.insert(e);
        self.loc.insert(id, Loc { tier: Tier::LongTerm, slot });
        self.maybe_cleanup();
        Ok(true)
    }

    /// Consolidate and protect a memory from TTL and context resets.
    pub fn pin(&mut self, id: MemoryId) -> Result<(), MemoryError> {
        self.consolidate(id)?;
        let l = self.loc[&id];
        self.meta_mut(l).pinned = true;
        Ok(())
    }

    /// Remove protection and return a pinned memory to the fast tier.
    pub fn unpin(&mut self, id: MemoryId) -> Result<bool, MemoryError> {
        let now = self.clock.now();
        let l = self.live_loc(id, now).ok_or(MemoryError::UnknownMemory(id))?;
        if !self.meta(l).pinned {
            return Ok(false);
        }
        let mut e = self.long.take(l.slot);
        e.meta.pinned = false;
        e.meta.expires = self.deadline(None, now);
        let slot = self.fast.insert(e);
        self.loc.insert(id, Loc { tier: Tier::Fast, slot });
        Ok(true)
    }

    // ----- inspection -----------------------------------------------------

    pub fn contains(&self, id: MemoryId) -> bool {
        self.live_loc(id, self.clock.now()).is_some()
    }

    pub fn get_memory(&self, id: MemoryId) -> Option<MemoryRecord<'_, P>> {
        let l = self.live_loc(id, self.clock.now())?;
        let bank = self.bank(l.tier);
        let m = self.meta(l);
        let opt = |x: MemoryId| (x != NO_ID).then_some(x);
        Some(MemoryRecord {
            id,
            context: m.ctx,
            tier: l.tier,
            created: m.created,
            expires: m.expires.is_finite().then_some(m.expires),
            strength: m.strength,
            recalls: m.recalls,
            pinned: m.pinned,
            polarity: m.polarity,
            prev: opt(m.prev),
            next: opt(m.next),
            pattern: bank.code(l.slot),
            states: bank.states(l.slot),
            payload: bank.payload(l.slot),
        })
    }

    pub fn payload(&self, id: MemoryId) -> Option<&P> {
        let l = self.live_loc(id, self.clock.now())?;
        self.bank(l.tier).payload(l.slot)
    }

    pub fn payload_mut(&mut self, id: MemoryId) -> Option<&mut P> {
        let l = self.live_loc(id, self.clock.now())?;
        self.bank_mut(l.tier).payload_mut(l.slot)
    }

    /// Number of live memories.
    pub fn len(&self) -> usize {
        let now = self.clock.now();
        self.loc.iter().filter(|(_, l)| self.bank(l.tier).is_live(l.slot, now)).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Ids of live memories (unordered).
    pub fn ids(&self) -> Vec<MemoryId> {
        let now = self.clock.now();
        self.loc.iter().filter(|(_, l)| self.bank(l.tier).is_live(l.slot, now)).map(|(&id, _)| id).collect()
    }

    /// Active neurons of the last settled recall (working-memory state).
    pub fn working_state(&self) -> &[u32] {
        &self.working.last_state
    }

    /// Recently activated live memories, newest first.
    pub fn recent(&self) -> Vec<MemoryId> {
        let mut seen = std::collections::HashSet::new();
        self.working.recent().filter(|&id| self.contains(id) && seen.insert(id)).collect()
    }

    pub fn stats(&self) -> Stats {
        let c = &self.counters;
        let avg = |ns: u128, n: u64| if n == 0 { 0.0 } else { ns as f64 / n as f64 / 1e3 };
        Stats {
            fast_memories: self.fast.n_active(),
            long_term_memories: self.long.n_active(),
            pinned: self.long.active_slots().filter(|&s| self.long.meta[s as usize].pinned).count(),
            pending_cleanup: self.fast.pending() + self.long.pending(),
            free_slots: self.fast.free_slots() + self.long.free_slots(),
            slots: self.fast.slots() + self.long.slots(),
            contexts: self.contexts.len(),
            synapses: self.fast.synapses() + self.long.synapses(),
            max_fan_out: self.fast.max_fan_out().max(self.long.max_fan_out()),
            working_traces: self.working.len(),
            learns: c.learns,
            recalls: c.recalls,
            forgets: c.forgets,
            avg_learn_us: avg(c.learn_ns, c.learns),
            avg_recall_us: avg(c.recall_ns, c.recalls),
            approx_bytes: self.fast.bytes() + self.long.bytes() + self.loc.capacity() * 32,
        }
    }
}
