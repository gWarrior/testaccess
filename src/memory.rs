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
    /// Override the novelty check (`None` = enabled iff configured).
    pub dedupe: Option<bool>,
    /// Link this memory as the successor of `after` (temporal association).
    pub after: Option<MemoryId>,
}

impl<P> Default for LearnOptions<P> {
    fn default() -> Self {
        Self { context: ContextId::DEFAULT, payload: None, ttl: None, pin: false, dedupe: None, after: None }
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
    /// The completed pattern (the engram's full ensemble).
    pub pattern: Vec<u32>,
    pub payload: Option<P>,
    /// Successors along the temporal chain (see `RecallOptions::follow`).
    pub sequence: Vec<MemoryId>,
}

/// Result of a recall. Empty `hits` means UNKNOWN.
#[derive(Clone, Debug)]
pub struct RecallResult<P> {
    pub hits: Vec<Hit<P>>,
    /// Candidates the index passed to the spiking stage.
    pub candidates: usize,
    /// Step at which the network settled.
    pub settle_step: usize,
    pub cue_size: usize,
}

impl<P> RecallResult<P> {
    fn unknown(cue_size: usize) -> Self {
        Self { hits: Vec::new(), candidates: 0, settle_step: 0, cue_size }
    }
    pub fn best(&self) -> Option<&Hit<P>> {
        self.hits.first()
    }
    pub fn id(&self) -> Option<MemoryId> {
        self.best().map(|h| h.id)
    }
    pub fn is_unknown(&self) -> bool {
        self.hits.is_empty()
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
    pub prev: Option<MemoryId>,
    pub next: Option<MemoryId>,
    pub pattern: &'a [u32],
    pub weights: Vec<f32>,
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
        if n == 0 {
            return Err(MemoryError::InvalidConfig("encoder has no neurons".into()));
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
        let fan_out = self.fast.postings.len(neuron) + self.long.postings.len(neuron);
        let total = self.fast.n_active() + self.long.n_active();
        (1.0 + (total as f32 + 1.0) / (fan_out as f32 + 1.0)).ln()
    }

    fn gains(&self, cue: &[u32]) -> Vec<f32> {
        cue.iter().map(|&n| self.gain(n)).collect()
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
            prev: NO_ID,
            next: NO_ID,
            len: code.len() as u32,
        };
        let weights = vec![1.0; code.len()];
        let slot = self.bank_mut(tier).insert(Engram { code, weights, meta, payload: opts.payload });
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
        let max_scan = self.cfg.max_scan;
        let mut best: Option<(f32, Loc)> = None;
        let mut found = Vec::new();
        for tier in TIERS {
            found.clear();
            self.bank_mut(tier).gather(code, &gains, now, Some(ctx), max_scan, &mut found);
            let bank = self.bank(tier);
            for &(score, slot) in &found {
                if score < 0.5 * thr * gsum {
                    continue;
                }
                let (cov, comp) = dynamics::similarity(code, &gains, bank.code(slot), &bank.weights(slot));
                let sym = cov.min(comp);
                if sym >= thr && best.map_or(true, |(s, _)| sym > s) {
                    best = Some((sym, Loc { tier, slot }));
                }
            }
        }
        best.map(|(_, l)| l)
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
        bank.reinforce(l.slot, code, &pcfg, l.tier == Tier::Fast);
        if let Some(p) = opts.payload {
            bank.set_payload(l.slot, p);
        }
        let m = &mut bank.meta[l.slot as usize];
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
        let thr = self.cfg.recall_threshold;
        if opts.top_k == 0 || opts.context.is_some_and(|c| self.check_context(c).is_err()) {
            return RecallResult::unknown(cue.len());
        }

        // Stage 1: index. Gain-weighted overlap through the cue's synapses.
        let gains = self.gains(cue);
        let gsum: f32 = gains.iter().sum();
        let max_scan = self.cfg.max_scan;
        let mut pool: Vec<(f32, Loc)> = Vec::new();
        let mut found = Vec::new();
        for tier in TIERS {
            found.clear();
            self.bank_mut(tier).gather(cue, &gains, now, opts.context, max_scan, &mut found);
            pool.extend(found.iter().map(|&(s, slot)| (s, Loc { tier, slot })));
        }
        let min_score = gsum * thr * self.cfg.prefilter;
        pool.retain(|c| c.0 >= min_score);
        let max_c = self.cfg.max_candidates;
        if pool.len() > max_c {
            pool.select_nth_unstable_by(max_c - 1, |a, b| b.0.total_cmp(&a.0));
            pool.truncate(max_c);
        }
        if pool.is_empty() {
            return RecallResult::unknown(cue.len());
        }

        // Stage 2: spiking dynamics over the candidates.
        let cands: Vec<Cand> = pool
            .iter()
            .map(|&(_, l)| {
                let bank = self.bank(l.tier);
                Cand {
                    code: bank.code(l.slot).to_vec(),
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
                    pattern: cands[j].code.clone(),
                    payload: self.bank(l.tier).payload(l.slot).cloned(),
                    sequence: self.successors(m.id, opts.follow, now),
                }
            })
            .collect();

        if opts.facilitate {
            self.working.last_state = settled.active;
            for h in &hits {
                self.working.facilitate(h.id, now);
            }
            if let Some(best) = hits.first() {
                self.on_recalled(best.id);
            }
        }
        RecallResult { hits, candidates: pool.len(), settle_step: settled.settle_step, cue_size: cue.len() }
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

    /// Forget every memory of a context. Long-term memories are kept unless
    /// `include_long_term`; pinned memories are always kept.
    pub fn reset_context(&mut self, ctx: ContextId, include_long_term: bool) -> usize {
        let mut victims = Vec::new();
        for tier in TIERS {
            if tier == Tier::LongTerm && !include_long_term {
                continue;
            }
            let bank = self.bank(tier);
            for slot in bank.active_slots() {
                let m = &bank.meta[slot as usize];
                if m.ctx == ctx && !m.pinned {
                    victims.push((m.id, Loc { tier, slot }));
                }
            }
        }
        for &(id, l) in &victims {
            self.remove(id, l);
        }
        self.counters.forgets += victims.len() as u64;
        self.maybe_cleanup();
        victims.len()
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
            prev: opt(m.prev),
            next: opt(m.next),
            pattern: bank.code(l.slot),
            weights: bank.weights(l.slot),
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

    /// Recently activated memories, newest first.
    pub fn recent(&self) -> Vec<MemoryId> {
        self.working.recent().collect()
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
            synapses: self.fast.postings.entries() + self.long.postings.entries(),
            max_fan_out: self.fast.postings.max_len().max(self.long.postings.max_len()),
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
