//! Engram bank: a population of engram (memory) neurons and the ternary
//! synapses they own.
//!
//! Every memory gets its own engram neuron with a private set of afferent
//! synapses (concept §14, variants A + C: ownership + sparse allocation).
//! No synapse is shared between memories, so forgetting one memory removes
//! exactly its own contribution and cannot damage any other memory.
//!
//! Storage is ternary. A synapse is addressed by its presynaptic neuron, one
//! [`Tryte`](crate::trit::Tryte) (9 trits), and holds two trits of state, a
//! slow and a fast component (concept §7), densely packed in a
//! [`TritVec`] (40 trits per 64-bit word):
//! `effective = clamp(slow + fast, -1, 1)` ∈ {`-1` inhibitory, `0` silent,
//! `+1` excitatory}. Fast-tier engrams live in the fast component;
//! consolidated engrams keep their knowledge in the slow one, while later
//! plasticity lands in the fast component and is wiped by a fast reset.
//! Plasticity is a discrete state machine (in the spirit of Amit–Fusi
//! bounded synapses) rather than a real-valued update.

use crate::config::PlasticityConfig;
use crate::index::{entry, entry_slot, PostingIndex, MAX_SLOTS};
use crate::rng::SplitMix64;
use crate::trit::TritVec;
use crate::types::{ContextId, MemoryId, Tier};

pub(crate) const NO_ID: MemoryId = 0;
const NO_NEURON: u16 = u16::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlotStatus {
    Free,
    Active,
    /// Logically deleted, waiting for physical cleanup.
    Deleted,
}

#[derive(Clone, Debug)]
pub(crate) struct SlotMeta {
    pub id: MemoryId,
    pub ctx: ContextId,
    pub status: SlotStatus,
    pub created: f64,
    pub expires: f64,
    pub strength: f32,
    pub recalls: u32,
    pub pinned: bool,
    /// `+1` for ordinary memories, `-1` for negative knowledge.
    pub polarity: i8,
    pub prev: MemoryId,
    pub next: MemoryId,
    pub len: u32,
}

impl SlotMeta {
    pub fn new(id: MemoryId) -> Self {
        Self {
            id,
            ctx: ContextId::DEFAULT,
            status: SlotStatus::Free,
            created: 0.0,
            expires: f64::INFINITY,
            strength: 0.0,
            recalls: 0,
            pinned: false,
            polarity: 1,
            prev: NO_ID,
            next: NO_ID,
            len: 0,
        }
    }
}

/// A memory detached from its bank (used to move between tiers).
pub(crate) struct Engram<P> {
    pub code: Vec<u32>,
    /// Effective ternary synapse states, aligned with `code`.
    pub states: Vec<i8>,
    pub meta: SlotMeta,
    pub payload: Option<P>,
}

/// Per-slot accumulator of the index stage (one cache line access per
/// synapse visited).
#[derive(Clone, Copy, Debug, Default)]
struct Acc {
    score: f32,
    exc: f32,
    stamp: u32,
}

/// Evidence one engram receives from a cue through its synapses.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gathered {
    /// Signed gain-weighted input (inhibitory synapses subtract).
    pub score: f32,
    /// Gain-weighted input through excitatory synapses only.
    pub excitatory: f32,
    pub slot: u32,
}

pub(crate) struct EngramBank<P> {
    tier: Tier,
    width: usize,
    /// Presynaptic neuron of every synapse position, one tryte each.
    ens: Vec<u16>,
    /// Two trits per synapse position: `[slow, fast]`.
    syn: TritVec,
    pub meta: Vec<SlotMeta>,
    payload: Vec<Option<P>>,
    free: Vec<u32>,
    pending: Vec<u32>,
    /// Index of excitatory synapses by presynaptic neuron.
    exc: PostingIndex,
    /// Index of inhibitory synapses. Silent synapses are not indexed, so
    /// the index stage never has to read a synapse state.
    inh: PostingIndex,
    n_active: usize,
    acc: Vec<Acc>,
    epoch: u32,
    touched: Vec<u32>,
    rng: SplitMix64,
}

impl<P> EngramBank<P> {
    pub fn new(tier: Tier, n_neurons: u32, width: usize) -> Self {
        Self {
            tier,
            width,
            ens: Vec::new(),
            syn: TritVec::new(),
            meta: Vec::new(),
            payload: Vec::new(),
            free: Vec::new(),
            pending: Vec::new(),
            exc: PostingIndex::new(n_neurons),
            inh: PostingIndex::new(n_neurons),
            n_active: 0,
            acc: Vec::new(),
            epoch: 0,
            touched: Vec::new(),
            rng: SplitMix64::new(0x5EED ^ width as u64),
        }
    }

    /// Engrams with `Active` status (including expired-but-unswept ones).
    pub fn n_active(&self) -> usize {
        self.n_active
    }

    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    pub fn free_slots(&self) -> usize {
        self.free.len()
    }

    pub fn slots(&self) -> usize {
        self.meta.len()
    }

    pub fn is_full(&self) -> bool {
        self.free.is_empty() && self.meta.len() >= MAX_SLOTS
    }

    pub fn insert(&mut self, e: Engram<P>) -> u32 {
        debug_assert!(e.code.len() <= self.width && e.code.len() == e.states.len());
        let slot = match self.free.pop() {
            Some(s) => s,
            None => {
                assert!(self.meta.len() < MAX_SLOTS, "engram bank is full");
                let s = self.meta.len() as u32;
                self.meta.push(SlotMeta::new(NO_ID));
                self.payload.push(None);
                self.ens.extend(std::iter::repeat(NO_NEURON).take(self.width));
                self.syn.extend_zeros(2 * self.width);
                self.acc.push(Acc::default());
                s
            }
        };
        let (s, base, len) = (slot as usize, slot as usize * self.width, e.code.len());
        for (i, (&n, &t)) in e.code.iter().zip(&e.states).enumerate() {
            self.ens[base + i] = n as u16;
            let (slow, fast) = match self.tier {
                Tier::Fast => (0, t),
                Tier::LongTerm => (t, 0),
            };
            self.syn.set(2 * (base + i), slow);
            self.syn.set(2 * (base + i) + 1, fast);
        }
        self.meta[s] = SlotMeta { status: SlotStatus::Active, len: len as u32, ..e.meta };
        self.payload[s] = e.payload;
        for (pos, (&n, &t)) in e.code.iter().zip(&e.states).enumerate() {
            self.index_add(n, slot, pos, t);
        }
        self.n_active += 1;
        slot
    }

    fn index_add(&mut self, neuron: u32, slot: u32, pos: usize, state: i8) {
        match state {
            1 => self.exc.add(neuron, entry(slot, pos)),
            -1 => self.inh.add(neuron, entry(slot, pos)),
            _ => {}
        }
    }

    fn index_remove(&mut self, neuron: u32, slot: u32, pos: usize, state: i8) {
        match state {
            1 => self.exc.remove(neuron, entry(slot, pos)),
            -1 => self.inh.remove(neuron, entry(slot, pos)),
            _ => true,
        };
    }

    /// Synapses (excitatory + inhibitory) leaving `neuron`.
    #[inline]
    pub fn fan_out(&self, neuron: u32) -> usize {
        self.exc.len(neuron) + self.inh.len(neuron)
    }

    /// Indexed (non-silent) synapses.
    pub fn synapses(&self) -> usize {
        self.exc.entries() + self.inh.entries()
    }

    pub fn max_fan_out(&self) -> usize {
        self.exc.max_len().max(self.inh.max_len())
    }

    #[inline]
    fn raw_code(&self, slot: u32) -> &[u16] {
        let base = slot as usize * self.width;
        &self.ens[base..base + self.meta[slot as usize].len as usize]
    }

    /// Presynaptic neurons of `slot`, in synapse-position order.
    pub fn code(&self, slot: u32) -> Vec<u32> {
        self.raw_code(slot).iter().map(|&n| n as u32).collect()
    }

    /// Effective state of synapse position `idx` (`slot * width + pos`).
    #[inline]
    fn eff(syn: &TritVec, idx: usize) -> i8 {
        (syn.get(2 * idx) + syn.get(2 * idx + 1)).clamp(-1, 1)
    }

    /// Effective ternary states of `slot`'s synapses.
    pub fn states(&self, slot: u32) -> Vec<i8> {
        let base = slot as usize * self.width;
        let len = self.meta[slot as usize].len as usize;
        (base..base + len).map(|i| Self::eff(&self.syn, i)).collect()
    }

    /// Effective states as `f32` weights for the dynamics.
    pub fn weights(&self, slot: u32) -> Vec<f32> {
        self.states(slot).into_iter().map(f32::from).collect()
    }

    pub fn payload(&self, slot: u32) -> Option<&P> {
        self.payload[slot as usize].as_ref()
    }

    pub fn payload_mut(&mut self, slot: u32) -> Option<&mut P> {
        self.payload[slot as usize].as_mut()
    }

    pub fn set_payload(&mut self, slot: u32, p: P) {
        self.payload[slot as usize] = Some(p);
    }

    #[inline]
    pub fn is_live(&self, slot: u32, now: f64) -> bool {
        let m = &self.meta[slot as usize];
        m.status == SlotStatus::Active && m.expires > now
    }

    /// Slots with `Active` status.
    pub fn active_slots(&self) -> impl Iterator<Item = u32> + '_ {
        self.meta.iter().enumerate().filter(|(_, m)| m.status == SlotStatus::Active).map(|(s, _)| s as u32)
    }

    /// Logical deletion: O(1). The engram is silenced immediately; its
    /// synapses are reclaimed later by [`cleanup`](Self::cleanup).
    pub fn mark_deleted(&mut self, slot: u32) -> bool {
        let s = slot as usize;
        if self.meta[s].status != SlotStatus::Active {
            return false;
        }
        self.meta[s].status = SlotStatus::Deleted;
        self.payload[s] = None;
        self.pending.push(slot);
        self.n_active -= 1;
        true
    }

    /// Detach a memory from this bank (it is logically deleted here).
    pub fn take(&mut self, slot: u32) -> Engram<P> {
        let e = Engram {
            code: self.code(slot),
            states: self.states(slot),
            meta: self.meta[slot as usize].clone(),
            payload: self.payload[slot as usize].take(),
        };
        self.mark_deleted(slot);
        e
    }

    /// Physical cleanup: drop synapses of deleted engrams and free slots.
    pub fn cleanup(&mut self) -> usize {
        if self.pending.is_empty() {
            return 0;
        }
        let mut neurons: Vec<u32> = Vec::new();
        for &slot in &self.pending {
            neurons.extend(self.raw_code(slot).iter().map(|&n| n as u32));
        }
        neurons.sort_unstable();
        neurons.dedup();
        let meta = &self.meta;
        let dead = |s: u32| meta[s as usize].status == SlotStatus::Deleted;
        self.exc.purge(&neurons, dead);
        self.inh.purge(&neurons, dead);

        let n = self.pending.len();
        for slot in std::mem::take(&mut self.pending) {
            let (s, base) = (slot as usize, slot as usize * self.width);
            self.ens[base..base + self.width].fill(NO_NEURON);
            self.syn.clear_range(2 * base, 2 * (base + self.width));
            self.meta[s] = SlotMeta::new(NO_ID);
            self.free.push(slot);
        }
        n
    }

    /// Drop everything (O(neurons + slots)).
    pub fn clear(&mut self) {
        *self = Self::new(self.tier, self.exc.n_neurons(), self.width);
    }

    /// Clear the fast component of every synapse and re-index.
    pub fn reset_fast_weights(&mut self) {
        for i in 0..self.syn.len() / 2 {
            self.syn.set(2 * i + 1, 0);
        }
        let n = self.exc.n_neurons();
        self.exc = PostingIndex::new(n);
        self.inh = PostingIndex::new(n);
        let slots: Vec<u32> = self.active_slots().collect();
        for slot in slots {
            let base = slot as usize * self.width;
            for pos in 0..self.meta[slot as usize].len as usize {
                let (neuron, state) = (self.ens[base + pos] as u32, Self::eff(&self.syn, base + pos));
                self.index_add(neuron, slot, pos, state);
            }
        }
    }

    /// Whether `slot` is live and belongs to `ctx` (if given).
    #[inline]
    pub fn accepts(&self, slot: u32, now: f64, ctx: Option<ContextId>) -> bool {
        let m = &self.meta[slot as usize];
        m.status == SlotStatus::Active && m.expires > now && ctx.map_or(true, |c| c == m.ctx)
    }

    /// Index stage: accumulate the signed, gain-weighted input every engram
    /// receives from the cue through its excitatory and inhibitory synapses.
    /// Appends every touched slot; liveness is checked later, and only for
    /// the few best candidates (see [`accepts`](Self::accepts)).
    pub fn gather(&mut self, cue: &[u32], gains: &[f32], out: &mut Vec<Gathered>) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.acc.iter_mut().for_each(|a| a.stamp = 0);
            self.epoch = 1;
        }
        let Self { exc, inh, acc, epoch, touched, .. } = self;
        let mut visit = |list: &[u32], dscore: f32, dexc: f32| {
            for &e in list {
                let s = entry_slot(e);
                let a = &mut acc[s as usize];
                if a.stamp != *epoch {
                    *a = Acc { score: 0.0, exc: 0.0, stamp: *epoch };
                    touched.push(s);
                }
                a.score += dscore;
                a.exc += dexc;
            }
        };
        for (&n, &g) in cue.iter().zip(gains) {
            visit(exc.list(n), g, g);
            visit(inh.list(n), -g, 0.0);
        }
        out.extend(touched.iter().map(|&s| {
            let a = acc[s as usize];
            Gathered { score: a.score, excitatory: a.exc, slot: s }
        }));
        touched.clear();
    }

    /// Move the effective state of `pos` in `slot` towards `target` by
    /// changing the fast component only, keeping the index in sync.
    fn set_state(&mut self, slot: u32, pos: usize, target: i8) {
        let idx = slot as usize * self.width + pos;
        let old = Self::eff(&self.syn, idx);
        let slow = self.syn.get(2 * idx);
        self.syn.set(2 * idx + 1, (target - slow).clamp(-1, 1));
        let new = Self::eff(&self.syn, idx);
        if new != old {
            let neuron = self.ens[idx] as u32;
            self.index_remove(neuron, slot, pos, old);
            self.index_add(neuron, slot, pos, new);
        }
    }

    /// A free position, or else the first silent synapse, if any. Silent
    /// synapses are recycled last: they keep a trace that co-activity can
    /// revive.
    fn vacant_position(&self, slot: u32) -> Option<usize> {
        let (base, len) = (slot as usize * self.width, self.meta[slot as usize].len as usize);
        (len < self.width).then_some(len).or_else(|| (0..len).find(|&i| Self::eff(&self.syn, base + i) == 0))
    }

    /// Wire neuron `n` into position `pos` of `slot` with state `target`.
    fn place(&mut self, slot: u32, pos: usize, n: u32, target: i8) {
        let base = slot as usize * self.width;
        let len = self.meta[slot as usize].len as usize;
        if pos < len {
            let old = Self::eff(&self.syn, base + pos);
            self.index_remove(self.ens[base + pos] as u32, slot, pos, old);
        } else {
            self.meta[slot as usize].len += 1;
        }
        self.ens[base + pos] = n as u16;
        self.syn.set(2 * (base + pos), 0);
        self.syn.set(2 * (base + pos) + 1, target);
        self.index_add(n, slot, pos, target);
    }

    /// Reinforce an existing engram with a new presentation `cue` (sorted).
    ///
    /// Discrete, rate-based STDP on ternary synapses:
    /// * presynaptic neuron active together with the engram: a silent
    ///   synapse becomes excitatory, an inhibitory one becomes silent
    ///   (probability `p_potentiate`);
    /// * presynaptic neuron silent: an excitatory synapse becomes silent
    ///   (probability `p_depress`);
    /// * newly active neurons are wired in as excitatory synapses on free or
    ///   silent positions (synaptogenesis / turnover).
    pub fn reinforce(&mut self, slot: u32, cue: &[u32], cfg: &PlasticityConfig) {
        let base = slot as usize * self.width;
        let len = self.meta[slot as usize].len as usize;
        for i in 0..len {
            let idx = base + i;
            let w = Self::eff(&self.syn, idx);
            let active = cue.binary_search(&(self.ens[idx] as u32)).is_ok();
            if active && w < 1 && self.chance(cfg.p_potentiate) {
                self.set_state(slot, i, w + 1);
            } else if !active && w == 1 && self.chance(cfg.p_depress) {
                self.set_state(slot, i, 0);
            }
        }
        let mut present = self.code(slot);
        present.sort_unstable();
        let novel: Vec<u32> = cue.iter().copied().filter(|n| present.binary_search(n).is_err()).collect();
        let mut added = 0;
        for n in novel {
            if added == cfg.max_new_synapses || !self.chance(cfg.p_potentiate) {
                continue;
            }
            let Some(pos) = self.vacant_position(slot) else { break };
            self.place(slot, pos, n, 1);
            added += 1;
        }
    }

    /// Anti-Hebbian correction: make `cue` count as evidence *against* this
    /// engram. Cue neurons that are not excitatory inputs get inhibitory
    /// synapses. Returns the number of synapses turned inhibitory.
    pub fn inhibit(&mut self, slot: u32, cue: &[u32], max_new: usize) -> usize {
        let base = slot as usize * self.width;
        let mut changed = 0;
        for &n in cue {
            if changed == max_new {
                break;
            }
            let len = self.meta[slot as usize].len as usize;
            match (0..len).find(|&i| self.ens[base + i] as u32 == n) {
                Some(i) if Self::eff(&self.syn, base + i) == 1 => {}
                Some(i) if Self::eff(&self.syn, base + i) == 0 => {
                    self.set_state(slot, i, -1);
                    changed += 1;
                }
                Some(_) => {}
                None => {
                    let Some(pos) = self.vacant_position(slot) else { break };
                    self.place(slot, pos, n, -1);
                    changed += 1;
                }
            }
        }
        changed
    }

    fn chance(&mut self, p: f32) -> bool {
        p >= 1.0 || (p > 0.0 && self.rng.next_f64() < p as f64)
    }

    /// Approximate heap usage in bytes.
    pub fn bytes(&self) -> usize {
        self.ens.capacity() * 2
            + self.syn.bytes()
            + self.acc.capacity() * std::mem::size_of::<Acc>()
            + self.meta.capacity() * std::mem::size_of::<SlotMeta>()
            + self.payload.capacity() * std::mem::size_of::<Option<P>>()
            + self.exc.bytes()
            + self.inh.bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engram(id: MemoryId, code: &[u32]) -> Engram<&'static str> {
        Engram { code: code.to_vec(), states: vec![1; code.len()], meta: SlotMeta::new(id), payload: Some("p") }
    }

    fn gather(bank: &mut EngramBank<&'static str>, cue: &[u32]) -> Vec<(f32, u32)> {
        let mut out = Vec::new();
        bank.gather(cue, &vec![1.0; cue.len()], &mut out);
        out.retain(|g| bank.accepts(g.slot, 0.0, None));
        let mut v: Vec<(f32, u32)> = out.iter().map(|g| (g.score, g.slot)).collect();
        v.sort_by_key(|&(_, s)| s);
        v
    }

    fn certain() -> PlasticityConfig {
        PlasticityConfig { p_potentiate: 1.0, p_depress: 1.0, max_new_synapses: 9 }
    }

    #[test]
    fn insert_gather_delete_cleanup_reuse() {
        let mut b = EngramBank::new(Tier::Fast, 81, 9);
        let a = b.insert(engram(1, &[1, 2, 3, 4]));
        let c = b.insert(engram(2, &[3, 4, 5, 6]));
        assert_eq!(gather(&mut b, &[3, 4, 5]), vec![(2.0, a), (3.0, c)]);

        assert!(b.mark_deleted(a));
        assert!(!b.mark_deleted(a));
        assert_eq!(gather(&mut b, &[3, 4, 5]), vec![(3.0, c)], "deleted engram is silent at once");
        assert_eq!(b.payload(a), None);

        assert_eq!(b.cleanup(), 1);
        assert_eq!(b.synapses(), 4);
        let d = b.insert(engram(3, &[7, 8]));
        assert_eq!(d, a, "freed slot is reused");
        assert_eq!(b.code(c), &[3, 4, 5, 6], "other engram untouched");
    }

    #[test]
    fn reinforce_depresses_silent_inputs_and_recruits_new_ones() {
        let mut b = EngramBank::new(Tier::Fast, 81, 6);
        let s = b.insert(engram(1, &[1, 2, 3, 4]));
        b.reinforce(s, &[1, 2, 9], &certain());
        assert_eq!(b.code(s), &[1, 2, 3, 4, 9]);
        assert_eq!(b.states(s), vec![1, 1, 0, 0, 1]);
        // Silent synapses score nothing.
        assert_eq!(gather(&mut b, &[3, 4]), vec![]);
        // A silent position is recycled for the next new input.
        b.reinforce(s, &[1, 2, 9, 10, 11, 12], &certain());
        let code = b.code(s).to_vec();
        assert!(code.contains(&10) && code.contains(&11) && code.contains(&12), "{code:?}");
        assert_eq!(b.states(s).iter().filter(|&&w| w == 1).count(), 6);
        assert_eq!(b.fan_out(3) + b.fan_out(4), 0, "replaced synapses are unindexed");
    }

    #[test]
    fn inhibitory_synapses_subtract() {
        let mut b = EngramBank::new(Tier::Fast, 81, 9);
        let s = b.insert(engram(1, &[1, 2, 3, 4]));
        assert_eq!(b.inhibit(s, &[1, 5, 6], 9), 2);
        assert_eq!(b.states(s), vec![1, 1, 1, 1, -1, -1]);
        assert_eq!(gather(&mut b, &[1, 2, 5, 6]), vec![(0.0, s)]);
    }

    #[test]
    fn long_term_reset_restores_slow_states() {
        let mut b = EngramBank::new(Tier::LongTerm, 81, 4);
        let s = b.insert(engram(1, &[1, 2, 3, 4]));
        b.reinforce(s, &[1, 2], &certain());
        assert_eq!(b.states(s), vec![1, 1, 0, 0]);
        b.reset_fast_weights();
        assert_eq!(b.states(s), vec![1, 1, 1, 1]);
    }
}
