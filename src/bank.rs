//! Engram bank: a population of engram (memory) neurons and the ternary
//! synapses they own.
//!
//! Every memory gets its own engram neuron with a private set of afferent
//! synapses (concept §14, variants A + C: ownership + sparse allocation).
//! No synapse is shared between memories, so forgetting one memory removes
//! exactly its own contribution and cannot damage any other memory.
//!
//! Synapses are ternary: `+1` excitatory, `0` silent, `-1` inhibitory.
//! Each synapse has a slow and a fast ternary component (concept §7) packed
//! into one byte, `effective = clamp(slow + fast, -1, 1)`. Fast-tier engrams
//! live in the fast component; consolidated engrams keep their knowledge in
//! the slow one, while later plasticity lands in the fast component and is
//! wiped by a fast reset. Plasticity is a discrete state machine (in the
//! spirit of Amit–Fusi bounded synapses) rather than a real-valued update.

use crate::config::PlasticityConfig;
use crate::index::{entry, entry_pos, entry_slot, PostingIndex, MAX_SLOTS};
use crate::rng::SplitMix64;
use crate::types::{ContextId, MemoryId, Tier};

pub(crate) const NO_ID: MemoryId = 0;
const NO_NEURON: u32 = u32::MAX;

#[inline]
fn enc(t: i8) -> u8 {
    match t {
        1 => 1,
        -1 => 2,
        _ => 0,
    }
}

#[inline]
fn dec(b: u8) -> i8 {
    match b & 3 {
        1 => 1,
        2 => -1,
        _ => 0,
    }
}

#[inline]
fn pack(slow: i8, fast: i8) -> u8 {
    enc(slow) | enc(fast) << 2
}

#[inline]
fn unpack(b: u8) -> (i8, i8) {
    (dec(b), dec(b >> 2))
}

#[inline]
fn effective(b: u8) -> i8 {
    let (s, f) = unpack(b);
    (s + f).clamp(-1, 1)
}

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
    ens: Vec<u32>,
    syn: Vec<u8>,
    pub meta: Vec<SlotMeta>,
    payload: Vec<Option<P>>,
    free: Vec<u32>,
    pending: Vec<u32>,
    pub postings: PostingIndex,
    n_active: usize,
    score: Vec<f32>,
    exc: Vec<f32>,
    stamp: Vec<u32>,
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
            syn: Vec::new(),
            meta: Vec::new(),
            payload: Vec::new(),
            free: Vec::new(),
            pending: Vec::new(),
            postings: PostingIndex::new(n_neurons),
            n_active: 0,
            score: Vec::new(),
            exc: Vec::new(),
            stamp: Vec::new(),
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
                self.syn.extend(std::iter::repeat(0).take(self.width));
                self.score.push(0.0);
                self.exc.push(0.0);
                self.stamp.push(0);
                s
            }
        };
        let (s, base, len) = (slot as usize, slot as usize * self.width, e.code.len());
        self.ens[base..base + len].copy_from_slice(&e.code);
        for (b, &t) in self.syn[base..base + len].iter_mut().zip(&e.states) {
            *b = match self.tier {
                Tier::Fast => pack(0, t),
                Tier::LongTerm => pack(t, 0),
            };
        }
        self.meta[s] = SlotMeta { status: SlotStatus::Active, len: len as u32, ..e.meta };
        self.payload[s] = e.payload;
        self.postings.add_engram(slot, &e.code);
        self.n_active += 1;
        slot
    }

    #[inline]
    pub fn code(&self, slot: u32) -> &[u32] {
        let base = slot as usize * self.width;
        &self.ens[base..base + self.meta[slot as usize].len as usize]
    }

    /// Effective ternary states of `slot`'s synapses.
    pub fn states(&self, slot: u32) -> Vec<i8> {
        let base = slot as usize * self.width;
        let len = self.meta[slot as usize].len as usize;
        self.syn[base..base + len].iter().map(|&b| effective(b)).collect()
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
        self.meta
            .iter()
            .enumerate()
            .filter(|(_, m)| m.status == SlotStatus::Active)
            .map(|(s, _)| s as u32)
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
            code: self.code(slot).to_vec(),
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
            neurons.extend_from_slice(self.code(slot));
        }
        neurons.sort_unstable();
        neurons.dedup();
        let meta = &self.meta;
        self.postings.purge(&neurons, |s| meta[s as usize].status == SlotStatus::Deleted);

        let n = self.pending.len();
        for slot in std::mem::take(&mut self.pending) {
            let (s, base) = (slot as usize, slot as usize * self.width);
            self.ens[base..base + self.width].fill(NO_NEURON);
            self.syn[base..base + self.width].fill(0);
            self.meta[s] = SlotMeta::new(NO_ID);
            self.free.push(slot);
        }
        n
    }

    /// Drop everything (O(neurons + slots)).
    pub fn clear(&mut self) {
        *self = Self::new(self.tier, self.postings.n_neurons(), self.width);
    }

    /// Clear the fast component of every synapse.
    pub fn reset_fast_weights(&mut self) {
        self.syn.iter_mut().for_each(|b| *b &= 3);
    }

    /// Index stage: accumulate the signed, gain-weighted input every live
    /// engram receives from the cue through its synapses, for live engrams
    /// passing the context filter.
    pub fn gather(
        &mut self,
        cue: &[u32],
        gains: &[f32],
        now: f64,
        ctx: Option<ContextId>,
        max_scan: usize,
        out: &mut Vec<Gathered>,
    ) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.fill(0);
            self.epoch = 1;
        }
        let Self { postings, score, exc, stamp, epoch, touched, meta, syn, width, .. } = self;
        for (&n, &g) in cue.iter().zip(gains) {
            let list = postings.list(n);
            if list.len() > max_scan {
                continue;
            }
            for &e in list {
                let s = entry_slot(e) as usize;
                let w = effective(syn[s * *width + entry_pos(e)]);
                if w == 0 {
                    continue;
                }
                if stamp[s] != *epoch {
                    stamp[s] = *epoch;
                    score[s] = 0.0;
                    exc[s] = 0.0;
                    touched.push(s as u32);
                }
                score[s] += g * w as f32;
                if w > 0 {
                    exc[s] += g;
                }
            }
        }
        for &s in touched.iter() {
            let m = &meta[s as usize];
            if m.status == SlotStatus::Active && m.expires > now && ctx.map_or(true, |c| c == m.ctx) {
                out.push(Gathered { score: score[s as usize], excitatory: exc[s as usize], slot: s });
            }
        }
        touched.clear();
    }

    fn set_effective(&mut self, idx: usize, target: i8) {
        let (slow, _) = unpack(self.syn[idx]);
        self.syn[idx] = pack(slow, (target - slow).clamp(-1, 1));
    }

    /// A free position, or else the first silent synapse, if any. Silent
    /// synapses are recycled last: they keep a trace that co-activity can
    /// revive.
    fn vacant_position(&self, slot: u32) -> Option<usize> {
        let (base, len) = (slot as usize * self.width, self.meta[slot as usize].len as usize);
        (len < self.width).then_some(len).or_else(|| (0..len).find(|&i| effective(self.syn[base + i]) == 0))
    }

    /// Wire neuron `n` into position `pos` of `slot` with state `target`.
    fn place(&mut self, slot: u32, pos: usize, n: u32, target: i8) {
        let base = slot as usize * self.width;
        let len = self.meta[slot as usize].len as usize;
        if pos < len {
            self.postings.remove(self.ens[base + pos], entry(slot, pos));
        } else {
            self.meta[slot as usize].len += 1;
        }
        self.ens[base + pos] = n;
        self.syn[base + pos] = pack(0, 0);
        self.set_effective(base + pos, target);
        self.postings.add(n, entry(slot, pos));
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
            let w = effective(self.syn[idx]);
            let active = cue.binary_search(&self.ens[idx]).is_ok();
            if active && w < 1 && self.chance(cfg.p_potentiate) {
                self.set_effective(idx, w + 1);
            } else if !active && w == 1 && self.chance(cfg.p_depress) {
                self.set_effective(idx, 0);
            }
        }
        let mut present = self.code(slot).to_vec();
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
            match (0..len).find(|&i| self.ens[base + i] == n) {
                Some(i) if effective(self.syn[base + i]) == 1 => {}
                Some(i) if effective(self.syn[base + i]) == 0 => {
                    self.set_effective(base + i, -1);
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
        self.ens.capacity() * 4
            + self.syn.capacity()
            + (self.score.capacity() + self.exc.capacity() + self.stamp.capacity()) * 4
            + self.meta.capacity() * std::mem::size_of::<SlotMeta>()
            + self.payload.capacity() * std::mem::size_of::<Option<P>>()
            + self.postings.bytes()
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
        bank.gather(cue, &vec![1.0; cue.len()], 0.0, None, usize::MAX, &mut out);
        let mut v: Vec<(f32, u32)> = out.iter().map(|g| (g.score, g.slot)).collect();
        v.sort_by_key(|&(_, s)| s);
        v
    }

    fn certain() -> PlasticityConfig {
        PlasticityConfig { p_potentiate: 1.0, p_depress: 1.0, max_new_synapses: 8 }
    }

    #[test]
    fn packing_roundtrip() {
        for s in -1..=1 {
            for f in -1..=1 {
                assert_eq!(unpack(pack(s, f)), (s, f));
                assert_eq!(effective(pack(s, f)), (s + f).clamp(-1, 1));
            }
        }
    }

    #[test]
    fn insert_gather_delete_cleanup_reuse() {
        let mut b = EngramBank::new(Tier::Fast, 64, 8);
        let a = b.insert(engram(1, &[1, 2, 3, 4]));
        let c = b.insert(engram(2, &[3, 4, 5, 6]));
        assert_eq!(gather(&mut b, &[3, 4, 5]), vec![(2.0, a), (3.0, c)]);

        assert!(b.mark_deleted(a));
        assert!(!b.mark_deleted(a));
        assert_eq!(gather(&mut b, &[3, 4, 5]), vec![(3.0, c)], "deleted engram is silent at once");
        assert_eq!(b.payload(a), None);

        assert_eq!(b.cleanup(), 1);
        assert_eq!(b.postings.entries(), 4);
        let d = b.insert(engram(3, &[7, 8]));
        assert_eq!(d, a, "freed slot is reused");
        assert_eq!(b.code(c), &[3, 4, 5, 6], "other engram untouched");
    }

    #[test]
    fn reinforce_depresses_silent_inputs_and_recruits_new_ones() {
        let mut b = EngramBank::new(Tier::Fast, 64, 6);
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
        assert!(b.postings.list(3).is_empty() && b.postings.list(4).is_empty());
    }

    #[test]
    fn inhibitory_synapses_subtract() {
        let mut b = EngramBank::new(Tier::Fast, 64, 8);
        let s = b.insert(engram(1, &[1, 2, 3, 4]));
        assert_eq!(b.inhibit(s, &[1, 5, 6], 8), 2);
        assert_eq!(b.states(s), vec![1, 1, 1, 1, -1, -1]);
        assert_eq!(gather(&mut b, &[1, 2, 5, 6]), vec![(0.0, s)]);
    }

    #[test]
    fn long_term_reset_restores_slow_states() {
        let mut b = EngramBank::new(Tier::LongTerm, 64, 4);
        let s = b.insert(engram(1, &[1, 2, 3, 4]));
        b.reinforce(s, &[1, 2], &certain());
        assert_eq!(b.states(s), vec![1, 1, 0, 0]);
        b.reset_fast_weights();
        assert_eq!(b.states(s), vec![1, 1, 1, 1]);
    }
}
