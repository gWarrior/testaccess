//! Engram bank: a population of engram (memory) neurons and the synapses
//! they own.
//!
//! Every memory gets its own engram neuron with a private set of afferent
//! synapses from its sparse ensemble (concept §14, variants A + C: ownership
//! + sparse allocation). Because no synapse is shared between memories,
//! forgetting one memory removes exactly its own contribution and cannot
//! damage any other memory.
//!
//! Each synapse has two components (concept §7):
//! `effective = max(0, w_slow + w_fast)`. Fast-tier engrams keep everything
//! in `w_fast`; consolidated engrams keep their knowledge in `w_slow`, while
//! later reinforcement lands in `w_fast` and is wiped by a fast reset.
//! Weights are relative: an ensemble's effective weights average to 1.

use crate::config::PlasticityConfig;
use crate::index::PostingIndex;
use crate::types::{ContextId, MemoryId, Tier};

pub(crate) const NO_ID: MemoryId = 0;
const NO_NEURON: u32 = u32::MAX;

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
    pub prev: MemoryId,
    pub next: MemoryId,
    pub len: u32,
}

impl SlotMeta {
    fn free() -> Self {
        Self {
            id: NO_ID,
            ctx: ContextId::DEFAULT,
            status: SlotStatus::Free,
            created: 0.0,
            expires: f64::INFINITY,
            strength: 0.0,
            recalls: 0,
            pinned: false,
            prev: NO_ID,
            next: NO_ID,
            len: 0,
        }
    }
}

/// A memory detached from its bank (used to move between tiers).
pub(crate) struct Engram<P> {
    pub code: Vec<u32>,
    /// Effective relative weights, aligned with `code`.
    pub weights: Vec<f32>,
    pub meta: SlotMeta,
    pub payload: Option<P>,
}

pub(crate) struct EngramBank<P> {
    tier: Tier,
    width: usize,
    ens: Vec<u32>,
    w_slow: Vec<f32>,
    w_fast: Vec<f32>,
    pub meta: Vec<SlotMeta>,
    payload: Vec<Option<P>>,
    free: Vec<u32>,
    pending: Vec<u32>,
    pub postings: PostingIndex,
    n_active: usize,
    score: Vec<f32>,
    touched: Vec<u32>,
}

impl<P> EngramBank<P> {
    pub fn new(tier: Tier, n_neurons: u32, width: usize) -> Self {
        Self {
            tier,
            width,
            ens: Vec::new(),
            w_slow: Vec::new(),
            w_fast: Vec::new(),
            meta: Vec::new(),
            payload: Vec::new(),
            free: Vec::new(),
            pending: Vec::new(),
            postings: PostingIndex::new(n_neurons),
            n_active: 0,
            score: Vec::new(),
            touched: Vec::new(),
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

    pub fn insert(&mut self, e: Engram<P>) -> u32 {
        debug_assert!(e.code.len() <= self.width && e.code.len() == e.weights.len());
        let slot = match self.free.pop() {
            Some(s) => s,
            None => {
                let s = self.meta.len() as u32;
                self.meta.push(SlotMeta::free());
                self.payload.push(None);
                self.ens.extend(std::iter::repeat(NO_NEURON).take(self.width));
                self.w_slow.extend(std::iter::repeat(0.0).take(self.width));
                self.w_fast.extend(std::iter::repeat(0.0).take(self.width));
                self.score.push(0.0);
                s
            }
        };
        let (s, base, len) = (slot as usize, slot as usize * self.width, e.code.len());
        self.ens[base..base + len].copy_from_slice(&e.code);
        let (target, zero) = match self.tier {
            Tier::Fast => (&mut self.w_fast, &mut self.w_slow),
            Tier::LongTerm => (&mut self.w_slow, &mut self.w_fast),
        };
        target[base..base + len].copy_from_slice(&e.weights);
        zero[base..base + len].fill(0.0);
        self.meta[s] = SlotMeta { status: SlotStatus::Active, len: len as u32, ..e.meta };
        self.payload[s] = e.payload;
        self.postings.add(slot, &e.code);
        self.n_active += 1;
        slot
    }

    #[inline]
    pub fn code(&self, slot: u32) -> &[u32] {
        let base = slot as usize * self.width;
        &self.ens[base..base + self.meta[slot as usize].len as usize]
    }

    /// Effective relative weights of `slot`.
    pub fn weights(&self, slot: u32) -> Vec<f32> {
        let base = slot as usize * self.width;
        let len = self.meta[slot as usize].len as usize;
        (base..base + len).map(|i| (self.w_slow[i] + self.w_fast[i]).max(0.0)).collect()
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
            weights: self.weights(slot),
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
            self.w_slow[base..base + self.width].fill(0.0);
            self.w_fast[base..base + self.width].fill(0.0);
            self.meta[s] = SlotMeta::free();
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
        self.w_fast.fill(0.0);
    }

    /// Index stage: accumulate gain-weighted overlap between `cue` and every
    /// live engram reachable through the cue's synapses. Appends
    /// `(score, slot)` for live engrams passing the context filter.
    pub fn gather(
        &mut self,
        cue: &[u32],
        gains: &[f32],
        now: f64,
        ctx: Option<ContextId>,
        max_scan: usize,
        out: &mut Vec<(f32, u32)>,
    ) {
        let Self { postings, score, touched, meta, .. } = self;
        for (&n, &g) in cue.iter().zip(gains) {
            let list = postings.list(n);
            if list.len() > max_scan {
                continue;
            }
            for &s in list {
                let sc = &mut score[s as usize];
                if *sc == 0.0 {
                    touched.push(s);
                }
                *sc += g.max(f32::MIN_POSITIVE);
            }
        }
        for &s in touched.iter() {
            let m = &meta[s as usize];
            if m.status == SlotStatus::Active && m.expires > now && ctx.map_or(true, |c| c == m.ctx) {
                out.push((score[s as usize], s));
            }
            score[s as usize] = 0.0;
        }
        touched.clear();
    }

    /// Reinforce an existing engram with a new presentation `cue` (sorted).
    ///
    /// Rate-based STDP: synapses whose presynaptic neuron fires together with
    /// the engram are potentiated (soft bound `w_max`), the others are
    /// depressed. With `structural`, weak synapses are replaced by synapses
    /// from newly active neurons (turnover) and the ensemble may grow up to
    /// its width. Finally weights are rescaled to mean 1 (synaptic scaling).
    pub fn reinforce(&mut self, slot: u32, cue: &[u32], cfg: &PlasticityConfig, structural: bool) {
        let (s, base) = (slot as usize, slot as usize * self.width);
        let mut len = self.meta[s].len as usize;
        let mut code = self.ens[base..base + len].to_vec();
        let mut eff = self.weights(slot);

        for (n, w) in code.iter().zip(eff.iter_mut()) {
            if cue.binary_search(n).is_ok() {
                *w += cfg.a_plus * (cfg.w_max - *w);
            } else {
                *w -= cfg.a_minus * *w;
            }
        }

        if structural {
            let mut sorted = code.clone();
            sorted.sort_unstable();
            let novel: Vec<u32> = cue
                .iter()
                .copied()
                .filter(|n| sorted.binary_search(n).is_err())
                .take(cfg.max_turnover)
                .collect();
            for n in novel {
                if len < self.width {
                    code.push(n);
                    eff.push(1.0);
                    len += 1;
                    self.postings.add_one(n, slot);
                    continue;
                }
                let (weakest, &w) = eff
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.total_cmp(b.1))
                    .expect("ensemble is not empty");
                if w >= cfg.turnover_below {
                    break;
                }
                self.postings.remove_one(code[weakest], slot);
                self.postings.add_one(n, slot);
                code[weakest] = n;
                eff[weakest] = 1.0;
            }
        }

        let mean = eff.iter().sum::<f32>() / len as f32;
        if mean > 0.0 {
            eff.iter_mut().for_each(|w| *w /= mean);
        }
        self.ens[base..base + len].copy_from_slice(&code);
        for (i, &w) in eff.iter().enumerate() {
            self.w_fast[base + i] = w - self.w_slow[base + i];
        }
        self.meta[s].len = len as u32;
    }

    /// Approximate heap usage in bytes.
    pub fn bytes(&self) -> usize {
        self.ens.capacity() * 4
            + (self.w_slow.capacity() + self.w_fast.capacity() + self.score.capacity()) * 4
            + self.meta.capacity() * std::mem::size_of::<SlotMeta>()
            + self.payload.capacity() * std::mem::size_of::<Option<P>>()
            + self.postings.bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engram(id: MemoryId, code: &[u32]) -> Engram<&'static str> {
        Engram {
            code: code.to_vec(),
            weights: vec![1.0; code.len()],
            meta: SlotMeta { id, ..SlotMeta::free() },
            payload: Some("p"),
        }
    }

    fn gather(bank: &mut EngramBank<&'static str>, cue: &[u32]) -> Vec<(f32, u32)> {
        let mut out = Vec::new();
        bank.gather(cue, &vec![1.0; cue.len()], 0.0, None, usize::MAX, &mut out);
        out.sort_by_key(|&(_, s)| s);
        out
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
    fn reinforce_potentiates_depresses_and_rescales() {
        let mut b = EngramBank::new(Tier::Fast, 64, 4);
        let s = b.insert(engram(1, &[1, 2, 3, 4]));
        b.reinforce(s, &[1, 2], &PlasticityConfig::default(), false);
        let w = b.weights(s);
        assert!(w[0] > 1.0 && w[1] > 1.0 && w[2] < 1.0 && w[3] < 1.0);
        assert!((w.iter().sum::<f32>() - 4.0).abs() < 1e-4);
    }

    #[test]
    fn structural_turnover_replaces_weak_synapses() {
        let mut b = EngramBank::new(Tier::Fast, 64, 4);
        let s = b.insert(engram(1, &[1, 2, 3, 4]));
        let cfg = PlasticityConfig::default();
        for _ in 0..6 {
            b.reinforce(s, &[1, 2, 9], &cfg, true);
        }
        let code = b.code(s).to_vec();
        assert!(code.contains(&9), "new neuron recruited: {code:?}");
        assert!(b.postings.list(9).contains(&s));
        let dropped: Vec<u32> = [3, 4].into_iter().filter(|n| !code.contains(n)).collect();
        for n in dropped {
            assert!(!b.postings.list(n).contains(&s));
        }
    }

    #[test]
    fn long_term_reset_keeps_slow_weights() {
        let mut b = EngramBank::new(Tier::LongTerm, 64, 4);
        let s = b.insert(engram(1, &[1, 2, 3, 4]));
        b.reinforce(s, &[1], &PlasticityConfig::default(), false);
        assert!(b.weights(s)[0] > 1.0);
        b.reset_fast_weights();
        assert_eq!(b.weights(s), vec![1.0; 4]);
    }
}
