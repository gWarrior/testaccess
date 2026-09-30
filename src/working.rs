//! Working memory: short-term facilitation of recently active engrams and
//! the last settled network state (concept §3.1).
//!
//! Every time an engram is written or recalled its utilization `u` jumps
//! towards 1 and then decays with time constant `tau` (Tsodyks–Markram style
//! facilitation). During recall a facilitated engram receives a slightly
//! higher drive, `gain = 1 + max_gain · u`, which breaks ties in favour of
//! what was active recently. Clearing working memory never touches engrams.

use std::collections::{HashMap, VecDeque};

use crate::config::StpConfig;
use crate::persist;
use crate::types::{MemoryError, MemoryId};

pub(crate) struct WorkingMemory {
    cfg: StpConfig,
    traces: HashMap<MemoryId, (f32, f64)>,
    recent: VecDeque<MemoryId>,
    /// Active ensemble neurons after the last recall settled.
    pub last_state: Vec<u32>,
}

impl WorkingMemory {
    pub fn new(cfg: StpConfig) -> Self {
        Self { cfg, traces: HashMap::new(), recent: VecDeque::new(), last_state: Vec::new() }
    }

    #[inline]
    fn decayed(&self, u: f32, t: f64, now: f64) -> f32 {
        u * (-(now - t).max(0.0) / self.cfg.tau).exp() as f32
    }

    /// Current facilitation of `id`, in `[0, 1]`.
    pub fn utilization(&self, id: MemoryId, now: f64) -> f32 {
        self.traces.get(&id).map_or(0.0, |&(u, t)| self.decayed(u, t, now))
    }

    /// Multiplicative drive gain for `id` (>= 1).
    pub fn gain(&self, id: MemoryId, now: f64) -> f32 {
        1.0 + self.cfg.max_gain * self.utilization(id, now)
    }

    pub fn facilitate(&mut self, id: MemoryId, now: f64) {
        let u0 = self.utilization(id, now);
        let u = u0 + self.cfg.utilization * (1.0 - u0);
        self.traces.insert(id, (u, now));
        self.recent.push_back(id);
        if self.recent.len() > self.cfg.capacity {
            self.recent.pop_front();
        }
        if self.traces.len() > 2 * self.cfg.capacity.max(1) {
            self.prune(now);
        }
    }

    /// Keep only the `capacity` most facilitated traces.
    fn prune(&mut self, now: f64) {
        let mut all: Vec<(f32, MemoryId)> =
            self.traces.iter().map(|(&id, &(u, t))| (self.decayed(u, t, now), id)).collect();
        let keep = self.cfg.capacity.min(all.len());
        // Ties go to the newest id, so pruning does not depend on the hash
        // map's iteration order (a restored copy prunes identically).
        all.select_nth_unstable_by(keep.saturating_sub(1), |a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)));
        for &(_, id) in &all[keep..] {
            self.traces.remove(&id);
        }
    }

    /// O(1): the id may linger in [`recent`](Self::recent); callers filter
    /// out memories that no longer exist.
    pub fn forget(&mut self, id: MemoryId) {
        self.traces.remove(&id);
    }

    pub fn clear(&mut self) {
        self.traces.clear();
        self.recent.clear();
        self.last_state.clear();
    }

    /// Exact state for a full snapshot (absolute times of the memory's
    /// clock; traces sorted by id so equal states give equal bytes).
    pub fn write_image(&self, out: &mut Vec<u8>) {
        let mut traces: Vec<(MemoryId, (f32, f64))> = self.traces.iter().map(|(&id, &t)| (id, t)).collect();
        traces.sort_unstable_by_key(|t| t.0);
        out.extend_from_slice(&(traces.len() as u64).to_le_bytes());
        for (id, (u, t)) in traces {
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&u.to_le_bytes());
            out.extend_from_slice(&t.to_le_bytes());
        }
        out.extend_from_slice(&(self.recent.len() as u64).to_le_bytes());
        for id in &self.recent {
            out.extend_from_slice(&id.to_le_bytes());
        }
        out.extend_from_slice(&(self.last_state.len() as u64).to_le_bytes());
        persist::put_u32s(out, &self.last_state);
    }

    pub fn read_image(&mut self, input: &mut &[u8]) -> Result<(), MemoryError> {
        let n = persist::read_u64(input)? as usize;
        let mut traces = HashMap::with_capacity(n.min(input.len() / 20));
        for _ in 0..n {
            let id = persist::read_u64(input)?;
            let u = persist::read_f32(input)?;
            let t = persist::read_f64(input)?;
            traces.insert(id, (u, t));
        }
        let n = persist::read_u64(input)? as usize;
        let recent = persist::read_u64s(input, n)?.into();
        let n = persist::read_u64(input)? as usize;
        let last_state = persist::read_u32s(input, n)?;
        self.traces = traces;
        self.recent = recent;
        self.last_state = last_state;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.traces.len()
    }

    /// Most recently activated memories, newest first.
    pub fn recent(&self) -> impl Iterator<Item = MemoryId> + '_ {
        self.recent.iter().rev().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facilitation_accumulates_and_decays() {
        let cfg = StpConfig { utilization: 0.5, tau: 9.0, max_gain: 0.2, capacity: 3 };
        let mut wm = WorkingMemory::new(cfg);
        assert_eq!(wm.gain(1, 0.0), 1.0);
        wm.facilitate(1, 0.0);
        wm.facilitate(1, 0.0);
        assert!((wm.utilization(1, 0.0) - 0.75).abs() < 1e-6);
        assert!((wm.gain(1, 0.0) - 1.15).abs() < 1e-6);
        assert!(wm.utilization(1, 54.0) < 0.01);
        wm.forget(1);
        assert_eq!(wm.utilization(1, 0.0), 0.0);
    }

    #[test]
    fn prune_keeps_strongest() {
        let cfg = StpConfig { utilization: 0.5, tau: 1.0, max_gain: 0.2, capacity: 3 };
        let mut wm = WorkingMemory::new(cfg);
        for id in 1..=9u64 {
            wm.facilitate(id, id as f64);
        }
        assert!(wm.len() <= 6);
        assert!(wm.utilization(9, 9.0) > 0.0);
        assert_eq!(wm.recent().next(), Some(9));
    }
}
