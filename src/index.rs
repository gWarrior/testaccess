//! Sparse synapse store, organised by presynaptic neuron.
//!
//! `lists[n]` holds one entry per synapse leaving neuron `n`. An entry packs
//! the target engram slot and the synapse's position inside that engram's
//! ensemble (`slot << 8 | pos`), so the synapse state can be read in O(1).
//! The store doubles as the fast candidate index (concept §10): a query only
//! touches the synapses of its own active neurons, so retrieval cost scales
//! with their fan-out, not with the number of memories.

/// Maximum synapses per engram (positions fit in 8 bits).
pub(crate) const MAX_WIDTH: usize = 256;
/// Maximum engram slots per bank (slots fit in 24 bits).
pub(crate) const MAX_SLOTS: usize = 1 << 24;

#[inline]
pub(crate) fn entry(slot: u32, pos: usize) -> u32 {
    debug_assert!((slot as usize) < MAX_SLOTS && pos < MAX_WIDTH);
    slot << 8 | pos as u32
}

#[inline]
pub(crate) fn entry_slot(e: u32) -> u32 {
    e >> 8
}

#[inline]
pub(crate) fn entry_pos(e: u32) -> usize {
    (e & 0xFF) as usize
}

pub(crate) struct PostingIndex {
    lists: Vec<Vec<u32>>,
    entries: usize,
}

impl PostingIndex {
    pub fn new(n_neurons: u32) -> Self {
        Self { lists: (0..n_neurons).map(|_| Vec::new()).collect(), entries: 0 }
    }

    pub fn n_neurons(&self) -> u32 {
        self.lists.len() as u32
    }

    /// Connect neuron `code[i]` to position `i` of `slot`.
    pub fn add_engram(&mut self, slot: u32, code: &[u32]) {
        for (pos, &n) in code.iter().enumerate() {
            self.add(n, entry(slot, pos));
        }
    }

    pub fn add(&mut self, neuron: u32, e: u32) {
        self.lists[neuron as usize].push(e);
        self.entries += 1;
    }

    /// Remove one synapse entry.
    pub fn remove(&mut self, neuron: u32, e: u32) -> bool {
        let list = &mut self.lists[neuron as usize];
        match list.iter().position(|&x| x == e) {
            Some(p) => {
                list.swap_remove(p);
                self.entries -= 1;
                true
            }
            None => false,
        }
    }

    /// Drop every entry of `neurons` whose target slot satisfies `dead`.
    pub fn purge(&mut self, neurons: &[u32], dead: impl Fn(u32) -> bool) -> usize {
        let mut removed = 0;
        for &n in neurons {
            let list = &mut self.lists[n as usize];
            let before = list.len();
            list.retain(|&e| !dead(entry_slot(e)));
            removed += before - list.len();
        }
        self.entries -= removed;
        removed
    }

    #[inline]
    pub fn list(&self, neuron: u32) -> &[u32] {
        &self.lists[neuron as usize]
    }

    /// Number of synapses leaving `neuron` (its fan-out).
    #[inline]
    pub fn len(&self, neuron: u32) -> usize {
        self.lists[neuron as usize].len()
    }

    pub fn entries(&self) -> usize {
        self.entries
    }

    pub fn max_len(&self) -> usize {
        self.lists.iter().map(Vec::len).max().unwrap_or(0)
    }

    /// Approximate heap usage in bytes.
    pub fn bytes(&self) -> usize {
        self.lists.len() * std::mem::size_of::<Vec<u32>>()
            + self.lists.iter().map(|l| l.capacity() * 4).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_packing_roundtrip() {
        let e = entry(123_456, 200);
        assert_eq!((entry_slot(e), entry_pos(e)), (123_456, 200));
    }

    #[test]
    fn add_remove_purge() {
        let mut idx = PostingIndex::new(16);
        idx.add_engram(0, &[1, 2, 3]);
        idx.add_engram(1, &[2, 3, 4]);
        idx.add_engram(2, &[3]);
        assert_eq!(idx.entries(), 7);
        let slots: Vec<u32> = idx.list(3).iter().map(|&e| entry_slot(e)).collect();
        assert_eq!(slots, vec![0, 1, 2]);
        assert_eq!(entry_pos(idx.list(3)[1]), 1);

        assert!(idx.remove(2, entry(0, 1)));
        assert!(!idx.remove(2, entry(0, 1)));
        assert_eq!(idx.list(2), &[entry(1, 0)]);

        let removed = idx.purge(&[2, 3, 4], |s| s == 1);
        assert_eq!(removed, 3);
        assert_eq!(idx.entries(), 3);
        assert_eq!(idx.max_len(), 2);
    }
}
