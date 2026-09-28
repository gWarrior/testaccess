//! Sparse synapse store, organised by presynaptic neuron.
//!
//! `lists[n]` holds one entry per synapse leaving neuron `n`. An entry packs
//! the target engram slot and the synapse's position inside that engram's
//! ensemble (`slot << 8 | pos`).
//! The store doubles as the fast candidate index (concept §10): a query only
//! touches the synapses of its own active neurons, so retrieval cost scales
//! with their fan-out, not with the number of memories.

/// Maximum synapses per engram, 3^5 (positions are stored in 8 bits).
pub(crate) const MAX_WIDTH: usize = 243;
/// Maximum engram slots per bank, 3^15 (slots are stored in 24 bits).
pub(crate) const MAX_SLOTS: usize = 14_348_907;

#[inline]
pub(crate) fn entry(slot: u32, pos: usize) -> u32 {
    debug_assert!((slot as usize) < MAX_SLOTS && pos < MAX_WIDTH);
    slot << 8 | pos as u32
}

#[inline]
pub(crate) fn entry_slot(e: u32) -> u32 {
    e >> 8
}

#[cfg(test)]
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
        self.lists.len() * std::mem::size_of::<Vec<u32>>() + self.lists.iter().map(|l| l.capacity() * 4).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_packing_roundtrip() {
        let e = entry(123_456, 242);
        assert_eq!((entry_slot(e), entry_pos(e)), (123_456, 242));
    }

    #[test]
    fn add_remove_purge() {
        let mut idx = PostingIndex::new(27);
        for (slot, code) in [(0, &[1, 2, 3][..]), (1, &[2, 3, 4]), (2, &[3])] {
            for (pos, &n) in code.iter().enumerate() {
                idx.add(n, entry(slot, pos));
            }
        }
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
