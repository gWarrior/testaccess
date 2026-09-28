//! Sparse synapse store, organised by presynaptic neuron.
//!
//! `lists[n]` holds the engram slots that neuron `n` projects to. It doubles
//! as the fast candidate index (concept §10): a query only touches the
//! synapses of its own active neurons, so retrieval cost scales with the
//! number of synapses of those neurons, not with the number of memories.

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

    /// Connect every neuron of `code` to `slot`.
    pub fn add(&mut self, slot: u32, code: &[u32]) {
        for &n in code {
            self.lists[n as usize].push(slot);
        }
        self.entries += code.len();
    }

    pub fn add_one(&mut self, neuron: u32, slot: u32) {
        self.lists[neuron as usize].push(slot);
        self.entries += 1;
    }

    /// Remove one synapse `neuron → slot`.
    pub fn remove_one(&mut self, neuron: u32, slot: u32) -> bool {
        let list = &mut self.lists[neuron as usize];
        match list.iter().position(|&s| s == slot) {
            Some(p) => {
                list.swap_remove(p);
                self.entries -= 1;
                true
            }
            None => false,
        }
    }

    /// Drop every synapse of `neurons` whose slot satisfies `dead`.
    pub fn purge(&mut self, neurons: &[u32], dead: impl Fn(u32) -> bool) -> usize {
        let mut removed = 0;
        for &n in neurons {
            let list = &mut self.lists[n as usize];
            let before = list.len();
            list.retain(|&s| !dead(s));
            removed += before - list.len();
        }
        self.entries -= removed;
        removed
    }

    #[inline]
    pub fn list(&self, neuron: u32) -> &[u32] {
        &self.lists[neuron as usize]
    }

    /// Number of synapses leaving `neuron` (its document frequency).
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
    fn add_remove_purge() {
        let mut idx = PostingIndex::new(16);
        idx.add(0, &[1, 2, 3]);
        idx.add(1, &[2, 3, 4]);
        idx.add(2, &[3]);
        assert_eq!(idx.entries(), 7);
        assert_eq!(idx.list(3), &[0, 1, 2]);

        assert!(idx.remove_one(2, 0));
        assert!(!idx.remove_one(2, 0));
        assert_eq!(idx.list(2), &[1]);

        let removed = idx.purge(&[2, 3, 4], |s| s == 1);
        assert_eq!(removed, 3);
        assert_eq!(idx.list(3), &[0, 2]);
        assert_eq!(idx.entries(), 3);
        assert_eq!(idx.max_len(), 2);
    }
}
