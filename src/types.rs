//! Core identifiers, inputs and errors shared across the crate.

use std::fmt;

/// Logical identifier of a memory. Ids are never reused; `0` means "none".
pub type MemoryId = u64;

/// Interned context (namespace) identifier. Obtain one with
/// [`SnnMemory::context`](crate::SnnMemory::context).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContextId(pub u32);

impl ContextId {
    /// The context every memory belongs to unless told otherwise.
    pub const DEFAULT: ContextId = ContextId(0);
}

/// Storage tier of a memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    /// Episodic memory: written in one shot, subject to TTL and fast resets.
    Fast,
    /// Consolidated memory: survives [`reset_fast_memory`](crate::SnnMemory::reset_fast_memory).
    LongTerm,
}

/// Something that can be turned into a sparse spike code.
#[derive(Clone, Copy, Debug)]
pub enum Input<'a> {
    /// Dense feature vector, e.g. an LLM hidden state.
    Dense(&'a [f32]),
    /// Sparse feature vector as `(feature_index, value)` pairs.
    Sparse(&'a [(u32, f32)]),
    /// Token sequence; its temporal structure (order) is encoded.
    Tokens(&'a [u32]),
    /// Pre-computed spike code: the set of active neuron ids.
    Code(&'a [u32]),
}

/// Errors reported by the memory.
#[derive(Clone, Debug, PartialEq)]
pub enum MemoryError {
    /// The configured encoder cannot handle this kind of input.
    UnsupportedInput(&'static str),
    /// Dense input has the wrong dimensionality.
    DimensionMismatch { expected: usize, got: usize },
    /// The input produced no active neurons.
    EmptyCode,
    /// The spike code is larger than an engram can hold.
    EnsembleTooLarge { len: usize, max: usize },
    /// A neuron id in a pre-computed code is outside the neuron space.
    NeuronOutOfRange { neuron: u32, n_neurons: u32 },
    /// No live memory with this id.
    UnknownMemory(MemoryId),
    /// The context id was never registered.
    UnknownContext(ContextId),
    /// Invalid configuration value.
    InvalidConfig(String),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedInput(what) => write!(f, "encoder does not support {what} input"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "input dimension mismatch: expected {expected}, got {got}")
            }
            Self::EmptyCode => write!(f, "input produced an empty spike code"),
            Self::EnsembleTooLarge { len, max } => {
                write!(f, "spike code has {len} neurons, engram capacity is {max}")
            }
            Self::NeuronOutOfRange { neuron, n_neurons } => {
                write!(f, "neuron {neuron} out of range (n_neurons = {n_neurons})")
            }
            Self::UnknownMemory(id) => write!(f, "unknown memory #{id}"),
            Self::UnknownContext(ctx) => write!(f, "unknown context {}", ctx.0),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for MemoryError {}
