//! # SNN Memory
//!
//! Fast, addressable memory built from sparse spiking ensembles.
//!
//! The crate implements the `Learn → Recall → Forget → Recall` cycle described
//! in `docs/concept.md`: patterns are written in one shot into sparse engram
//! ensembles, recalled associatively from partial cues through LIF attractor
//! dynamics, and forgotten selectively without disturbing other memories.

pub mod clock;
pub mod config;
pub mod encoder;
pub mod memory;
pub mod rng;
pub mod shared;
pub mod trit;
pub mod types;

mod bank;
mod dynamics;
mod index;
mod working;

pub use clock::{Clock, ManualClock, SystemClock};
pub use config::{DynamicsConfig, MemoryConfig, PlasticityConfig, StpConfig};
pub use encoder::{CodeEncoder, Encoder, FlyHashEncoder, NGramEncoder};
pub use memory::{
    Basis, BatchOptions, Hit, LearnOptions, MaintenanceReport, MemoryRecord, RecallOptions, RecallResult,
    SnnMemory, Stats, Verdict,
};
pub use shared::{MaintenanceHandle, SharedMemory};
pub use trit::{TritVec, Tryte};
pub use types::{ContextId, Input, MemoryError, MemoryId, Tier};
