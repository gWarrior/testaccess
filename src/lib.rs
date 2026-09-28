//! # SNN Memory
//!
//! Fast, addressable memory built from sparse spiking ensembles.
//!
//! The crate implements the `Learn → Recall → Forget → Recall` cycle described
//! in `docs/concept.md`: patterns are written in one shot into sparse engram
//! ensembles, recalled associatively from partial cues through LIF attractor
//! dynamics, and forgotten selectively without disturbing other memories.
