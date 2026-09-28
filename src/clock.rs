//! Time sources. Memory lifetimes (TTL) and short-term plasticity decay are
//! expressed in seconds of this clock.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Monotonic time source, in seconds.
pub trait Clock: Send + Sync {
    fn now(&self) -> f64;
}

/// Wall-clock time since the clock was created.
#[derive(Clone, Debug)]
pub struct SystemClock {
    origin: Instant,
}

impl SystemClock {
    pub fn new() -> Self {
        Self { origin: Instant::now() }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn now(&self) -> f64 {
        self.origin.elapsed().as_secs_f64()
    }
}

/// Manually driven clock for tests and simulations. Clones share the time.
#[derive(Clone, Debug, Default)]
pub struct ManualClock {
    bits: Arc<AtomicU64>,
}

impl ManualClock {
    pub fn new(start: f64) -> Self {
        let c = Self::default();
        c.set(start);
        c
    }

    pub fn set(&self, t: f64) {
        self.bits.store(t.to_bits(), Ordering::SeqCst);
    }

    pub fn advance(&self, dt: f64) {
        self.set(self.now() + dt);
    }
}

impl Clock for ManualClock {
    fn now(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::SeqCst))
    }
}
