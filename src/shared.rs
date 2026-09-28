//! Thread-safe handle with asynchronous physical cleanup (concept §13):
//! `forget` stays O(1) on the caller's thread, while a background worker
//! expires TTLs and reclaims synapses of deleted engrams.

use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::memory::SnnMemory;

/// Shared, lockable memory.
pub struct SharedMemory<P = ()> {
    inner: Arc<Mutex<SnnMemory<P>>>,
}

impl<P> Clone for SharedMemory<P> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl<P: Clone + Send + 'static> SharedMemory<P> {
    pub fn new(memory: SnnMemory<P>) -> Self {
        Self { inner: Arc::new(Mutex::new(memory)) }
    }

    /// Lock the memory. A panic in another holder does not poison access.
    pub fn lock(&self) -> MutexGuard<'_, SnnMemory<P>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Start a worker that calls [`SnnMemory::maintain`] every `interval`.
    /// The worker stops when the returned handle is dropped.
    pub fn spawn_maintenance(&self, interval: Duration) -> MaintenanceHandle {
        let (stop, rx) = mpsc::channel::<()>();
        let memory = self.clone();
        let thread = std::thread::Builder::new()
            .name("snn-memory-maintenance".into())
            .spawn(move || loop {
                match rx.recv_timeout(interval) {
                    Err(RecvTimeoutError::Timeout) => {
                        memory.lock().maintain();
                    }
                    _ => break,
                }
            })
            .expect("failed to spawn maintenance thread");
        MaintenanceHandle { stop: Some(stop), thread: Some(thread) }
    }
}

/// Stops the maintenance worker on drop.
pub struct MaintenanceHandle {
    stop: Option<Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl MaintenanceHandle {
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        drop(self.stop.take());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for MaintenanceHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CodeEncoder, Input, LearnOptions, ManualClock, MemoryConfig};

    #[test]
    fn background_worker_expires_and_cleans() {
        let clock = ManualClock::new(0.0);
        let cfg = MemoryConfig { auto_cleanup: 0, ..Default::default() };
        let mem: SnnMemory<()> = SnnMemory::new(CodeEncoder::new(1024), cfg).unwrap().with_clock(clock.clone());
        let shared = SharedMemory::new(mem);
        {
            let mut m = shared.lock();
            m.learn(Input::Code(&[1, 2, 3]), LearnOptions::new().ttl(1.0)).unwrap();
            let id = m.learn(Input::Code(&[4, 5, 6]), LearnOptions::new()).unwrap();
            m.forget(id);
            assert_eq!(m.stats().pending_cleanup, 1);
        }
        clock.advance(2.0);
        let handle = shared.spawn_maintenance(Duration::from_millis(5));
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let stats = shared.lock().stats();
            if stats.slots == stats.free_slots {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "worker did not clean up");
            std::thread::sleep(Duration::from_millis(5));
        }
        handle.stop();
        assert!(shared.lock().is_empty());
    }
}
