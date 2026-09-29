#!/usr/bin/env bash
# Optional build tuned for this machine's CPU (AVX-512, VNNI, FMA, ...).
# The binary runs only on CPUs with the same instruction sets, so it goes to
# its own target directory: target/native/release/snn-lm.
# Usage: scripts/build-native.sh [extra cargo args]
set -euo pipefail
cd "$(dirname "$0")/.."
RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native" cargo build --release --target-dir target/native -p snn-lm "$@"
echo "built target/native/release/snn-lm"
