#!/usr/bin/env bash
# Run the full criterion suite and the cold-start timing report (Linux/macOS).
set -euo pipefail
cargo bench -- --noplot
cargo run --release --example quick_timing
cargo run --release --example bench_smoke
