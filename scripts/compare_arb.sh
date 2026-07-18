#!/usr/bin/env bash
# Reproduce the rigor-vs-Arb-vs-MPFR table from the README (Linux).
#
#   sudo apt-get install libflint-dev libmpfr-dev
#   ./scripts/compare_arb.sh
set -euo pipefail
cargo run --release --manifest-path tools/arb-diff/Cargo.toml
cargo run --release --manifest-path tools/arb-bench/Cargo.toml
