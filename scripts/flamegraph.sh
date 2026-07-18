#!/usr/bin/env bash
# Generate the README flamegraph (Linux; needs `cargo install flamegraph`
# and perf). Profiles exp+ln+sin at 10k digits.
set -euo pipefail
mkdir -p docs
cargo flamegraph --example quick_timing -o docs/flamegraph.svg
echo "wrote docs/flamegraph.svg"
