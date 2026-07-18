# Run the full criterion suite and the cold-start timing report (Windows).
# Results land in target/criterion; summary lines are printed to stdout.
$ErrorActionPreference = "Stop"
cargo bench -- --noplot
cargo run --release --example quick_timing
cargo run --release --example bench_smoke
