//! Benchmark regression canary for CI.
//!
//! Asserts that key operations finish inside generous time budgets (≈8×
//! the measured numbers on a 2023 desktop). Not a benchmark — a tripwire:
//! it fails only on order-of-magnitude regressions, which is what a shared
//! CI runner can reliably detect. Full numbers come from `cargo bench`.
//!
//! Run: `cargo run --release --example bench_smoke`

use rigor::{ball::Ball, binsplit, elementary, gamma, zeta};
use std::time::Instant;

struct Check {
    name: &'static str,
    budget_ms: u128,
    actual_ms: u128,
}

fn timed(name: &'static str, budget_ms: u128, f: impl FnOnce()) -> Check {
    let t0 = Instant::now();
    f();
    Check {
        name,
        budget_ms,
        actual_ms: t0.elapsed().as_millis(),
    }
}

fn main() {
    let p1k = 3339u32; // 1000 digits
    let p10k = 33_235u32; // 10 000 digits
    let x = Ball::from_f64(1.5);

    // Warm constant/Bernoulli caches outside the timed regions.
    let _ = elementary::ln(&x, p10k);
    let _ = gamma::gamma(&x, p1k);
    let _ = zeta::zeta(&Ball::from_i64(3), p1k);

    let checks = vec![
        timed("exp 10k digits", 400, || {
            let _ = elementary::exp(&x, p10k);
        }),
        timed("ln 10k digits", 1200, || {
            let _ = elementary::ln(&x, p10k);
        }),
        timed("sin 10k digits", 1200, || {
            let _ = elementary::sin(&x, p10k);
        }),
        timed("atan 10k digits", 2000, || {
            let _ = elementary::atan(&x, p10k);
        }),
        timed("pi 100k digits", 2000, || {
            let _ = binsplit::pi_chudnovsky(332_270);
        }),
        timed("e 100k digits", 800, || {
            let _ = binsplit::e_binsplit(332_270);
        }),
        timed("gamma 1k digits (warm)", 800, || {
            let _ = gamma::gamma(&x, p1k);
        }),
        timed("zeta(3) 1k digits (warm)", 800, || {
            let _ = zeta::zeta(&Ball::from_i64(3), p1k);
        }),
    ];

    let mut failed = false;
    for c in &checks {
        let status = if c.actual_ms <= c.budget_ms {
            "ok  "
        } else {
            "FAIL"
        };
        if c.actual_ms > c.budget_ms {
            failed = true;
        }
        println!(
            "{status} {:28} {:>6} ms (budget {} ms)",
            c.name, c.actual_ms, c.budget_ms
        );
    }
    if failed {
        eprintln!("benchmark regression canary tripped");
        std::process::exit(1);
    }
}
