//! Elementary-function throughput across precisions.
//!
//! Caches (π, ln 2, Bernoulli) are warmed before measurement so the numbers
//! reflect steady-state per-call cost; cold-start costs are reported by
//! `examples/quick_timing.rs`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rigor::{ball::Ball, constants, elementary};

const DIGITS: &[u32] = &[100, 1_000, 10_000];

fn prec_for(digits: u32) -> u32 {
    (digits as f64 * std::f64::consts::LOG2_10).ceil() as u32 + 16
}

fn bench_elementary(c: &mut Criterion) {
    let x = Ball::from_f64(1.5);
    for &digits in DIGITS {
        let prec = prec_for(digits);
        // Warm constant caches at this precision tier.
        let _ = constants::pi(prec + 256);
        let _ = constants::ln2(prec + 256);

        let mut group = c.benchmark_group(format!("{digits}_digits"));
        if digits >= 10_000 {
            group.sample_size(10);
        }
        group.bench_function(BenchmarkId::new("exp", digits), |b| {
            b.iter(|| elementary::exp(std::hint::black_box(&x), prec))
        });
        group.bench_function(BenchmarkId::new("ln", digits), |b| {
            b.iter(|| elementary::ln(std::hint::black_box(&x), prec))
        });
        group.bench_function(BenchmarkId::new("sin", digits), |b| {
            b.iter(|| elementary::sin(std::hint::black_box(&x), prec))
        });
        group.bench_function(BenchmarkId::new("atan", digits), |b| {
            b.iter(|| elementary::atan(std::hint::black_box(&x), prec))
        });
        group.bench_function(BenchmarkId::new("sqrt", digits), |b| {
            b.iter(|| std::hint::black_box(&x).sqrt(prec))
        });
        group.bench_function(BenchmarkId::new("mul", digits), |b| {
            let pi = constants::pi(prec);
            let l2 = constants::ln2(prec);
            b.iter(|| std::hint::black_box(&pi).mul(std::hint::black_box(&l2), prec))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_elementary);
criterion_main!(benches);
