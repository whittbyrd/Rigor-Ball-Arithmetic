//! Special functions: gamma and zeta. Bernoulli/tangent caches are warmed
//! first (their one-time generation cost is the honest headline gap vs Arb
//! at high precision — see README).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rigor::{ball::Ball, gamma, zeta};

const DIGITS: &[u32] = &[100, 1_000, 5_000];

fn prec_for(digits: u32) -> u32 {
    (digits as f64 * std::f64::consts::LOG2_10).ceil() as u32 + 16
}

fn bench_special(c: &mut Criterion) {
    let x = Ball::from_f64(1.5);
    let s3 = Ball::from_i64(3);
    for &digits in DIGITS {
        let prec = prec_for(digits);
        // Warm Bernoulli + constant caches.
        let _ = gamma::gamma(&x, prec);
        let _ = zeta::zeta(&s3, prec);

        let mut group = c.benchmark_group(format!("special_{digits}d"));
        if digits >= 1_000 {
            group.sample_size(10);
        }
        group.bench_function(BenchmarkId::new("gamma_1_5", digits), |b| {
            b.iter(|| gamma::gamma(std::hint::black_box(&x), prec))
        });
        group.bench_function(BenchmarkId::new("zeta_3", digits), |b| {
            b.iter(|| zeta::zeta(std::hint::black_box(&s3), prec))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_special);
criterion_main!(benches);
