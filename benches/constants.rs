//! Constants via binary splitting: digits/second for π, e, ln 2.
//! Calls the binsplit functions directly (bypassing the cache) so each
//! iteration does full work.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rigor::binsplit;

const DIGITS: &[u32] = &[100, 1_000, 10_000, 100_000];

fn prec_for(digits: u32) -> u32 {
    (digits as f64 * std::f64::consts::LOG2_10).ceil() as u32 + 16
}

fn bench_constants(c: &mut Criterion) {
    let mut group = c.benchmark_group("constants");
    for &digits in DIGITS {
        let prec = prec_for(digits);
        if digits >= 10_000 {
            group.sample_size(10);
        }
        group.bench_function(BenchmarkId::new("pi_chudnovsky", digits), |b| {
            b.iter(|| binsplit::pi_chudnovsky(std::hint::black_box(prec)))
        });
        group.bench_function(BenchmarkId::new("e_binsplit", digits), |b| {
            b.iter(|| binsplit::e_binsplit(std::hint::black_box(prec)))
        });
        group.bench_function(BenchmarkId::new("ln2_binsplit", digits), |b| {
            b.iter(|| binsplit::ln2_binsplit(std::hint::black_box(prec)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_constants);
criterion_main!(benches);
