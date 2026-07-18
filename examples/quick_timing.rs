//! Rough timing sanity check: `cargo run --release --example quick_timing`
use rigor::{ball::Ball, constants, elementary, gamma, zeta};
use std::time::Instant;

fn time<F: FnMut() -> Ball>(name: &str, mut f: F) {
    let t0 = Instant::now();
    let r = f();
    let dt = t0.elapsed();
    println!(
        "{name:24} {:>10.3} ms   acc {} bits",
        dt.as_secs_f64() * 1e3,
        r.rel_accuracy_bits()
    );
}

fn main() {
    for digits in [100u32, 1_000, 10_000] {
        let prec = (digits as f64 * 3.3219).ceil() as u32 + 16;
        println!("--- {digits} digits ({prec} bits) ---");
        let x = Ball::from_f64(1.5);
        time("exp(1.5)", || elementary::exp(&x, prec));
        time("ln(1.5)", || elementary::ln(&x, prec));
        time("sin(1.5)", || elementary::sin(&x, prec));
        time("atan(1.5)", || elementary::atan(&x, prec));
        time("pi (cold-ish)", || constants::pi(prec));
        if digits <= 1_000 {
            time("gamma(1.5)", || gamma::gamma(&x, prec));
            time("zeta(3)", || zeta::zeta(&Ball::from_i64(3), prec));
        }
    }
}
