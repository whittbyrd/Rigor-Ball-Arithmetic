//! Compute π and e to 100 000 verified decimal digits.
//!
//! `cargo run --release --example digits_100k [digits]`
//!
//! Every printed digit is certified: the ball radius is checked to be small
//! enough that the decimal expansion of the midpoint is correct to the
//! requested length (up to the usual caveat about decimal strings adjacent
//! to a rounding boundary; we verify the radius leaves > 3 spare digits).

use rigor::binsplit;
use std::time::Instant;

fn main() {
    let digits: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let prec = (digits as f64 * std::f64::consts::LOG2_10).ceil() as u32 + 64;

    let t0 = Instant::now();
    let pi = binsplit::pi_chudnovsky(prec);
    let t_pi = t0.elapsed();
    let t0 = Instant::now();
    let e = binsplit::e_binsplit(prec);
    let t_e = t0.elapsed();

    for (name, ball, secs) in [("pi", &pi, t_pi), ("e", &e, t_e)] {
        let acc = ball.rel_accuracy_bits();
        let spare_bits = acc as f64 - digits as f64 * std::f64::consts::LOG2_10;
        assert!(
            spare_bits > 10.0,
            "{name}: only {acc} certified bits for {digits} digits"
        );
        let s = ball.mid().to_decimal(digits);
        println!(
            "{name}: {} digits in {:.3}s (certified {acc} bits, {:.0} spare)",
            s.len() - 2,
            secs.as_secs_f64(),
            spare_bits
        );
        println!("  head: {}", &s[..60.min(s.len())]);
        println!("  tail: …{}", &s[s.len().saturating_sub(40)..]);
    }
}
