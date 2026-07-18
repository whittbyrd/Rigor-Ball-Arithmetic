//! The dependency problem, demonstrated rather than hidden.
//!
//! Ball (and interval) arithmetic forgets correlations between operands:
//! `x − x` over a fat ball is not 0 but a ball of width 2r, and evaluating
//! a polynomial in different algebraic forms yields different widths.
//! This example demonstrates the effect and the standard mitigations
//! (rewriting expressions, and subdividing input intervals).
//!
//! Run: `cargo run --release --example dependency_problem`

use rigor::ball::Ball;
use rigor::fp::Float;
use rigor::mag::Mag;

fn width_bits(b: &Ball) -> i64 {
    if b.is_exact() {
        i64::MIN
    } else {
        b.rad().exp()
    }
}

fn main() {
    let prec = 64;

    // 1. x − x with a fat ball.
    let x = Ball::new(Float::from_i64(1), Mag::two_exp(-8)); // [1 ± 2^-8]
    let d = x.sub(&x, prec);
    println!("x = [1 ± 2^-8]");
    println!(
        "x − x           = {}   (true value: exactly 0; width ~2^{})",
        d.to_string_digits(12),
        width_bits(&d)
    );

    // 2. The same function, three algebraic forms: f(x) = x² − x on [0,1]:
    //    (a) x·x − x          (b) x·(x − 1)         (c) (x − ½)² − ¼
    // Form (c) is centered and gives the tightest enclosure.
    let x = Ball::new(Float::from_f64(0.5), Mag::two_exp(-1)); // [0, 1]
    let one = Ball::from_i64(1);
    let half = Ball::exact(Float::from_f64(0.5));
    let quarter = Ball::exact(Float::from_f64(0.25));

    let fa = x.mul(&x, prec).sub(&x, prec);
    let fb = x.mul(&x.sub(&one, prec), prec);
    let xc = x.sub(&half, prec);
    let fc = xc.mul(&xc, prec).sub(&quarter, prec);
    println!("\nf(x) = x² − x on x = [0, 1]  (true range: [−1/4, 0])");
    println!("  x·x − x       = {}", fa.to_string_digits(8));
    println!("  x·(x − 1)     = {}", fb.to_string_digits(8));
    println!("  (x−½)² − ¼    = {}", fc.to_string_digits(8));

    // 3. Mitigation by subdivision: split [0,1] into 2^k pieces and hull.
    println!("\nsubdivision of x·(x − 1) on [0, 1]:");
    for k in [1u32, 3, 6] {
        let n = 1u64 << k;
        let mut lo = Float::from_i64(1);
        let mut hi = Float::from_i64(-1);
        for i in 0..n {
            // piece midpoint (2i+1)/2n, radius 1/2n
            let mid = Float::from_u64(2 * i + 1).mul_2exp(-(k as i64 + 1));
            let piece = Ball::new(mid, Mag::two_exp(-(k as i64 + 1)));
            let f = piece.mul(&piece.sub(&one, prec), prec);
            let (plo, phi) = f.endpoints();
            if plo.cmp(&lo) == core::cmp::Ordering::Less {
                lo = plo;
            }
            if phi.cmp(&hi) == core::cmp::Ordering::Greater {
                hi = phi;
            }
        }
        println!(
            "  {n:3} pieces -> [{}, {}]",
            lo.to_decimal(6),
            hi.to_decimal(6)
        );
    }
    println!("\nBalls never lie — but correlated expressions deserve rewriting.");
}
