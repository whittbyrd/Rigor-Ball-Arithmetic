//! Inclusion property tests: the central correctness contract.
//!
//! For random exact inputs x, evaluate f at precision p (a ball B) and at a
//! much higher precision P (the oracle ball O, which encloses the true value
//! tightly). The truth lies in both, so:
//!   1. `B − O` must contain zero (the balls overlap), and
//!   2. `B` must contain `O`'s midpoint once widened by `O`'s (tiny) radius —
//!      practically: O.mid ∈ B widened by O.rad.
//! We check the stronger form (2) via the exact `contains` predicate.
//!
//! Any failure here is a soundness bug, not a quality regression.

use rigor::ball::Ball;
use rigor::elementary;
use rigor::fp::Float;
use rigor::gamma;
use rigor::testutil::Rng;
use rigor::zeta;

/// Check that the low-precision ball contains the oracle's midpoint modulo
/// the oracle's own radius, and report the seed context on failure.
fn assert_inclusion(b: &Ball, oracle: &Ball, what: &str) {
    let widened = b.add_error(oracle.rad());
    assert!(
        widened.contains(oracle.mid()),
        "{what}: inclusion violated\n  ball   = {}\n  oracle = {}",
        b.to_string_digits(30),
        oracle.to_string_digits(30),
    );
}

/// Tightness guard: accuracy should be within a modest distance of prec.
/// (Not a soundness property — catches silently-useless output.)
fn assert_useful(b: &Ball, prec: u32, what: &str) {
    assert!(
        b.rel_accuracy_bits() >= prec as i64 - 16,
        "{what}: ball too wide: {} certified bits at precision {prec}",
        b.rel_accuracy_bits()
    );
}

fn random_exact(rng: &mut Rng, scale_pow: i64) -> Ball {
    let v = rng.next() as i64;
    Ball::exact(Float::from_i64(v).mul_2exp(rng.range_i64(-scale_pow, scale_pow) - 62))
}

#[test]
fn exp_inclusion_random() {
    let mut rng = Rng::new(600);
    for i in 0..40 {
        let x = random_exact(&mut rng, 8); // |x| up to ~2^8
        let prec = 64 + rng.below(400) as u32;
        let b = elementary::exp(&x, prec);
        let o = elementary::exp(&x, 2 * prec + 128);
        assert_inclusion(&b, &o, &format!("exp #{i}"));
        assert_useful(&b, prec, &format!("exp #{i}"));
    }
}

#[test]
fn ln_inclusion_random() {
    let mut rng = Rng::new(601);
    for i in 0..40 {
        let mut x = random_exact(&mut rng, 30);
        if x.mid().signum() <= 0 {
            x = x.neg();
        }
        if x.mid().is_zero() {
            continue;
        }
        let prec = 64 + rng.below(400) as u32;
        let b = elementary::ln(&x, prec);
        let o = elementary::ln(&x, 2 * prec + 128);
        assert_inclusion(&b, &o, &format!("ln #{i}"));
        assert_useful(&b, prec, &format!("ln #{i}"));
    }
}

#[test]
fn sin_cos_inclusion_random() {
    let mut rng = Rng::new(602);
    for i in 0..30 {
        let x = random_exact(&mut rng, 10);
        let prec = 64 + rng.below(300) as u32;
        let (s, c) = elementary::sin_cos(&x, prec);
        let (so, co) = elementary::sin_cos(&x, 2 * prec + 128);
        assert_inclusion(&s, &so, &format!("sin #{i}"));
        assert_inclusion(&c, &co, &format!("cos #{i}"));
    }
}

#[test]
fn atan_inclusion_random() {
    let mut rng = Rng::new(603);
    for i in 0..40 {
        let x = random_exact(&mut rng, 40);
        let prec = 64 + rng.below(300) as u32;
        let b = elementary::atan(&x, prec);
        let o = elementary::atan(&x, 2 * prec + 128);
        assert_inclusion(&b, &o, &format!("atan #{i}"));
        assert_useful(&b, prec, &format!("atan #{i}"));
    }
}

#[test]
fn gamma_inclusion_random() {
    let mut rng = Rng::new(604);
    for i in 0..15 {
        // Positive, away from 0 (poles are rejected separately).
        let v = (rng.below(20_000) as f64 + 1.0) / 512.0;
        let x = Ball::from_f64(v);
        let prec = 64 + rng.below(200) as u32;
        let b = gamma::gamma(&x, prec);
        let o = gamma::gamma(&x, 2 * prec + 128);
        assert_inclusion(&b, &o, &format!("gamma #{i} (x={v})"));
        assert_useful(&b, prec, &format!("gamma #{i} (x={v})"));
    }
}

#[test]
fn gamma_inclusion_negative() {
    let mut rng = Rng::new(605);
    for i in 0..10 {
        // Negative non-integer arguments through the reflection formula.
        let v = -((rng.below(5000) as f64 + 1.0) / 512.0) - 0.25 / 512.0;
        let x = Ball::from_f64(v);
        if let Some(b) = gamma::try_gamma(&x, 160) {
            let o = gamma::gamma(&x, 512);
            assert_inclusion(&b, &o, &format!("gamma− #{i} (x={v})"));
        }
    }
}

#[test]
fn zeta_inclusion_random() {
    let mut rng = Rng::new(606);
    for i in 0..8 {
        let s = Ball::from_u64(2 + rng.below(40));
        let prec = 64 + rng.below(200) as u32;
        let b = zeta::zeta(&s, prec);
        let o = zeta::zeta(&s, 2 * prec + 128);
        assert_inclusion(&b, &o, &format!("zeta #{i}"));
        assert_useful(&b, prec, &format!("zeta #{i}"));
    }
    // A couple of non-integer points through the slow path.
    for v in [0.5f64, 2.5, 7.25] {
        let s = Ball::from_f64(v);
        if let Some(b) = zeta::try_zeta(&s, 96) {
            let o = zeta::zeta(&s, 320);
            assert_inclusion(&b, &o, &format!("zeta s={v}"));
        }
    }
}

#[test]
fn composed_expression_inclusion() {
    // Compose several functions; inclusion must survive composition.
    let mut rng = Rng::new(607);
    for i in 0..10 {
        let x = random_exact(&mut rng, 4);
        let prec = 128 + rng.below(200) as u32;
        let f = |p: u32| -> Ball {
            // f(x) = ln(1 + exp(x)) · cos(x)² + atan(x)
            let ex = elementary::exp(&x, p);
            let l = elementary::ln(&ex.add(&Ball::from_i64(1), p), p);
            let c = elementary::cos(&x, p);
            l.mul(&c.mul(&c, p), p).add(&elementary::atan(&x, p), p)
        };
        let b = f(prec);
        let o = f(2 * prec + 128);
        assert_inclusion(&b, &o, &format!("composed #{i}"));
    }
}
