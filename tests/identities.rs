//! Cross-algorithm identity tests: different code paths must produce
//! overlapping balls for mathematically equal quantities. These catch
//! systematic bias that inclusion self-tests (same code at two precisions)
//! could miss.

use rigor::ball::Ball;
use rigor::binsplit;
use rigor::constants;
use rigor::elementary;
use rigor::fp::Float;
use rigor::gamma;
use rigor::zeta;

fn assert_same(a: &Ball, b: &Ball, prec: u32, what: &str) {
    let d = a.sub(b, prec + 64);
    assert!(
        d.contains(&Float::zero()),
        "{what}: balls disagree\n  a = {}\n  b = {}",
        a.to_string_digits(30),
        b.to_string_digits(30)
    );
}

#[test]
fn pi_three_ways() {
    let prec = 4000;
    let chud = binsplit::pi_chudnovsky(prec);
    // 4·atan(1) — exercises atan reduction + series.
    let atan_pi = elementary::atan(&Ball::from_i64(1), prec).mul_2exp(2);
    assert_same(&chud, &atan_pi, prec, "pi: Chudnovsky vs 4·atan(1)");
    // Γ(1/2)² — exercises Stirling + reflection machinery.
    let g = gamma::gamma(&Ball::from_f64(0.5), prec);
    let g2 = g.mul(&g, prec);
    assert_same(&chud, &g2, prec, "pi: Chudnovsky vs Γ(1/2)²");
}

#[test]
fn e_two_ways() {
    let prec = 4000;
    let bs = binsplit::e_binsplit(prec);
    let ex = elementary::exp(&Ball::from_i64(1), prec);
    assert_same(&bs, &ex, prec, "e: binsplit vs exp(1)");
}

#[test]
fn ln2_two_ways() {
    let prec = 4000;
    let bs = binsplit::ln2_binsplit(prec);
    let ln = elementary::ln(&Ball::from_i64(2), prec);
    assert_same(&bs, &ln, prec, "ln2: binsplit vs ln(2)");
}

#[test]
fn gamma_duplication() {
    // Γ(2x) = Γ(x)·Γ(x+1/2)·2^(2x−1)/√π.
    let prec = 400;
    for xv in [0.75f64, 1.25, 3.5, 6.75] {
        let x = Ball::from_f64(xv);
        let lhs = gamma::gamma(&x.mul_2exp(1), prec);
        let gx = gamma::gamma(&x, prec);
        let gxh = gamma::gamma(&x.add(&Ball::from_f64(0.5), prec + 32), prec);
        let sp = constants::pi(prec + 32).sqrt(prec + 32);
        // 2^(2x−1) with 2x−1 an exact dyadic: use exp((2x−1)·ln2).
        let ln2 = constants::ln2(prec + 32);
        let e2 = elementary::exp(
            &Ball::from_f64(2.0 * xv - 1.0).mul(&ln2, prec + 32),
            prec + 32,
        );
        let rhs = gx.mul(&gxh, prec).mul(&e2, prec).div(&sp, prec);
        assert_same(&lhs, &rhs, prec, &format!("duplication at x={xv}"));
    }
}

#[test]
fn zeta_even_values_vs_bernoulli_closed_form() {
    // ζ(6) = π⁶/945, ζ(8) = π⁸/9450.
    let prec = 300;
    let pi = constants::pi(prec + 32);
    let p2 = pi.mul(&pi, prec + 32);
    let p4 = p2.mul(&p2, prec + 32);
    let p6 = p4.mul(&p2, prec + 32);
    let p8 = p4.mul(&p4, prec + 32);
    assert_same(
        &zeta::zeta(&Ball::from_i64(6), prec),
        &p6.div_u64(945, prec),
        prec,
        "zeta(6)",
    );
    assert_same(
        &zeta::zeta(&Ball::from_i64(8), prec),
        &p8.div_u64(9450, prec),
        prec,
        "zeta(8)",
    );
}

#[test]
fn trig_addition_formula() {
    // sin(a+b) = sin a cos b + cos a sin b at ball level.
    let prec = 300;
    let a = Ball::from_f64(1.1);
    let b = Ball::from_f64(-2.7);
    let (sa, ca) = elementary::sin_cos(&a, prec);
    let (sb, cb) = elementary::sin_cos(&b, prec);
    let lhs = elementary::sin(&a.add(&b, prec + 32), prec);
    let rhs = sa.mul(&cb, prec).add(&ca.mul(&sb, prec), prec);
    assert_same(&lhs, &rhs, prec, "sin addition formula");
}

#[test]
fn atan_of_large_and_small_sum_to_half_pi() {
    // atan(x) + atan(1/x) = π/2 for x > 0 — exercises both atan branches.
    let prec = 300;
    for xv in [3.0f64, 17.5, 1000.25] {
        let x = Ball::from_f64(xv);
        let inv = Ball::from_i64(1).div(&x, prec + 32);
        let s = elementary::atan(&x, prec).add(&elementary::atan(&inv, prec), prec);
        let hp = constants::pi(prec + 32).mul_2exp(-1);
        assert_same(&s, &hp, prec, &format!("atan pair at {xv}"));
    }
}
