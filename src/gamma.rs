//! The gamma function via the Stirling asymptotic series with a rigorous
//! remainder, plus the reflection formula for arguments left of 1/2.
//!
//! For real z > 0:
//!   ln Γ(z) = (z − 1/2) ln z − z + ½ ln 2π
//!           + Σ_{k=1}^{M} B_{2k} / (2k(2k−1) z^(2k−1)) + R_M(z),
//! and the classical remainder bound for real positive z is
//!   |R_M(z)| ≤ |B_{2M+2}| / ((2M+2)(2M+1) z^(2M+1)),
//! i.e. the magnitude of the first omitted term (Whittaker & Watson §12.33).
//! The bound is monotone decreasing in z, so evaluating it at a certified
//! lower bound of the argument ball covers every point of the ball.
//!
//! Arguments with midpoint < 1/2 go through Γ(x)·Γ(1−x) = π/sin(πx); balls
//! containing a pole (nonpositive integer) yield `None` from [`try_gamma`]
//! because the sin ball then contains zero.

use crate::ball::Ball;
use crate::bernoulli;
use crate::constants;
use crate::elementary;
use crate::fp::{Float, Round};
use crate::mag::Mag;

const MAX_ATTEMPTS: u32 = 3;

/// Γ(x). Panics if the ball contains a pole or the parameters are hopeless;
/// use [`try_gamma`] for the fallible form.
pub fn gamma(x: &Ball, prec: u32) -> Ball {
    try_gamma(x, prec).expect("gamma: ball contains a pole of Γ")
}

/// Γ(x), or None if the ball contains a nonpositive integer (a pole).
pub fn try_gamma(x: &Ball, prec: u32) -> Option<Ball> {
    for attempt in 0..=MAX_ATTEMPTS {
        let extra = 64 + prec / 2 * attempt;
        let r = gamma_attempt(x, prec + extra)?;
        #[cfg(feature = "trace-retries")]
        eprintln!(
            "gamma attempt {attempt}: acc {} (want {prec})",
            r.rel_accuracy_bits()
        );
        if r.rel_accuracy_bits() >= prec as i64 || attempt == MAX_ATTEMPTS {
            return Some(r.round(prec));
        }
    }
    unreachable!()
}

/// ln Γ(x) for balls strictly right of 0.
pub fn ln_gamma(x: &Ball, prec: u32) -> Ball {
    assert!(
        x.lower_bound().signum() > 0,
        "ln_gamma requires a strictly positive ball"
    );
    for attempt in 0..=MAX_ATTEMPTS {
        let extra = 64 + prec / 2 * attempt;
        let wp = prec + extra;
        let (shifted, poch) = shift_argument(x, wp);
        let lg = ln_gamma_stirling(&shifted, wp);
        let r = if let Some(p) = poch {
            lg.sub(&elementary::ln(&p, wp), wp)
        } else {
            lg
        };
        if r.rel_accuracy_bits() >= prec as i64 || attempt == MAX_ATTEMPTS {
            return r.round(prec);
        }
    }
    unreachable!()
}

fn gamma_attempt(x: &Ball, wp: u32) -> Option<Ball> {
    let mid_neg_or_left = x.mid().cmp(&Float::from_f64(0.5)) == core::cmp::Ordering::Less;
    if mid_neg_or_left {
        // Reflection: Γ(x) = π / (sin(πx) · Γ(1−x)).
        let pi = constants::pi(wp + 8);
        let one_minus = Ball::from_i64(1).sub(x, wp);
        let g = gamma_pos(&one_minus, wp)?;
        let s = elementary::sin(&pi.mul(x, wp), wp);
        return pi.try_div(&s.mul(&g, wp), wp);
    }
    gamma_pos(x, wp)
}

/// Γ for balls with midpoint ≥ 1/2 (still may touch 0 on the left: caught
/// by the shifted-product division).
fn gamma_pos(x: &Ball, wp: u32) -> Option<Ball> {
    let (shifted, poch) = shift_argument(x, wp);
    let lg = ln_gamma_stirling(&shifted, wp);
    let g = elementary::exp(&lg, wp);
    match poch {
        Some(p) => g.try_div(&p, wp),
        None => Some(g),
    }
}

/// Series-length budget. Tangent-number generation costs O(M²) big-integer
/// operations (≈ M³ bit ops), so M is capped and the argument shift takes
/// up the slack — shift multiplications are only O(n) each. This is the
/// honest cost center vs Arb, which generates Bernoulli numbers much
/// faster (see README).
const MAX_STIRLING_TERMS: f64 = 1500.0;

/// Stirling series length and argument target for `wp` bits:
/// the M-th term is ≈ (2M/(2πez))^(2M), so requiring 2^−wp gives
/// z ≈ (M/2π)·exp(wp·ln2/(2M)).
fn stirling_params(wp: u32) -> (usize, f64) {
    let m = ((wp as f64) * 0.17 + 8.0).min(MAX_STIRLING_TERMS);
    let z = (m / (2.0 * core::f64::consts::PI))
        * (((wp as f64) * core::f64::consts::LN_2 + 16.0) / (2.0 * m)).exp()
        + 4.0;
    (m as usize, z)
}

/// Shift x up by an integer r so Stirling converges fast:
/// Γ(x) = Γ(x + r) / (x(x+1)…(x+r−1)).
/// Returns (x + r, product ball or None if r == 0).
fn shift_argument(x: &Ball, wp: u32) -> (Ball, Option<Ball>) {
    let (_, target) = stirling_params(wp);
    let xm = x.mid().to_f64();
    let r = (target - xm).ceil().max(0.0) as i64;
    if r == 0 {
        return (x.clone(), None);
    }
    let mut prod = x.clone();
    let mut term = x.clone();
    for _ in 1..r {
        term = term.add(&Ball::from_i64(1), wp);
        prod = prod.mul(&term, wp);
    }
    (x.add(&Ball::from_i64(r), wp), Some(prod))
}

/// Stirling series for ln Γ(z), z a ball with certified lower bound
/// z_low ≳ wp·ln2/(2π) (arranged by [`shift_argument`]).
fn ln_gamma_stirling(z: &Ball, wp: u32) -> Ball {
    let z_low = z.lower_bound();
    assert!(z_low.signum() > 0, "Stirling argument must be positive");
    let zf = z_low.to_f64();

    // Choose M so the first omitted term is ≲ 2^−(wp+8), estimated in f64
    // via ln|B_2k| ≈ ln 2 + lnΓ(2k+1) − 2k ln(2π). The rigorous bound below
    // is computed with the actual B_{2M+2}; a bad M only widens the ball.
    let (m_budget, _) = stirling_params(wp);
    let mut m = 1usize;
    let lo_target = -((wp as f64) + 8.0) * core::f64::consts::LN_2;
    while m < m_budget + 64 {
        let k2 = (2 * m + 2) as f64;
        let ln_b = core::f64::consts::LN_2 + ln_gamma_f64(k2 + 1.0)
            - k2 * (2.0 * core::f64::consts::PI).ln();
        let ln_term = ln_b - (k2 - 1.0) * zf.ln() - (k2 * (k2 - 1.0)).ln();
        if ln_term < lo_target {
            break;
        }
        m += 1;
    }

    // Series: Σ_{k=1}^{M} B_2k / (2k(2k−1)) · z^(1−2k), incremental in 1/z².
    let inv_z = Ball::from_i64(1).div(z, wp);
    let inv_z2 = inv_z.mul(&inv_z, wp);
    let mut pow = inv_z.clone(); // z^(1−2k) for k = 1
    let mut sum = Ball::zero();
    for k in 1..=m {
        let b2k = bernoulli::bernoulli(k, wp);
        let t = b2k
            .mul(&pow, wp)
            .div_u64((2 * k as u64) * (2 * k as u64 - 1), wp);
        sum = sum.add(&t, wp);
        if k < m {
            pow = pow.mul(&inv_z2, wp);
        }
    }

    // Rigorous remainder: |R_M| ≤ |B_{2M+2}| / ((2M+2)(2M+1) z_low^(2M+1)).
    let rem = {
        let b_next = bernoulli::bernoulli(m + 1, 64);
        let num = b_next.abs_upper();
        let (den1, _) = Float::from_u64((2 * m as u64 + 2) * (2 * m as u64 + 1)).mul(
            &pow_down(&z_low, 2 * m as u64 + 1),
            64,
            Round::Down,
        );
        let (bound, _) = num.div(&den1, 64, Round::Up);
        Mag::from_float_upper(&bound)
    };
    let sum = sum.add_error(&rem);

    // (z − 1/2) ln z − z + ½ ln 2π + series.
    let half = Ball::exact(Float::from_f64(0.5));
    let lnz = elementary::ln(z, wp);
    let main = z.sub(&half, wp).mul(&lnz, wp).sub(z, wp);
    let ln_2pi = constants::ln_2pi(wp + 8);
    main.add(&ln_2pi.mul_2exp(-1), wp).add(&sum, wp)
}

/// x^n rounded down (x > 0), 64-bit — for remainder denominators.
fn pow_down(x: &Float, mut n: u64) -> Float {
    let mut base = x.clone();
    let mut acc = Float::from_i64(1);
    while n > 0 {
        if n & 1 == 1 {
            acc = acc.mul(&base, 64, Round::Down).0;
        }
        base = base.mul(&base, 64, Round::Down).0;
        n >>= 1;
    }
    acc
}

/// lnΓ for f64 (parameter tuning only; no rigor needed): Stirling.
fn ln_gamma_f64(x: f64) -> f64 {
    // x ≥ 3 here always (called with 2m+3).
    let z = x;
    (z - 0.5) * z.ln() - z + 0.5 * (2.0 * core::f64::consts::PI).ln() + 1.0 / (12.0 * z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_integers_are_factorials() {
        let mut fact = 1u64;
        for n in 1..=10u64 {
            if n > 1 {
                fact *= n - 1;
            }
            let g = gamma(&Ball::from_u64(n), 128);
            assert!(
                g.contains(&Float::from_u64(fact)),
                "Γ({n}) should be {fact}, got {}",
                g.to_string_digits(25)
            );
            assert!(g.rel_accuracy_bits() >= 120);
        }
    }

    #[test]
    fn gamma_half_is_sqrt_pi() {
        let prec = 512;
        let g = gamma(&Ball::from_f64(0.5), prec);
        let sp = constants::pi(prec + 16).sqrt(prec + 16);
        let d = g.sub(&sp, prec);
        assert!(d.contains(&Float::zero()), "Γ(1/2) ≠ √π");
        assert!(g.rel_accuracy_bits() >= prec as i64 - 8);
    }

    #[test]
    fn functional_equation() {
        // Γ(x+1) = x Γ(x) for a few x.
        for xv in [0.75f64, 1.5, 3.25, 10.125] {
            let prec = 256;
            let x = Ball::from_f64(xv);
            let lhs = gamma(&x.add(&Ball::from_i64(1), prec + 32), prec);
            let rhs = x.mul(&gamma(&x, prec), prec);
            let d = lhs.sub(&rhs, prec);
            assert!(d.contains(&Float::zero()), "Γ(x+1) ≠ xΓ(x) at {xv}");
        }
    }

    #[test]
    fn reflection() {
        // Γ(1/3)Γ(2/3) = π / sin(π/3) = 2π/√3.
        let prec = 320;
        let third = Ball::from_i64(1).div_u64(3, prec + 32);
        let g13 = gamma(&third, prec);
        let g23 = gamma(&Ball::from_i64(2).div_u64(3, prec + 32), prec);
        let lhs = g13.mul(&g23, prec);
        let pi = constants::pi(prec + 32);
        let rhs = pi.mul_2exp(1).div(&Ball::from_i64(3).sqrt(prec + 32), prec);
        let d = lhs.sub(&rhs, prec);
        assert!(d.contains(&Float::zero()), "reflection identity failed");
    }

    #[test]
    fn negative_argument() {
        // Γ(−1/2) = −2√π.
        let prec = 256;
        let g = gamma(&Ball::from_f64(-0.5), prec);
        let want = constants::pi(prec + 16)
            .sqrt(prec + 16)
            .mul_i64(-2, prec + 16);
        let d = g.sub(&want, prec);
        assert!(d.contains(&Float::zero()), "Γ(−1/2) ≠ −2√π");
    }

    #[test]
    fn pole_detection() {
        assert!(try_gamma(&Ball::from_i64(0), 64).is_none());
        assert!(try_gamma(&Ball::from_i64(-3), 64).is_none());
        assert!(try_gamma(&Ball::from_i64(5), 64).is_some());
    }

    #[test]
    fn ln_gamma_consistency() {
        let prec = 256;
        let x = Ball::from_f64(7.25);
        let lg = ln_gamma(&x, prec);
        let g = gamma(&x, prec);
        let d = elementary::exp(&lg, prec).sub(&g, prec);
        assert!(d.contains(&Float::zero()), "exp(lnΓ) ≠ Γ");
    }
}
