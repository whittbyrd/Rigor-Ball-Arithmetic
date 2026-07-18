//! Elementary functions on balls: exp, log, sin/cos, atan.
//!
//! Strategy shared by all functions:
//! 1. **Argument reduction** with exact 2-adic scaling (`x/2^k`) or ball
//!    square roots, with all reduction error carried by ball arithmetic.
//! 2. **Taylor series** on the reduced argument, evaluated in ball
//!    arithmetic at working precision, truncated after `N` terms.
//! 3. **Rigorous tail bound**: the first omitted term is bounded with
//!    64-bit directed floating point (round-up), then a geometric-series
//!    factor `1/(1 − ratio)` covers the rest; for alternating series with
//!    decreasing terms the first omitted term alone suffices. The bound is
//!    *computed*, never assumed — a bad choice of `N` widens the ball but
//!    cannot break inclusion.
//! 4. **Reconstruction** (repeated squaring, double-angle, doubling), again
//!    in ball arithmetic.
//! 5. **Adaptive retry**: if the result certifies fewer than `prec` bits and
//!    the input ball is not the limiting factor, retry with more working
//!    precision.

use crate::ball::Ball;
use crate::constants;
use crate::fp::{Float, Round};
use crate::mag::Mag;

/// Largest |x| accepted by exp/sin/cos (exponent-range guard).
const MAX_ARG_EXP: i64 = 40;

/// Retry schedule: extra working bits on each attempt.
fn attempt_extra(prec: u32, attempt: u32) -> u32 {
    64 + (prec / 2) * attempt * attempt
}
const MAX_ATTEMPTS: u32 = 4;

/// Integer square root of a u32 (for reduction-parameter tuning only).
fn isqrt32(v: u32) -> u32 {
    let r = (v as f64).sqrt() as u32;
    r.max(1)
}

// ---------------------------------------------------------------------
// exp
// ---------------------------------------------------------------------

/// `exp(x)` with rigorous inclusion.
pub fn exp(x: &Ball, prec: u32) -> Ball {
    if x.is_exact() && x.mid().is_zero() {
        return Ball::from_i64(1);
    }
    let up = x.abs_upper();
    assert!(
        up.exponent().unwrap_or(i64::MIN) <= MAX_ARG_EXP,
        "exp: |argument| exceeds 2^{MAX_ARG_EXP}"
    );
    for attempt in 0..=MAX_ATTEMPTS {
        let r = exp_attempt(x, prec, attempt_extra(prec, attempt));
        if r.rel_accuracy_bits() >= prec as i64 || attempt == MAX_ATTEMPTS {
            return r.round(prec);
        }
    }
    unreachable!()
}

fn exp_attempt(x: &Ball, prec: u32, extra: u32) -> Ball {
    let up = x.abs_upper();
    let e = up.exponent().unwrap_or(-(prec as i64)); // |x| < 2^e
    let j = isqrt32(prec) as i64 + 2; // target: |r| ≤ 2^−j
    let k = (e + j).max(0) as u32; // halvings
    let wp = prec + extra + k + 16;

    // r = x / 2^k (exact); |r| ≤ 2^(e−k) ≤ 2^−j.
    let r = x.mul_2exp(-(k as i64));
    let r_up = r.abs_upper();

    // Series length: |r|^n/n! < |r|^n ≤ 2^(−jr·n); jr bits per term.
    let jr = (-(r_up.exponent().unwrap_or(-(wp as i64)))).max(1);
    let n = ((wp as i64) / jr + 2).max(2) as u64;

    // Horner: P = 1 + r(1 + r/2(1 + r/3(…(1 + r/(n−1))…)))
    //           = Σ_{m<n} r^m/m!.
    let one = Ball::from_i64(1);
    let mut s = one.clone();
    for m in (1..n).rev() {
        s = one.add(&r.mul(&s, wp).div_u64(m, wp), wp);
    }
    // Tail: Σ_{m≥n} |r|^m/m! ≤ (|r|^n/n!) · 1/(1 − |r|)   (ratio ≤ |r| < 1).
    let tail = exp_tail_bound(&r_up, n);
    let mut s = s.add_error(&tail);

    // exp(x) = exp(r)^(2^k).
    for _ in 0..k {
        s = s.mul(&s, wp);
    }
    s
}

/// Upper bound for `|r|^n/n! · 1/(1−|r|)`, all in 64-bit directed floats.
fn exp_tail_bound(r_up: &Float, n: u64) -> Mag {
    if r_up.is_zero() {
        return Mag::zero();
    }
    debug_assert!(r_up.cmp(&Float::from_f64(0.5)) != core::cmp::Ordering::Greater);
    let num = pow_up(r_up, n);
    let den = factorial_down(n);
    let (t, _) = num.div(&den, 64, Round::Up);
    let (om, _) = Float::from_i64(1).sub(r_up, 64, Round::Down);
    let (t, _) = t.div(&om, 64, Round::Up);
    Mag::from_float_upper(&t)
}

/// `x^n` rounded up (x ≥ 0).
fn pow_up(x: &Float, mut n: u64) -> Float {
    let mut base = x.clone();
    let mut acc = Float::from_i64(1);
    while n > 0 {
        if n & 1 == 1 {
            acc = acc.mul(&base, 64, Round::Up).0;
        }
        base = base.mul(&base, 64, Round::Up).0;
        n >>= 1;
    }
    acc
}

/// `n!` rounded down.
fn factorial_down(n: u64) -> Float {
    let mut acc = Float::from_i64(1);
    for m in 2..=n {
        acc = acc.mul_u64(m, 64, Round::Down).0;
    }
    acc
}

// ---------------------------------------------------------------------
// log
// ---------------------------------------------------------------------

/// `ln(x)`; panics unless the ball is strictly positive
/// (use [`try_ln`] for an Option).
pub fn ln(x: &Ball, prec: u32) -> Ball {
    try_ln(x, prec).expect("ln: ball is not strictly positive")
}

/// `ln(x)`, or None if the ball touches `(−∞, 0]`.
pub fn try_ln(x: &Ball, prec: u32) -> Option<Ball> {
    if x.lower_bound().signum() <= 0 {
        return None;
    }
    if x.is_exact() && x.mid().cmp(&Float::from_i64(1)) == core::cmp::Ordering::Equal {
        return Some(Ball::zero());
    }
    for attempt in 0..=MAX_ATTEMPTS {
        let r = ln_attempt(x, prec, attempt_extra(prec, attempt));
        #[cfg(feature = "trace-retries")]
        eprintln!("ln attempt {attempt}: acc {}", r.rel_accuracy_bits());
        if r.rel_accuracy_bits() >= prec as i64 || attempt == MAX_ATTEMPTS {
            return Some(r.round(prec));
        }
    }
    unreachable!()
}

fn ln_attempt(x: &Ball, prec: u32, extra: u32) -> Ball {
    let wp = prec + extra + 2 * isqrt32(prec) + 16;
    // x = m · 2^E with m ∈ [1/2, 1)  ⇒  x = (2m) · 2^(E−1), 2m ∈ [1, 2).
    let e2 = x.mid().exponent().unwrap() - 1;
    let m2 = x.mul_2exp(-e2);
    let lnm = ln_core(&m2, wp);
    if e2 == 0 {
        return lnm;
    }
    let ln2 = constants::ln2(wp + 8);
    lnm.add(&ln2.mul_i64(e2, wp), wp)
}

/// `ln(y)` for a ball near `[1, 2]` (no exponent decomposition).
/// Also used to compute the ln 2 constant itself.
pub(crate) fn ln_core(y: &Ball, wp: u32) -> Ball {
    // Reduce by k square roots: ln y = 2^k · ln(y^(1/2^k)).
    // A ball sqrt costs ≈ 5 multiplications, a series term 1: balancing
    // wp/(2k) series muls against 5k sqrt muls gives k ≈ √(wp/10).
    let k = isqrt32(wp / 10);
    let mut v = y.clone();
    for _ in 0..k {
        v = v.sqrt(wp);
    }
    // t = v − 1 is tiny (|t| ≈ |ln y|/2^k);
    // ln(1+t) = 2 atanh(z), z = t/(2+t).
    let one = Ball::from_i64(1);
    let t = v.sub(&one, wp);
    let z = t.div(&t.add(&Ball::from_i64(2), wp), wp);
    let s = atanh_series(&z, wp);
    // ln y = 2^(k+1) · atanh(z).
    s.mul_2exp(k as i64 + 1)
}

/// `atanh(z) = Σ z^(2n+1)/(2n+1)` for |z| < 1/2, with geometric tail bound.
fn atanh_series(z: &Ball, wp: u32) -> Ball {
    let z_up = z.abs_upper();
    if z_up.is_zero() {
        return z.clone();
    }
    let jz = (-(z_up.exponent().unwrap())).max(1);
    let n = ((wp as i64) / (2 * jz) + 2).max(2) as u64;

    let u = z.mul(z, wp);
    let mut term = z.clone();
    let mut sum = z.clone();
    for m in 1..n {
        term = term.mul(&u, wp);
        sum = sum.add(&term.div_u64(2 * m + 1, wp), wp);
    }
    // Tail: Σ_{m≥n} |z|^(2m+1)/(2m+1) ≤ |z|^(2n+1)/(2n+1) · 1/(1−|z|²).
    let num = pow_up(&z_up, 2 * n + 1);
    let (num, _) = num.div(&Float::from_u64(2 * n + 1), 64, Round::Up);
    let (z2, _) = z_up.mul(&z_up, 64, Round::Up);
    let (om, _) = Float::from_i64(1).sub(&z2, 64, Round::Down);
    let (t, _) = num.div(&om, 64, Round::Up);
    sum.add_error(&Mag::from_float_upper(&t))
}

// ---------------------------------------------------------------------
// atan
// ---------------------------------------------------------------------

/// `atan(x)` with rigorous inclusion (defined for all x).
pub fn atan(x: &Ball, prec: u32) -> Ball {
    // A hopeless input ball: atan is bounded by π/2 < 2 anyway.
    if x.rad().exp() > MAX_ARG_EXP {
        return Ball::new(Float::zero(), Mag::two_exp(1));
    }
    for attempt in 0..=MAX_ATTEMPTS {
        let r = atan_attempt(x, prec, attempt_extra(prec, attempt));
        if r.rel_accuracy_bits() >= prec as i64 || attempt == MAX_ATTEMPTS {
            return r.round(prec);
        }
    }
    unreachable!()
}

fn atan_attempt(x: &Ball, prec: u32, extra: u32) -> Ball {
    let wp = prec + extra + 2 * isqrt32(prec) + 16;
    // |x| > 1: atan(x) = sign(x)·π/2 − atan(1/x); requires x bounded away
    // from 0, which |x| > 1 guarantees.
    let low = x.lower_bound();
    let up = x.abs_upper();
    if up.cmp(&Float::from_i64(1)) == core::cmp::Ordering::Greater {
        if low.signum() > 0 && low.cmp(&Float::from_i64(1)) != core::cmp::Ordering::Less {
            let inv = Ball::from_i64(1).div(x, wp);
            let half_pi = constants::pi(wp + 8).mul_2exp(-1);
            return half_pi.sub(&atan_core(&inv, wp), wp);
        }
        let neg_x = x.neg();
        let nlow = neg_x.lower_bound();
        if nlow.signum() > 0 && nlow.cmp(&Float::from_i64(1)) != core::cmp::Ordering::Less {
            return atan_attempt(&neg_x, prec, extra).neg();
        }
        // Ball straddles ±1: fall through (series still converges for
        // |x| slightly above 1 after reduction; reduction handles it).
    }
    atan_core(x, wp)
}

/// atan via k half-angle reductions then the alternating Taylor series.
fn atan_core(x: &Ball, wp: u32) -> Ball {
    // Reduction: atan(x) = 2·atan( x / (1 + √(1+x²)) ), applied k times.
    // Each reduction ≈ 11 muls (sqrt 5 + div 5 + mul 1); a series term ≈ 2.
    // Balancing wp/(2k)·2 against 11k gives k ≈ √(wp/11).
    let k = isqrt32(wp / 11) + 1;
    let one = Ball::from_i64(1);
    let mut z = x.clone();
    for _ in 0..k {
        let s = one.add(&z.mul(&z, wp), wp).sqrt(wp);
        z = z.div(&one.add(&s, wp), wp);
    }
    let s = atan_series(&z, wp);
    s.mul_2exp(k as i64)
}

/// `atan(z) = Σ (−1)^n z^(2n+1)/(2n+1)` for |z| < 1; alternating-series
/// tail bound (first omitted term).
fn atan_series(z: &Ball, wp: u32) -> Ball {
    let z_up = z.abs_upper();
    if z_up.is_zero() {
        return z.clone();
    }
    let jz = (-(z_up.exponent().unwrap())).max(1);
    let n = ((wp as i64) / (2 * jz) + 2).max(2) as u64;

    let u = z.mul(z, wp);
    let mut term = z.clone();
    let mut sum = z.clone();
    for m in 1..n {
        term = term.mul(&u, wp);
        let contrib = term.div_u64(2 * m + 1, wp);
        sum = if m % 2 == 1 {
            sum.sub(&contrib, wp)
        } else {
            sum.add(&contrib, wp)
        };
    }
    // |z| < 1 ⇒ terms decrease ⇒ alternating tail ≤ first omitted term.
    let num = pow_up(&z_up, 2 * n + 1);
    let (t, _) = num.div(&Float::from_u64(2 * n + 1), 64, Round::Up);
    sum.add_error(&Mag::from_float_upper(&t))
}

// ---------------------------------------------------------------------
// sin / cos
// ---------------------------------------------------------------------

/// `(sin x, cos x)` with rigorous inclusion.
pub fn sin_cos(x: &Ball, prec: u32) -> (Ball, Ball) {
    let wide = || {
        (
            Ball::new(Float::zero(), Mag::two_exp(0)),
            Ball::new(Float::zero(), Mag::two_exp(0)),
        )
    };
    let up = x.abs_upper();
    if up.exponent().unwrap_or(i64::MIN) > MAX_ARG_EXP || x.rad().exp() > 2 {
        // Radius spans more than a full period, or argument out of range:
        // sin/cos are still certainly in [−1, 1].
        assert!(
            up.exponent().unwrap_or(i64::MIN) <= MAX_ARG_EXP,
            "sin_cos: |argument| exceeds 2^{MAX_ARG_EXP}"
        );
        return wide();
    }
    for attempt in 0..=MAX_ATTEMPTS {
        let (s, c) = sin_cos_attempt(x, prec, attempt_extra(prec, attempt));
        let done = s.rel_accuracy_bits() >= prec as i64 && c.rel_accuracy_bits() >= prec as i64;
        #[cfg(feature = "trace-retries")]
        eprintln!(
            "sin_cos attempt {attempt}: acc {} / {}",
            s.rel_accuracy_bits(),
            c.rel_accuracy_bits()
        );
        if done || attempt == MAX_ATTEMPTS {
            return (clamp_unit(s.round(prec)), clamp_unit(c.round(prec)));
        }
    }
    unreachable!()
}

pub fn sin(x: &Ball, prec: u32) -> Ball {
    sin_cos(x, prec).0
}

pub fn cos(x: &Ball, prec: u32) -> Ball {
    sin_cos(x, prec).1
}

/// If a sin/cos ball is wider than [−1,1] is tall, replace it by [0 ± 1].
fn clamp_unit(b: Ball) -> Ball {
    if b.rad().exp() > 0 {
        Ball::new(Float::zero(), Mag::two_exp(0))
    } else {
        b
    }
}

fn sin_cos_attempt(x: &Ball, prec: u32, extra: u32) -> (Ball, Ball) {
    let e = x.abs_upper().exponent().unwrap_or(0).max(0);
    let wp = prec + extra + 2 * isqrt32(prec) + e as u32 + 16;

    // Reduce mod π/2: t = x − q·(π/2), quadrant = q mod 4.
    // q need not be the exactly-nearest integer — any q with |t| ≲ π/2
    // preserves correctness since t is computed in ball arithmetic.
    let half_pi = constants::pi(wp + e as u32 + 8).mul_2exp(-1);
    let q_approx = x.mid().to_f64() / half_pi.mid().to_f64();
    let q = q_approx.round() as i64;
    let t = x.sub(&half_pi.mul(&Ball::exact(Float::from_i64(q)), wp), wp);
    let quadrant = q.rem_euclid(4);

    // Halve k times, then Taylor, then double-angle k times.
    let k = isqrt32(wp) / 2 + 1;
    let t = t.mul_2exp(-(k as i64));
    let (mut s, mut c) = sin_cos_series(&t, wp);
    for _ in 0..k {
        // sin 2a = 2 sin a cos a;  cos 2a = 1 − 2 sin² a.
        let s2 = s.mul(&c, wp).mul_2exp(1);
        let c2 = Ball::from_i64(1).sub(&s.mul(&s, wp).mul_2exp(1), wp);
        s = s2;
        c = c2;
    }
    match quadrant {
        0 => (s, c),
        1 => (c, s.neg()),
        2 => (s.neg(), c.neg()),
        _ => (c.neg(), s),
    }
}

/// Taylor series for sin and cos of a small ball (|t| ≤ 1/2 after reduction),
/// with geometric tail bounds.
fn sin_cos_series(t: &Ball, wp: u32) -> (Ball, Ball) {
    let t_up = t.abs_upper();
    if t_up.is_zero() {
        return (t.clone(), Ball::from_i64(1));
    }
    let jt = (-(t_up.exponent().unwrap())).max(1);
    let m = ((wp as i64) / (2 * jt) + 2).max(2) as u64;
    let u = t.mul(t, wp);
    let one = Ball::from_i64(1);

    // sin t = t · P_s,  P_s = 1 − u/(2·3)(1 − u/(4·5)(1 − …)).
    let mut ps = one.clone();
    for i in (1..m).rev() {
        ps = one.sub(&u.mul(&ps, wp).div_u64(2 * i * (2 * i + 1), wp), wp);
    }
    let sin = t.mul(&ps, wp);
    // cos t = P_c,  P_c = 1 − u/(1·2)(1 − u/(3·4)(1 − …)).
    let mut pc = one.clone();
    for i in (1..m).rev() {
        pc = one.sub(&u.mul(&pc, wp).div_u64((2 * i - 1) * 2 * i, wp), wp);
    }

    // Tails (ratio of consecutive terms ≤ |t|² ≤ 1/4):
    //   |sin tail| ≤ |t|^(2m+1)/(2m+1)! · 1/(1−|t|²), likewise for cos.
    let (t2, _) = t_up.mul(&t_up, 64, Round::Up);
    let (om, _) = Float::from_i64(1).sub(&t2, 64, Round::Down);
    let sin_tail = {
        let num = pow_up(&t_up, 2 * m + 1);
        let (v, _) = num.div(&factorial_down(2 * m + 1), 64, Round::Up);
        let (v, _) = v.div(&om, 64, Round::Up);
        Mag::from_float_upper(&v)
    };
    let cos_tail = {
        let num = pow_up(&t_up, 2 * m);
        let (v, _) = num.div(&factorial_down(2 * m), 64, Round::Up);
        let (v, _) = v.div(&om, 64, Round::Up);
        Mag::from_float_upper(&v)
    };
    (sin.add_error(&sin_tail), pc.add_error(&cos_tail))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Rng;

    /// f64 reference must land inside the ball, allowing f64's own error.
    fn assert_close_f64(b: &Ball, want: f64, what: &str) {
        let widened = b.add_error(&Mag::from_float_upper(&Float::from_f64(
            want.abs() * 1e-13 + 1e-300,
        )));
        assert!(
            widened.contains(&Float::from_f64(want)),
            "{what}: ball {} vs f64 {want}",
            b.to_string_digits(25)
        );
    }

    #[test]
    fn exp_matches_f64() {
        let mut rng = Rng::new(40);
        for _ in 0..60 {
            let v = (rng.range_i64(-6000, 6000) as f64) / 1000.0;
            let b = exp(&Ball::from_f64(v), 64);
            assert_close_f64(&b, v.exp(), &format!("exp({v})"));
        }
    }

    #[test]
    fn exp_known_values() {
        // e to 50 digits.
        let e = exp(&Ball::from_i64(1), 200);
        let s = e.mid().to_decimal(45);
        assert!(
            s.starts_with("2.7182818284590452353602874713526624977572"),
            "e = {s}"
        );
        assert!(
            e.rel_accuracy_bits() >= 190,
            "acc {}",
            e.rel_accuracy_bits()
        );
        // exp(0) = 1 exactly.
        let one = exp(&Ball::zero(), 64);
        assert!(one.is_exact());
    }

    #[test]
    fn exp_functional_equation() {
        // exp(a)·exp(b) must overlap exp(a+b).
        let mut rng = Rng::new(41);
        for _ in 0..20 {
            let a = Ball::from_f64((rng.range_i64(-3000, 3000) as f64) / 1000.0);
            let b = Ball::from_f64((rng.range_i64(-3000, 3000) as f64) / 1000.0);
            let prec = 128;
            let lhs = exp(&a, prec).mul(&exp(&b, prec), prec);
            let rhs = exp(&a.add(&b, prec), prec);
            // Both contain the true value: they must overlap.
            let diff = lhs.sub(&rhs, prec);
            assert!(diff.contains(&Float::zero()), "exp homomorphism violated");
        }
    }

    #[test]
    fn ln_matches_f64() {
        let mut rng = Rng::new(42);
        for _ in 0..60 {
            let v = (rng.below(1_000_000) as f64 + 1.0) / 1000.0;
            let b = ln(&Ball::from_f64(v), 64);
            assert_close_f64(&b, v.ln(), &format!("ln({v})"));
        }
    }

    #[test]
    fn ln_exp_roundtrip() {
        let mut rng = Rng::new(43);
        for _ in 0..15 {
            let v = (rng.range_i64(-4000, 4000) as f64) / 1000.0;
            let prec = 192;
            let x = Ball::from_f64(v);
            let back = ln(&exp(&x, prec), prec);
            let diff = back.sub(&x, prec);
            assert!(diff.contains(&Float::zero()), "ln(exp({v})) != {v}");
            assert!(
                diff.rad().exp() < -150,
                "roundtrip too imprecise: rad 2^{}",
                diff.rad().exp()
            );
        }
    }

    #[test]
    fn ln_rejects_nonpositive() {
        assert!(try_ln(&Ball::zero(), 64).is_none());
        assert!(try_ln(&Ball::from_i64(-3), 64).is_none());
        // Ball straddling zero.
        assert!(try_ln(&Ball::new(Float::from_f64(0.001), Mag::two_exp(0)), 64).is_none());
    }

    #[test]
    fn atan_matches_f64() {
        let mut rng = Rng::new(44);
        for _ in 0..60 {
            let v = (rng.range_i64(-500_000, 500_000) as f64) / 1000.0;
            let b = atan(&Ball::from_f64(v), 64);
            assert_close_f64(&b, v.atan(), &format!("atan({v})"));
        }
    }

    #[test]
    fn atan_one_is_quarter_pi() {
        let prec = 256;
        let q = atan(&Ball::from_i64(1), prec);
        let pi4 = constants::pi(prec).mul_2exp(-2);
        let diff = q.sub(&pi4, prec);
        assert!(diff.contains(&Float::zero()));
        assert!(q.rel_accuracy_bits() >= 240);
    }

    #[test]
    fn sin_cos_match_f64() {
        let mut rng = Rng::new(45);
        for _ in 0..60 {
            let v = (rng.range_i64(-100_000, 100_000) as f64) / 1000.0;
            let (s, c) = sin_cos(&Ball::from_f64(v), 64);
            assert_close_f64(&s, v.sin(), &format!("sin({v})"));
            assert_close_f64(&c, v.cos(), &format!("cos({v})"));
        }
    }

    #[test]
    fn sin_cos_pythagorean() {
        let mut rng = Rng::new(46);
        for _ in 0..15 {
            let v = (rng.range_i64(-50_000, 50_000) as f64) / 1000.0;
            let prec = 160;
            let (s, c) = sin_cos(&Ball::from_f64(v), prec);
            let one = s.mul(&s, prec).add(&c.mul(&c, prec), prec);
            let diff = one.sub(&Ball::from_i64(1), prec);
            assert!(diff.contains(&Float::zero()), "sin²+cos² ∌ 1 at {v}");
            assert!(diff.rad().exp() < -120, "pythagorean too wide at {v}");
        }
    }

    #[test]
    fn high_precision_self_consistency() {
        // 1000-digit sanity: exp(ln(7)) contains 7 with tight radius.
        let prec = 3500;
        let x = Ball::from_i64(7);
        let y = exp(&ln(&x, prec), prec);
        let diff = y.sub(&x, prec);
        assert!(diff.contains(&Float::zero()));
        assert!(
            diff.rad().exp() < -(prec as i64) + 60,
            "rad 2^{}",
            diff.rad().exp()
        );
    }
}
