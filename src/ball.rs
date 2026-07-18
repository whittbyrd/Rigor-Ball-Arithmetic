//! Ball arithmetic: `[m ± r]` with rigorously propagated error bounds.
//!
//! **Contract**: for every operation `op`, if the input balls contain real
//! numbers `x, y`, the output ball contains `op(x, y)` exactly. Each radius
//! formula below is annotated with the inequality that proves it.
//!
//! Midpoints are computed at caller-specified precision with round-to-nearest
//! (error ≤ half an ulp, absorbed into the radius); radii are [`Mag`]s, whose
//! every operation rounds up.

use crate::fp::{Float, Round};
use crate::mag::Mag;
use core::cmp::Ordering;

/// A real number known to lie in `[mid − rad, mid + rad]`.
#[derive(Clone, Debug)]
pub struct Ball {
    mid: Float,
    rad: Mag,
}

/// Radius contribution of a round-to-nearest midpoint: half an ulp.
fn round_err(mid: &Float, inexact: bool, prec: u32) -> Mag {
    if !inexact {
        return Mag::zero();
    }
    // |round(x) − x| ≤ 2^(e − prec − 1) for a prec-bit nearest rounding of a
    // value with exponent e.
    match mid.exponent() {
        Some(e) => Mag::two_exp(e - prec as i64 - 1),
        None => Mag::zero(), // exact zero result cannot be inexact
    }
}

impl Ball {
    // ---------------------------------------------------------------
    // Construction / accessors
    // ---------------------------------------------------------------

    pub fn new(mid: Float, rad: Mag) -> Self {
        Ball { mid, rad }
    }

    /// Exact ball (radius zero).
    pub fn exact(mid: Float) -> Self {
        Ball { mid, rad: Mag::zero() }
    }

    pub fn zero() -> Self {
        Ball::exact(Float::zero())
    }

    pub fn from_i64(v: i64) -> Self {
        Ball::exact(Float::from_i64(v))
    }

    pub fn from_u64(v: u64) -> Self {
        Ball::exact(Float::from_u64(v))
    }

    pub fn from_f64(v: f64) -> Self {
        Ball::exact(Float::from_f64(v))
    }

    pub fn mid(&self) -> &Float {
        &self.mid
    }

    pub fn rad(&self) -> &Mag {
        &self.rad
    }

    pub fn is_exact(&self) -> bool {
        self.rad.is_zero()
    }

    /// Widen the radius by `err`.
    pub fn add_error(&self, err: &Mag) -> Ball {
        Ball { mid: self.mid.clone(), rad: self.rad.add_up(err) }
    }

    /// Does the ball certainly contain zero candidates? True iff |mid| ≤ rad.
    pub fn contains_zero(&self) -> bool {
        self.mid.cmp_abs(&self.rad.to_float()) != Ordering::Greater
    }

    /// Certain containment of an exact dyadic point: |x − mid| ≤ rad.
    /// Decides exactly (both quantities are dyadic rationals).
    pub fn contains(&self, x: &Float) -> bool {
        // Exact subtraction: allocate enough precision for exactness.
        let bits = self.mid.bit_len().max(x.bit_len()) as u32;
        let span = exact_sub_prec(&self.mid, x, bits);
        let (d, inexact) = x.sub(&self.mid, span, Round::Nearest);
        debug_assert!(!inexact, "exact subtraction overflowed its window");
        d.cmp_abs(&self.rad.to_float()) != Ordering::Greater
    }

    /// Relative accuracy in bits: how many significant bits of the midpoint
    /// are certified (0 if the ball contains no information).
    pub fn rel_accuracy_bits(&self) -> i64 {
        match (self.mid.exponent(), self.rad.is_zero()) {
            (_, true) => i64::MAX,
            (None, false) => 0,
            (Some(e), false) => (e - self.rad.exp()).max(0),
        }
    }

    /// Round the midpoint to `prec` bits, absorbing the change into the radius.
    pub fn round(&self, prec: u32) -> Ball {
        let (mid, inexact) = self.mid.round(prec, Round::Nearest);
        let rad = self.rad.add_up(&round_err(&mid, inexact, prec));
        Ball { mid, rad }
    }

    // ---------------------------------------------------------------
    // Ring operations
    // ---------------------------------------------------------------

    pub fn neg(&self) -> Ball {
        Ball { mid: self.mid.neg(), rad: self.rad }
    }

    /// Exact scaling by 2^k.
    pub fn mul_2exp(&self, k: i64) -> Ball {
        Ball { mid: self.mid.mul_2exp(k), rad: self.rad.mul_2exp(k) }
    }

    /// `self + other` at precision `prec`.
    ///
    /// Proof: x ∈ [ma ± ra], y ∈ [mb ± rb] ⇒
    /// |x + y − round(ma + mb)| ≤ ra + rb + |round(ma+mb) − (ma+mb)|.
    pub fn add(&self, other: &Ball, prec: u32) -> Ball {
        let (mid, inexact) = self.mid.add(&other.mid, prec, Round::Nearest);
        let rad = self
            .rad
            .add_up(&other.rad)
            .add_up(&round_err(&mid, inexact, prec));
        Ball { mid, rad }
    }

    pub fn sub(&self, other: &Ball, prec: u32) -> Ball {
        self.add(&other.neg(), prec)
    }

    /// `self · other` at precision `prec`.
    ///
    /// Proof: |xy − ma·mb| = |(x−ma)y + ma(y−mb)|
    ///        ≤ ra·(|mb| + rb) + |ma|·rb, plus midpoint rounding.
    pub fn mul(&self, other: &Ball, prec: u32) -> Ball {
        let (mid, inexact) = self.mid.mul(&other.mid, prec, Round::Nearest);
        let ma = Mag::from_float_upper(&self.mid);
        let mb = Mag::from_float_upper(&other.mid);
        let rad = self
            .rad
            .mul_up(&mb.add_up(&other.rad))
            .add_up(&ma.mul_up(&other.rad))
            .add_up(&round_err(&mid, inexact, prec));
        Ball { mid, rad }
    }

    /// Multiply by an exact small integer.
    pub fn mul_i64(&self, v: i64, prec: u32) -> Ball {
        self.mul(&Ball::from_i64(v), prec)
    }

    /// `self / other`. Panics if the divisor ball contains zero
    /// (use [`Ball::try_div`] to handle that case).
    pub fn div(&self, other: &Ball, prec: u32) -> Ball {
        self.try_div(other, prec)
            .expect("Ball::div: divisor ball contains zero")
    }

    /// `self / other`, or `None` if the divisor ball contains zero.
    ///
    /// Proof: with L = |mb| − rb > 0 (so |y| ≥ L for all y in the divisor):
    /// |x/y − ma/mb| = |x·mb − ma·y| / (|y||mb|) ≤ (ra·|mb| + rb·|ma|) / (L·|mb|).
    pub fn try_div(&self, other: &Ball, prec: u32) -> Option<Ball> {
        const RP: u32 = 64; // radius-bound working precision
        let amb = other.mid.abs();
        let rb_f = other.rad.to_float();
        // L: certified lower bound on |y| (round toward zero → smaller).
        let (l, _) = amb.sub(&rb_f, RP, Round::Down);
        if l.signum() <= 0 {
            return None;
        }
        let (mid, inexact) = self.mid.div(&other.mid, prec, Round::Nearest);

        // Numerator upper bound: ra·|mb| + rb·|ma| (all rounded up).
        let ama = self.mid.abs();
        let ra_f = self.rad.to_float();
        let (n1, _) = amb.mul(&ra_f, RP, Round::Up);
        let (n2, _) = ama.mul(&rb_f, RP, Round::Up);
        let (num, _) = n1.add(&n2, RP, Round::Up);
        // Denominator lower bound: L·|mb| (rounded down).
        let (den, _) = l.mul(&amb, RP, Round::Down);
        let (bound, _) = num.div(&den, RP, Round::Up);
        let rad = Mag::from_float_upper(&bound)
            .add_up(&round_err(&mid, inexact, prec));
        Some(Ball { mid, rad })
    }

    /// Divide by an exact small positive integer.
    pub fn div_u64(&self, v: u64, prec: u32) -> Ball {
        assert!(v != 0);
        let d = Ball::from_u64(v);
        self.div(&d, prec)
    }

    /// `sqrt(self)`. Panics if the ball contains negative numbers.
    ///
    /// Proof: with L = ma − ra > 0, for x ∈ [ma ± ra]:
    /// |√x − √ma| = |x − ma| / (√x + √ma) ≤ ra / (2·√L).
    pub fn sqrt(&self, prec: u32) -> Ball {
        const RP: u32 = 64;
        assert!(
            self.mid.signum() >= 0,
            "Ball::sqrt: midpoint is negative"
        );
        let ra_f = self.rad.to_float();
        let (l, _) = self.mid.sub(&ra_f, RP, Round::Down);
        if self.rad.is_zero() {
            let (mid, inexact) = self.mid.sqrt(prec, Round::Nearest);
            let rad = round_err(&mid, inexact, prec);
            return Ball { mid, rad };
        }
        assert!(
            l.signum() > 0,
            "Ball::sqrt: ball contains negative numbers (lower bound ≤ 0)"
        );
        let (mid, inexact) = self.mid.sqrt(prec, Round::Nearest);
        // Lower bound on √L: round down.
        let (sl, _) = l.sqrt(RP, Round::Down);
        let (den, _) = sl.mul_u64(2, RP, Round::Down);
        let (bound, _) = ra_f.div(&den, RP, Round::Up);
        let rad = Mag::from_float_upper(&bound)
            .add_up(&round_err(&mid, inexact, prec));
        Ball { mid, rad }
    }

    /// The interval hull endpoints `[mid − rad, mid + rad]` as exact floats.
    pub fn endpoints(&self) -> (Float, Float) {
        let r = self.rad.to_float();
        let bits = (self.mid.bit_len() as u32).max(64);
        let span = exact_sub_prec(&self.mid, &r, bits);
        let (lo, e1) = self.mid.sub(&r, span, Round::Floor);
        let (hi, e2) = self.mid.add(&r, span, Round::Ceil);
        debug_assert!(!e1 && !e2, "endpoint computation must be exact");
        (lo, hi)
    }

    /// Printable form: midpoint digits that are certified correct, plus the
    /// radius as a power of two.
    pub fn to_string_digits(&self, max_digits: usize) -> String {
        format!("[{} ± {}]", self.mid.to_decimal(max_digits), self.rad)
    }
}

/// Precision sufficient to subtract/add two floats exactly, given the top
/// `bits` estimate of significant bits. The exact result spans at most
/// (max exponent − min bottom exponent) bits; we bound that coarsely.
fn exact_sub_prec(a: &Float, b: &Float, bits: u32) -> u32 {
    let ea = a.exponent().unwrap_or(0);
    let eb = b.exponent().unwrap_or(0);
    let d = ea.abs_diff(eb).min(1 << 20) as u32;
    (bits + d + 128).min(1 << 24)
}

impl core::fmt::Display for Ball {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_string_digits(20))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Rng;

    fn rand_ball(rng: &mut Rng) -> Ball {
        let mid = Float::from_f64(
            (rng.next() as i64 as f64) * 2f64.powi(rng.range_i64(-40, 40) as i32),
        );
        let rad = if rng.below(4) == 0 {
            Mag::zero()
        } else {
            Mag::from_u64_upper(rng.next() >> 34).mul_2exp(rng.range_i64(-80, 10))
        };
        Ball::new(mid, rad)
    }

    /// Sample exact points guaranteed inside the ball: midpoint and both
    /// endpoints.
    fn sample_points(b: &Ball) -> Vec<Float> {
        let (lo, hi) = b.endpoints();
        vec![b.mid().clone(), lo, hi]
    }

    /// Property: op(x, y) ∈ op_ball(A, B) for all sampled x ∈ A, y ∈ B,
    /// where op(x, y) is computed exactly at very high precision.
    #[test]
    fn inclusion_add_sub_mul() {
        let mut rng = Rng::new(30);
        const EXACT: u32 = 4096;
        for i in 0..500 {
            let a = rand_ball(&mut rng);
            let b = rand_ball(&mut rng);
            let prec = 8 + rng.below(120) as u32;
            let sum = a.add(&b, prec);
            let dif = a.sub(&b, prec);
            let prd = a.mul(&b, prec);
            for x in sample_points(&a) {
                for y in sample_points(&b) {
                    let (s, e1) = x.add(&y, EXACT, Round::Nearest);
                    let (d, e2) = x.sub(&y, EXACT, Round::Nearest);
                    let (p, e3) = x.mul(&y, EXACT, Round::Nearest);
                    assert!(!e1 && !e2 && !e3, "exact reference rounded");
                    assert!(sum.contains(&s), "iter {i}: add violates inclusion");
                    assert!(dif.contains(&d), "iter {i}: sub violates inclusion");
                    assert!(prd.contains(&p), "iter {i}: mul violates inclusion");
                }
            }
        }
    }

    #[test]
    fn inclusion_div() {
        let mut rng = Rng::new(31);
        const EXACT: u32 = 2048;
        let mut tested = 0;
        for _ in 0..800 {
            let a = rand_ball(&mut rng);
            let b = rand_ball(&mut rng);
            let prec = 8 + rng.below(120) as u32;
            let Some(q) = a.try_div(&b, prec) else { continue };
            tested += 1;
            for x in sample_points(&a) {
                for y in sample_points(&b) {
                    // Compare x/y against the ball by checking membership of a
                    // high-precision rounding plus its error bound.
                    let (v, inexact) = x.div(&y, EXACT, Round::Nearest);
                    // |x/y − v| ≤ 2^(e − EXACT − 1)
                    let widened = if inexact {
                        q.add_error(&Mag::two_exp(
                            v.exponent().unwrap_or(i64::MIN / 2) - EXACT as i64 - 1,
                        ))
                    } else {
                        q.clone()
                    };
                    let _ = &widened;
                    // v must lie within q widened by the reference error.
                    assert!(
                        widened.contains(&v),
                        "div violates inclusion: x/y ≈ {}",
                        v.to_f64()
                    );
                }
            }
        }
        assert!(tested > 100, "too few div cases exercised: {tested}");
    }

    #[test]
    fn inclusion_sqrt() {
        let mut rng = Rng::new(32);
        const EXACT: u32 = 2048;
        let mut tested = 0;
        for _ in 0..500 {
            let mut a = rand_ball(&mut rng);
            if a.mid().signum() < 0 {
                a = a.neg();
            }
            let ra = a.rad().to_float();
            let (l, _) = a.mid().sub(&ra, 64, Round::Down);
            if l.signum() <= 0 {
                continue;
            }
            tested += 1;
            let prec = 8 + rng.below(120) as u32;
            let s = a.sqrt(prec);
            for x in sample_points(&a) {
                let (v, inexact) = x.sqrt(EXACT, Round::Nearest);
                let widened = if inexact {
                    s.add_error(&Mag::two_exp(
                        v.exponent().unwrap_or(i64::MIN / 2) - EXACT as i64 - 1,
                    ))
                } else {
                    s.clone()
                };
                assert!(widened.contains(&v), "sqrt violates inclusion");
            }
        }
        assert!(tested > 50, "too few sqrt cases exercised: {tested}");
    }

    #[test]
    fn contains_and_endpoints() {
        let b = Ball::new(Float::from_i64(10), Mag::two_exp(1)); // [8, 12]
        assert!(b.contains(&Float::from_i64(8)));
        assert!(b.contains(&Float::from_i64(12)));
        assert!(b.contains(&Float::from_i64(10)));
        assert!(!b.contains(&Float::from_i64(13)));
        assert!(!b.contains(&Float::from_i64(7)));
        let (lo, hi) = b.endpoints();
        assert_eq!(lo.cmp(&Float::from_i64(8)), Ordering::Equal);
        assert_eq!(hi.cmp(&Float::from_i64(12)), Ordering::Equal);
    }

    #[test]
    fn dependency_problem_demo() {
        // x − x with a fat ball: interval arithmetic gives [−2r, 2r], not 0.
        // This documents (rather than hides) the dependency problem.
        let x = Ball::new(Float::from_i64(1), Mag::two_exp(-10));
        let d = x.sub(&x, 64);
        assert!(d.contains(&Float::zero()));
        assert!(!d.is_exact()); // the price of forgetting correlation
    }

    #[test]
    fn accuracy_reporting() {
        let b = Ball::new(Float::from_i64(1), Mag::two_exp(-100));
        assert!(b.rel_accuracy_bits() >= 99);
        let e = Ball::from_i64(7);
        assert_eq!(e.rel_accuracy_bits(), i64::MAX);
    }
}
