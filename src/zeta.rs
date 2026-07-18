//! The Riemann zeta function for real s > 0 (s ≠ 1) via Euler–Maclaurin
//! summation with a rigorous remainder bound.
//!
//! ζ(s) = Σ_{j=1}^{N−1} j^(−s) + N^(−s)/2 + N^(1−s)/(s−1)
//!      + Σ_{k=1}^{M} B_{2k}/(2k)! · (s)_{2k−1} · N^(−s−2k+1) + R_M,
//! where (s)_{2k−1} = s(s+1)…(s+2k−2) is the rising factorial, and for
//! real s the remainder satisfies
//!   |R_M| ≤ |first omitted term| · (s + 2M + 1)/(σ + 2M + 1) = |t_{M+1}|
//! (Edwards, *Riemann's Zeta Function*, §6.4: the quotient is 1 on the
//! real axis). Requires s > −(2M+1), amply satisfied for s > 0.
//!
//! Integer s uses exact powers (fast); non-integer s computes j^(−s) as
//! exp(−s ln j), which is honest but slow — see README.

use crate::ball::Ball;
use crate::bernoulli;
use crate::elementary;
use crate::fp::{Float, Round};
use crate::mag::Mag;

const MAX_ATTEMPTS: u32 = 3;

/// ζ(s) for a real ball with s > 0 and 1 ∉ s. Panics otherwise;
/// use [`try_zeta`] for the fallible form.
pub fn zeta(s: &Ball, prec: u32) -> Ball {
    try_zeta(s, prec).expect("zeta: ball must satisfy s > 0 and exclude 1")
}

/// ζ(s), or None if the ball leaves the supported region (s > 0, 1 ∉ s).
pub fn try_zeta(s: &Ball, prec: u32) -> Option<Ball> {
    if s.lower_bound().signum() <= 0 || s.contains(&Float::from_i64(1)) {
        return None;
    }
    for attempt in 0..=MAX_ATTEMPTS {
        let extra = 64 + prec / 2 * attempt;
        let r = zeta_em(s, prec + extra);
        if r.rel_accuracy_bits() >= prec as i64 || attempt == MAX_ATTEMPTS {
            return Some(r.round(prec));
        }
    }
    unreachable!()
}

/// Exact small-integer exponent, if the ball is one.
fn as_small_int(s: &Ball) -> Option<u64> {
    if !s.is_exact() {
        return None;
    }
    let v = s.mid().to_i64_trunc();
    if v >= 2 && s.mid().cmp(&Float::from_i64(v)) == core::cmp::Ordering::Equal {
        Some(v as u64)
    } else {
        None
    }
}

/// j^(−s) at working precision.
fn j_pow_neg_s(j: u64, s: &Ball, int_s: Option<u64>, wp: u32) -> Ball {
    match int_s {
        Some(e) => {
            // Exact integer power by squaring, then one division.
            let mut acc = Ball::from_u64(1);
            let mut base = Ball::from_u64(j);
            let mut n = e;
            while n > 0 {
                if n & 1 == 1 {
                    acc = acc.mul(&base, wp);
                }
                base = base.mul(&base, wp);
                n >>= 1;
            }
            Ball::from_i64(1).div(&acc, wp)
        }
        None => {
            let lnj = elementary::ln(&Ball::from_u64(j), wp);
            elementary::exp(&s.neg().mul(&lnj, wp), wp)
        }
    }
}

fn zeta_em(s: &Ball, wp: u32) -> Ball {
    let int_s = as_small_int(s);

    // Parameter choice: with N = M the k-th correction term is roughly
    //   2(2k)!/((2π)^(2k) N^(2k−1)) ⇒ log2 t_M ≈ 2M·[log2(2M/e) − log2(2πN)]
    // ≈ −6.2·M bits, so M ≈ wp/6.2 terms suffice. The remainder is computed
    // rigorously below; this choice only affects performance.
    let nm = ((wp as f64) * 0.17 + 10.0).ceil() as u64;
    let n = nm;
    let m = nm as usize;

    // Direct part: Σ_{j=1}^{N−1} j^(−s).
    let mut direct = Ball::from_i64(1);
    for j in 2..n {
        direct = direct.add(&j_pow_neg_s(j, s, int_s, wp), wp);
    }

    // N^(−s), N^(1−s).
    let n_neg_s = j_pow_neg_s(n, s, int_s, wp);
    let n_ball = Ball::from_u64(n);
    let n_1ms = n_neg_s.mul(&n_ball, wp);

    // Tail head: N^(−s)/2 + N^(1−s)/(s−1).
    let s_m1 = s.sub(&Ball::from_i64(1), wp);
    let head = n_neg_s.mul_2exp(-1).add(&n_1ms.div(&s_m1, wp), wp);

    // Correction sum: Σ_k B_2k/(2k)! · (s)_{2k−1} · N^(−s−2k+1).
    // Incremental pieces:
    //   fact_inv: 1/(2k)!   rise: (s)(s+1)…(s+2k−2)   npow: N^(−s−2k+1).
    let inv_n2 = Ball::from_i64(1).div_u64(n * n, wp);
    let mut rise = s.clone(); // (s)_1 for k=1
    let mut npow = n_neg_s.div_u64(n, wp); // N^(−s−1) for k=1
    let mut sum = Ball::zero();
    for k in 1..=m {
        let b = bernoulli::bernoulli(k, wp);
        let t = b
            .mul(&rise, wp)
            .mul(&npow, wp)
            .div(&Ball::exact(factorial_float(2 * k as u64)), wp);
        sum = sum.add(&t, wp);
        if k < m {
            // rise *= (s+2k−1)(s+2k), npow *= 1/N².
            let a1 = s.add(&Ball::from_u64(2 * k as u64 - 1), wp);
            let a2 = s.add(&Ball::from_u64(2 * k as u64), wp);
            rise = rise.mul(&a1, wp).mul(&a2, wp);
            npow = npow.mul(&inv_n2, wp);
        }
    }

    // Rigorous remainder: |R_M| ≤ |t_{M+1}|. Compute the first omitted term
    // with 64-bit directed bounds (upper).
    let rem = {
        let k = m + 1;
        let b = bernoulli::bernoulli(k, 64).abs_upper();
        // (s)_{2k−1} = Π_{i=0}^{2k−2} (s+i), upper-bounded term by term.
        let s_up = s.abs_upper();
        let mut rise_up = Float::from_i64(1);
        for i in 0..(2 * k as u64 - 1) {
            let (f, _) = s_up.add(&Float::from_u64(i), 64, Round::Up);
            rise_up = rise_up.mul(&f, 64, Round::Up).0;
        }
        // N^(−s−2k+1) ≤ N^(−s_low−2k+1); s_low ≥ 0 ⇒ ≤ N^(−2k+1)… use s_low.
        let s_low = s.lower_bound();
        let e = (2 * k as u64 - 1) as i64 + s_low.to_i64_trunc().max(0);
        let n_f = Float::from_u64(n);
        let npow_dn = pow_dir(&n_f, e as u64, Round::Down);
        let (t, _) = b.mul(&rise_up, 64, Round::Up);
        let (t, _) = t.div(&factorial_dir(2 * k as u64, Round::Down), 64, Round::Up);
        let (t, _) = t.div(&npow_dn, 64, Round::Up);
        Mag::from_float_upper(&t)
    };

    direct.add(&head, wp).add(&sum, wp).add_error(&rem)
}

/// (2k)! as an exact Float (integer product, exact conversion).
fn factorial_float(n: u64) -> Float {
    let mut acc = crate::int::Int::from_u64(1);
    for m in 2..=n {
        acc = acc.mul_u64(m);
    }
    acc.to_float()
}

fn factorial_dir(n: u64, rnd: Round) -> Float {
    let mut acc = Float::from_i64(1);
    for m in 2..=n {
        acc = acc.mul_u64(m, 64, rnd).0;
    }
    acc
}

fn pow_dir(x: &Float, mut n: u64, rnd: Round) -> Float {
    let mut base = x.clone();
    let mut acc = Float::from_i64(1);
    while n > 0 {
        if n & 1 == 1 {
            acc = acc.mul(&base, 64, rnd).0;
        }
        base = base.mul(&base, 64, rnd).0;
        n >>= 1;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;

    #[test]
    fn zeta_two_is_pi2_over_6() {
        let prec = 512;
        let z = zeta(&Ball::from_i64(2), prec);
        let pi = constants::pi(prec + 32);
        let want = pi.mul(&pi, prec + 32).div_u64(6, prec + 32);
        let d = z.sub(&want, prec);
        assert!(
            d.contains(&Float::zero()),
            "ζ(2) ≠ π²/6: {}",
            z.to_string_digits(30)
        );
        assert!(z.rel_accuracy_bits() >= prec as i64 - 8);
    }

    #[test]
    fn zeta_four_is_pi4_over_90() {
        let prec = 320;
        let z = zeta(&Ball::from_i64(4), prec);
        let pi = constants::pi(prec + 32);
        let p2 = pi.mul(&pi, prec + 32);
        let want = p2.mul(&p2, prec + 32).div_u64(90, prec + 32);
        let d = z.sub(&want, prec);
        assert!(d.contains(&Float::zero()), "ζ(4) ≠ π⁴/90");
    }

    #[test]
    fn zeta_three_apery() {
        // Apéry's constant to 30 digits.
        let z = zeta(&Ball::from_i64(3), 200);
        let s = z.mid().to_decimal(30);
        assert!(
            s.starts_with("1.2020569031595942853997381615"),
            "ζ(3) = {s}"
        );
    }

    #[test]
    fn zeta_non_integer() {
        // ζ(3/2) ≈ 2.612375348685488343348567567924…
        let prec = 128;
        let s = Ball::from_f64(1.5);
        let z = zeta(&s, prec);
        let ds = z.mid().to_decimal(20);
        assert!(ds.starts_with("2.61237534868548834"), "ζ(3/2) = {ds}");
        assert!(z.rel_accuracy_bits() >= 120);
    }

    #[test]
    fn rejects_bad_domain() {
        assert!(try_zeta(&Ball::from_i64(1), 64).is_none());
        assert!(try_zeta(&Ball::from_i64(0), 64).is_none());
        assert!(try_zeta(&Ball::from_i64(-2), 64).is_none());
    }
}
