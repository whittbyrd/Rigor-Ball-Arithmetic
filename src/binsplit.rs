//! Binary splitting for fast evaluation of rational hypergeometric-style
//! series: π (Chudnovsky), e (Σ 1/k!), ln 2 (2·atanh(1/3)).
//!
//! For a series Σ_k a_k · Π_{j≤k} (p_j/q_j) with integer p, q, a, binary
//! splitting computes the exact rational partial sum T(0,N)/Q(0,N) using
//! O(M(n) log N) work instead of N full-precision operations, by combining
//! half-ranges with the recursions
//!   P(a,b) = P(a,m)·P(m,b),  Q(a,b) = Q(a,m)·Q(m,b),
//!   T(a,b) = T(a,m)·Q(m,b) + P(a,m)·T(m,b).
//! Truncation tails are bounded rigorously per series (each bound derives
//! from an explicit term-ratio inequality, stated at the call site).

use crate::ball::Ball;
use crate::fp::{Float, Round};
use crate::int::Int;
use crate::mag::Mag;

/// A series definition for plain binary splitting (integer a_k folded
/// into T at the leaves).
trait Series {
    /// Leaf values (p_k, q_k, t_k) where t_k = a_k·p_k (sign included).
    fn leaf(&self, k: u64) -> (Int, Int, Int);
}

/// (P, Q, T) over `[a, b)`.
fn split<S: Series>(s: &S, a: u64, b: u64) -> (Int, Int, Int) {
    debug_assert!(a < b);
    if b - a == 1 {
        return s.leaf(a);
    }
    let m = a + (b - a) / 2;
    let (pl, ql, tl) = split(s, a, m);
    let (pr, qr, tr) = split(s, m, b);
    let p = pl.mul(&pr);
    let q = ql.mul(&qr);
    let t = tl.mul(&qr).add(&pl.mul(&tr));
    (p, q, t)
}

// ---------------------------------------------------------------------
// π via Chudnovsky
// ---------------------------------------------------------------------

// π = 426880·√10005 · Q(0,N) / T(0,N), with
//   p_k = (6k−5)(2k−1)(6k−1),  q_k = k³·640320³/24,
//   a_k = (−1)^k (13591409 + 545140134 k).
// Each term adds log2(q/p) ≈ 47.11 bits.
const C3_OVER_24: u64 = 10_939_058_860_032_000; // 640320³ / 24
const A0: u64 = 13_591_409;
const A1: u64 = 545_140_134;

struct Chudnovsky;

impl Series for Chudnovsky {
    fn leaf(&self, k: u64) -> (Int, Int, Int) {
        if k == 0 {
            return (Int::from_u64(1), Int::from_u64(1), Int::from_u64(A0));
        }
        let p = Int::from_u64(6 * k - 5)
            .mul_u64(2 * k - 1)
            .mul_u64(6 * k - 1);
        let q = Int::from_u64(k).mul_u64(k).mul_u64(k).mul_u64(C3_OVER_24);
        let mut t = p.mul(&Int::from_u64(A0 + A1 * k));
        if k % 2 == 1 {
            t = t.neg();
        }
        (p, q, t)
    }
}

/// π to `prec` bits via Chudnovsky binary splitting.
pub fn pi_chudnovsky(prec: u32) -> Ball {
    let wp = prec + 64;
    // 47.11 bits per term; the tail bound below is computed, not assumed.
    let n = (wp as u64) / 47 + 2;
    let (p, q, t) = split(&Chudnovsky, 0, n);
    let _ = &p;

    // S = T/Q as a ball; truncation tail: the term ratio is
    //   |t_{k+1}/t_k| ≤ p_{k+1}·a_{k+1} / (q_{k+1}·a_k) ≤ 2^−45 for k ≥ 1,
    // so |tail| ≤ |t_N|·2^−44 with t_N = a_N·P(0,N)/Q(0,N).
    let qf = q.to_float();
    let tf = t.to_float();
    let s = Ball::exact(tf.clone()).div(&Ball::exact(qf.clone()), wp);
    let tail = {
        let (pq, _) = p.to_float().div(&qf, 64, Round::Up);
        let (tn, _) = pq.mul(&Float::from_u64(A0 + A1 * n), 64, Round::Up);
        Mag::from_float_upper(&tn.mul_2exp(-44))
    };
    let s = s.add_error(&tail);

    // π = 426880·√10005 / S.
    let sqrt = Ball::from_u64(10005).sqrt(wp);
    sqrt.mul_i64(426_880, wp).div(&s, wp).round(prec)
}

// ---------------------------------------------------------------------
// e via Σ 1/k!
// ---------------------------------------------------------------------

struct ExpOne;

impl Series for ExpOne {
    fn leaf(&self, k: u64) -> (Int, Int, Int) {
        // p_k = 1, q_k = max(k, 1)  ⇒  Π q = k!, a_k = 1.
        (Int::from_u64(1), Int::from_u64(k.max(1)), Int::from_u64(1))
    }
}

/// e to `prec` bits via binary splitting of Σ 1/k!.
pub fn e_binsplit(prec: u32) -> Ball {
    let wp = prec + 64;
    // Choose N with N! > 2^(wp+4): accumulate log2 k.
    let mut n = 2u64;
    let mut lg = 0.0f64;
    while lg < (wp + 4) as f64 {
        lg += (n as f64).log2();
        n += 1;
    }
    let (_p, q, t) = split(&ExpOne, 0, n);
    // Tail: Σ_{k≥N} 1/k! ≤ 2/N! = 2/Q(0,N)   (ratio ≤ 1/2).
    let qf = q.to_float();
    let s = Ball::exact(t.to_float()).div(&Ball::exact(qf.clone()), wp);
    let (inv_q, _) = Float::from_u64(1).div(&qf, 64, Round::Up);
    s.add_error(&Mag::from_float_upper(&inv_q.mul_2exp(1))).round(prec)
}

// ---------------------------------------------------------------------
// ln 2 via 2·atanh(1/3)
// ---------------------------------------------------------------------

// ln 2 = 2·Σ_{k≥0} (1/3)^(2k+1) / (2k+1)
//      = (2/3)·Σ_k (1/9)^k / (2k+1).
// With p_k = 1 (k=0: 1), q_k = 9 (k=0: 1) and the harmonic denominator
// (2k+1) folded in via the B-extended recursion:
//   T(a,b) = T(a,m)·B(m,b)·Q(m,b) + B(a,m)·P(a,m)·T(m,b),
//   B(a,b) = Π (2k+1).
fn split_atanh_third(a: u64, b: u64) -> (Int, Int, Int) {
    // Returns (Q, B, T) over [a, b); P is 9^−range folded into Q (p_k = 1).
    if b - a == 1 {
        let k = a;
        let q = Int::from_u64(if k == 0 { 1 } else { 9 });
        let bb = Int::from_u64(2 * k + 1);
        // t_k = a_k·p_k with the 1/(2k+1) carried by B: t contributes 1.
        (q, bb, Int::from_u64(1))
    } else {
        let m = a + (b - a) / 2;
        let (ql, bl, tl) = split_atanh_third(a, m);
        let (qr, br, tr) = split_atanh_third(m, b);
        let q = ql.mul(&qr);
        let bb = bl.mul(&br);
        // P(a,m) = 1 (all p_k = 1): T = T_l·B_r·Q_r + B_l·T_r.
        let t = tl.mul(&br).mul(&qr).add(&bl.mul(&tr));
        (q, bb, t)
    }
}

/// ln 2 to `prec` bits via binary splitting of 2·atanh(1/3).
pub fn ln2_binsplit(prec: u32) -> Ball {
    let wp = prec + 64;
    // (1/9)^k: 3.17 bits per term.
    let n = (wp as u64) * 10 / 31 + 4;
    let (q, b, t) = split_atanh_third(0, n);
    // S = T/(B·Q) = Σ_{k<N} (1/9)^k/(2k+1); ln2 = (2/3)·S.
    let den = b.mul(&q);
    let s = Ball::exact(t.to_float()).div(&Ball::exact(den.to_float()), wp);
    // Tail: Σ_{k≥N} (1/9)^k/(2k+1) ≤ (1/9)^N · 9/8.
    let tail_f = {
        let (nth, _) = Float::from_u64(1).div(&Float::from_u64(9), 64, Round::Up);
        let mut acc = Float::from_u64(1);
        let mut base = nth;
        let mut e = n;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(&base, 64, Round::Up).0;
            }
            base = base.mul(&base, 64, Round::Up).0;
            e >>= 1;
        }
        let (t, _) = acc.mul(&Float::from_f64(9.0 / 8.0), 64, Round::Up);
        t
    };
    let s = s.add_error(&Mag::from_float_upper(&tail_f));
    s.mul_i64(2, wp).div_u64(3, wp).round(prec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;

    #[test]
    fn pi_matches_machin() {
        // Independent formulas must agree: difference contains zero.
        let prec = 2000;
        let a = pi_chudnovsky(prec);
        let b = constants::compute_pi_machin(prec);
        let d = a.sub(&b, prec + 64);
        assert!(d.contains(&Float::zero()), "Chudnovsky vs Machin disagree");
        assert!(a.rel_accuracy_bits() >= prec as i64 - 8);
    }

    #[test]
    fn pi_digits_1000() {
        let prec = 3350;
        let p = pi_chudnovsky(prec);
        let s = p.mid().to_decimal(50);
        assert!(
            s.starts_with("3.141592653589793238462643383279502884197169399375"),
            "pi = {s}"
        );
    }

    #[test]
    fn e_matches_exp_one() {
        let prec = 2000;
        let a = e_binsplit(prec);
        let b = crate::elementary::exp(&Ball::from_i64(1), prec);
        let d = a.sub(&b, prec + 64);
        assert!(d.contains(&Float::zero()), "binsplit e vs exp(1) disagree");
        assert!(a.rel_accuracy_bits() >= prec as i64 - 8);
    }

    #[test]
    fn ln2_matches_series() {
        let prec = 2000;
        let a = ln2_binsplit(prec);
        let b = crate::elementary::ln_core(&Ball::from_i64(2), prec + 32);
        let d = a.sub(&b, prec + 64);
        assert!(d.contains(&Float::zero()), "binsplit ln2 vs sqrt-reduction ln2 disagree");
        assert!(a.rel_accuracy_bits() >= prec as i64 - 8);
    }
}
