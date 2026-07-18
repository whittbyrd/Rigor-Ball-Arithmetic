//! Bernoulli numbers B_{2k} as balls, via exact tangent numbers.
//!
//! Tangent numbers T_k (T_1 = 1, T_2 = 2, T_3 = 16, …) satisfy
//!   tan x = Σ T_k x^(2k−1)/(2k−1)!
//! and are computed exactly by Luschny's O(k²) integer recurrence. Then
//!   B_{2k} = (−1)^(k−1) · 2k · T_k / (2^(2k) · (2^(2k) − 1)),
//! an exact rational, which we convert to a ball at the requested precision.
//!
//! Cost note (honest): the O(k²) recurrence on integers of O(k log k) bits
//! makes Bernoulli generation the dominant one-time cost for gamma/zeta
//! above ~5000 digits (Arb's zeta-based multi-evaluation is the known
//! faster approach). Both the tangent table and the per-precision ball
//! table are cached and grown geometrically.

use crate::ball::Ball;
use crate::int::Int;
use std::sync::Mutex;

static TANGENT: Mutex<Vec<Int>> = Mutex::new(Vec::new());
/// (precision, balls[k] = B_{2(k+1)}) — rebuilt when a higher precision is
/// requested; served for any lower precision.
static BALLS: Mutex<(u32, Vec<Ball>)> = Mutex::new((0, Vec::new()));

/// Tangent numbers T_1..T_m (1-indexed conceptually; `out[k-1]` = T_k).
///
/// Luschny's algorithm:
///   T[1] = 1;  T[k] = (k−1)·T[k−1]
///   for k in 2..=m: for j in k..=m: T[j] = (j−k)·T[j−1] + (j−k+2)·T[j]
fn tangent_numbers(m: usize) -> Vec<Int> {
    let mut t: Vec<Int> = Vec::with_capacity(m + 1);
    t.push(Int::zero()); // index 0 unused
    t.push(Int::from_u64(1));
    for k in 2..=m {
        let prev = t[k - 1].mul_u64(k as u64 - 1);
        t.push(prev);
    }
    for k in 2..=m {
        for j in k..=m {
            let a = t[j - 1].mul_u64((j - k) as u64);
            let b = t[j].mul_u64((j - k + 2) as u64);
            t[j] = a.add(&b);
        }
    }
    t.remove(0);
    t
}

fn ensure_tangent(count: usize) {
    let mut guard = TANGENT.lock().unwrap();
    if guard.len() < count {
        // Geometric growth: the recurrence is not incremental.
        let m = count.max(guard.len() * 2).max(16);
        *guard = tangent_numbers(m);
    }
}

/// B_{2k} for k ≥ 1 as a ball with ≥ `prec` certified bits.
pub fn bernoulli(k: usize, prec: u32) -> Ball {
    assert!(k >= 1);
    {
        let guard = BALLS.lock().unwrap();
        if guard.0 >= prec && guard.1.len() >= k {
            return guard.1[k - 1].clone();
        }
    }
    ensure_tangent(k);
    let mut guard = BALLS.lock().unwrap();
    if guard.0 < prec {
        // Precision tier raised: rebuild the ball table. Generous headroom —
        // adaptive retry loops upstream probe successively larger working
        // precisions, and each tier bump discards the whole table.
        *guard = (prec * 2 + 64, Vec::new());
    }
    let cp = guard.0;
    let balls = &mut guard.1;
    let tangent = TANGENT.lock().unwrap();
    while balls.len() < k {
        let i = balls.len() + 1; // computing B_{2i}
        let num = tangent[i - 1].mul_u64(2 * i as u64);
        let den = Int::pow2(2 * i as u64).sub(&Int::from_u64(1));
        let mut b = Ball::exact(num.to_float())
            .div(&Ball::exact(den.to_float()), cp)
            .mul_2exp(-(2 * i as i64));
        if i % 2 == 0 {
            b = b.neg();
        }
        balls.push(b);
    }
    balls[k - 1].clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp::{Float, Round};

    fn assert_is(k: usize, num: i64, den: i64) {
        let b = bernoulli(k, 128);
        let (want, _) = Float::from_i64(num).div(&Float::from_i64(den), 256, Round::Nearest);
        assert!(
            b.contains(&want) || {
                // 1/den may be inexact in binary: widen by reference error.
                let widened =
                    b.add_error(&crate::mag::Mag::two_exp(want.exponent().unwrap() - 250));
                widened.contains(&want)
            },
            "B_{} should be {num}/{den}, got {}",
            2 * k,
            b.to_string_digits(25)
        );
    }

    #[test]
    fn known_values() {
        // B_2 = 1/6, B_4 = −1/30, B_6 = 1/42, B_8 = −1/30, B_10 = 5/66,
        // B_12 = −691/2730.
        assert_is(1, 1, 6);
        assert_is(2, -1, 30);
        assert_is(3, 1, 42);
        assert_is(4, -1, 30);
        assert_is(5, 5, 66);
        assert_is(6, -691, 2730);
    }

    #[test]
    fn tangent_small() {
        let t = tangent_numbers(5);
        let want = [1u64, 2, 16, 272, 7936];
        for (i, w) in want.iter().enumerate() {
            assert_eq!(
                t[i].cmp(&Int::from_u64(*w)),
                core::cmp::Ordering::Equal,
                "T_{}",
                i + 1
            );
        }
    }
}
