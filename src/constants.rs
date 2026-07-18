//! Cached fundamental constants as balls.
//!
//! Each constant is computed once at (slightly more than) the highest
//! precision requested so far and served from cache, rounded down to the
//! caller's precision. All cache hits are certified inclusions — a cached
//! ball rounded to lower precision is still a correct enclosure.

use crate::ball::Ball;
use crate::elementary;
use std::sync::Mutex;

struct Cache {
    slot: Mutex<Option<(u32, Ball)>>,
}

impl Cache {
    const fn new() -> Self {
        Cache { slot: Mutex::new(None) }
    }

    fn get(&self, prec: u32, compute: impl FnOnce(u32) -> Ball) -> Ball {
        let mut guard = self.slot.lock().unwrap();
        if let Some((p, b)) = guard.as_ref() {
            if *p >= prec {
                return b.round(prec);
            }
        }
        // Compute with headroom so nearby future requests hit the cache.
        let cp = prec + prec / 4 + 64;
        let b = compute(cp);
        *guard = Some((cp, b.clone()));
        b.round(prec)
    }
}

static PI: Cache = Cache::new();
static LN2: Cache = Cache::new();
static E: Cache = Cache::new();

/// π as a ball with ≥ `prec` certified bits (Chudnovsky binary splitting).
pub fn pi(prec: u32) -> Ball {
    PI.get(prec, crate::binsplit::pi_chudnovsky)
}

/// ln 2 as a ball with ≥ `prec` certified bits (binary splitting of
/// 2·atanh(1/3)).
pub fn ln2(prec: u32) -> Ball {
    LN2.get(prec, crate::binsplit::ln2_binsplit)
}

/// e as a ball with ≥ `prec` certified bits (binary splitting of Σ 1/k!).
pub fn e(prec: u32) -> Ball {
    E.get(prec, crate::binsplit::e_binsplit)
}

/// Machin's formula: π = 16·atan(1/5) − 4·atan(1/239).
///
/// Both arguments are < 1, so the atan path never needs π (no circularity).
/// Slower than Chudnovsky; kept as an algorithmically independent
/// cross-check of the production π (see binsplit tests).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn compute_pi_machin(wp: u32) -> Ball {
    let wp = wp + 32;
    let a5 = elementary::atan(&Ball::from_i64(1).div_u64(5, wp), wp);
    let a239 = elementary::atan(&Ball::from_i64(1).div_u64(239, wp), wp);
    a5.mul_2exp(4).sub(&a239.mul_2exp(2), wp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pi_digits() {
        let p = pi(400);
        let s = p.mid().to_decimal(45);
        assert!(
            s.starts_with("3.14159265358979323846264338327950288419716"),
            "pi = {s}"
        );
        assert!(p.rel_accuracy_bits() >= 390);
    }

    #[test]
    fn ln2_digits() {
        let l = ln2(300);
        let s = l.mid().to_decimal(35);
        assert!(s.starts_with("0.693147180559945309417232121458"), "ln2 = {s}");
        assert!(l.rel_accuracy_bits() >= 290);
    }

    #[test]
    fn cache_serves_lower_precision() {
        let hi = pi(1000);
        let lo = pi(100);
        // Both must contain the true π: their difference contains 0.
        let d = hi.sub(&lo, 1100);
        assert!(d.contains(&crate::fp::Float::zero()));
        assert!(lo.rel_accuracy_bits() >= 100);
    }
}
