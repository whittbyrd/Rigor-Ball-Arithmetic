//! Signed arbitrary-precision integers over the mpn layer.
//!
//! Deliberately minimal: exactly the operations binary splitting and the
//! Bernoulli recurrence need. All results are exact.

use crate::fp::Float;
use crate::mpn::{self, Limb};
use core::cmp::Ordering;

/// A signed big integer. `mag` is normalized (no high zero limbs);
/// zero is the empty magnitude with `neg == false`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Int {
    neg: bool,
    mag: Vec<Limb>,
}

impl Int {
    pub const fn zero() -> Self {
        Int { neg: false, mag: Vec::new() }
    }

    pub fn from_u64(v: u64) -> Self {
        Int { neg: false, mag: if v == 0 { Vec::new() } else { vec![v] } }
    }

    pub fn from_i64(v: i64) -> Self {
        Int { neg: v < 0, mag: if v == 0 { Vec::new() } else { vec![v.unsigned_abs()] } }
    }

    pub fn is_zero(&self) -> bool {
        self.mag.is_empty()
    }

    pub fn is_neg(&self) -> bool {
        self.neg
    }

    pub fn bit_len(&self) -> u64 {
        mpn::bit_len(&self.mag)
    }

    pub fn neg(&self) -> Int {
        if self.is_zero() {
            self.clone()
        } else {
            Int { neg: !self.neg, mag: self.mag.clone() }
        }
    }

    pub fn add(&self, other: &Int) -> Int {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }
        if self.neg == other.neg {
            Int { neg: self.neg, mag: mag_add(&self.mag, &other.mag) }
        } else {
            match mpn::cmp(&self.mag, &other.mag) {
                Ordering::Equal => Int::zero(),
                Ordering::Greater => Int { neg: self.neg, mag: mag_sub(&self.mag, &other.mag) },
                Ordering::Less => Int { neg: other.neg, mag: mag_sub(&other.mag, &self.mag) },
            }
        }
    }

    pub fn sub(&self, other: &Int) -> Int {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &Int) -> Int {
        if self.is_zero() || other.is_zero() {
            return Int::zero();
        }
        let mut r = vec![0 as Limb; self.mag.len() + other.mag.len()];
        mpn::mul(&mut r, &self.mag, &other.mag);
        r.truncate(mpn::normalized_len(&r));
        Int { neg: self.neg != other.neg, mag: r }
    }

    pub fn mul_u64(&self, v: u64) -> Int {
        if self.is_zero() || v == 0 {
            return Int::zero();
        }
        let mut r = vec![0 as Limb; self.mag.len() + 1];
        r[self.mag.len()] = mpn::mul_1(&mut r[..self.mag.len()], &self.mag, v);
        r.truncate(mpn::normalized_len(&r));
        Int { neg: self.neg, mag: r }
    }

    pub fn mul_i64(&self, v: i64) -> Int {
        let r = self.mul_u64(v.unsigned_abs());
        if v < 0 { r.neg() } else { r }
    }

    /// Exact quotient and remainder (round toward zero).
    pub fn divrem(&self, other: &Int) -> (Int, Int) {
        assert!(!other.is_zero(), "Int division by zero");
        let (q, r) = mpn::divrem(&self.mag, &other.mag);
        let q = Int { neg: self.neg != other.neg && !q.is_empty(), mag: q };
        let r = Int { neg: self.neg && !r.is_empty(), mag: r };
        (q, r)
    }

    /// Exact conversion to [`Float`].
    pub fn to_float(&self) -> Float {
        if self.is_zero() {
            return Float::zero();
        }
        let f = Float::from_limbs(&self.mag);
        if self.neg { f.neg() } else { f }
    }

    pub fn cmp(&self, other: &Int) -> Ordering {
        match (self.is_zero(), other.is_zero()) {
            (true, true) => return Ordering::Equal,
            (true, false) => return if other.neg { Ordering::Greater } else { Ordering::Less },
            (false, true) => return if self.neg { Ordering::Less } else { Ordering::Greater },
            _ => {}
        }
        match (self.neg, other.neg) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => mpn::cmp(&self.mag, &other.mag),
            (true, true) => mpn::cmp(&other.mag, &self.mag),
        }
    }
}

fn mag_add(a: &[Limb], b: &[Limb]) -> Vec<Limb> {
    let (a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut r = vec![0 as Limb; a.len() + 1];
    let carry = mpn::add(&mut r[..a.len()], a, b);
    r[a.len()] = carry;
    r.truncate(mpn::normalized_len(&r));
    r
}

/// a − b for |a| > |b|.
fn mag_sub(a: &[Limb], b: &[Limb]) -> Vec<Limb> {
    let mut r = vec![0 as Limb; a.len()];
    let borrow = mpn::sub(&mut r, a, b);
    debug_assert_eq!(borrow, 0);
    r.truncate(mpn::normalized_len(&r));
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Rng;

    fn from_i128(v: i128) -> Int {
        let neg = v < 0;
        let m = v.unsigned_abs();
        let mut mag = vec![m as Limb, (m >> 64) as Limb];
        mag.truncate(mpn::normalized_len(&mag));
        Int { neg, mag }
    }

    fn to_i128(v: &Int) -> i128 {
        assert!(v.mag.len() <= 2);
        let m = match v.mag.len() {
            0 => 0u128,
            1 => v.mag[0] as u128,
            _ => v.mag[0] as u128 | (v.mag[1] as u128) << 64,
        };
        if v.neg { -(m as i128) } else { m as i128 }
    }

    #[test]
    fn arithmetic_matches_i128() {
        let mut rng = Rng::new(50);
        for _ in 0..3000 {
            let a = rng.next() as i64 as i128;
            let b = rng.next() as i64 as i128;
            let (ia, ib) = (from_i128(a), from_i128(b));
            assert_eq!(to_i128(&ia.add(&ib)), a + b, "add {a} {b}");
            assert_eq!(to_i128(&ia.sub(&ib)), a - b, "sub {a} {b}");
            assert_eq!(to_i128(&ia.mul(&ib)), a * b, "mul {a} {b}");
            if b != 0 {
                let (q, r) = ia.divrem(&ib);
                assert_eq!(to_i128(&q), a / b, "div {a} {b}");
                assert_eq!(to_i128(&r), a % b, "rem {a} {b}");
            }
        }
    }

    #[test]
    fn to_float_exact() {
        let v = from_i128(123456789012345678901234567i128);
        assert_eq!(v.to_float().to_decimal(30), "123456789012345678901234567");
        let v = from_i128(-42);
        assert_eq!(v.to_float().to_decimal(5), "-42");
    }
}
