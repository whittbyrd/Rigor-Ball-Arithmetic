//! Unsigned magnitude bounds — the radius type (Arb's `mag_t`).
//!
//! A [`Mag`] stores `man · 2^(exp − 30)` with `man = 0` or `man ∈ [2^29, 2^30)`.
//! Every operation rounds **up**: a `Mag` is only ever used as an upper bound,
//! so all arithmetic here must preserve "result ≥ true value". Each op's
//! comment carries its one-line over-approximation argument.

use crate::fp::Float;

/// Mantissa bits of a magnitude bound.
pub const MAG_BITS: u32 = 30;
const MAN_MAX: u32 = 1 << MAG_BITS; // exclusive

/// A nonnegative upper bound with a 30-bit mantissa: `man · 2^(exp − 30)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mag {
    man: u32,
    exp: i64,
}

impl Mag {
    pub const fn zero() -> Self {
        Mag { man: 0, exp: 0 }
    }

    pub fn is_zero(&self) -> bool {
        self.man == 0
    }

    /// The bound `2^e`.
    pub fn two_exp(e: i64) -> Self {
        Mag { man: 1 << (MAG_BITS - 1), exp: e + 1 }
    }

    /// Exponent bound: value ≤ 2^exp (tight to within one part in 2^29).
    pub fn exp(&self) -> i64 {
        self.exp
    }

    fn normalized(mut man: u64, mut exp: i64) -> Self {
        debug_assert!(man != 0);
        // Bring man into [2^29, 2^30), rounding up on right shifts.
        while man >= MAN_MAX as u64 {
            man = (man + 1) >> 1; // ceil(man/2)·2^(exp+1) ≥ man·2^exp
            exp += 1;
        }
        while man < (MAN_MAX >> 1) as u64 {
            man <<= 1;
            exp -= 1;
        }
        Mag { man: man as u32, exp }
    }

    /// Upper bound for a `u64`.
    pub fn from_u64_upper(v: u64) -> Self {
        if v == 0 {
            return Mag::zero();
        }
        let bits = 64 - v.leading_zeros();
        if bits <= MAG_BITS {
            // value = v·2^(30−30) = v.
            Mag::normalized(v, MAG_BITS as i64)
        } else {
            let shift = bits - MAG_BITS;
            // ((v >> s) + 1)·2^s ≥ v.
            Mag::normalized((v >> shift) as u64 + 1, (MAG_BITS + shift) as i64)
        }
    }

    /// Upper bound for `|f|`.
    pub fn from_float_upper(f: &Float) -> Self {
        match f.exponent() {
            None => Mag::zero(),
            Some(e) => {
                // |f| < 2^e always; take the top 30 fraction bits and add one
                // ulp to cover the discarded tail: top30·2^(e−30) + 2^(e−30) ≥ |f|.
                let top = f.top_bits(MAG_BITS);
                Mag::normalized(top + 1, e)
            }
        }
    }

    /// Exact conversion to [`Float`] (a Mag is a dyadic rational).
    pub fn to_float(&self) -> Float {
        if self.is_zero() {
            Float::zero()
        } else {
            Float::from_u64(self.man as u64).mul_2exp(self.exp - MAG_BITS as i64)
        }
    }

    /// Upper bound for `a + b`.
    pub fn add_up(&self, other: &Mag) -> Mag {
        if self.is_zero() {
            return *other;
        }
        if other.is_zero() {
            return *self;
        }
        let (a, b) = if self.exp >= other.exp { (self, other) } else { (other, self) };
        let d = (a.exp - b.exp) as u64;
        if d >= 34 {
            // b < 2^b.exp ≤ 2^(a.exp−34); adding 1 to a's mantissa adds
            // 2^(a.exp−30) > b, so this over-approximates.
            Mag::normalized(a.man as u64 + 1, a.exp)
        } else {
            // ceil-shift b onto a's scale: (b.man >> d) + 1 ≥ b.man / 2^d.
            Mag::normalized(a.man as u64 + (b.man as u64 >> d) + 1, a.exp)
        }
    }

    /// Upper bound for `a · b`.
    pub fn mul_up(&self, other: &Mag) -> Mag {
        if self.is_zero() || other.is_zero() {
            return Mag::zero();
        }
        // a·b = ma·mb · 2^(ea−30) · 2^(eb−30) = (ma·mb) · 2^((ea+eb−30) − 30).
        let p = self.man as u64 * other.man as u64; // < 2^60, exact
        Mag::normalized(p, self.exp + other.exp - MAG_BITS as i64)
    }

    /// Exact scaling by 2^k.
    pub fn mul_2exp(&self, k: i64) -> Mag {
        if self.is_zero() {
            *self
        } else {
            Mag { man: self.man, exp: self.exp + k }
        }
    }

    /// max(a, b) — an upper bound for both.
    pub fn max(&self, other: &Mag) -> Mag {
        if self.cmp(other) == core::cmp::Ordering::Less {
            *other
        } else {
            *self
        }
    }

    pub fn cmp(&self, other: &Mag) -> core::cmp::Ordering {
        match (self.is_zero(), other.is_zero()) {
            (true, true) => return core::cmp::Ordering::Equal,
            (true, false) => return core::cmp::Ordering::Less,
            (false, true) => return core::cmp::Ordering::Greater,
            _ => {}
        }
        (self.exp, self.man).cmp(&(other.exp, other.man))
    }

    /// Approximate value for display/diagnostics.
    pub fn to_f64(&self) -> f64 {
        if self.is_zero() {
            0.0
        } else {
            self.man as f64 * 2f64.powi((self.exp - MAG_BITS as i64).clamp(-1060, 1020) as i32)
        }
    }
}

impl core::fmt::Display for Mag {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_zero() {
            write!(f, "0")
        } else {
            write!(f, "2^{}", self.exp)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp::Round;
    use crate::testutil::Rng;

    #[test]
    fn upper_bound_from_float() {
        let mut rng = Rng::new(20);
        for _ in 0..2000 {
            let v = f64::from_bits(rng.next() % (1023u64 << 52));
            if !v.is_finite() || v == 0.0 {
                continue;
            }
            let f = Float::from_f64(v.abs());
            let m = Mag::from_float_upper(&f);
            // m ≥ |f|
            assert!(
                m.to_float().cmp(&f) != core::cmp::Ordering::Less,
                "Mag {} < float {}",
                m.to_f64(),
                v.abs()
            );
        }
    }

    #[test]
    fn add_mul_are_upper_bounds() {
        let mut rng = Rng::new(21);
        for _ in 0..2000 {
            let a = Mag::from_u64_upper(rng.next() >> (rng.next() % 60));
            let b = Mag::from_u64_upper(rng.next() >> (rng.next() % 60));
            let s = a.add_up(&b);
            let p = a.mul_up(&b);
            // Verify with exact Float arithmetic.
            let (fs, _) = a.to_float().add(&b.to_float(), 128, Round::Nearest);
            let (fp, _) = a.to_float().mul(&b.to_float(), 128, Round::Nearest);
            assert!(s.to_float().cmp(&fs) != core::cmp::Ordering::Less, "add not upper");
            assert!(p.to_float().cmp(&fp) != core::cmp::Ordering::Less, "mul not upper");
            // And not absurdly loose (within a factor 1 + 2^-25).
            let slack = Float::from_f64(1.0 + 1e-7);
            let (fs_hi, _) = fs.mul(&slack, 64, Round::Up);
            let (fp_hi, _) = fp.mul(&slack, 64, Round::Up);
            assert!(s.to_float().cmp(&fs_hi) != core::cmp::Ordering::Greater, "add too loose");
            assert!(p.to_float().cmp(&fp_hi) != core::cmp::Ordering::Greater, "mul too loose");
        }
    }

    #[test]
    fn two_exp_value() {
        let m = Mag::two_exp(10);
        assert_eq!(m.to_float().cmp(&Float::from_i64(1024)), core::cmp::Ordering::Equal);
        let m = Mag::two_exp(-3);
        let (eighth, _) = Float::from_i64(1).div(&Float::from_i64(8), 64, Round::Nearest);
        assert_eq!(m.to_float().cmp(&eighth), core::cmp::Ordering::Equal);
    }
}
