//! Arbitrary-precision binary floating point with directed rounding.
//!
//! Representation: `value = (-1)^neg * 0.L * 2^exp`, where `L` is the limb
//! vector read as a binary fraction (most significant bit first). Invariants
//! for nonzero values:
//! - the top bit of the top limb is set (fraction in `[1/2, 1)`),
//! - the lowest limb is nonzero (minimal representation).
//!
//! Precision is a *per-operation* argument (as in Arb's `arf_t`), not a field
//! of the number: a `Float` is always an exactly-represented dyadic rational.
//!
//! ## Rounding correctness
//!
//! Every arithmetic operation reduces its exact result to a **window**: a limb
//! vector `w` denoting `0.w × 2^exp_top`, exact except for an optional sticky
//! term `eps` with `true = window + eps`, `0 < eps < 2·ulp(window bottom)`.
//! Construction of each window guarantees the top set bit lies ≥ 126 bits
//! above the window bottom whenever `eps` can be nonzero, so `eps` can never
//! influence any bit at or above the rounding position — it only matters as a
//! sticky "there is something nonzero below" flag. [`Float::from_window`] then
//! performs the only rounding step in the whole crate.

use crate::mpn::{self, Limb, LIMB_BITS};
use core::cmp::Ordering;

/// Rounding mode for [`Float`] operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Round {
    /// Toward -infinity.
    Floor,
    /// Toward +infinity.
    Ceil,
    /// Toward zero (truncate).
    Down,
    /// Away from zero.
    Up,
    /// To nearest, ties to even.
    Nearest,
}

/// Magnitude-space rounding decision (sign already folded in).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MagRound {
    Down,
    Up,
    Nearest,
}

impl Round {
    #[inline]
    fn mag(self, neg: bool) -> MagRound {
        match self {
            Round::Nearest => MagRound::Nearest,
            Round::Down => MagRound::Down,
            Round::Up => MagRound::Up,
            Round::Floor => {
                if neg {
                    MagRound::Up
                } else {
                    MagRound::Down
                }
            }
            Round::Ceil => {
                if neg {
                    MagRound::Down
                } else {
                    MagRound::Up
                }
            }
        }
    }
}

/// An arbitrary-precision binary floating-point number (a dyadic rational).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Float {
    neg: bool,
    /// Exponent of the value: nonzero value lies in `[2^(exp-1), 2^exp)`.
    exp: i64,
    /// Little-endian fraction limbs; empty means zero.
    limbs: Vec<Limb>,
}

impl Float {
    pub const fn zero() -> Self {
        Float { neg: false, exp: 0, limbs: Vec::new() }
    }

    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    pub fn is_neg(&self) -> bool {
        self.neg
    }

    /// Exponent `e` with `|x| ∈ [2^(e-1), 2^e)`; `None` for zero.
    pub fn exponent(&self) -> Option<i64> {
        if self.is_zero() { None } else { Some(self.exp) }
    }

    /// Number of significant bits in the fraction (0 for zero).
    pub fn bit_len(&self) -> u64 {
        if self.is_zero() {
            return 0;
        }
        let n = self.limbs.len() as u64;
        let tz = self.limbs[0].trailing_zeros() as u64;
        n * LIMB_BITS as u64 - tz
    }

    pub fn from_u64(v: u64) -> Self {
        if v == 0 {
            return Float::zero();
        }
        let lz = v.leading_zeros();
        Float { neg: false, exp: 64 - lz as i64, limbs: vec![v << lz] }
    }

    pub fn from_i64(v: i64) -> Self {
        let mut f = Float::from_u64(v.unsigned_abs());
        f.neg = v < 0;
        f
    }

    /// Exact conversion from an f64 (must be finite).
    pub fn from_f64(v: f64) -> Self {
        assert!(v.is_finite(), "Float::from_f64 requires a finite value");
        if v == 0.0 {
            return Float::zero();
        }
        let bits = v.to_bits();
        let neg = bits >> 63 == 1;
        let biased = ((bits >> 52) & 0x7FF) as i64;
        let frac = bits & ((1u64 << 52) - 1);
        let (mant, exp2) = if biased == 0 {
            (frac, -1074i64) // subnormal
        } else {
            (frac | 1 << 52, biased - 1075)
        };
        // value = mant * 2^exp2, normalized so the limb's top bit is set.
        let lz = mant.leading_zeros();
        Float { neg, exp: exp2 + 64 - lz as i64, limbs: vec![mant << lz] }
    }

    /// Approximate conversion to f64 (round to nearest; may overflow to inf).
    pub fn to_f64(&self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }
        let n = self.limbs.len();
        let top = self.limbs[n - 1] as f64; // fraction top limb / 2^64
        let next = if n >= 2 { self.limbs[n - 2] as f64 } else { 0.0 };
        let frac = top / 2f64.powi(64) + next / 2f64.powi(128);
        let v = frac * 2f64.powi(self.exp.clamp(-2000, 2000) as i32);
        if self.neg { -v } else { v }
    }

    pub fn neg_assign(&mut self) {
        if !self.is_zero() {
            self.neg = !self.neg;
        }
    }

    pub fn neg(&self) -> Self {
        let mut r = self.clone();
        r.neg_assign();
        r
    }

    pub fn abs(&self) -> Self {
        let mut r = self.clone();
        r.neg = false;
        r
    }

    /// Exact scaling by a power of two.
    pub fn mul_2exp(&self, k: i64) -> Self {
        let mut r = self.clone();
        if !r.is_zero() {
            r.exp += k;
        }
        r
    }

    /// Compare by magnitude.
    pub fn cmp_abs(&self, other: &Float) -> Ordering {
        match (self.is_zero(), other.is_zero()) {
            (true, true) => return Ordering::Equal,
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            _ => {}
        }
        if self.exp != other.exp {
            return self.exp.cmp(&other.exp);
        }
        // Same exponent: compare fraction limbs from the top.
        let (a, b) = (&self.limbs, &other.limbs);
        let (na, nb) = (a.len(), b.len());
        let n = na.max(nb);
        for i in 1..=n {
            let la = if i <= na { a[na - i] } else { 0 };
            let lb = if i <= nb { b[nb - i] } else { 0 };
            if la != lb {
                return la.cmp(&lb);
            }
        }
        Ordering::Equal
    }

    /// Total order (-inf .. +inf).
    pub fn cmp(&self, other: &Float) -> Ordering {
        match (self.is_zero(), other.is_zero()) {
            (true, true) => return Ordering::Equal,
            (true, false) => return if other.neg { Ordering::Greater } else { Ordering::Less },
            (false, true) => return if self.neg { Ordering::Less } else { Ordering::Greater },
            _ => {}
        }
        match (self.neg, other.neg) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => self.cmp_abs(other),
            (true, true) => other.cmp_abs(self),
        }
    }

    /// Sign as -1 / 0 / +1.
    pub fn signum(&self) -> i32 {
        if self.is_zero() {
            0
        } else if self.neg {
            -1
        } else {
            1
        }
    }

    // ------------------------------------------------------------------
    // Window rounding: the single place where precision is lost.
    // ------------------------------------------------------------------

    /// Build a rounded `Float` from a fraction window.
    ///
    /// `w` denotes `0.w × 2^exp_top` (MSB of `w.last()` sits just below the
    /// binary point). If `sticky`, the true value is `window + eps` for some
    /// `0 < eps < 2^(exp_top - 64·w.len() + 1)` (up to two bottom ulps), and
    /// the caller guarantees the top set bit of `w` is at least 126 bits above
    /// the window bottom. Returns the rounded value and an `inexact` flag.
    fn from_window(
        neg: bool,
        w: &[Limb],
        exp_top: i64,
        sticky: bool,
        prec: u32,
        rnd: Round,
    ) -> (Float, bool) {
        assert!(prec >= 2, "precision must be at least 2 bits");
        let mrnd = rnd.mag(neg);
        let wbits = (w.len() as i64) * LIMB_BITS as i64;

        // Locate the top set bit (position from window bottom, 0-based).
        let wn = mpn::normalized_len(w);
        if wn == 0 {
            assert!(!sticky, "window vanished with sticky bits set");
            return (Float::zero(), false);
        }
        let t = (wn as i64) * 64 - 1 - w[wn - 1].leading_zeros() as i64;
        debug_assert!(!sticky || t >= 126, "sticky window with catastrophic cancellation");

        let e = exp_top - (wbits - 1 - t); // value in [2^(e-1), 2^e)
        let nl = prec.div_ceil(LIMB_BITS) as usize;
        let drop = (nl as u32) * LIMB_BITS - prec; // low zero bits in the output

        let bit = |i: i64| -> u64 {
            if i < 0 || i >= wbits {
                0
            } else {
                w[(i / 64) as usize] >> (i % 64) & 1
            }
        };
        // Any set bit strictly below position `p`?
        let any_below = |p: i64| -> bool {
            if p <= 0 {
                return false;
            }
            let p = p.min(wbits);
            let full = (p / 64) as usize;
            if w[..full].iter().any(|&l| l != 0) {
                return true;
            }
            let rem = (p % 64) as u32;
            rem > 0 && w[full] << (64 - rem) != 0
        };

        let rb_pos = t - prec as i64; // round bit position
        let round_bit = bit(rb_pos);
        let sticky2 = sticky || any_below(rb_pos);
        let inexact = round_bit == 1 || sticky2;

        // Extract the kept bits, left-aligned into nl limbs.
        let off = t + 1 - 64 * nl as i64;
        let window_limb = |o: i64| -> Limb {
            if o <= -64 || o >= wbits {
                return 0;
            }
            let q = o.div_euclid(64);
            let sh = o.rem_euclid(64) as u32;
            let lo = if q >= 0 && (q as usize) < w.len() { w[q as usize] } else { 0 };
            let hi = if q + 1 >= 0 && ((q + 1) as usize) < w.len() { w[(q + 1) as usize] } else { 0 };
            if sh == 0 { lo } else { lo >> sh | hi << (64 - sh) }
        };
        let mut m: Vec<Limb> = (0..nl).map(|k| window_limb(off + 64 * k as i64)).collect();
        if drop > 0 {
            m[0] &= !0 << drop;
        }

        let increment = match mrnd {
            MagRound::Down => false,
            MagRound::Up => inexact,
            MagRound::Nearest => {
                let last_kept = bit(rb_pos + 1);
                round_bit == 1 && (sticky2 || last_kept == 1)
            }
        };

        let mut e = e;
        if increment {
            let mut carry = 1u128 << drop;
            for limb in m.iter_mut() {
                let t = *limb as u128 + carry;
                *limb = t as Limb;
                carry = t >> 64;
                if carry == 0 {
                    break;
                }
            }
            if carry != 0 {
                // 0.111..1 rounded up to 1.0: renormalize.
                m.fill(0);
                *m.last_mut().unwrap() = 1 << 63;
                e += 1;
            }
        }

        // Minimal representation: strip low zero limbs.
        let lowz = m.iter().position(|&l| l != 0).unwrap_or(m.len());
        let limbs: Vec<Limb> = m[lowz..].to_vec();
        debug_assert!(!limbs.is_empty(), "kept bits cannot all vanish");
        debug_assert!(limbs.last().unwrap() >> 63 == 1);
        (Float { neg, exp: e, limbs }, inexact)
    }

    /// Round `self` to `prec` bits.
    pub fn round(&self, prec: u32, rnd: Round) -> (Float, bool) {
        if self.is_zero() {
            return (Float::zero(), false);
        }
        Float::from_window(self.neg, &self.limbs, self.exp, false, prec, rnd)
    }

    // ------------------------------------------------------------------
    // Addition / subtraction
    // ------------------------------------------------------------------

    /// `self + other`, rounded to `prec` bits. Returns (result, inexact).
    pub fn add(&self, other: &Float, prec: u32, rnd: Round) -> (Float, bool) {
        if self.is_zero() {
            return other.round(prec, rnd);
        }
        if other.is_zero() {
            return self.round(prec, rnd);
        }
        if self.neg == other.neg {
            Float::mag_add(self, other, self.neg, prec, rnd)
        } else {
            match self.cmp_abs(other) {
                Ordering::Equal => (Float::zero(), false),
                Ordering::Greater => Float::mag_sub(self, other, self.neg, prec, rnd),
                Ordering::Less => Float::mag_sub(other, self, other.neg, prec, rnd),
            }
        }
    }

    /// `self - other`, rounded to `prec` bits.
    pub fn sub(&self, other: &Float, prec: u32, rnd: Round) -> (Float, bool) {
        // Cheap: negate a copy's sign only (no limb copy needed conceptually,
        // but clone is fine here; operands are small relative to the op cost).
        self.add(&other.neg(), prec, rnd)
    }

    /// |a| + |b| with result sign `neg`; a, b nonzero.
    fn mag_add(a: &Float, b: &Float, neg: bool, prec: u32, rnd: Round) -> (Float, bool) {
        let (a, b) = if a.exp >= b.exp { (a, b) } else { (b, a) };
        let d = (a.exp - b.exp) as u64;
        let nl = prec.div_ceil(LIMB_BITS) as usize;
        let wl = nl + 3; // top limb reserved for carry; ≥128 guard bits
        let exp_top = a.exp + 64;
        let mut w = vec![0 as Limb; wl];
        let mut sticky = false;

        // Place a: fraction occupies window bits [wbits-64-64*na, wbits-64).
        sticky |= place_shifted(&mut w, &a.limbs, 64);
        // Place b at additional offset d.
        let db = 64i64 + d as i64;
        sticky |= add_shifted(&mut w, &b.limbs, db);

        Float::from_window(neg, &w, exp_top, sticky, prec, rnd)
    }

    /// |a| - |b| with |a| > |b|, result sign `neg`; a, b nonzero.
    fn mag_sub(a: &Float, b: &Float, neg: bool, prec: u32, rnd: Round) -> (Float, bool) {
        let d = (a.exp - b.exp) as u64; // a.exp >= b.exp since |a| > |b|
        if d <= 1 {
            // Possible catastrophic cancellation: compute exactly.
            // Common bottom exponent:
            let abot = a.exp - 64 * a.limbs.len() as i64;
            let bbot = b.exp - 64 * b.limbs.len() as i64;
            let bot = abot.min(bbot);
            let len = ((a.exp + 64 - bot) / 64) as usize; // a.exp+1..: 1 guard limb
            let mut w = vec![0 as Limb; len];
            let top = bot + 64 * len as i64;
            let sa = place_shifted(&mut w, &a.limbs, top - a.exp);
            debug_assert!(!sa);
            let mut bw = vec![0 as Limb; len];
            let sb = place_shifted(&mut bw, &b.limbs, top - b.exp);
            debug_assert!(!sb);
            let borrow = mpn::sub_in_place(&mut w, &bw);
            debug_assert_eq!(borrow, 0);
            return Float::from_window(neg, &w, top, false, prec, rnd);
        }

        // d >= 2: cancellation is at most one bit; windowed subtraction.
        let nl = prec.div_ceil(LIMB_BITS) as usize;
        let wl = nl + 3;
        let exp_top = a.exp;
        let mut w = vec![0 as Limb; wl];
        let eps_a = place_shifted(&mut w, &a.limbs, 0);
        let mut bw = vec![0 as Limb; wl];
        let eps_b = place_shifted(&mut bw, &b.limbs, d as i64);
        let borrow = mpn::sub_in_place(&mut w, &bw);
        debug_assert_eq!(borrow, 0);

        let sticky = if eps_b {
            // true = w + eps_a - eps_b: decrement so the leftover is positive.
            let src = w.clone();
            let borrow = mpn::sub_1(&mut w, &src, 1);
            debug_assert_eq!(borrow, 0);
            true
        } else {
            eps_a
        };
        Float::from_window(neg, &w, exp_top, sticky, prec, rnd)
    }

    // ------------------------------------------------------------------
    // Multiplication
    // ------------------------------------------------------------------

    /// `self * other`, rounded to `prec` bits.
    pub fn mul(&self, other: &Float, prec: u32, rnd: Round) -> (Float, bool) {
        if self.is_zero() || other.is_zero() {
            return (Float::zero(), false);
        }
        let neg = self.neg != other.neg;
        let (na, nb) = (self.limbs.len(), other.limbs.len());
        let mut w = vec![0 as Limb; na + nb];
        mpn::mul(&mut w, &self.limbs, &other.limbs);
        // 0.a × 0.b ∈ [1/4, 1): exact product, exp_top = ea + eb.
        Float::from_window(neg, &w, self.exp + other.exp, false, prec, rnd)
    }

    /// `self * m` for a small unsigned integer, rounded to `prec` bits.
    pub fn mul_u64(&self, m: u64, prec: u32, rnd: Round) -> (Float, bool) {
        self.mul(&Float::from_u64(m), prec, rnd)
    }

    // ------------------------------------------------------------------
    // Division
    // ------------------------------------------------------------------

    /// `self / other`, rounded to `prec` bits. Panics on division by zero.
    pub fn div(&self, other: &Float, prec: u32, rnd: Round) -> (Float, bool) {
        assert!(!other.is_zero(), "division by zero");
        if self.is_zero() {
            return (Float::zero(), false);
        }
        let neg = self.neg != other.neg;
        let (na, nb) = (self.limbs.len(), other.limbs.len());
        let nl = prec.div_ceil(LIMB_BITS) as usize;
        let qn = nl + 2;
        let s = (qn + nb).saturating_sub(na); // numerator shift, in limbs
        let mut num = vec![0 as Limb; na + s];
        num[s..].copy_from_slice(&self.limbs);
        let (q, r) = mpn::divrem(&num, &other.limbs);
        let nq = q.len();
        // value = (Q + rem/B) · 2^(exp_a - exp_b - 64(na - nb) - 64 s)
        let exp_top =
            self.exp - other.exp - 64 * (na as i64 - nb as i64) - 64 * s as i64 + 64 * nq as i64;
        Float::from_window(neg, &q, exp_top, !r.is_empty(), prec, rnd)
    }

    // ------------------------------------------------------------------
    // Square root
    // ------------------------------------------------------------------

    /// `sqrt(self)`, rounded to `prec` bits. Panics if negative.
    pub fn sqrt(&self, prec: u32, rnd: Round) -> (Float, bool) {
        assert!(!self.neg || self.is_zero(), "sqrt of negative number");
        if self.is_zero() {
            return (Float::zero(), false);
        }
        let na = self.limbs.len() as i64;
        let nl = prec.div_ceil(LIMB_BITS) as i64;
        // X = A · 2^E with A the fraction as an integer.
        let mut e = self.exp - 64 * na;
        // Scale A by 2^(2·64·t) (+1 bit if E is odd) so isqrt yields ≥ nl+2 limbs.
        let t = (nl + 2 - na / 2).max(0) as usize + 1;
        let odd = e & 1 != 0;
        let mut scaled = vec![0 as Limb; self.limbs.len() + 2 * t];
        scaled[2 * t..].copy_from_slice(&self.limbs);
        if odd {
            let src = scaled.clone();
            let carry = mpn::lshift(&mut scaled, &src, 1);
            if carry != 0 {
                scaled.push(carry);
            }
            e -= 1;
        }
        let (s, exact) = isqrt(&scaled);
        let ns = s.len() as i64;
        // sqrt(X) = (S + frac) · 2^(E/2 - 64 t), 0 <= frac < 1.
        let exp_top = e / 2 - 64 * t as i64 + 64 * ns;
        Float::from_window(false, &s, exp_top, !exact, prec, rnd)
    }

    /// Top `n` fraction bits (1 ≤ n ≤ 64) as an integer in `[2^(n-1), 2^n)`.
    /// Zero for a zero value.
    pub fn top_bits(&self, n: u32) -> u64 {
        debug_assert!(n >= 1 && n <= 64);
        match self.limbs.last() {
            None => 0,
            Some(&top) => top >> (64 - n),
        }
    }

    /// Exponent of one unit in the last place for a `prec`-bit rounding of
    /// this number: `ulp = 2^(exponent - prec)`.
    pub fn ulp_exp(&self, prec: u32) -> Option<i64> {
        self.exponent().map(|e| e - prec as i64)
    }

    // ------------------------------------------------------------------
    // Decimal output
    // ------------------------------------------------------------------

    /// Decimal string with `digits` significant digits (round-to-nearest on
    /// the last digit is NOT performed; the digits are truncated — intended
    /// for verified-digit output where the caller controls slack).
    pub fn to_decimal(&self, digits: usize) -> String {
        if self.is_zero() {
            return "0".to_string();
        }
        // |x| = 0.L × 2^exp. Split into integer and fractional parts.
        let mut out = String::new();
        if self.neg {
            out.push('-');
        }
        // Integer part: floor(|x|).
        let e = self.exp;
        let nbits = 64 * self.limbs.len() as i64;
        let int_limbs: Vec<Limb> = if e <= 0 {
            Vec::new()
        } else {
            // Integer part = top e bits of the fraction.
            let nl = (e as usize).div_ceil(64);
            let mut v = vec![0 as Limb; nl];
            for (k, limb) in v.iter_mut().enumerate() {
                // bit i of the integer = fraction bit (nbits - e + i)
                let o = nbits - e + 64 * k as i64;
                *limb = window_limb_of(&self.limbs, o);
            }
            v.truncate(mpn::normalized_len(&v));
            v
        };
        let int_str = limbs_to_decimal(&int_limbs);
        out.push_str(&int_str);

        let int_digits = if int_str == "0" { 0 } else { int_str.len() };
        if int_digits >= digits {
            return out;
        }
        let frac_digits = digits - int_digits;

        // Fractional part: |x| - floor(|x|), as fraction bits below the point.
        // frac = 0.F where F = fraction bits below position e.
        let frac_bits = (nbits - e).max(0);
        if frac_bits == 0 {
            return out;
        }
        let fl = (frac_bits as u64).div_ceil(64) as usize;
        let mut f = vec![0 as Limb; fl];
        for (k, limb) in f.iter_mut().enumerate() {
            // Window top = fraction bit position (nbits - e); we build the
            // fractional value left-aligned in fl limbs.
            let o = nbits - e - 64 * (fl - k) as i64;
            *limb = window_limb_of(&self.limbs, o);
        }
        out.push('.');
        // Repeatedly multiply by 10^19 and peel the integer part.
        let mut remaining = frac_digits;
        while remaining > 0 {
            let chunk = remaining.min(19);
            let mul = 10u64.pow(chunk as u32);
            let carry = {
                let src = f.clone();
                mpn::mul_1(&mut f, &src, mul)
            };
            let s = format!("{:0width$}", carry, width = chunk);
            out.push_str(&s);
            remaining -= chunk;
        }
        let trimmed = out.trim_end_matches('0').trim_end_matches('.');
        trimmed.to_string()
    }
}

/// 64 window bits of `w` starting at bit offset `o` (bits outside are 0).
fn window_limb_of(w: &[Limb], o: i64) -> Limb {
    let wbits = 64 * w.len() as i64;
    if o <= -64 || o >= wbits {
        return 0;
    }
    let q = o.div_euclid(64);
    let sh = o.rem_euclid(64) as u32;
    let lo = if q >= 0 && (q as usize) < w.len() { w[q as usize] } else { 0 };
    let hi = if q + 1 >= 0 && ((q + 1) as usize) < w.len() { w[(q + 1) as usize] } else { 0 };
    if sh == 0 { lo } else { lo >> sh | hi << (64 - sh) }
}

/// Place `src` into window `w` such that the top of `src` lands `down_bits`
/// below the top of `w`. Returns true if nonzero bits fell below the window.
fn place_shifted(w: &mut [Limb], src: &[Limb], down_bits: i64) -> bool {
    add_or_place(w, src, down_bits, false)
}

/// Add `src` (shifted `down_bits` below the window top) into `w`.
/// Returns true if nonzero bits fell below the window. Panics on carry out
/// of the window top (callers reserve headroom).
fn add_shifted(w: &mut [Limb], src: &[Limb], down_bits: i64) -> bool {
    add_or_place(w, src, down_bits, true)
}

fn add_or_place(w: &mut [Limb], src: &[Limb], down_bits: i64, accumulate: bool) -> bool {
    let wbits = 64 * w.len() as i64;
    let sbits = 64 * src.len() as i64;
    // src bit j (0-based from src bottom) maps to window bit
    //   j + (wbits - down_bits - sbits)
    let shift = wbits - down_bits - sbits;
    let mut sticky = false;
    if shift >= 0 {
        let ls = (shift / 64) as usize;
        let bs = (shift % 64) as u32;
        debug_assert!(ls + src.len() <= w.len(), "source overflows the window top");
        if bs == 0 {
            if accumulate {
                let c = mpn::add_in_place(&mut w[ls..], src);
                debug_assert_eq!(c, 0);
            } else {
                w[ls..ls + src.len()].copy_from_slice(src);
            }
        } else {
            let mut shifted = vec![0 as Limb; src.len() + 1];
            shifted[src.len()] = mpn::lshift(&mut shifted[..src.len()], src, bs);
            let sn = mpn::normalized_len(&shifted);
            if accumulate {
                let c = mpn::add_in_place(&mut w[ls..], &shifted[..sn]);
                debug_assert_eq!(c, 0);
            } else {
                w[ls..ls + sn].copy_from_slice(&shifted[..sn]);
            }
        }
    } else {
        // Some low bits of src fall below the window.
        let cut = (-shift) as u64; // number of src bits below the window
        let ls = (cut / 64) as usize;
        let bs = (cut % 64) as u32;
        if ls >= src.len() {
            return src.iter().any(|&l| l != 0);
        }
        sticky |= src[..ls].iter().any(|&l| l != 0);
        let kept: Vec<Limb> = if bs == 0 {
            src[ls..].to_vec()
        } else {
            sticky |= src[ls] << (64 - bs) != 0;
            let mut v = vec![0 as Limb; src.len() - ls];
            mpn::rshift(&mut v, &src[ls..], bs);
            v
        };
        let kn = mpn::normalized_len(&kept);
        if accumulate {
            let c = mpn::add_in_place(w, &kept[..kn]);
            debug_assert_eq!(c, 0);
        } else {
            w[..kn].copy_from_slice(&kept[..kn]);
        }
    }
    sticky
}

/// Integer square root of a limb vector: returns (floor(sqrt(a)), is_exact).
pub fn isqrt(a: &[Limb]) -> (Vec<Limb>, bool) {
    let an = mpn::normalized_len(a);
    let a = &a[..an];
    if an == 0 {
        return (Vec::new(), true);
    }
    if an <= 2 {
        let v = if an == 1 { a[0] as u128 } else { a[0] as u128 | (a[1] as u128) << 64 };
        let s = isqrt_u128(v);
        let exact = s * s == v;
        let mut out = vec![s as Limb, (s >> 64) as Limb];
        out.truncate(mpn::normalized_len(&out));
        return (out, exact);
    }

    // Newton from above: x_{k+1} = (x_k + a/x_k) / 2, starting from a value
    // guaranteed >= sqrt(a); the iteration is monotonically decreasing and
    // terminates at floor(sqrt(a)) (Cohen, Alg. 1.7.1).
    let bits = mpn::bit_len(a);
    // Initial guess: 2^ceil(bits/2) (top-heavy but converges quadratically).
    let gb = bits.div_ceil(2) + 1;
    let mut x = vec![0 as Limb; (gb / 64) as usize + 1];
    let nx = x.len();
    x[nx - 1] = 0;
    x[(gb / 64) as usize] = 1 << (gb % 64);
    x.truncate(mpn::normalized_len(&x));

    loop {
        let (q, _r) = mpn::divrem(a, &x);
        // y = (x + q) / 2
        let n = x.len().max(q.len()) + 1;
        let mut y = vec![0 as Limb; n];
        y[..x.len()].copy_from_slice(&x);
        let c = mpn::add_in_place(&mut y, &q);
        debug_assert_eq!(c, 0);
        let src = y.clone();
        mpn::rshift(&mut y, &src, 1);
        y.truncate(mpn::normalized_len(&y));
        if mpn::cmp(&y, &x) != Ordering::Less {
            break;
        }
        x = y;
    }
    // x = floor(sqrt(a)); check exactness.
    let mut sq = vec![0 as Limb; 2 * x.len()];
    mpn::sqr(&mut sq, &x);
    sq.truncate(mpn::normalized_len(&sq));
    let exact = mpn::cmp(&sq, a) == Ordering::Equal;
    (x, exact)
}

fn isqrt_u128(v: u128) -> u128 {
    if v == 0 {
        return 0;
    }
    let mut x = 1u128 << (v.ilog2() / 2 + 1);
    loop {
        let y = (x + v / x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// Convert a nonnegative limb integer to decimal.
fn limbs_to_decimal(a: &[Limb]) -> String {
    let mut v = a[..mpn::normalized_len(a)].to_vec();
    if v.is_empty() {
        return "0".to_string();
    }
    let mut chunks: Vec<u64> = Vec::new();
    while !v.is_empty() {
        let src = v.clone();
        let rem = mpn::divrem_1(&mut v, &src, 10_000_000_000_000_000_000);
        v.truncate(mpn::normalized_len(&v));
        chunks.push(rem);
    }
    let mut s = chunks.last().unwrap().to_string();
    for c in chunks.iter().rev().skip(1) {
        s.push_str(&format!("{c:019}"));
    }
    s
}

impl core::fmt::Display for Float {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_decimal(20))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Rng;

    fn rand_float(rng: &mut Rng, max_limbs: usize, max_exp: i64) -> Float {
        let n = 1 + rng.below(max_limbs as u64) as usize;
        let mut limbs: Vec<Limb> = (0..n).map(|_| rng.next()).collect();
        *limbs.last_mut().unwrap() |= 1 << 63;
        if limbs[0] == 0 {
            limbs[0] = 1;
        }
        // Strip low zero limbs to satisfy the minimality invariant.
        let lowz = limbs.iter().position(|&l| l != 0).unwrap();
        let limbs = limbs[lowz..].to_vec();
        Float {
            neg: rng.below(2) == 1,
            exp: rng.range_i64(-max_exp, max_exp),
            limbs,
        }
    }

    #[test]
    fn from_to_small_ints() {
        for v in [0i64, 1, -1, 2, 7, -1000, i64::MAX, i64::MIN + 1] {
            let f = Float::from_i64(v);
            assert_eq!(f.to_f64(), v as f64, "roundtrip {v}");
        }
    }

    #[test]
    fn add_matches_f64() {
        let mut rng = Rng::new(10);
        for _ in 0..2000 {
            let a = (rng.next() as i32 as f64) * 2f64.powi((rng.below(40) as i32) - 20);
            let b = (rng.next() as i32 as f64) * 2f64.powi((rng.below(40) as i32) - 20);
            let fa = Float::from_f64(a);
            let fb = Float::from_f64(b);
            // f64 arithmetic on these inputs can be inexact; use high prec and
            // compare to f64 sum only when that sum is exact (integers scaled).
            let (sum, _) = fa.add(&fb, 120, Round::Nearest);
            let expect = a + b;
            // 120 bits is enough to represent the exact sum of two doubles
            // whose exponents differ by < 64.
            if (a + b).is_finite() {
                let diff = (sum.to_f64() - expect).abs();
                let tol = expect.abs() * 1e-15 + 1e-300;
                assert!(diff <= tol, "a={a} b={b} got={} want={expect}", sum.to_f64());
            }
        }
    }

    #[test]
    fn add_exact_small() {
        let two = Float::from_i64(2);
        let three = Float::from_i64(3);
        let (five, inexact) = two.add(&three, 64, Round::Nearest);
        assert!(!inexact);
        assert_eq!(five.cmp(&Float::from_i64(5)), Ordering::Equal);

        let (neg1, inexact) = two.sub(&three, 64, Round::Nearest);
        assert!(!inexact);
        assert_eq!(neg1.cmp(&Float::from_i64(-1)), Ordering::Equal);
    }

    #[test]
    fn cancellation_exact() {
        // (2^100 + 1) - 2^100 must give exactly 1 at any precision.
        let big = Float::from_i64(1).mul_2exp(100);
        let (x, in1) = big.add(&Float::from_i64(1), 200, Round::Nearest);
        assert!(!in1);
        let (one, in2) = x.sub(&big, 8, Round::Nearest);
        assert!(!in2);
        assert_eq!(one.cmp(&Float::from_i64(1)), Ordering::Equal);
    }

    #[test]
    fn directed_rounding_brackets_truth() {
        // For random a, b: round-down result <= round-up result, and
        // nearest is between them.
        let mut rng = Rng::new(11);
        for _ in 0..500 {
            let a = rand_float(&mut rng, 4, 100);
            let b = rand_float(&mut rng, 4, 100);
            let prec = 2 + rng.below(150) as u32;
            for op in 0..2 {
                let (lo, _) = if op == 0 {
                    a.add(&b, prec, Round::Floor)
                } else {
                    a.mul(&b, prec, Round::Floor)
                };
                let (hi, _) = if op == 0 {
                    a.add(&b, prec, Round::Ceil)
                } else {
                    a.mul(&b, prec, Round::Ceil)
                };
                let (mid, _) = if op == 0 {
                    a.add(&b, prec, Round::Nearest)
                } else {
                    a.mul(&b, prec, Round::Nearest)
                };
                assert!(lo.cmp(&hi) != Ordering::Greater);
                assert!(lo.cmp(&mid) != Ordering::Greater);
                assert!(mid.cmp(&hi) != Ordering::Greater);
                // exact high-precision result must lie in [lo, hi]
                let (exact, ie) = if op == 0 {
                    a.add(&b, 64 * 8 + 128, Round::Nearest)
                } else {
                    a.mul(&b, 64 * 8 + 128, Round::Nearest)
                };
                assert!(!ie, "high-precision reference should be exact here");
                assert!(lo.cmp(&exact) != Ordering::Greater);
                assert!(exact.cmp(&hi) != Ordering::Greater);
            }
        }
    }

    #[test]
    fn mul_matches_u128() {
        let mut rng = Rng::new(12);
        for _ in 0..1000 {
            let a = rng.next() >> rng.below(32);
            let b = rng.next() >> rng.below(32);
            let fa = Float::from_u64(a);
            let fb = Float::from_u64(b);
            let (p, inexact) = fa.mul(&fb, 128, Round::Nearest);
            assert!(!inexact);
            let expect = a as u128 * b as u128;
            // Reconstruct: p = 0.L × 2^exp
            let s = p.to_decimal(40);
            assert_eq!(s, expect.to_string());
        }
    }

    #[test]
    fn div_mul_roundtrip() {
        let mut rng = Rng::new(13);
        for _ in 0..300 {
            let a = rand_float(&mut rng, 3, 60);
            let b = rand_float(&mut rng, 3, 60);
            if b.is_zero() {
                continue;
            }
            let prec = 320;
            let (q, _) = a.div(&b, prec, Round::Nearest);
            let (back, _) = q.mul(&b, prec, Round::Nearest);
            // |back - a| <= 4 ulp-ish of a
            let (diff, _) = back.sub(&a, prec, Round::Up);
            if !diff.is_zero() {
                let tol_exp = a.exponent().unwrap() - prec as i64 + 4;
                assert!(
                    diff.exponent().unwrap() <= tol_exp,
                    "residual too large: {} vs 2^{tol_exp}",
                    diff.to_f64()
                );
            }
        }
    }

    #[test]
    fn div_exact_small() {
        let six = Float::from_i64(6);
        let three = Float::from_i64(3);
        let (two, inexact) = six.div(&three, 64, Round::Nearest);
        assert!(!inexact);
        assert_eq!(two.cmp(&Float::from_i64(2)), Ordering::Equal);
        // 1/3 must be inexact in binary.
        let (_, inexact) = Float::from_i64(1).div(&three, 64, Round::Nearest);
        assert!(inexact);
    }

    #[test]
    fn sqrt_squares() {
        for v in [1u64, 4, 9, 144, 1 << 40, 10_000_000_000_000_000_002 /* not square */] {
            let f = Float::from_u64(v);
            let (s, inexact) = f.sqrt(200, Round::Nearest);
            let (sq, _) = s.mul(&s, 200, Round::Nearest);
            let (diff, _) = sq.sub(&f, 200, Round::Up);
            if (v as f64).sqrt().fract() == 0.0 && v < 1 << 52 {
                assert!(!inexact, "sqrt({v}) should be exact");
                assert!(diff.is_zero());
            } else if !diff.is_zero() {
                assert!(diff.exponent().unwrap() < f.exponent().unwrap() - 190);
            }
        }
    }

    #[test]
    fn sqrt_directed_brackets() {
        let mut rng = Rng::new(14);
        for _ in 0..200 {
            let mut a = rand_float(&mut rng, 3, 80);
            a.neg = false;
            if a.is_zero() {
                continue;
            }
            let prec = 2 + rng.below(200) as u32;
            let (lo, _) = a.sqrt(prec, Round::Floor);
            let (hi, _) = a.sqrt(prec, Round::Ceil);
            assert!(lo.cmp(&hi) != Ordering::Greater);
            // lo^2 <= a <= hi^2 (with exact squaring at high precision)
            let (lo2, _) = lo.mul(&lo, 2048, Round::Nearest);
            let (hi2, _) = hi.mul(&hi, 2048, Round::Nearest);
            assert!(lo2.cmp(&a) != Ordering::Greater, "lo^2 > a");
            assert!(hi2.cmp(&a) != Ordering::Less, "hi^2 < a");
        }
    }

    #[test]
    fn decimal_output() {
        assert_eq!(Float::from_i64(0).to_decimal(10), "0");
        assert_eq!(Float::from_i64(42).to_decimal(10), "42");
        assert_eq!(Float::from_i64(-42).to_decimal(10), "-42");
        // 1/4 = 0.25
        let (q, _) = Float::from_i64(1).div(&Float::from_i64(4), 64, Round::Nearest);
        assert_eq!(q.to_decimal(6), "0.25");
        // 1/3 = 0.3333...
        let (q, _) = Float::from_i64(1).div(&Float::from_i64(3), 256, Round::Nearest);
        assert_eq!(q.to_decimal(11), "0.33333333333");
    }

    #[test]
    fn rounding_to_nearest_ties_even() {
        // 0b10.1 = 2.5 rounded to 2 bits: tie -> even -> 2 (0b10).
        let f = Float::from_f64(2.5);
        let (r, inexact) = f.round(2, Round::Nearest);
        assert!(inexact);
        assert_eq!(r.cmp(&Float::from_i64(2)), Ordering::Equal);
        // 3.5 = 0b11.1 rounded to 2 bits: tie -> even -> 4 (0b100).
        let f = Float::from_f64(3.5);
        let (r, _) = f.round(2, Round::Nearest);
        assert_eq!(r.cmp(&Float::from_i64(4)), Ordering::Equal);
    }
}
