//! Low-level limb-vector arithmetic, modeled on GMP's `mpn` layer.
//!
//! Conventions (shared with GMP):
//! - A number is a little-endian slice of [`Limb`]s (least significant first).
//! - Callers guarantee output slices are large enough; functions return
//!   carries/borrows rather than reallocating.
//! - "Normalized" means the most significant limb is nonzero (empty slice = 0).
//!
//! Everything here is safe Rust. The `u128` widening operations compile to
//! single `MUL`/`ADC` chains on x86-64 and AArch64; we verified the codegen
//! rather than reaching for inline asm (see README §Performance).

pub type Limb = u64;

/// Bits per limb.
pub const LIMB_BITS: u32 = 64;

/// Karatsuba crossover, in limbs. Tuned on Zen 4; see `benches/` and
/// `scripts/tune.ps1`.
pub const KARATSUBA_THRESHOLD: usize = 24;

#[inline(always)]
fn adc(a: Limb, b: Limb, carry: Limb) -> (Limb, Limb) {
    let t = a as u128 + b as u128 + carry as u128;
    (t as Limb, (t >> LIMB_BITS) as Limb)
}

#[inline(always)]
fn sbb(a: Limb, b: Limb, borrow: Limb) -> (Limb, Limb) {
    let t = (a as u128).wrapping_sub(b as u128).wrapping_sub(borrow as u128);
    (t as Limb, (t >> LIMB_BITS) as Limb & 1)
}

/// Full 64x64 -> 128 multiply, returned as (low, high).
#[inline(always)]
fn umul(a: Limb, b: Limb) -> (Limb, Limb) {
    let t = a as u128 * b as u128;
    (t as Limb, (t >> LIMB_BITS) as Limb)
}

/// Strip high zero limbs: `&a[..len]` is the normalized value.
#[inline]
pub fn normalized_len(a: &[Limb]) -> usize {
    let mut n = a.len();
    while n > 0 && a[n - 1] == 0 {
        n -= 1;
    }
    n
}

/// r = a + b for equal-length operands. Returns the carry (0 or 1).
/// `r` may alias `a` or `b`.
pub fn add_n(r: &mut [Limb], a: &[Limb], b: &[Limb]) -> Limb {
    debug_assert!(a.len() == b.len() && r.len() == a.len());
    let mut carry = 0;
    for i in 0..a.len() {
        let (s, c) = adc(a[i], b[i], carry);
        r[i] = s;
        carry = c;
    }
    carry
}

/// r = a + b where a.len() >= b.len(); r.len() == a.len(). Returns carry.
pub fn add(r: &mut [Limb], a: &[Limb], b: &[Limb]) -> Limb {
    debug_assert!(a.len() >= b.len() && r.len() == a.len());
    let (bl, ah) = (b.len(), a.len());
    let carry = add_n(&mut r[..bl], &a[..bl], b);
    add_1(&mut r[bl..ah], &a[bl..ah], carry)
}

/// r = a + b (single limb). Returns carry. `r` may alias `a`.
pub fn add_1(r: &mut [Limb], a: &[Limb], b: Limb) -> Limb {
    debug_assert!(r.len() == a.len());
    let mut carry = b;
    for i in 0..a.len() {
        let (s, c) = adc(a[i], 0, carry);
        r[i] = s;
        carry = c;
        if carry == 0 && !core::ptr::eq(r.as_ptr(), a.as_ptr()) {
            r[i + 1..].copy_from_slice(&a[i + 1..]);
            return 0;
        }
    }
    carry
}

/// r = a - b for equal-length operands. Returns the borrow (0 or 1).
/// `r` may alias `a` or `b`.
pub fn sub_n(r: &mut [Limb], a: &[Limb], b: &[Limb]) -> Limb {
    debug_assert!(a.len() == b.len() && r.len() == a.len());
    let mut borrow = 0;
    for i in 0..a.len() {
        let (d, bw) = sbb(a[i], b[i], borrow);
        r[i] = d;
        borrow = bw;
    }
    borrow
}

/// r = a - b where a.len() >= b.len(). Returns borrow.
pub fn sub(r: &mut [Limb], a: &[Limb], b: &[Limb]) -> Limb {
    debug_assert!(a.len() >= b.len() && r.len() == a.len());
    let bl = b.len();
    let borrow = sub_n(&mut r[..bl], &a[..bl], b);
    sub_1(&mut r[bl..], &a[bl..], borrow)
}

/// r = a - b (single limb). Returns borrow. `r` may alias `a`.
pub fn sub_1(r: &mut [Limb], a: &[Limb], b: Limb) -> Limb {
    debug_assert!(r.len() == a.len());
    let mut borrow = b;
    for i in 0..a.len() {
        let (d, bw) = sbb(a[i], 0, borrow);
        r[i] = d;
        borrow = bw;
        if borrow == 0 && !core::ptr::eq(r.as_ptr(), a.as_ptr()) {
            r[i + 1..].copy_from_slice(&a[i + 1..]);
            return 0;
        }
    }
    borrow
}

/// r += b in place, r.len() >= b.len(). Returns the carry out of the top.
pub fn add_in_place(r: &mut [Limb], b: &[Limb]) -> Limb {
    debug_assert!(r.len() >= b.len());
    let mut carry = 0;
    for i in 0..b.len() {
        let (s, c) = adc(r[i], b[i], carry);
        r[i] = s;
        carry = c;
    }
    let mut i = b.len();
    while carry != 0 && i < r.len() {
        let (s, c) = adc(r[i], 0, carry);
        r[i] = s;
        carry = c;
        i += 1;
    }
    carry
}

/// r -= b in place, r.len() >= b.len(). Returns the borrow out of the top.
pub fn sub_in_place(r: &mut [Limb], b: &[Limb]) -> Limb {
    debug_assert!(r.len() >= b.len());
    let mut borrow = 0;
    for i in 0..b.len() {
        let (d, bw) = sbb(r[i], b[i], borrow);
        r[i] = d;
        borrow = bw;
    }
    let mut i = b.len();
    while borrow != 0 && i < r.len() {
        let (d, bw) = sbb(r[i], 0, borrow);
        r[i] = d;
        borrow = bw;
        i += 1;
    }
    borrow
}

/// Compare equal-length limb vectors.
pub fn cmp_n(a: &[Limb], b: &[Limb]) -> core::cmp::Ordering {
    debug_assert_eq!(a.len(), b.len());
    for i in (0..a.len()).rev() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    core::cmp::Ordering::Equal
}

/// Compare two normalized limb vectors of possibly different lengths.
pub fn cmp(a: &[Limb], b: &[Limb]) -> core::cmp::Ordering {
    debug_assert_eq!(a.len(), normalized_len(a));
    debug_assert_eq!(b.len(), normalized_len(b));
    a.len().cmp(&b.len()).then_with(|| cmp_n(a, b))
}

/// r = a * b (single limb). Returns the high carry limb. `r` may alias `a`.
pub fn mul_1(r: &mut [Limb], a: &[Limb], b: Limb) -> Limb {
    debug_assert!(r.len() == a.len());
    let mut carry = 0;
    for i in 0..a.len() {
        let (lo, hi) = umul(a[i], b);
        let (lo, c) = adc(lo, carry, 0);
        r[i] = lo;
        carry = hi + c;
    }
    carry
}

/// r += a * b (single limb). Returns the carry limb out of the top.
pub fn addmul_1(r: &mut [Limb], a: &[Limb], b: Limb) -> Limb {
    debug_assert!(r.len() >= a.len());
    let mut carry = 0;
    for i in 0..a.len() {
        let (lo, hi) = umul(a[i], b);
        let (lo, c1) = adc(lo, r[i], 0);
        let (lo, c2) = adc(lo, carry, 0);
        r[i] = lo;
        carry = hi + c1 + c2;
    }
    carry
}

/// r -= a * b (single limb). Returns the borrow limb out of the top.
pub fn submul_1(r: &mut [Limb], a: &[Limb], b: Limb) -> Limb {
    debug_assert!(r.len() >= a.len());
    let mut borrow = 0;
    for i in 0..a.len() {
        let (lo, hi) = umul(a[i], b);
        let (d, bw1) = sbb(r[i], lo, 0);
        let (d, bw2) = sbb(d, borrow, 0);
        r[i] = d;
        borrow = hi + bw1 + bw2;
    }
    borrow
}

/// Schoolbook multiplication: r = a * b, r.len() == a.len() + b.len().
/// `r` must not alias the inputs.
pub fn mul_basecase(r: &mut [Limb], a: &[Limb], b: &[Limb]) {
    debug_assert_eq!(r.len(), a.len() + b.len());
    if a.is_empty() || b.is_empty() {
        r.fill(0);
        return;
    }
    r[a.len()] = mul_1(&mut r[..a.len()], a, b[0]);
    r[a.len() + 1..].fill(0);
    for j in 1..b.len() {
        let c = addmul_1(&mut r[j..j + a.len()], a, b[j]);
        r[j + a.len()] = c;
    }
}

/// Full product r = a * b with r.len() == a.len() + b.len().
/// Uses Karatsuba above [`KARATSUBA_THRESHOLD`]. `r` must not alias inputs.
pub fn mul(r: &mut [Limb], a: &[Limb], b: &[Limb]) {
    debug_assert_eq!(r.len(), a.len() + b.len());
    let (a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    if b.len() < KARATSUBA_THRESHOLD {
        mul_basecase(r, a, b);
    } else if b.len() * 2 <= a.len() {
        // Unbalanced: chop `a` into b.len()-sized chunks.
        mul_unbalanced(r, a, b);
    } else {
        mul_karatsuba(r, a, b);
    }
}

/// Handle a.len() much larger than b.len() by chunking `a`.
fn mul_unbalanced(r: &mut [Limb], a: &[Limb], b: &[Limb]) {
    let n = b.len();
    r.fill(0);
    let mut tmp = vec![0 as Limb; 2 * n];
    let mut i = 0;
    while i < a.len() {
        let chunk = core::cmp::min(n, a.len() - i);
        let t = &mut tmp[..chunk + n];
        mul(t, &a[i..i + chunk], b);
        let carry = add_in_place(&mut r[i..], t);
        debug_assert_eq!(carry, 0);
        i += chunk;
    }
}

/// Karatsuba: split at n = ceil(a.len()/2) >= b.len()/2.
fn mul_karatsuba(r: &mut [Limb], a: &[Limb], b: &[Limb]) {
    debug_assert!(a.len() >= b.len());
    let n = a.len().div_ceil(2);
    let (a0, a1) = a.split_at(n);
    let (b0, b1) = b.split_at(core::cmp::min(n, b.len()));
    let a0n = normalized_len(a0);
    let b0n = normalized_len(b0);

    // z0 = a0*b0, z2 = a1*b1
    let mut z0 = vec![0 as Limb; a0n + b0n];
    mul(&mut z0, &a0[..a0n], &b0[..b0n]);
    let mut z2 = vec![0 as Limb; a1.len() + b1.len()];
    if !a1.is_empty() && !b1.is_empty() {
        mul(&mut z2, a1, b1);
    }

    // s_a = a0 + a1, s_b = b0 + b1 (may carry one extra limb each)
    let mut sa = vec![0 as Limb; n + 1];
    sa[n] = add(&mut sa[..n], a0, a1);
    let san = normalized_len(&sa);
    let mut sb = vec![0 as Limb; n + 1];
    {
        let (lo, _) = sb.split_at_mut(n);
        let m = b0.len();
        lo[..m].copy_from_slice(b0);
        let carry = if b1.is_empty() { 0 } else { add(&mut lo[..m], b0, b1) };
        sb[n] = carry;
    }
    let sbn = normalized_len(&sb);

    // z1 = s_a * s_b - z0 - z2
    let mut z1 = vec![0 as Limb; san + sbn];
    mul(&mut z1, &sa[..san], &sb[..sbn]);
    let bw = sub_in_place(&mut z1, &z0);
    debug_assert_eq!(bw, 0);
    let z2n = normalized_len(&z2);
    let bw = sub_in_place(&mut z1, &z2[..z2n]);
    debug_assert_eq!(bw, 0);

    // r = z0 + z1 << (64 n) + z2 << (128 n)
    r.fill(0);
    r[..z0.len()].copy_from_slice(&z0);
    r[2 * n..2 * n + z2.len()].copy_from_slice(&z2);
    let z1n = normalized_len(&z1);
    let carry = add_in_place(&mut r[n..], &z1[..z1n]);
    debug_assert_eq!(carry, 0);
}

/// Squaring convenience: r = a^2.
pub fn sqr(r: &mut [Limb], a: &[Limb]) {
    mul(r, a, a);
}

/// Left shift by `cnt` bits (0 < cnt < 64): r = a << cnt.
/// Returns the bits shifted out of the top. `r` may alias `a`.
pub fn lshift(r: &mut [Limb], a: &[Limb], cnt: u32) -> Limb {
    debug_assert!(cnt > 0 && cnt < LIMB_BITS && r.len() == a.len());
    let mut out = 0;
    for i in 0..a.len() {
        let v = a[i];
        r[i] = (v << cnt) | out;
        out = v >> (LIMB_BITS - cnt);
    }
    out
}

/// Right shift by `cnt` bits (0 < cnt < 64): r = a >> cnt.
/// Returns the bits shifted out of the bottom (in the high bits of the limb).
pub fn rshift(r: &mut [Limb], a: &[Limb], cnt: u32) -> Limb {
    debug_assert!(cnt > 0 && cnt < LIMB_BITS && r.len() == a.len());
    let mut out = 0;
    for i in (0..a.len()).rev() {
        let v = a[i];
        r[i] = (v >> cnt) | out;
        out = v << (LIMB_BITS - cnt);
    }
    out
}

/// q = a / d, returns a % d. `q` may alias `a`. Single-limb divisor.
pub fn divrem_1(q: &mut [Limb], a: &[Limb], d: Limb) -> Limb {
    debug_assert!(d != 0 && q.len() == a.len());
    let mut rem: u128 = 0;
    for i in (0..a.len()).rev() {
        let cur = (rem << LIMB_BITS) | a[i] as u128;
        q[i] = (cur / d as u128) as Limb;
        rem = cur % d as u128;
    }
    rem as Limb
}

/// Knuth Algorithm D: divide `num` by `den` (both normalized, den.len() >= 2,
/// num.len() >= den.len()). On return `q` holds the quotient
/// (num.len() - den.len() + 1 limbs) and `num` holds the remainder in its low
/// den.len() limbs. `den` must have its top bit set (caller pre-shifts).
fn div_schoolbook_normalized(q: &mut [Limb], num: &mut [Limb], den: &[Limb]) {
    let n = den.len();
    let m = num.len() - n;
    debug_assert!(den[n - 1] >> (LIMB_BITS - 1) == 1);
    debug_assert!(q.len() == m + 1);

    // q[m]: top quotient limb (0 or 1 after normalization).
    if cmp_n(&num[m..], den) != core::cmp::Ordering::Less {
        let src = num[m..].to_vec();
        sub_n(&mut num[m..], &src, den);
        q[m] = 1;
    } else {
        q[m] = 0;
    }

    let dh = den[n - 1];
    let dl = den[n - 2];
    for j in (0..m).rev() {
        let n2 = ((num[j + n] as u128) << LIMB_BITS) | num[j + n - 1] as u128;
        let mut qhat: u128;
        let mut rhat: u128;
        if num[j + n] == dh {
            qhat = (1u128 << LIMB_BITS) - 1;
            rhat = n2 - qhat * dh as u128;
        } else {
            qhat = n2 / dh as u128;
            rhat = n2 % dh as u128;
        }
        // Refine qhat using the second divisor limb.
        while rhat >> LIMB_BITS == 0
            && qhat * dl as u128 > ((rhat << LIMB_BITS) | num[j + n - 2] as u128)
        {
            qhat -= 1;
            rhat += dh as u128;
        }
        // num[j .. j+n+1] -= qhat * den
        let borrow = submul_1(&mut num[j..j + n], den, qhat as Limb);
        let (top, bw) = sbb(num[j + n], borrow, 0);
        num[j + n] = top;
        if bw != 0 {
            // qhat was one too large: add back.
            qhat -= 1;
            let carry = {
                let src = num[j..j + n].to_vec();
                add_n(&mut num[j..j + n], &src, den)
            };
            num[j + n] = num[j + n].wrapping_add(carry);
        }
        q[j] = qhat as Limb;
    }
}

/// Divide normalized `a` by normalized nonzero `d`.
/// Returns (quotient, remainder), both normalized.
pub fn divrem(a: &[Limb], d: &[Limb]) -> (Vec<Limb>, Vec<Limb>) {
    let an = normalized_len(a);
    let dn = normalized_len(d);
    assert!(dn > 0, "division by zero");
    let a = &a[..an];
    let d = &d[..dn];
    if an < dn {
        return (Vec::new(), a.to_vec());
    }
    if dn == 1 {
        let mut q = vec![0; an];
        let r = divrem_1(&mut q, a, d[0]);
        q.truncate(normalized_len(&q));
        return (q, if r == 0 { Vec::new() } else { vec![r] });
    }
    // Normalize: shift so the divisor's top bit is set.
    let shift = d[dn - 1].leading_zeros();
    let mut den = d.to_vec();
    let mut num = vec![0; an + 1];
    if shift > 0 {
        lshift(&mut den, d, shift);
        num[an] = lshift(&mut num[..an], a, shift);
    } else {
        num[..an].copy_from_slice(a);
    }
    let mut q = vec![0; an - dn + 2];
    div_schoolbook_normalized(&mut q, &mut num, &den);
    let mut rem = num[..dn].to_vec();
    if shift > 0 {
        let src = rem.clone();
        rshift(&mut rem, &src, shift);
    }
    q.truncate(normalized_len(&q));
    rem.truncate(normalized_len(&rem));
    (q, rem)
}

/// Total bit length of a normalized limb vector.
pub fn bit_len(a: &[Limb]) -> u64 {
    let n = normalized_len(a);
    if n == 0 {
        0
    } else {
        n as u64 * LIMB_BITS as u64 - a[n - 1].leading_zeros() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Rng;

    fn to_u128(a: &[Limb]) -> u128 {
        match a.len() {
            0 => 0,
            1 => a[0] as u128,
            _ => a[0] as u128 | (a[1] as u128) << 64,
        }
    }

    #[test]
    fn add_sub_roundtrip() {
        let mut rng = Rng::new(1);
        for _ in 0..1000 {
            let n = 1 + (rng.next() as usize % 8);
            let a: Vec<Limb> = (0..n).map(|_| rng.next()).collect();
            let b: Vec<Limb> = (0..n).map(|_| rng.next()).collect();
            let mut s = vec![0; n];
            let carry = add_n(&mut s, &a, &b);
            let mut back = vec![0; n];
            let borrow = sub_n(&mut back, &s, &b);
            assert_eq!(back, a);
            assert_eq!(carry, borrow);
        }
    }

    #[test]
    fn mul_1_matches_u128() {
        let mut rng = Rng::new(2);
        for _ in 0..1000 {
            let a = [rng.next()];
            let b = rng.next();
            let mut r = [0];
            let hi = mul_1(&mut r, &a, b);
            let expect = a[0] as u128 * b as u128;
            assert_eq!(to_u128(&[r[0], hi]), expect);
        }
    }

    #[test]
    fn mul_matches_basecase() {
        let mut rng = Rng::new(3);
        for _ in 0..40 {
            let an = 1 + (rng.next() as usize % 90);
            let bn = 1 + (rng.next() as usize % 90);
            let a: Vec<Limb> = (0..an).map(|_| rng.next()).collect();
            let b: Vec<Limb> = (0..bn).map(|_| rng.next()).collect();
            let mut r1 = vec![0; an + bn];
            let mut r2 = vec![0; an + bn];
            mul_basecase(&mut r1, &a, &b);
            mul(&mut r2, &a, &b);
            assert_eq!(r1, r2, "an={an} bn={bn}");
        }
    }

    #[test]
    fn karatsuba_forced() {
        // Sizes straddling and well above the threshold, including unbalanced.
        let mut rng = Rng::new(4);
        for &(an, bn) in &[(24, 24), (25, 24), (48, 31), (100, 25), (200, 26), (37, 129)] {
            let a: Vec<Limb> = (0..an).map(|_| rng.next()).collect();
            let b: Vec<Limb> = (0..bn).map(|_| rng.next()).collect();
            let mut r1 = vec![0; an + bn];
            let mut r2 = vec![0; an + bn];
            mul_basecase(&mut r1, &a, &b);
            mul(&mut r2, &a, &b);
            assert_eq!(r1, r2, "an={an} bn={bn}");
        }
    }

    #[test]
    fn divrem_reconstructs() {
        let mut rng = Rng::new(5);
        for _ in 0..300 {
            let an = 1 + (rng.next() as usize % 12);
            let dn = 1 + (rng.next() as usize % an.max(1));
            let mut a: Vec<Limb> = (0..an).map(|_| rng.next()).collect();
            let mut d: Vec<Limb> = (0..dn).map(|_| rng.next()).collect();
            a.truncate(normalized_len(&a));
            d.truncate(normalized_len(&d));
            if d.is_empty() {
                continue;
            }
            let (q, r) = divrem(&a, &d);
            // Check a == q*d + r and r < d.
            assert!(r.is_empty() || cmp(&r, &d) == core::cmp::Ordering::Less);
            let mut qd = vec![0; q.len() + d.len()];
            mul(&mut qd, &q, &d);
            let mut sum = qd.clone();
            if !r.is_empty() {
                let src = sum.clone();
                let c = add(&mut sum, &src, &r);
                assert_eq!(c, 0);
            }
            sum.truncate(normalized_len(&sum));
            assert_eq!(sum, a);
        }
    }

    #[test]
    fn shifts_roundtrip() {
        let mut rng = Rng::new(6);
        for _ in 0..200 {
            let n = 1 + (rng.next() as usize % 6);
            let a: Vec<Limb> = (0..n).map(|_| rng.next()).collect();
            let cnt = 1 + (rng.next() as u32 % 63);
            let mut l = vec![0; n];
            let hi = lshift(&mut l, &a, cnt);
            let mut back = vec![0; n];
            let lo = rshift(&mut back, &l, cnt);
            let _ = lo;
            // back should equal a with the top cnt bits cleared, plus hi holds them.
            let mut expect = a.clone();
            let top = &mut expect[n - 1];
            let kept = *top << cnt >> cnt;
            let lost = *top >> (64 - cnt);
            *top = kept;
            assert_eq!(back, expect);
            assert_eq!(hi, lost);
        }
    }
}
