//! Differential testing against Arb (FLINT ≥ 3).
//!
//! For a battery of inputs and precisions, evaluate each function with both
//! rigor and Arb and require the *certified digit prefixes* to agree.
//! `arb_get_str(n, 0)` prints only digits that are correct given the radius;
//! we compare it with rigor's midpoint digits truncated to the certified
//! length, allowing 2 boundary digits of slack on each side.
//!
//! Exit code 0 = all comparisons agree; nonzero = genuine disagreement,
//! which given both libraries' inclusion contracts means a bug in one.

use rigor::ball::Ball;
use rigor::{binsplit, elementary, gamma, zeta};
use std::ffi::CStr;
use std::os::raw::{c_char, c_longlong, c_void};

/// Opaque, over-sized stand-in for arb_struct (48 bytes on x86-64;
/// we allocate 128 to be layout-agnostic — always used behind a pointer).
#[repr(C)]
struct ArbT {
    _data: [u64; 16],
}

impl ArbT {
    fn new() -> Self {
        ArbT { _data: [0; 16] }
    }
}

type Slong = c_longlong;

#[link(name = "flint")]
extern "C" {
    fn arb_init(x: *mut ArbT);
    fn arb_clear(x: *mut ArbT);
    fn arb_set_d(x: *mut ArbT, v: f64);
    fn arb_exp(z: *mut ArbT, x: *const ArbT, prec: Slong);
    fn arb_log(z: *mut ArbT, x: *const ArbT, prec: Slong);
    fn arb_sin(z: *mut ArbT, x: *const ArbT, prec: Slong);
    fn arb_cos(z: *mut ArbT, x: *const ArbT, prec: Slong);
    fn arb_atan(z: *mut ArbT, x: *const ArbT, prec: Slong);
    fn arb_gamma(z: *mut ArbT, x: *const ArbT, prec: Slong);
    fn arb_zeta(z: *mut ArbT, s: *const ArbT, prec: Slong);
    fn arb_const_pi(z: *mut ArbT, prec: Slong);
    fn arb_get_str(x: *const ArbT, n: Slong, flags: u64) -> *mut c_char;
    fn flint_free(p: *mut c_void);
}

/// Digits-only view of a decimal string (sign and separators stripped).
fn digit_prefix(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_digit()).collect()
}

/// Extract the printed midpoint digits from an arb string like
/// "[3.141592653 +/- 5.61e-10]" or a plain "3.141592653".
fn arb_digits(s: &str) -> String {
    let core = s
        .trim_start_matches('[')
        .split("+/-")
        .next()
        .unwrap_or("")
        .trim();
    digit_prefix(core.split(['e', 'E']).next().unwrap_or(""))
}

fn compare(name: &str, ours: &Ball, arb: &ArbT, digits: usize) -> bool {
    let ours_str = ours.mid().to_decimal(digits);
    let ours_digits = digit_prefix(&ours_str);
    let arb_str = unsafe {
        let p = arb_get_str(arb, digits as Slong, 0);
        let s = CStr::from_ptr(p).to_string_lossy().into_owned();
        flint_free(p as *mut c_void);
        s
    };
    let theirs = arb_digits(&arb_str);
    // Compare the common certified prefix, with 2 digits of slack for
    // decimal-boundary rounding on either side.
    let n = ours_digits.len().min(theirs.len()).saturating_sub(2);
    let ok = n >= digits.saturating_sub(6) && ours_digits[..n] == theirs[..n];
    if !ok {
        eprintln!("MISMATCH {name}:\n  rigor: {ours_str}\n  arb:   {arb_str}");
    } else {
        println!("ok {name} ({n} digits agree)");
    }
    ok
}

fn main() {
    let mut failures = 0u32;
    let cases: &[(&str, f64)] = &[
        ("x=1.5", 1.5),
        ("x=0.125", 0.125),
        ("x=3.75", 3.75),
        ("x=10.0625", 10.0625),
        ("x=0.9990234375", 0.9990234375),
    ];
    for &(label, xv) in cases {
        for &prec in &[128u32, 1024, 8192] {
            let digits = (prec as f64 / 3.33) as usize - 8;
            let x = Ball::from_f64(xv);
            unsafe {
                let mut ax = ArbT::new();
                let mut az = ArbT::new();
                arb_init(&mut ax);
                arb_init(&mut az);
                arb_set_d(&mut ax, xv);

                type ArbFn = unsafe extern "C" fn(*mut ArbT, *const ArbT, Slong);
                let fns: &[(&str, ArbFn, fn(&Ball, u32) -> Ball)] = &[
                    ("exp", arb_exp, elementary::exp),
                    ("ln", arb_log, |b, p| elementary::ln(b, p)),
                    ("sin", arb_sin, elementary::sin),
                    ("cos", arb_cos, elementary::cos),
                    ("atan", arb_atan, elementary::atan),
                    ("gamma", arb_gamma, |b, p| gamma::gamma(b, p)),
                ];
                for (fname, afn, rfn) in fns {
                    afn(&mut az, &ax, prec as Slong + 32);
                    let ours = rfn(&x, prec + 32);
                    if !compare(&format!("{fname}({label}) @{prec}"), &ours, &az, digits) {
                        failures += 1;
                    }
                }
                // zeta only for s > 1.
                if xv > 1.0 {
                    arb_zeta(&mut az, &ax, prec as Slong + 32);
                    let ours = zeta::zeta(&x, prec + 32);
                    if !compare(&format!("zeta({label}) @{prec}"), &ours, &az, digits) {
                        failures += 1;
                    }
                }
                arb_clear(&mut ax);
                arb_clear(&mut az);
            }
        }
    }
    // Constants.
    for &prec in &[1024u32, 65536] {
        let digits = (prec as f64 / 3.33) as usize - 8;
        unsafe {
            let mut az = ArbT::new();
            arb_init(&mut az);
            arb_const_pi(&mut az, prec as Slong + 32);
            let ours = binsplit::pi_chudnovsky(prec + 32);
            if !compare(&format!("pi @{prec}"), &ours, &az, digits) {
                failures += 1;
            }
            arb_clear(&mut az);
        }
    }

    if failures > 0 {
        eprintln!("{failures} differential failures");
        std::process::exit(1);
    }
    println!("all differential tests passed");
}
