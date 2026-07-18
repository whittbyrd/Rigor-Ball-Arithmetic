//! rigor vs Arb vs MPFR: wall-clock comparison at matched precisions.
//!
//! Methodology: each cell is the best of `REPS` runs (best-of to shed
//! scheduler noise; these are single-threaded CPU-bound kernels). Arb and
//! rigor compute certified enclosures; MPFR computes a correctly-rounded
//! point value (no error bound) — it is the "how much does rigor cost"
//! baseline, not an apples-to-apples competitor.
//!
//! Output: a Markdown table on stdout, ready to paste into the README.

use rigor::ball::Ball;
use rigor::{binsplit, elementary, gamma, zeta};
use std::os::raw::{c_int, c_longlong};
use std::time::Instant;

#[repr(C)]
struct ArbT {
    _data: [u64; 16],
}
impl ArbT {
    fn new() -> Self {
        ArbT { _data: [0; 16] }
    }
}

#[repr(C)]
struct MpfrT {
    _data: [u64; 8],
}
impl MpfrT {
    fn new() -> Self {
        MpfrT { _data: [0; 8] }
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
    fn arb_atan(z: *mut ArbT, x: *const ArbT, prec: Slong);
    fn arb_gamma(z: *mut ArbT, x: *const ArbT, prec: Slong);
    fn arb_zeta(z: *mut ArbT, s: *const ArbT, prec: Slong);
    fn arb_const_pi(z: *mut ArbT, prec: Slong);
}

#[link(name = "mpfr")]
extern "C" {
    fn mpfr_init2(x: *mut MpfrT, prec: Slong);
    fn mpfr_clear(x: *mut MpfrT);
    fn mpfr_set_d(x: *mut MpfrT, v: f64, rnd: c_int) -> c_int;
    fn mpfr_exp(z: *mut MpfrT, x: *const MpfrT, rnd: c_int) -> c_int;
    fn mpfr_log(z: *mut MpfrT, x: *const MpfrT, rnd: c_int) -> c_int;
    fn mpfr_sin(z: *mut MpfrT, x: *const MpfrT, rnd: c_int) -> c_int;
    fn mpfr_atan(z: *mut MpfrT, x: *const MpfrT, rnd: c_int) -> c_int;
    fn mpfr_gamma(z: *mut MpfrT, x: *const MpfrT, rnd: c_int) -> c_int;
    fn mpfr_zeta(z: *mut MpfrT, x: *const MpfrT, rnd: c_int) -> c_int;
    fn mpfr_const_pi(z: *mut MpfrT, rnd: c_int) -> c_int;
    fn mpfr_free_cache();
}

const REPS: usize = 5;

fn best_of(mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
}

fn fmt_time(s: f64) -> String {
    if s < 1e-3 {
        format!("{:.1} µs", s * 1e6)
    } else if s < 1.0 {
        format!("{:.2} ms", s * 1e3)
    } else {
        format!("{:.2} s", s)
    }
}

fn main() {
    let digit_sets: &[u32] = &[100, 1_000, 10_000];
    let xv = 1.5f64;

    println!("| function | digits | rigor | Arb | MPFR | rigor/Arb |");
    println!("|---|---|---|---|---|---|");

    for &digits in digit_sets {
        let prec = (digits as f64 * 3.3219).ceil() as Slong + 16;
        let precu = prec as u32;
        let x = Ball::from_f64(xv);

        // Warm rigor caches.
        let _ = elementary::ln(&x, precu);
        let _ = gamma::gamma(&x, precu);
        let _ = zeta::zeta(&Ball::from_i64(3), precu);

        unsafe {
            let mut ax = ArbT::new();
            let mut az = ArbT::new();
            arb_init(&mut ax);
            arb_init(&mut az);
            arb_set_d(&mut ax, xv);
            let mut a3 = ArbT::new();
            arb_init(&mut a3);
            arb_set_d(&mut a3, 3.0);
            let mut mx = MpfrT::new();
            let mut mz = MpfrT::new();
            let mut m3 = MpfrT::new();
            mpfr_init2(&mut mx, prec);
            mpfr_init2(&mut mz, prec);
            mpfr_init2(&mut m3, prec);
            mpfr_set_d(&mut mx, xv, 0);
            mpfr_set_d(&mut m3, 3.0, 0);

            type Row = (
                &'static str,
                Box<dyn FnMut() -> ()>,
                Box<dyn FnMut() -> ()>,
                Box<dyn FnMut() -> ()>,
            );
            let axp = &mut ax as *mut ArbT;
            let azp = &mut az as *mut ArbT;
            let a3p = &mut a3 as *mut ArbT;
            let mxp = &mut mx as *mut MpfrT;
            let mzp = &mut mz as *mut MpfrT;
            let m3p = &mut m3 as *mut MpfrT;
            let s3 = Ball::from_i64(3);

            let mut rows: Vec<Row> = vec![
                (
                    "exp",
                    Box::new({
                        let x = x.clone();
                        move || {
                            let _ = elementary::exp(&x, precu);
                        }
                    }),
                    Box::new(move || arb_exp(azp, axp, prec)),
                    Box::new(move || {
                        mpfr_exp(mzp, mxp, 0);
                    }),
                ),
                (
                    "ln",
                    Box::new({
                        let x = x.clone();
                        move || {
                            let _ = elementary::ln(&x, precu);
                        }
                    }),
                    Box::new(move || arb_log(azp, axp, prec)),
                    Box::new(move || {
                        mpfr_log(mzp, mxp, 0);
                    }),
                ),
                (
                    "sin",
                    Box::new({
                        let x = x.clone();
                        move || {
                            let _ = elementary::sin(&x, precu);
                        }
                    }),
                    Box::new(move || arb_sin(azp, axp, prec)),
                    Box::new(move || {
                        mpfr_sin(mzp, mxp, 0);
                    }),
                ),
                (
                    "atan",
                    Box::new({
                        let x = x.clone();
                        move || {
                            let _ = elementary::atan(&x, precu);
                        }
                    }),
                    Box::new(move || arb_atan(azp, axp, prec)),
                    Box::new(move || {
                        mpfr_atan(mzp, mxp, 0);
                    }),
                ),
                (
                    "gamma",
                    Box::new({
                        let x = x.clone();
                        move || {
                            let _ = gamma::gamma(&x, precu);
                        }
                    }),
                    Box::new(move || arb_gamma(azp, axp, prec)),
                    Box::new(move || {
                        mpfr_gamma(mzp, mxp, 0);
                    }),
                ),
                (
                    "zeta(3)",
                    Box::new({
                        let s3 = s3.clone();
                        move || {
                            let _ = zeta::zeta(&s3, precu);
                        }
                    }),
                    Box::new(move || arb_zeta(azp, a3p, prec)),
                    Box::new(move || {
                        mpfr_zeta(mzp, m3p, 0);
                    }),
                ),
                (
                    "pi",
                    Box::new(move || {
                        let _ = binsplit::pi_chudnovsky(precu);
                    }),
                    Box::new(move || arb_const_pi(azp, prec)),
                    Box::new(move || {
                        mpfr_const_pi(mzp, 0);
                        mpfr_free_cache();
                    }),
                ),
            ];

            for (name, ours, arb, mpfr) in rows.iter_mut() {
                let t_r = best_of(ours);
                let t_a = best_of(arb);
                let t_m = best_of(mpfr);
                println!(
                    "| {name} | {digits} | {} | {} | {} | {:.1}× |",
                    fmt_time(t_r),
                    fmt_time(t_a),
                    fmt_time(t_m),
                    t_r / t_a
                );
            }
            arb_clear(&mut ax);
            arb_clear(&mut az);
            arb_clear(&mut a3);
            mpfr_clear(&mut mx);
            mpfr_clear(&mut mz);
            mpfr_clear(&mut m3);
        }
    }
}
