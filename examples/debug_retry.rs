//! Diagnose retry behavior: `cargo run --release --example debug_retry`
use rigor::{ball::Ball, elementary};

fn main() {
    let prec = 33_235u32;
    let x = Ball::from_f64(1.5);
    let s = elementary::sin(&x, prec);
    println!("sin acc {}  (prec {prec})", s.rel_accuracy_bits());
    let l = elementary::ln(&x, prec);
    println!("ln  acc {}", l.rel_accuracy_bits());
    let a = elementary::atan(&x, prec);
    println!("atan acc {}", a.rel_accuracy_bits());
}
