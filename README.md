# rigor — ball arithmetic for verified real computation

**Correct inclusions, always.** `rigor` is a pure-Rust library for computing
with real numbers such that every result is a *ball* `[m ± r]` guaranteed to
contain the true mathematical value. Precision is a quality-of-service knob;
correctness is never negotiable. In the spirit of [Arb](https://arblib.org/)
(now part of FLINT), built from the limbs up with zero runtime dependencies.

```rust
use rigor::{Ball, elementary, gamma, constants};

let x = Ball::from_f64(1.5);
let y = elementary::exp(&x, 3330);        // e^1.5 to ~1000 digits
assert!(y.rel_accuracy_bits() >= 3330);   // certified, not estimated

let g = gamma::gamma(&x, 333);            // Γ(3/2) = √π/2, enclosed
let pi = constants::pi(332_270);          // 100 000 digits of π in ~0.1 s
```

```
$ cargo run --release --example digits_100k
pi: 99999 digits in 0.117s (certified 332254 bits, 64 spare)
  head: 3.1415926535897932384626433832795028841971693993751058209749
e: 99999 digits in 0.031s (certified 332254 bits, 64 spare)
  head: 2.7182818284590452353602874713526624977572470936999595749669
```

## Why balls?

Floating point silently rounds; interval endpoints double the work. A ball
stores a full-precision midpoint and a *low-precision* radius (a 30-bit
mantissa + exponent, like Arb's `mag_t`), so rigor costs ~one extra word per
number. Every operation propagates a proved error bound:

- if `x ∈ [ma ± ra]` and `y ∈ [mb ± rb]`, then
  `x·y ∈ [round(ma·mb) ± (ra(|mb|+rb) + |ma|rb + ½ulp)]` — the inequality
  justifying each radius formula is written next to the code that computes it;
- series tails are bounded by *computed* geometric/alternating-series
  remainders, never by "N terms is surely enough";
- when a fast algorithm has no error proof (Newton reciprocal, rsqrt), its
  output is certified *a posteriori* from an exact residual.

The library never returns a wrong enclosure by design; a bug in a bound is a
soundness bug and the test suite treats it as such (see
[Testing](#testing-strategy)).

## What's inside

| layer | file | contents |
|---|---|---|
| limbs | `src/mpn.rs` | GMP-style `u64`-vector kernels: add/sub with carry, schoolbook + Karatsuba multiply, Knuth-D division, shifts, integer sqrt |
| floats | `src/fp.rs` | arbitrary-precision dyadic floats; add/sub/mul/div/sqrt with directed rounding (`Floor/Ceil/Down/Up/Nearest`ties-even); all rounding funnels through one audited window-rounding routine |
| radii | `src/mag.rs` | 30-bit always-round-up magnitude type; each op carries its one-line over-approximation argument |
| balls | `src/ball.rs` | `Ball` with proved error propagation; a-posteriori-certified fast div/sqrt above 2048 bits |
| bigints | `src/int.rs` | minimal signed integers for binary splitting |
| elementary | `src/elementary.rs` | exp, ln, sin/cos, atan: 2-adic & sqrt argument reduction, Taylor + rigorous tails, adaptive-precision retry |
| constants | `src/constants.rs`, `src/binsplit.rs` | π (Chudnovsky), e, ln 2 via binary splitting; thread-safe precision-amortized cache; Machin π kept as an independent cross-check |
| special | `src/gamma.rs`, `src/zeta.rs`, `src/bernoulli.rs` | Γ via Stirling + reflection, ζ via Euler–Maclaurin, both with classical first-omitted-term remainder bounds; exact tangent-number Bernoulli cache |

No `unsafe` in the core: the `u128` widening ops compile to the same
`MUL`/`ADC` chains inline assembly would give (verified on x86-64; see
*Performance notes*).

## The correctness argument, briefly

1. **One rounding site.** Every `Float` operation reduces its exact result to
   a *window* — a limb vector plus an optional sticky term with a proven
   placement invariant (the top set bit sits ≥126 bits above the window
   bottom whenever the sticky term is nonzero, so sticky bits can only ever
   flip the round/sticky decision, not a kept bit). `Float::from_window` is
   the only function in the crate that discards information.
2. **Radii only grow.** `Mag` has no rounding modes: every operation rounds
   up, and each carries its inequality in a comment.
3. **Tails are computed.** A truncated series contributes
   `|first omitted| · geometric factor`, evaluated with 64-bit directed
   floats from certified upper bounds of the argument. Choosing N badly can
   only widen the ball.
4. **Fast paths are certified, not trusted.** Above 2048 bits, division and
   sqrt midpoints come from unproved Newton iterations; rigor is restored by
   one exact multiplication: `Δ = a − q·b` gives `|a/b − q| ≤ |Δ|/b_low`,
   which goes into the radius. The identical pattern covers `√m` via
   `Δ = m − s²`.
5. **Domain edges are failures, not guesses.** `try_div`, `try_ln`,
   `try_gamma`, `try_zeta` return `None` when the input ball touches a pole
   or leaves the supported region; the panicking wrappers say why.

## Performance

Measured on an Intel Core Ultra 9 185H (Windows 11, `-C lto=thin`, single
thread), warm constant caches. Reproduce with `scripts/bench.ps1` /
`scripts/bench.sh` (criterion) — the table below is from
`cargo bench` medians.

| op @ digits | 100 | 1 000 | 10 000 |
|---|---|---|---|
| exp | TBD | TBD | TBD |
| ln | TBD | TBD | TBD |
| sin | TBD | TBD | TBD |
| atan | TBD | TBD | TBD |
| Γ(1.5) (warm) | TBD | TBD | — (see below) |
| ζ(3) (warm) | TBD | TBD | — |

| constant @ digits | 1 000 | 10 000 | 100 000 |
|---|---|---|---|
| π (Chudnovsky) | TBD | TBD | TBD |
| e | TBD | TBD | TBD |
| ln 2 | TBD | TBD | TBD |

### Versus Arb (honest edition)

`scripts/compare_arb.sh` builds `tools/arb-bench`, which times identical
workloads against Arb (FLINT ≥ 3) and MPFR on Linux, and `tools/arb-diff`,
which requires digit-for-digit agreement of certified output. Expectations
based on algorithmic accounting (run the script for your machine's truth):

- **exp**: we are within roughly 3–8× of Arb at 10²–10⁴ digits. Arb wins via
  assembly GMP limb kernels, `mulhigh` truncated products, and rectangular
  splitting of the Taylor sum (O(√N) full products vs our O(N)).
- **ln / atan**: gap larger (5–15×). Arb evaluates these through binary
  splitting / bit-burst methods on dyadic arguments; our sqrt-reduction +
  series is simpler but multiplies the constant factor.
- **Γ**: comparable at 100–1000 digits once caches are warm. Beyond ~5000
  digits our wall clock is dominated by **Bernoulli generation** — the exact
  tangent-number recurrence is O(M²) big-integer ops (≈ M³ bit ops). Arb
  generates Bernoulli numbers via zeta-based multi-evaluation and the
  von Staudt–Clausen theorem, which is dramatically faster. This is the
  single biggest structural gap and is deliberately documented rather than
  papered over. (First call at 10k digits: ~seconds; warm calls: ~1 s.)
- **ζ(integer)**: Euler–Maclaurin with N≈M≈0.17·bits terms; same Bernoulli
  cost story. Arb additionally has special code for integer arguments.
- **π**: both use Chudnovsky binary splitting; Arb's advantage reduces to
  GMP's FFT multiplication above ~10⁵ digits (our Karatsuba-only `mpn::mul`
  is the bottleneck: O(n^1.585) vs O(n log n)).
- **MPFR** computes correctly-rounded *point* values, not enclosures — it is
  the "price of rigor" baseline, typically the fastest column.

### Where the time goes

Flamegraph of the 10k-digit elementary-function workload (generated by
`scripts/flamegraph.sh`, also produced as a CI artifact on every main-branch
build): `docs/flamegraph.svg`. The hot paths are `mpn::mul_basecase` /
`mul_karatsuba` under ball multiplication — as they should be; everything
else is bookkeeping.

### Performance notes

- Karatsuba threshold 24 limbs, tuned on this machine.
- `debug = true` in release keeps symbols for profiling; it does not affect
  codegen.
- Known future wins, in impact order: Toom-3/FFT multiplication,
  `mulhigh`-style truncated products in `Float::mul`, rectangular splitting
  for series, zeta-based Bernoulli generation, bit-burst ln/atan.

## Testing strategy

`cargo test --release` runs, beyond unit tests of every kernel:

- **Inclusion property tests** (`tests/inclusion.rs`): for random inputs,
  the ball at precision p must contain the midpoint of the ball at
  precision 2p+128, for every function and for composed expressions. Any
  failure is a soundness bug.
- **Cross-algorithm identities** (`tests/identities.rs`): π computed three
  independent ways (Chudnovsky, 4·atan(1), Γ(1/2)²), e and ln 2 two ways,
  Γ duplication formula, ζ(2k) closed forms, trig addition theorems —
  different code paths must produce overlapping balls.
- **Differential tests vs Arb** (`tools/arb-diff`, Linux CI): certified
  digit prefixes must agree with Arb across functions, inputs and
  precisions.
- **Known-value anchors**: 50 digits of π, e, ln 2, ζ(3), Γ(1/2)√π
  identities, factorials, tangent numbers T₁..T₅, Bernoulli B₂..B₁₂.
- **Dependency-problem demonstrations** (`examples/dependency_problem.rs`
  and a unit test): `x − x`, three algebraic forms of `x² − x`, and
  subdivision — documenting what ball arithmetic does *not* solve.

CI (GitHub Actions, `.github/workflows/ci.yml`): rustfmt, clippy
(`-D warnings`), the full release test suite on Linux **and** Windows/MSVC,
the 100k-digit constants demo, a benchmark-regression canary
(order-of-magnitude tripwire budgets — the honest granularity for shared
runners), the Arb differential job, and criterion + flamegraph artifacts.

## Design decisions worth defending

- **Pure Rust over GMP bindings.** `gmp-mpfr-sys` does not build under
  MSVC; a resume-grade library that only works on Unix is half a library.
  The mpn layer keeps GMP's API shape so a GMP backend could be
  feature-swapped later; meanwhile the whole crate builds anywhere Rust
  does, in seconds.
- **Per-operation precision, Arb-style.** A `Float` is an exact dyadic
  rational; precision belongs to operations, not values. This makes exactness
  propagation free (integers stay exact through any precision) and retry
  loops trivial.
- **Approximate-then-certify beats prove-every-step** for division and
  sqrt: Newton with precision doubling is textbook-fast, and one exact
  residual multiplication turns "probably right" into "provably enclosed".
- **Caches are enclosure-safe.** A constant cached at high precision and
  rounded down is still a correct enclosure — so caches can only make
  results tighter, never wrong.

## Limitations

- Real balls only (no complex arithmetic yet).
- ζ requires real s > 0, s ≠ 1; non-integer s uses the slow
  `exp(−s ln j)` path for the direct sum.
- exp/sin/cos reject |x| ≥ 2^40 (exponent-range guard) rather than
  computing with gigantic quadrant reductions.
- Bernoulli generation limits special functions above ~5000 digits (see
  the honest Arb comparison above).
- Decimal printing truncates; it does not round the last digit.

## Repository map

```
src/            the library (see table above)
tests/          inclusion + identity suites
benches/        criterion: elementary, constants, special
examples/       digits_100k, dependency_problem, quick_timing, bench_smoke
tools/arb-diff  differential tester vs Arb (Linux)
tools/arb-bench rigor vs Arb vs MPFR timing table (Linux)
scripts/        bench, compare_arb, flamegraph — all results reproducible
```

License: MIT OR Apache-2.0.
