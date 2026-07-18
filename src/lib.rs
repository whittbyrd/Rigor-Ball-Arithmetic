//! # rigor — ball arithmetic for verified real computation
//!
//! A ball `[m ± r]` represents the set of real numbers within distance `r`
//! of `m`. Every operation returns a ball guaranteed to contain the exact
//! mathematical result: **inclusions are always correct**; precision is a
//! quality-of-service knob, never a correctness knob.
//!
//! Layers (bottom up):
//! - [`mpn`]: GMP-style limb-vector kernels (add/sub/mul/div/shift).
//! - [`fp`]: arbitrary-precision binary floating point with directed rounding.
//! - [`mag`]: fixed-precision unsigned magnitude bounds (the radius type).
//! - [`ball`]: the [`Ball`] type with rigorously propagated error bounds.
//! - elementary and special functions on balls: [`elementary`], [`constants`],
//!   [`gamma`], [`zeta`].

pub mod mpn;
pub mod fp;

#[doc(hidden)]
pub mod testutil;

pub use fp::{Float, Round};
