//! Deterministic PRNG for tests and benches (xoshiro256**).
//!
//! Dependency-free on purpose: reproducible across platforms and toolchains,
//! seeds are printed on failure so any run can be replayed.

pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to spread the seed over the state.
        let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        Rng {
            s: [next(), next(), next(), next()],
        }
    }

    #[allow(clippy::should_implement_trait)] // an RNG, not an Iterator
    pub fn next(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, n).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// Uniform i64 in [lo, hi].
    pub fn range_i64(&mut self, lo: i64, hi: i64) -> i64 {
        lo.wrapping_add(self.below((hi - lo + 1) as u64) as i64)
    }
}
