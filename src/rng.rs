//! Deterministic randomness. Fixed seed + fixed timestep must reproduce a
//! frame exactly, so nothing here touches wall time or OS entropy.

/// The one seed. Everything derives from it.
pub const SEED: u64 = 0x6b69_7474_795f_766a; // "kitty_vj"

/// splitmix64 — stateless hash for per-cell / per-column determinism.
pub fn hash(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Hash a few coordinates together with the global seed.
pub fn hash3(a: u64, b: u64, c: u64) -> u64 {
    hash(SEED ^ hash(a ^ hash(b ^ hash(c))))
}

/// Uniform f64 in [0, 1) from a hash value.
pub fn unit_f64(h: u64) -> f64 {
    (h >> 11) as f64 / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_stable() {
        assert_eq!(hash3(1, 2, 3), hash3(1, 2, 3));
        assert_ne!(hash3(1, 2, 3), hash3(3, 2, 1));
    }

    #[test]
    fn unit_range() {
        for i in 0..1000 {
            let v = unit_f64(hash(i));
            assert!((0.0..1.0).contains(&v));
        }
    }
}
