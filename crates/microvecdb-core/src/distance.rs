use crate::quantize::{BinaryVec, DIMS, WORDS};

/// Compute the Hamming distance between two [`BinaryVec`]s.
///
/// Hamming distance = number of bit positions that differ = `popcount(a XOR b)`.
/// Range: `[0, 384]`. A distance of `0` means the vectors are identical; `384`
/// means they are perfectly opposite.
///
/// This function dispatches to a SIMD path on wasm32 (when the `simd` feature
/// is enabled) and falls back to a branchless scalar loop otherwise.
/// LLVM auto-vectorises the scalar loop on every modern architecture.
#[inline]
pub fn hamming_distance(a: &BinaryVec, b: &BinaryVec) -> u32 {
    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    {
        // SAFETY: we compile with +simd128, so this is always available.
        unsafe { hamming_simd(a, b) }
    }
    #[cfg(not(all(feature = "simd", target_arch = "wasm32")))]
    {
        hamming_scalar(a, b)
    }
}

/// Scalar fallback — always compiled (used by native tests and non-simd builds).
#[inline]
pub fn hamming_scalar(a: &BinaryVec, b: &BinaryVec) -> u32 {
    let mut dist = 0u32;
    for i in 0..WORDS {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Explicit SIMD path for wasm32 with +simd128.
///
/// Processes 4 × u32 = 128 bits per v128_xor instruction (3 iterations for 12
/// words).  Each XOR result is split into lanes, and `count_ones()` (which maps
/// to `i32.popcnt` in wasm32) is applied per lane.
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
#[target_feature(enable = "simd128")]
unsafe fn hamming_simd(a: &BinaryVec, b: &BinaryVec) -> u32 {
    use core::arch::wasm32::*;

    // WORDS = 12, so we have exactly 3 chunks of 4 u32 (128 bits each).
    debug_assert_eq!(WORDS % 4, 0);

    let a_ptr = a.as_ptr() as *const v128;
    let b_ptr = b.as_ptr() as *const v128;

    let mut total = 0u32;
    for i in 0..(WORDS / 4) {
        let va = v128_load(a_ptr.add(i));
        let vb = v128_load(b_ptr.add(i));
        let xored = v128_xor(va, vb);
        // Extract the 4 × i32 lanes; count_ones() maps to wasm i32.popcnt.
        total += (i32x4_extract_lane::<0>(xored) as u32).count_ones();
        total += (i32x4_extract_lane::<1>(xored) as u32).count_ones();
        total += (i32x4_extract_lane::<2>(xored) as u32).count_ones();
        total += (i32x4_extract_lane::<3>(xored) as u32).count_ones();
    }
    total
}

/// Maximum possible Hamming distance for the configured dimensionality.
pub const MAX_DISTANCE: u32 = DIMS as u32;

/// Normalised similarity score: `1.0 - (distance / MAX_DISTANCE)`.
/// Returns `1.0` for identical vectors, `0.0` for completely opposite vectors.
#[inline]
pub fn similarity_score(distance: u32) -> f32 {
    1.0 - (distance as f32 / MAX_DISTANCE as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_ones() -> BinaryVec { [0xFFFF_FFFF_u32; WORDS] }
    fn all_zeros() -> BinaryVec { [0u32; WORDS] }

    #[test]
    fn identical_vectors_have_zero_distance() {
        assert_eq!(hamming_distance(&all_ones(), &all_ones()), 0);
        assert_eq!(hamming_distance(&all_zeros(), &all_zeros()), 0);
    }

    #[test]
    fn opposite_vectors_have_max_distance() {
        assert_eq!(hamming_distance(&all_ones(), &all_zeros()), DIMS as u32);
    }

    #[test]
    fn symmetry() {
        let a: BinaryVec = core::array::from_fn(|i| i as u32 * 0x11111111);
        let b: BinaryVec = core::array::from_fn(|i| (i as u32 + 3) * 0x12345678);
        assert_eq!(hamming_distance(&a, &b), hamming_distance(&b, &a));
    }

    #[test]
    fn triangle_inequality() {
        let a: BinaryVec = core::array::from_fn(|i| (i as u32).wrapping_mul(0xDEADBEEF));
        let b: BinaryVec = core::array::from_fn(|i| (i as u32).wrapping_mul(0xCAFEBABE));
        let c: BinaryVec = core::array::from_fn(|i| (i as u32).wrapping_mul(0xABADCAFE));
        let d_ab = hamming_distance(&a, &b);
        let d_bc = hamming_distance(&b, &c);
        let d_ac = hamming_distance(&a, &c);
        assert!(d_ac <= d_ab + d_bc, "triangle inequality violated");
    }

    #[test]
    fn single_bit_difference() {
        let a = all_zeros();
        let mut b = all_zeros();
        b[0] = 1;
        assert_eq!(hamming_distance(&a, &b), 1);
    }

    #[test]
    fn similarity_score_range() {
        assert!((similarity_score(0) - 1.0).abs() < 1e-6);
        assert!((similarity_score(DIMS as u32)).abs() < 1e-6);
        assert!(similarity_score(DIMS as u32 / 2) > 0.0);
        assert!(similarity_score(DIMS as u32 / 2) < 1.0);
    }
}
