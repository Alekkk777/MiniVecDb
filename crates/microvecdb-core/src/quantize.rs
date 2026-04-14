use crate::error::DbError;

/// Number of float32 dimensions supported (all-MiniLM-L6-v2 standard).
pub const DIMS: usize = 384;

/// Number of u32 words needed to store one quantized vector (384 / 32 = 12).
pub const WORDS: usize = DIMS / 32; // = 12

/// A 1-bit quantized vector: 384 dimensions packed into 12 × u32.
///
/// Bit layout: word `i` holds bits for dimensions `[i*32 .. i*32+32)`.
/// Within each word, dimension `i*32 + j` occupies bit `j` (LSB = dim 0).
/// A bit is `1` if the original float was **strictly positive**, `0` otherwise.
pub type BinaryVec = [u32; WORDS];

/// Quantize a 384-dimensional `f32` vector into a [`BinaryVec`].
///
/// Each dimension is mapped: `v > 0.0 → 1`, `v ≤ 0.0 → 0`.
/// This compresses `384 × 4 = 1536` bytes down to `12 × 4 = 48` bytes (96 %).
///
/// # Errors
/// Returns [`DbError::InvalidDimension`] when `floats.len() != 384`.
#[inline]
pub fn quantize_f32(floats: &[f32]) -> Result<BinaryVec, DbError> {
    if floats.len() != DIMS {
        return Err(DbError::InvalidDimension { got: floats.len() });
    }
    let mut out = [0u32; WORDS];
    for (word_idx, chunk) in floats.chunks_exact(32).enumerate() {
        let mut word = 0u32;
        for (bit, &f) in chunk.iter().enumerate() {
            if f > 0.0 {
                word |= 1u32 << bit;
            }
        }
        out[word_idx] = word;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_positive_sets_all_bits() {
        let floats = [1.0_f32; DIMS];
        let bv = quantize_f32(&floats).unwrap();
        assert_eq!(bv, [0xFFFF_FFFF_u32; WORDS]);
    }

    #[test]
    fn all_negative_clears_all_bits() {
        let floats = [-1.0_f32; DIMS];
        let bv = quantize_f32(&floats).unwrap();
        assert_eq!(bv, [0u32; WORDS]);
    }

    #[test]
    fn zero_maps_to_zero_bit() {
        let mut floats = [1.0_f32; DIMS];
        floats[0] = 0.0; // first bit of first word should be 0
        let bv = quantize_f32(&floats).unwrap();
        assert_eq!(bv[0] & 1, 0, "zero should map to 0 bit");
        assert_eq!(bv[0] & !1u32, 0xFFFF_FFFEu32, "other bits should be set");
    }

    #[test]
    fn bit_positions_are_correct() {
        let mut floats = [-1.0_f32; DIMS];
        floats[0] = 1.0;   // word 0, bit 0
        floats[31] = 1.0;  // word 0, bit 31
        floats[32] = 1.0;  // word 1, bit 0
        floats[383] = 1.0; // word 11, bit 31
        let bv = quantize_f32(&floats).unwrap();
        assert_eq!(bv[0], (1u32 << 0) | (1u32 << 31));
        assert_eq!(bv[1], 1u32);
        assert_eq!(bv[11], 1u32 << 31);
        for i in 2..11 {
            assert_eq!(bv[i], 0);
        }
    }

    #[test]
    fn wrong_dimension_returns_error() {
        assert!(quantize_f32(&[1.0_f32; 100]).is_err());
        assert!(quantize_f32(&[]).is_err());
    }
}
