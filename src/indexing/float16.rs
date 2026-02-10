//! # Half-Precision Floating Point Support
//!
//! This module provides Float16 (IEEE 754 half-precision) and BFloat16 (Brain Float)
//! vector support for memory-efficient storage with minimal accuracy loss.
//!
//! ## Formats
//!
//! - **Float16 (f16)**: IEEE 754 half-precision, 1 sign + 5 exponent + 10 mantissa bits
//! - **BFloat16 (bf16)**: Brain float, 1 sign + 8 exponent + 7 mantissa bits
//!
//! Float16 has better precision, while BFloat16 has larger dynamic range (same as f32).
//!
//! ## Example
//!
//! ```rust,ignore
//! use needle::float16::{F16Vector, Bf16Vector};
//!
//! // Convert f32 vectors to half precision
//! let f32_vec = vec![1.0f32, 2.0, 3.0, 4.0];
//! let f16_vec = F16Vector::from_f32(&f32_vec);
//!
//! // Convert back to f32
//! let recovered = f16_vec.to_f32();
//!
//! // Compute distance directly in reduced precision
//! let distance = f16_vec.euclidean_distance(&other_f16_vec);
//! ```

/// IEEE 754 half-precision float (16-bit)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct F16(u16);

impl F16 {
    /// Zero value
    pub const ZERO: Self = Self(0);

    /// One value
    pub const ONE: Self = Self(0x3C00);

    /// Positive infinity
    pub const INFINITY: Self = Self(0x7C00);

    /// Negative infinity
    pub const NEG_INFINITY: Self = Self(0xFC00);

    /// NaN value
    pub const NAN: Self = Self(0x7E00);

    /// Create from raw bits
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Get raw bits
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert from f32
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();

        // Extract components
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x7FFFFF;

        // Handle special cases
        if exp == 255 {
            // Inf or NaN
            if mantissa == 0 {
                return Self(((sign << 15) | 0x7C00) as u16);
            } else {
                return Self(((sign << 15) | 0x7E00) as u16);
            }
        }

        // Bias conversion: f32 bias is 127, f16 bias is 15
        let new_exp = exp - 127 + 15;

        if new_exp <= 0 {
            // Subnormal or zero
            if new_exp < -10 {
                return Self((sign << 15) as u16);
            }
            // Subnormal
            let m = (mantissa | 0x800000) >> (1 - new_exp + 13);
            return Self(((sign << 15) | m) as u16);
        } else if new_exp >= 31 {
            // Overflow to infinity
            return Self(((sign << 15) | 0x7C00) as u16);
        }

        // Normal case
        let m = mantissa >> 13;
        Self(((sign << 15) | ((new_exp as u32) << 10) | m) as u16)
    }

    /// Convert to f32
    pub fn to_f32(self) -> f32 {
        let bits = self.0 as u32;

        let sign = (bits >> 15) & 1;
        let exp = (bits >> 10) & 0x1F;
        let mantissa = bits & 0x3FF;

        if exp == 0 {
            if mantissa == 0 {
                // Zero
                return f32::from_bits(sign << 31);
            }
            // Subnormal
            let mut m = mantissa;
            let mut e = 1i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            let new_exp = (127 - 15 + e) as u32;
            let new_mantissa = (m & 0x3FF) << 13;
            return f32::from_bits((sign << 31) | (new_exp << 23) | new_mantissa);
        } else if exp == 31 {
            // Inf or NaN
            if mantissa == 0 {
                return f32::from_bits((sign << 31) | 0x7F800000);
            } else {
                return f32::from_bits((sign << 31) | 0x7FC00000);
            }
        }

        // Normal
        let new_exp = (exp as i32 - 15 + 127) as u32;
        let new_mantissa = mantissa << 13;
        f32::from_bits((sign << 31) | (new_exp << 23) | new_mantissa)
    }

    /// Check if NaN
    pub fn is_nan(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        let mantissa = self.0 & 0x3FF;
        exp == 31 && mantissa != 0
    }

    /// Check if infinite
    pub fn is_infinite(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        let mantissa = self.0 & 0x3FF;
        exp == 31 && mantissa == 0
    }

    /// Check if zero
    pub fn is_zero(self) -> bool {
        (self.0 & 0x7FFF) == 0
    }
}

impl From<f32> for F16 {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<F16> for f32 {
    fn from(value: F16) -> Self {
        value.to_f32()
    }
}

/// Brain Float 16-bit floating point
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Bf16(u16);

impl Bf16 {
    /// Zero value
    pub const ZERO: Self = Self(0);

    /// One value
    pub const ONE: Self = Self(0x3F80);

    /// Positive infinity
    pub const INFINITY: Self = Self(0x7F80);

    /// Negative infinity
    pub const NEG_INFINITY: Self = Self(0xFF80);

    /// NaN value
    pub const NAN: Self = Self(0x7FC0);

    /// Create from raw bits
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Get raw bits
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert from f32 (simple truncation of lower 16 bits)
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        // BFloat16 is just the upper 16 bits of f32
        Self((bits >> 16) as u16)
    }

    /// Convert from f32 with rounding
    pub fn from_f32_round(value: f32) -> Self {
        let bits = value.to_bits();
        // Round to nearest even
        let round = (bits >> 15) & 1;
        let sticky = (bits & 0x7FFF) != 0;
        let upper = (bits >> 16) as u16;
        if round != 0 && (sticky || (upper & 1) != 0) {
            Self(upper.wrapping_add(1))
        } else {
            Self(upper)
        }
    }

    /// Convert to f32
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    /// Check if NaN
    pub fn is_nan(self) -> bool {
        let exp = (self.0 >> 7) & 0xFF;
        let mantissa = self.0 & 0x7F;
        exp == 255 && mantissa != 0
    }

    /// Check if infinite
    pub fn is_infinite(self) -> bool {
        let exp = (self.0 >> 7) & 0xFF;
        let mantissa = self.0 & 0x7F;
        exp == 255 && mantissa == 0
    }

    /// Check if zero
    pub fn is_zero(self) -> bool {
        (self.0 & 0x7FFF) == 0
    }
}

impl From<f32> for Bf16 {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<Bf16> for f32 {
    fn from(value: Bf16) -> Self {
        value.to_f32()
    }
}

/// A vector stored in Float16 format
#[derive(Clone, Debug)]
pub struct F16Vector {
    data: Vec<F16>,
}

impl F16Vector {
    /// Create from f32 slice
    pub fn from_f32(values: &[f32]) -> Self {
        Self {
            data: values.iter().map(|&v| F16::from_f32(v)).collect(),
        }
    }

    /// Convert to f32 vector
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|v| v.to_f32()).collect()
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get raw data
    pub fn as_slice(&self) -> &[F16] {
        &self.data
    }

    /// Euclidean distance to another F16Vector
    pub fn euclidean_distance(&self, other: &F16Vector) -> f32 {
        // Convert to f32 for computation
        let a = self.to_f32();
        let b = other.to_f32();

        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Dot product with another F16Vector
    pub fn dot(&self, other: &F16Vector) -> f32 {
        let a = self.to_f32();
        let b = other.to_f32();

        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Cosine similarity with another F16Vector
    pub fn cosine_similarity(&self, other: &F16Vector) -> f32 {
        let dot = self.dot(other);
        let norm_a: f32 = self.to_f32().iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.to_f32().iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<F16>()
    }
}

/// A vector stored in BFloat16 format
#[derive(Clone, Debug)]
pub struct Bf16Vector {
    data: Vec<Bf16>,
}

impl Bf16Vector {
    /// Create from f32 slice
    pub fn from_f32(values: &[f32]) -> Self {
        Self {
            data: values.iter().map(|&v| Bf16::from_f32(v)).collect(),
        }
    }

    /// Create from f32 slice with rounding
    pub fn from_f32_round(values: &[f32]) -> Self {
        Self {
            data: values.iter().map(|&v| Bf16::from_f32_round(v)).collect(),
        }
    }

    /// Convert to f32 vector
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|v| v.to_f32()).collect()
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get raw data
    pub fn as_slice(&self) -> &[Bf16] {
        &self.data
    }

    /// Euclidean distance to another Bf16Vector
    pub fn euclidean_distance(&self, other: &Bf16Vector) -> f32 {
        let a = self.to_f32();
        let b = other.to_f32();

        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Dot product with another Bf16Vector
    pub fn dot(&self, other: &Bf16Vector) -> f32 {
        let a = self.to_f32();
        let b = other.to_f32();

        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Cosine similarity with another Bf16Vector
    pub fn cosine_similarity(&self, other: &Bf16Vector) -> f32 {
        let dot = self.dot(other);
        let norm_a: f32 = self.to_f32().iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.to_f32().iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<Bf16>()
    }
}

/// Trait for vector types that can be converted to/from f32
pub trait HalfPrecision: Sized {
    /// Create from f32 slice
    fn from_f32(values: &[f32]) -> Self;

    /// Convert to f32 vector
    fn to_f32(&self) -> Vec<f32>;

    /// Get dimensions
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl HalfPrecision for F16Vector {
    fn from_f32(values: &[f32]) -> Self {
        F16Vector::from_f32(values)
    }

    fn to_f32(&self) -> Vec<f32> {
        F16Vector::to_f32(self)
    }

    fn len(&self) -> usize {
        F16Vector::len(self)
    }
}

impl HalfPrecision for Bf16Vector {
    fn from_f32(values: &[f32]) -> Self {
        Bf16Vector::from_f32(values)
    }

    fn to_f32(&self) -> Vec<f32> {
        Bf16Vector::to_f32(self)
    }

    fn len(&self) -> usize {
        Bf16Vector::len(self)
    }
}

/// Calculate memory savings from using half-precision
pub fn memory_savings(dimensions: usize, num_vectors: usize) -> (usize, usize, f32) {
    let f32_size = dimensions * num_vectors * std::mem::size_of::<f32>();
    let f16_size = dimensions * num_vectors * std::mem::size_of::<F16>();
    let savings = ((f32_size - f16_size) as f32 / f32_size as f32) * 100.0;
    (f32_size, f16_size, savings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 100.0, -0.001];

        for &v in &values {
            let f16 = F16::from_f32(v);
            let back = f16.to_f32();
            // Allow some error due to precision loss
            assert!((v - back).abs() < v.abs() * 0.01 + 0.001, "Failed for {}", v);
        }
    }

    #[test]
    fn test_f16_special_values() {
        // Zero
        assert!(F16::from_f32(0.0).is_zero());

        // Infinity
        assert!(F16::from_f32(f32::INFINITY).is_infinite());
        assert!(F16::from_f32(f32::NEG_INFINITY).is_infinite());

        // NaN
        assert!(F16::from_f32(f32::NAN).is_nan());
    }

    #[test]
    fn test_bf16_conversion() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 100.0, -0.001, 1e10, 1e-10];

        for &v in &values {
            let bf16 = Bf16::from_f32(v);
            let back = bf16.to_f32();
            // BFloat16 has less precision but same range
            let relative_error = if v != 0.0 {
                ((v - back) / v).abs()
            } else {
                back.abs()
            };
            assert!(relative_error < 0.01, "Failed for {}: got {}", v, back);
        }
    }

    #[test]
    fn test_bf16_special_values() {
        assert!(Bf16::from_f32(0.0).is_zero());
        assert!(Bf16::from_f32(f32::INFINITY).is_infinite());
        assert!(Bf16::from_f32(f32::NAN).is_nan());
    }

    #[test]
    fn test_f16_vector() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let f16_vec = F16Vector::from_f32(&values);

        assert_eq!(f16_vec.len(), 4);
        assert_eq!(f16_vec.memory_bytes(), 8); // 4 * 2 bytes

        let back = f16_vec.to_f32();
        for (a, b) in values.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_f16_vector_distance() {
        let a = F16Vector::from_f32(&[1.0, 0.0, 0.0]);
        let b = F16Vector::from_f32(&[0.0, 1.0, 0.0]);

        let dist = a.euclidean_distance(&b);
        let expected = 2.0f32.sqrt();
        assert!((dist - expected).abs() < 0.1);
    }

    #[test]
    fn test_bf16_vector_dot() {
        let a = Bf16Vector::from_f32(&[1.0, 2.0, 3.0]);
        let b = Bf16Vector::from_f32(&[4.0, 5.0, 6.0]);

        let dot = a.dot(&b);
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0;
        assert!((dot - expected).abs() < 1.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = F16Vector::from_f32(&[1.0, 0.0]);
        let b = F16Vector::from_f32(&[1.0, 0.0]);

        let sim = a.cosine_similarity(&b);
        assert!((sim - 1.0).abs() < 0.01);

        let c = F16Vector::from_f32(&[0.0, 1.0]);
        let sim2 = a.cosine_similarity(&c);
        assert!(sim2.abs() < 0.01);
    }

    #[test]
    fn test_memory_savings() {
        let (f32_size, f16_size, savings) = memory_savings(384, 1_000_000);

        assert_eq!(f32_size, 384 * 1_000_000 * 4);
        assert_eq!(f16_size, 384 * 1_000_000 * 2);
        assert!((savings - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_half_precision_trait() {
        fn test_generic<V: HalfPrecision>(values: &[f32]) -> V {
            V::from_f32(values)
        }

        let f16_vec: F16Vector = test_generic(&[1.0, 2.0, 3.0]);
        assert_eq!(f16_vec.len(), 3);

        let bf16_vec: Bf16Vector = test_generic(&[1.0, 2.0, 3.0]);
        assert_eq!(bf16_vec.len(), 3);
    }
}
