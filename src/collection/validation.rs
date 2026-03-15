//! Input validation methods for Collection.

use super::Collection;
use crate::error::{NeedleError, Result};

impl Collection {
    /// Validate that a vector contains only finite values (no NaN or Inf)
    pub(crate) fn validate_vector(vector: &[f32]) -> Result<()> {
        for (i, &val) in vector.iter().enumerate() {
            if val.is_nan() {
                return Err(NeedleError::InvalidVector(format!(
                    "Vector contains NaN at index {}",
                    i
                )));
            }
            if val.is_infinite() {
                return Err(NeedleError::InvalidVector(format!(
                    "Vector contains Inf at index {}",
                    i
                )));
            }
        }
        Ok(())
    }

    /// Validate query vector dimensions and values
    pub(super) fn validate_query(&self, query: &[f32]) -> Result<()> {
        if query.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }
        Self::validate_vector(query)
    }

    /// Clamp k to the collection size to avoid wasting resources
    #[inline]
    pub(super) fn clamp_k(&self, k: usize) -> usize {
        k.min(self.len())
    }

    /// Validate insert input (dimensions and vector values)
    pub(super) fn validate_insert_input(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }
        Self::validate_vector(vector)
    }

    /// Maximum length for a vector ID in bytes.
    const MAX_VECTOR_ID_BYTES: usize = 1024;

    /// Validate that a vector ID is non-empty, within length bounds, and free of
    /// control characters. These rules match the server-layer validation so that
    /// vectors inserted via the Rust API can always be served via HTTP.
    pub fn validate_vector_id(id: &str) -> Result<()> {
        if id.is_empty() {
            return Err(NeedleError::InvalidInput(
                "Vector ID must not be empty".to_string(),
            ));
        }
        if id.len() > Self::MAX_VECTOR_ID_BYTES {
            return Err(NeedleError::InvalidInput(format!(
                "Vector ID exceeds maximum length of {} bytes",
                Self::MAX_VECTOR_ID_BYTES
            )));
        }
        if id.chars().any(|c| c.is_control()) {
            return Err(NeedleError::InvalidInput(
                "Vector ID must not contain control characters".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NeedleError;

    // ── validate_vector ─────────────────────────────────────────────────

    #[test]
    fn test_validate_vector_ok() {
        assert!(Collection::validate_vector(&[1.0, 2.0, 3.0]).is_ok());
    }

    #[test]
    fn test_validate_vector_empty() {
        assert!(Collection::validate_vector(&[]).is_ok());
    }

    #[test]
    fn test_validate_vector_nan() {
        let result = Collection::validate_vector(&[1.0, f32::NAN, 3.0]);
        assert!(matches!(result, Err(NeedleError::InvalidVector(_))));
    }

    #[test]
    fn test_validate_vector_infinity() {
        let result = Collection::validate_vector(&[f32::INFINITY, 0.0]);
        assert!(matches!(result, Err(NeedleError::InvalidVector(_))));
    }

    #[test]
    fn test_validate_vector_neg_infinity() {
        let result = Collection::validate_vector(&[0.0, f32::NEG_INFINITY]);
        assert!(matches!(result, Err(NeedleError::InvalidVector(_))));
    }

    #[test]
    fn test_validate_vector_zeros() {
        assert!(Collection::validate_vector(&[0.0, 0.0, 0.0]).is_ok());
    }

    // ── validate_query ──────────────────────────────────────────────────

    #[test]
    fn test_validate_query_ok() {
        let col = Collection::with_dimensions("test", 3);
        assert!(col.validate_query(&[1.0, 2.0, 3.0]).is_ok());
    }

    #[test]
    fn test_validate_query_wrong_dims() {
        let col = Collection::with_dimensions("test", 4);
        let result = col.validate_query(&[1.0, 2.0]);
        assert!(matches!(result, Err(NeedleError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_validate_query_nan() {
        let col = Collection::with_dimensions("test", 2);
        let result = col.validate_query(&[1.0, f32::NAN]);
        assert!(matches!(result, Err(NeedleError::InvalidVector(_))));
    }

    // ── validate_insert_input ───────────────────────────────────────────

    #[test]
    fn test_validate_insert_input_ok() {
        let col = Collection::with_dimensions("test", 3);
        assert!(col.validate_insert_input(&[1.0, 2.0, 3.0]).is_ok());
    }

    #[test]
    fn test_validate_insert_input_wrong_dims() {
        let col = Collection::with_dimensions("test", 4);
        let result = col.validate_insert_input(&[1.0]);
        assert!(matches!(result, Err(NeedleError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_validate_insert_input_nan() {
        let col = Collection::with_dimensions("test", 2);
        let result = col.validate_insert_input(&[f32::NAN, 1.0]);
        assert!(matches!(result, Err(NeedleError::InvalidVector(_))));
    }

    // ── clamp_k ─────────────────────────────────────────────────────────

    #[test]
    fn test_clamp_k_empty_collection() {
        let col = Collection::with_dimensions("test", 4);
        assert_eq!(col.clamp_k(10), 0);
    }

    #[test]
    fn test_clamp_k_larger_than_size() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert_eq!(col.clamp_k(100), 1);
    }

    #[test]
    fn test_clamp_k_smaller_than_size() {
        let mut col = Collection::with_dimensions("test", 4);
        for i in 0..10 {
            col.insert(format!("v{i}"), &[i as f32, 0.0, 0.0, 0.0], None)
                .unwrap();
        }
        assert_eq!(col.clamp_k(5), 5);
    }

    #[test]
    fn test_clamp_k_zero() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert_eq!(col.clamp_k(0), 0);
    }

    // ── validate_vector_id ──────────────────────────────────────────────

    #[test]
    fn test_validate_vector_id_ok() {
        assert!(Collection::validate_vector_id("doc-123").is_ok());
        assert!(Collection::validate_vector_id("a").is_ok());
    }

    #[test]
    fn test_validate_vector_id_empty() {
        let result = Collection::validate_vector_id("");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_vector_id_too_long() {
        let long_id = "x".repeat(1025);
        let result = Collection::validate_vector_id(&long_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_vector_id_at_max_length() {
        let max_id = "x".repeat(1024);
        assert!(Collection::validate_vector_id(&max_id).is_ok());
    }

    #[test]
    fn test_validate_vector_id_control_chars() {
        let result = Collection::validate_vector_id("id\x00null");
        assert!(result.is_err());
        let result = Collection::validate_vector_id("id\nnewline");
        assert!(result.is_err());
        let result = Collection::validate_vector_id("id\ttab");
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_rejects_empty_id() {
        let mut col = Collection::with_dimensions("test", 3);
        let result = col.insert("", &[1.0, 2.0, 3.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_rejects_control_char_id() {
        let mut col = Collection::with_dimensions("test", 3);
        let result = col.insert("bad\x00id", &[1.0, 2.0, 3.0], None);
        assert!(result.is_err());
    }
}
