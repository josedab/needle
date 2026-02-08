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
}
