//! # Needle Core
//!
//! Foundational types for the Needle vector database.
//! This crate provides error types and distance functions used across the Needle ecosystem.

pub mod distance;
pub mod error;

// Re-export primary types
pub use distance::DistanceFunction;
pub use error::{ErrorCode, NeedleError, Recoverable, RecoveryHint, Result};
