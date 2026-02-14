use needle::{DistanceFunction, NeedleError, Result};

mod admin;
mod alias;
mod backup;
mod collection;
mod drift;
mod federation;
mod ops;
mod search;
mod ttl;
mod vector;

pub use admin::*;
pub use alias::*;
pub use backup::*;
pub use collection::*;
#[cfg(feature = "observability")]
pub use drift::*;
pub use federation::*;
pub use ops::*;
pub use search::*;
pub use ttl::*;
pub use vector::*;

// ============================================================================
// Utility functions
// ============================================================================

pub fn parse_distance(distance: &str) -> Option<DistanceFunction> {
    match distance.to_lowercase().as_str() {
        "cosine" => Some(DistanceFunction::Cosine),
        "euclidean" | "l2" => Some(DistanceFunction::Euclidean),
        "dot" | "dotproduct" => Some(DistanceFunction::DotProduct),
        "manhattan" | "l1" => Some(DistanceFunction::Manhattan),
        _ => None,
    }
}

pub fn parse_query_vector(query_str: &str) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    for part in query_str.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = trimmed.parse::<f32>().map_err(|_| {
            NeedleError::InvalidVector(format!("Invalid float value '{}'", trimmed))
        })?;
        values.push(value);
    }

    if values.is_empty() {
        return Err(NeedleError::InvalidVector(
            "Query vector must contain at least one value".to_string(),
        ));
    }

    Ok(values)
}
