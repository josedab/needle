use needle::{DistanceFunction, NeedleError, Result};

mod admin;
mod alias;
mod backup;
mod cache;
mod collection;
mod drift;
mod federation;
mod ingestion;
mod ops;
mod search;
mod ttl;
mod vector;

pub use admin::*;
pub use alias::*;
pub use backup::*;
pub use cache::*;
pub use collection::*;
#[cfg(feature = "observability")]
pub use drift::*;
pub use federation::*;
pub use ingestion::*;
pub use ops::*;
pub use search::*;
pub use ttl::*;
pub use vector::*;

// ============================================================================
// Utility functions
// ============================================================================

pub fn parse_distance(distance: &str) -> Option<DistanceFunction> {
    distance.parse().ok()
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_distance ───────────────────────────────────────────────────

    #[test]
    fn test_parse_distance_cosine() {
        assert!(matches!(parse_distance("cosine"), Some(DistanceFunction::Cosine)));
        assert!(matches!(parse_distance("COSINE"), Some(DistanceFunction::Cosine)));
        assert!(matches!(parse_distance("Cosine"), Some(DistanceFunction::Cosine)));
    }

    #[test]
    fn test_parse_distance_euclidean() {
        assert!(matches!(parse_distance("euclidean"), Some(DistanceFunction::Euclidean)));
        assert!(matches!(parse_distance("l2"), Some(DistanceFunction::Euclidean)));
        assert!(matches!(parse_distance("L2"), Some(DistanceFunction::Euclidean)));
    }

    #[test]
    fn test_parse_distance_dot_product() {
        assert!(matches!(parse_distance("dot"), Some(DistanceFunction::DotProduct)));
        assert!(matches!(parse_distance("dotproduct"), Some(DistanceFunction::DotProduct)));
    }

    #[test]
    fn test_parse_distance_manhattan() {
        assert!(matches!(parse_distance("manhattan"), Some(DistanceFunction::Manhattan)));
        assert!(matches!(parse_distance("l1"), Some(DistanceFunction::Manhattan)));
    }

    #[test]
    fn test_parse_distance_invalid() {
        assert!(parse_distance("unknown").is_none());
        assert!(parse_distance("").is_none());
        assert!(parse_distance("hamming").is_none());
    }

    // ── parse_query_vector ───────────────────────────────────────────────

    #[test]
    fn test_parse_query_vector_basic() {
        let result = parse_query_vector("1.0,2.0,3.0").unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_query_vector_with_spaces() {
        let result = parse_query_vector(" 1.0 , 2.0 , 3.0 ").unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_query_vector_single_value() {
        let result = parse_query_vector("42.5").unwrap();
        assert_eq!(result, vec![42.5]);
    }

    #[test]
    fn test_parse_query_vector_negative_values() {
        let result = parse_query_vector("-1.0,0.0,1.0").unwrap();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_parse_query_vector_empty_string() {
        let err = parse_query_vector("").unwrap_err();
        assert!(matches!(err, NeedleError::InvalidVector(_)));
    }

    #[test]
    fn test_parse_query_vector_invalid_float() {
        let err = parse_query_vector("1.0,abc,3.0").unwrap_err();
        assert!(matches!(err, NeedleError::InvalidVector(_)));
    }

    #[test]
    fn test_parse_query_vector_trailing_comma() {
        let result = parse_query_vector("1.0,2.0,").unwrap();
        assert_eq!(result, vec![1.0, 2.0]);
    }

    #[test]
    fn test_parse_query_vector_integers() {
        let result = parse_query_vector("1,2,3").unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_distance_case_variations() {
        assert!(matches!(parse_distance("DOT"), Some(DistanceFunction::DotProduct)));
        assert!(matches!(parse_distance("DOTPRODUCT"), Some(DistanceFunction::DotProduct)));
        assert!(matches!(parse_distance("Manhattan"), Some(DistanceFunction::Manhattan)));
        assert!(matches!(parse_distance("L1"), Some(DistanceFunction::Manhattan)));
        assert!(matches!(parse_distance("Euclidean"), Some(DistanceFunction::Euclidean)));
    }

    #[test]
    fn test_parse_query_vector_whitespace_only() {
        let err = parse_query_vector("  ,  ,  ").unwrap_err();
        assert!(matches!(err, NeedleError::InvalidVector(_)));
    }

    #[test]
    fn test_parse_query_vector_large_values() {
        let result = parse_query_vector("1e10,-1e10,0.0").unwrap();
        assert_eq!(result.len(), 3);
        assert!(result[0] > 0.0);
        assert!(result[1] < 0.0);
    }

    #[test]
    fn test_parse_query_vector_multiple_trailing_commas() {
        let result = parse_query_vector("1.0,2.0,,,").unwrap();
        assert_eq!(result, vec![1.0, 2.0]);
    }
}
