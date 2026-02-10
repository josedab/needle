//! Search result types and query explanation.
//!
//! This module contains the output types from vector similarity searches.

use serde_json::Value;

/// A single result from a vector similarity search.
///
/// Contains the vector's identifier, its distance from the query, and optionally
/// the associated metadata. Results are typically returned sorted by distance
/// (ascending for most distance functions).
///
/// # Distance Interpretation
///
/// The meaning of the `distance` field depends on the collection's distance function:
///
/// | Function | Range | Interpretation |
/// |----------|-------|----------------|
/// | Cosine | 0.0 - 2.0 | 0 = identical, 2 = opposite |
/// | Euclidean | 0.0 - ∞ | 0 = identical |
/// | DotProduct | -∞ - +∞ | Higher (less negative) = more similar |
/// | Manhattan | 0.0 - ∞ | 0 = identical |
///
/// # Example
///
/// ```
/// use needle::SearchResult;
///
/// let result = SearchResult::new("doc_123", 0.15, None);
/// println!("Found {} with distance {}", result.id, result.distance);
///
/// if let Some(meta) = &result.metadata {
///     println!("Metadata: {}", meta);
/// }
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The unique string identifier of the vector.
    pub id: String,

    /// Distance from the query vector. Lower values indicate greater similarity
    /// for Cosine, Euclidean, and Manhattan; for DotProduct, higher (less negative)
    /// values indicate greater similarity.
    pub distance: f32,

    /// Optional JSON metadata associated with this vector. Only populated if
    /// the vector has metadata and `include_metadata` was not set to false
    /// in the search builder.
    pub metadata: Option<Value>,
}

impl SearchResult {
    /// Create a new search result
    #[must_use]
    pub fn new(id: impl Into<String>, distance: f32, metadata: Option<Value>) -> Self {
        Self {
            id: id.into(),
            distance,
            metadata,
        }
    }
}

impl From<(String, f32)> for SearchResult {
    fn from((id, distance): (String, f32)) -> Self {
        Self {
            id,
            distance,
            metadata: None,
        }
    }
}

impl From<(String, f32, Option<Value>)> for SearchResult {
    fn from((id, distance, metadata): (String, f32, Option<Value>)) -> Self {
        Self {
            id,
            distance,
            metadata,
        }
    }
}

impl From<SearchResult> for (String, f32, Option<Value>) {
    fn from(result: SearchResult) -> Self {
        (result.id, result.distance, result.metadata)
    }
}

/// Detailed query execution plan and profiling information.
///
/// Returned by search operations when explain mode is enabled, providing
/// insights into query performance for optimization and debugging.
///
/// # Example
///
/// ```
/// use needle::Collection;
///
/// let mut collection = Collection::with_dimensions("test", 4);
/// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
///
/// let (results, explain) = collection.search_explain(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();
/// println!("Total time: {}μs", explain.total_time_us);
/// println!("Nodes visited: {}", explain.hnsw_stats.visited_nodes);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SearchExplain {
    /// Total search time in microseconds
    pub total_time_us: u64,
    /// Time spent in HNSW index traversal (microseconds)
    pub index_time_us: u64,
    /// Time spent evaluating metadata filters (microseconds)
    pub filter_time_us: u64,
    /// Time spent enriching results with metadata (microseconds)
    pub enrich_time_us: u64,
    /// Number of results before filtering
    pub candidates_before_filter: usize,
    /// Number of results after filtering
    pub candidates_after_filter: usize,
    /// HNSW index statistics
    pub hnsw_stats: crate::hnsw::SearchStats,
    /// Collection dimensions
    pub dimensions: usize,
    /// Collection vector count
    pub collection_size: usize,
    /// Requested k value
    pub requested_k: usize,
    /// Effective k (clamped to collection size)
    pub effective_k: usize,
    /// ef_search parameter used
    pub ef_search: usize,
    /// Whether a filter was applied
    pub filter_applied: bool,
    /// Distance function used
    pub distance_function: String,
}
