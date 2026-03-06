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

/// Strategy for normalizing search result scores into a \[0, 1\] range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoreNormalization {
    /// Min-max normalization: maps the best distance to 1.0, worst to 0.0.
    /// Best for comparing scores within a single result set.
    MinMax,
    /// Inverse-distance normalization: `1 / (1 + distance)`.
    /// Produces scores in (0, 1] without needing multiple results for context.
    InverseDistance,
}

/// Normalize search result distances into similarity scores in \[0, 1\].
///
/// Higher scores indicate greater similarity. The original `distance` field is
/// replaced with the normalized score.
///
/// # Examples
///
/// ```
/// use needle::{SearchResult, normalize_scores, ScoreNormalization};
///
/// let mut results = vec![
///     SearchResult::new("closest", 0.1, None),
///     SearchResult::new("mid", 0.5, None),
///     SearchResult::new("far", 1.0, None),
/// ];
/// normalize_scores(&mut results, ScoreNormalization::MinMax);
/// assert!((results[0].distance - 1.0).abs() < f32::EPSILON);
/// assert!((results[2].distance - 0.0).abs() < f32::EPSILON);
/// ```
pub fn normalize_scores(results: &mut [SearchResult], method: ScoreNormalization) {
    if results.is_empty() {
        return;
    }

    match method {
        ScoreNormalization::MinMax => {
            let min_dist = results
                .iter()
                .map(|r| r.distance)
                .fold(f32::INFINITY, f32::min);
            let max_dist = results
                .iter()
                .map(|r| r.distance)
                .fold(f32::NEG_INFINITY, f32::max);
            let range = max_dist - min_dist;

            if range < f32::EPSILON {
                // All distances equal — assign score 1.0 (perfect match)
                for r in results.iter_mut() {
                    r.distance = 1.0;
                }
            } else {
                for r in results.iter_mut() {
                    r.distance = 1.0 - (r.distance - min_dist) / range;
                }
            }
        }
        ScoreNormalization::InverseDistance => {
            for r in results.iter_mut() {
                r.distance = 1.0 / (1.0 + r.distance);
            }
        }
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

impl SearchExplain {
    /// Format the explain output as a human-readable ASCII report.
    ///
    /// Includes timing breakdown, index traversal stats, filter selectivity,
    /// and an ASCII bar chart showing time distribution.
    pub fn format_ascii(&self) -> String {
        let mut out = String::new();

        out.push_str("╔══════════════════════════════════════════════════╗\n");
        out.push_str("║           NEEDLE SEARCH EXPLAIN                 ║\n");
        out.push_str("╠══════════════════════════════════════════════════╣\n");

        // Collection info
        out.push_str(&format!(
            "║ Collection: {} vectors, {} dims, {}\n",
            self.collection_size, self.dimensions, self.distance_function
        ));
        out.push_str(&format!(
            "║ Query: k={} (effective={}), ef_search={}\n",
            self.requested_k, self.effective_k, self.ef_search
        ));

        out.push_str("╠══════════════════════════════════════════════════╣\n");

        // Timing breakdown
        out.push_str("║ TIMING BREAKDOWN:\n");
        out.push_str(&format!("║   Total:    {:>8}μs\n", self.total_time_us));
        out.push_str(&format!("║   Index:    {:>8}μs\n", self.index_time_us));
        out.push_str(&format!("║   Filter:   {:>8}μs\n", self.filter_time_us));
        out.push_str(&format!("║   Enrich:   {:>8}μs\n", self.enrich_time_us));

        // ASCII bar chart
        if self.total_time_us > 0 {
            let total = self.total_time_us as f64;
            let idx_pct = (self.index_time_us as f64 / total * 30.0) as usize;
            let flt_pct = (self.filter_time_us as f64 / total * 30.0) as usize;
            let enr_pct = (self.enrich_time_us as f64 / total * 30.0) as usize;

            out.push_str("║\n");
            out.push_str(&format!(
                "║   Index:  [{}{}]\n",
                "█".repeat(idx_pct.min(30)),
                "░".repeat(30_usize.saturating_sub(idx_pct))
            ));
            out.push_str(&format!(
                "║   Filter: [{}{}]\n",
                "█".repeat(flt_pct.min(30)),
                "░".repeat(30_usize.saturating_sub(flt_pct))
            ));
            out.push_str(&format!(
                "║   Enrich: [{}{}]\n",
                "█".repeat(enr_pct.min(30)),
                "░".repeat(30_usize.saturating_sub(enr_pct))
            ));
        }

        out.push_str("╠══════════════════════════════════════════════════╣\n");

        // HNSW traversal stats
        out.push_str("║ HNSW TRAVERSAL:\n");
        out.push_str(&format!(
            "║   Nodes visited:         {}\n",
            self.hnsw_stats.visited_nodes
        ));
        out.push_str(&format!(
            "║   Layers traversed:      {}\n",
            self.hnsw_stats.layers_traversed
        ));
        out.push_str(&format!(
            "║   Distance computations: {}\n",
            self.hnsw_stats.distance_computations
        ));

        // Filter selectivity
        if self.filter_applied {
            out.push_str("╠══════════════════════════════════════════════════╣\n");
            out.push_str("║ FILTER SELECTIVITY:\n");
            out.push_str(&format!(
                "║   Candidates: {} → {} ({:.1}% pass rate)\n",
                self.candidates_before_filter,
                self.candidates_after_filter,
                if self.candidates_before_filter > 0 {
                    self.candidates_after_filter as f64 / self.candidates_before_filter as f64
                        * 100.0
                } else {
                    0.0
                }
            ));
        }

        out.push_str("╚══════════════════════════════════════════════════╝\n");

        out
    }
}

impl std::fmt::Display for SearchExplain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_ascii())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_explain_format_ascii_basic() {
        let explain = SearchExplain {
            total_time_us: 1000,
            index_time_us: 700,
            filter_time_us: 100,
            enrich_time_us: 200,
            candidates_before_filter: 50,
            candidates_after_filter: 10,
            hnsw_stats: crate::hnsw::SearchStats {
                visited_nodes: 42,
                layers_traversed: 3,
                distance_computations: 150,
                traversal_time_us: 700,
            },
            dimensions: 384,
            collection_size: 10000,
            requested_k: 10,
            effective_k: 10,
            ef_search: 50,
            filter_applied: true,
            distance_function: "Cosine".to_string(),
        };

        let output = explain.format_ascii();
        assert!(output.contains("NEEDLE SEARCH EXPLAIN"));
        assert!(output.contains("10000 vectors"));
        assert!(output.contains("384 dims"));
        assert!(output.contains("Cosine"));
        assert!(output.contains("1000μs")); // total time
        assert!(output.contains("700μs")); // index time
        assert!(output.contains("42")); // visited nodes
        assert!(output.contains("FILTER SELECTIVITY"));
        assert!(output.contains("50 →")); // filter candidates
        assert!(output.contains("█")); // bar chart
    }

    #[test]
    fn test_search_explain_format_no_filter() {
        let explain = SearchExplain {
            filter_applied: false,
            ..Default::default()
        };
        let output = explain.format_ascii();
        assert!(!output.contains("FILTER SELECTIVITY"));
    }

    #[test]
    fn test_search_explain_display_impl() {
        let explain = SearchExplain {
            total_time_us: 500,
            dimensions: 128,
            collection_size: 100,
            ..Default::default()
        };
        let display = format!("{}", explain);
        assert!(display.contains("NEEDLE SEARCH EXPLAIN"));
    }

    #[test]
    fn test_search_trace_format_ascii() {
        let trace = crate::hnsw::SearchTrace {
            hops: vec![
                crate::hnsw::TraceHop {
                    layer: 1,
                    node_id: 5,
                    distance: 0.5,
                    added_to_candidates: true,
                    neighbors_explored: 3,
                },
                crate::hnsw::TraceHop {
                    layer: 0,
                    node_id: 2,
                    distance: 0.2,
                    added_to_candidates: true,
                    neighbors_explored: 8,
                },
                crate::hnsw::TraceHop {
                    layer: 0,
                    node_id: 7,
                    distance: 0.8,
                    added_to_candidates: false,
                    neighbors_explored: 4,
                },
            ],
            entry_point: Some(5),
            entry_layer: 1,
            results: vec![(2, 0.2), (5, 0.5)],
            distance_computations: 15,
            nodes_visited: 3,
            layers_traversed: vec![1, 0],
        };

        let output = trace.format_ascii();
        assert!(output.contains("HNSW Search Trace"));
        assert!(output.contains("Entry point: node 5 (layer 1)"));
        assert!(output.contains("3 nodes visited"));
        assert!(output.contains("15 distance computations"));
        assert!(output.contains("Layer 1:"));
        assert!(output.contains("Layer 0:"));
        assert!(output.contains("Results:"));
    }

    #[test]
    fn test_search_trace_display() {
        let trace = crate::hnsw::SearchTrace::default();
        let output = format!("{}", trace);
        assert!(output.contains("HNSW Search Trace"));
    }

    // ── Score normalization tests ─────────────────────────────────────

    #[test]
    fn test_normalize_scores_minmax_basic() {
        let mut results = vec![
            SearchResult::new("a", 0.1, None),
            SearchResult::new("b", 0.5, None),
            SearchResult::new("c", 1.0, None),
        ];
        normalize_scores(&mut results, ScoreNormalization::MinMax);

        // Closest → 1.0, farthest → 0.0
        assert!((results[0].distance - 1.0).abs() < f32::EPSILON);
        assert!((results[2].distance - 0.0).abs() < f32::EPSILON);
        // Mid should be ~0.444
        let expected_mid = 1.0 - (0.5 - 0.1) / (1.0 - 0.1);
        assert!((results[1].distance - expected_mid).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_scores_minmax_single_result() {
        let mut results = vec![SearchResult::new("only", 0.5, None)];
        normalize_scores(&mut results, ScoreNormalization::MinMax);
        assert!((results[0].distance - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalize_scores_minmax_equal_distances() {
        let mut results = vec![
            SearchResult::new("a", 0.3, None),
            SearchResult::new("b", 0.3, None),
        ];
        normalize_scores(&mut results, ScoreNormalization::MinMax);
        assert!((results[0].distance - 1.0).abs() < f32::EPSILON);
        assert!((results[1].distance - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalize_scores_minmax_empty() {
        let mut results: Vec<SearchResult> = vec![];
        normalize_scores(&mut results, ScoreNormalization::MinMax);
        assert!(results.is_empty());
    }

    #[test]
    fn test_normalize_scores_inverse_distance() {
        let mut results = vec![
            SearchResult::new("a", 0.0, None),
            SearchResult::new("b", 1.0, None),
            SearchResult::new("c", 3.0, None),
        ];
        normalize_scores(&mut results, ScoreNormalization::InverseDistance);

        assert!((results[0].distance - 1.0).abs() < f32::EPSILON); // 1/(1+0)
        assert!((results[1].distance - 0.5).abs() < f32::EPSILON); // 1/(1+1)
        assert!((results[2].distance - 0.25).abs() < f32::EPSILON); // 1/(1+3)
    }

    #[test]
    fn test_normalize_scores_preserves_metadata() {
        let mut results = vec![
            SearchResult::new("a", 0.1, Some(serde_json::json!({"key": "val"}))),
        ];
        normalize_scores(&mut results, ScoreNormalization::InverseDistance);
        assert!(results[0].metadata.is_some());
        assert_eq!(results[0].metadata.as_ref().unwrap()["key"], "val");
    }
}
