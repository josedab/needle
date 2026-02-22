//! Query Replay & Regression Testing
//!
//! Record production queries as replay logs and replay them against new
//! configurations to measure recall and latency regressions before deploying.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::query_replay::{
//!     QueryRecorder, QueryReplayer, ReplayReport, RecordedQuery,
//! };
//!
//! let mut recorder = QueryRecorder::new(100);
//!
//! // Record queries during production
//! recorder.record(RecordedQuery::new(
//!     "docs", vec![0.1f32; 4], 10,
//!     vec![("d1".into(), 0.05), ("d2".into(), 0.1)],
//!     3.2,
//! ));
//!
//! // Replay against new results to detect regressions
//! let new_results = vec![("d1".into(), 0.06), ("d3".into(), 0.11)];
//! let mut replayer = QueryReplayer::new(recorder.queries());
//! replayer.add_result(0, &new_results, 2.8);
//!
//! let report = replayer.report();
//! println!("Recall@10: {:.2}%", report.avg_recall * 100.0);
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── Recorded Query ───────────────────────────────────────────────────────────

/// A recorded query with its original results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedQuery {
    /// Collection name.
    pub collection: String,
    /// Query vector.
    pub query: Vec<f32>,
    /// K (number of results).
    pub k: usize,
    /// Original results (id, distance).
    pub results: Vec<(String, f32)>,
    /// Original latency in milliseconds.
    pub latency_ms: f64,
    /// Recording timestamp.
    pub timestamp: u64,
}

impl RecordedQuery {
    /// Create a new recorded query.
    pub fn new(
        collection: &str,
        query: Vec<f32>,
        k: usize,
        results: Vec<(String, f32)>,
        latency_ms: f64,
    ) -> Self {
        Self {
            collection: collection.into(),
            query,
            k,
            results,
            latency_ms,
            timestamp: now_secs(),
        }
    }
}

// ── Query Recorder ───────────────────────────────────────────────────────────

/// Records queries for replay testing.
pub struct QueryRecorder {
    queries: Vec<RecordedQuery>,
    max_queries: usize,
    sample_rate: f32,
    counter: u64,
}

impl QueryRecorder {
    /// Create a new recorder with max query capacity.
    pub fn new(max_queries: usize) -> Self {
        Self {
            queries: Vec::new(),
            max_queries,
            sample_rate: 1.0,
            counter: 0,
        }
    }

    /// Set sampling rate (0.0 to 1.0).
    #[must_use]
    pub fn with_sample_rate(mut self, rate: f32) -> Self {
        self.sample_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Record a query.
    pub fn record(&mut self, query: RecordedQuery) {
        self.counter += 1;
        // Simple deterministic sampling
        if self.sample_rate < 1.0 {
            let hash = self.counter.wrapping_mul(0x517c_c1b7_2722_0a95);
            let threshold = (self.sample_rate * u32::MAX as f32) as u64;
            if (hash & 0xFFFF_FFFF) > threshold { return; }
        }
        if self.queries.len() >= self.max_queries {
            self.queries.remove(0);
        }
        self.queries.push(query);
    }

    /// Get recorded queries.
    pub fn queries(&self) -> &[RecordedQuery] {
        &self.queries
    }

    /// Number of recorded queries.
    pub fn len(&self) -> usize {
        self.queries.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }

    /// Clear all recorded queries.
    pub fn clear(&mut self) {
        self.queries.clear();
    }

    /// Serialize to JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(&self.queries).map_err(|e| NeedleError::Serialization(e))
    }

    /// Load from JSON bytes.
    pub fn from_bytes(data: &[u8], max_queries: usize) -> Result<Self> {
        let queries: Vec<RecordedQuery> =
            serde_json::from_slice(data).map_err(|e| NeedleError::Serialization(e))?;
        Ok(Self {
            queries,
            max_queries,
            sample_rate: 1.0,
            counter: 0,
        })
    }
}

// ── Replay Result ────────────────────────────────────────────────────────────

/// Result of replaying a single query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayResult {
    /// Query index.
    pub index: usize,
    /// Recall@k (overlap between original and new results).
    pub recall: f32,
    /// Original latency.
    pub original_latency_ms: f64,
    /// New latency.
    pub new_latency_ms: f64,
    /// Latency change (positive = slower).
    pub latency_change_pct: f64,
    /// IDs present in original but missing in new.
    pub missing_ids: Vec<String>,
    /// IDs present in new but not in original.
    pub new_ids: Vec<String>,
}

// ── Replay Report ────────────────────────────────────────────────────────────

/// Summary report from a replay session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayReport {
    /// Total queries replayed.
    pub total_queries: usize,
    /// Average recall@k.
    pub avg_recall: f32,
    /// Minimum recall.
    pub min_recall: f32,
    /// Average latency change (%).
    pub avg_latency_change_pct: f64,
    /// Queries with recall regression (below threshold).
    pub regressions: usize,
    /// Per-query results.
    pub details: Vec<ReplayResult>,
    /// Regression threshold used.
    pub regression_threshold: f32,
    /// Overall pass/fail.
    pub passed: bool,
}

// ── Query Replayer ───────────────────────────────────────────────────────────

/// Replays recorded queries and compares results.
pub struct QueryReplayer {
    original: Vec<RecordedQuery>,
    results: Vec<Option<(Vec<(String, f32)>, f64)>>,
    regression_threshold: f32,
}

impl QueryReplayer {
    /// Create a replayer from recorded queries.
    pub fn new(queries: &[RecordedQuery]) -> Self {
        let len = queries.len();
        Self {
            original: queries.to_vec(),
            results: vec![None; len],
            regression_threshold: 0.8,
        }
    }

    /// Set regression threshold (recall below this = regression).
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.regression_threshold = threshold;
        self
    }

    /// Add replay results for a query at the given index.
    pub fn add_result(&mut self, index: usize, new_results: &[(String, f32)], latency_ms: f64) {
        if index < self.results.len() {
            self.results[index] = Some((new_results.to_vec(), latency_ms));
        }
    }

    /// Generate the replay report.
    pub fn report(&self) -> ReplayReport {
        let mut details = Vec::new();
        let mut total_recall: f32 = 0.0;
        let mut min_recall: f32 = 1.0;
        let mut total_latency_change: f64 = 0.0;
        let mut regressions = 0;
        let mut count = 0;

        for (i, original) in self.original.iter().enumerate() {
            if let Some((new_results, new_latency)) = &self.results[i] {
                let original_ids: std::collections::HashSet<&str> =
                    original.results.iter().map(|(id, _)| id.as_str()).collect();
                let new_ids_set: std::collections::HashSet<&str> =
                    new_results.iter().map(|(id, _)| id.as_str()).collect();

                let overlap = original_ids.intersection(&new_ids_set).count();
                let recall = if original_ids.is_empty() {
                    1.0
                } else {
                    overlap as f32 / original_ids.len() as f32
                };

                let latency_change = if original.latency_ms > 0.0 {
                    (new_latency - original.latency_ms) / original.latency_ms * 100.0
                } else {
                    0.0
                };

                let missing: Vec<String> = original_ids
                    .difference(&new_ids_set)
                    .map(|s| s.to_string())
                    .collect();
                let new: Vec<String> = new_ids_set
                    .difference(&original_ids)
                    .map(|s| s.to_string())
                    .collect();

                if recall < self.regression_threshold {
                    regressions += 1;
                }

                total_recall += recall;
                min_recall = min_recall.min(recall);
                total_latency_change += latency_change;
                count += 1;

                details.push(ReplayResult {
                    index: i,
                    recall,
                    original_latency_ms: original.latency_ms,
                    new_latency_ms: *new_latency,
                    latency_change_pct: latency_change,
                    missing_ids: missing,
                    new_ids: new,
                });
            }
        }

        let avg_recall = if count > 0 { total_recall / count as f32 } else { 1.0 };
        let avg_latency_change = if count > 0 { total_latency_change / count as f64 } else { 0.0 };

        ReplayReport {
            total_queries: count,
            avg_recall,
            min_recall,
            avg_latency_change_pct: avg_latency_change,
            regressions,
            details,
            regression_threshold: self.regression_threshold,
            passed: regressions == 0,
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_replay() {
        let mut recorder = QueryRecorder::new(100);
        recorder.record(RecordedQuery::new(
            "docs", vec![0.1; 4], 10,
            vec![("d1".into(), 0.05), ("d2".into(), 0.1)],
            3.2,
        ));

        let mut replayer = QueryReplayer::new(recorder.queries());
        replayer.add_result(0, &[("d1".into(), 0.06), ("d2".into(), 0.11)], 2.8);

        let report = replayer.report();
        assert_eq!(report.total_queries, 1);
        assert!((report.avg_recall - 1.0).abs() < 0.01); // perfect recall
        assert!(report.passed);
    }

    #[test]
    fn test_regression_detection() {
        let mut recorder = QueryRecorder::new(100);
        recorder.record(RecordedQuery::new(
            "docs", vec![0.1; 4], 10,
            vec![("d1".into(), 0.05), ("d2".into(), 0.1), ("d3".into(), 0.15)],
            3.0,
        ));

        let mut replayer = QueryReplayer::new(recorder.queries()).with_threshold(0.8);
        // Only 1 of 3 original IDs present → recall = 0.33
        replayer.add_result(0, &[("d1".into(), 0.06), ("d4".into(), 0.2)], 5.0);

        let report = replayer.report();
        assert!(report.avg_recall < 0.5);
        assert!(!report.passed);
        assert_eq!(report.regressions, 1);
    }

    #[test]
    fn test_latency_tracking() {
        let mut recorder = QueryRecorder::new(100);
        recorder.record(RecordedQuery::new("docs", vec![0.1; 4], 5, vec![], 10.0));

        let mut replayer = QueryReplayer::new(recorder.queries());
        replayer.add_result(0, &[], 15.0); // 50% slower

        let report = replayer.report();
        assert!((report.avg_latency_change_pct - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_serialization() {
        let mut recorder = QueryRecorder::new(100);
        recorder.record(RecordedQuery::new("docs", vec![0.1; 4], 5, vec![("d1".into(), 0.05)], 3.0));

        let bytes = recorder.to_bytes().unwrap();
        let restored = QueryRecorder::from_bytes(&bytes, 100).unwrap();
        assert_eq!(restored.len(), 1);
    }

    #[test]
    fn test_max_capacity() {
        let mut recorder = QueryRecorder::new(3);
        for i in 0..5 {
            recorder.record(RecordedQuery::new("d", vec![i as f32], 5, vec![], 1.0));
        }
        assert_eq!(recorder.len(), 3);
    }

    #[test]
    fn test_missing_new_ids() {
        let mut recorder = QueryRecorder::new(100);
        recorder.record(RecordedQuery::new("d", vec![0.1; 4], 5,
            vec![("d1".into(), 0.1), ("d2".into(), 0.2)], 1.0));

        let mut replayer = QueryReplayer::new(recorder.queries());
        replayer.add_result(0, &[("d1".into(), 0.1), ("d3".into(), 0.3)], 1.0);

        let report = replayer.report();
        assert_eq!(report.details[0].missing_ids, vec!["d2".to_string()]);
        assert_eq!(report.details[0].new_ids, vec!["d3".to_string()]);
    }
}
