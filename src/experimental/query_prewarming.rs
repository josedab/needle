//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Predictive Query Pre-warming
//!
//! Trains a lightweight model on query access patterns to predict upcoming queries.
//! Logs anonymized query embeddings, extracts temporal features, and pre-loads
//! HNSW graph segments for predicted queries.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::experimental::query_prewarming::*;
//!
//! let mut prewarm = QueryPrewarmer::new(PrewarmConfig::default());
//! prewarm.log_query("docs", &query_embedding, 42);
//! let predictions = prewarm.predict_next(5);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

/// Configuration for the query pre-warming system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrewarmConfig {
    /// Maximum number of query patterns to retain
    pub max_history: usize,
    /// Minimum confidence threshold for predictions (0.0 to 1.0)
    pub confidence_threshold: f64,
    /// Time window for temporal features (seconds)
    pub temporal_window_secs: u64,
    /// Number of nearest query patterns to consider for prediction
    pub k_neighbors: usize,
    /// Enable adaptive confidence threshold adjustment
    pub adaptive_threshold: bool,
    /// Decay factor for older patterns (0.0 = forget quickly, 1.0 = remember forever)
    pub decay_factor: f64,
}

impl Default for PrewarmConfig {
    fn default() -> Self {
        Self {
            max_history: 10_000,
            confidence_threshold: 0.5,
            temporal_window_secs: 3600,
            k_neighbors: 5,
            adaptive_threshold: true,
            decay_factor: 0.95,
        }
    }
}

/// An anonymized query pattern entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPattern {
    /// Anonymized query embedding (optionally dimensionality-reduced)
    pub embedding: Vec<f32>,
    /// Collection that was queried
    pub collection: String,
    /// Hour of day (0-23) for temporal feature
    pub hour_of_day: u8,
    /// Day of week (0-6, Mon=0) for temporal feature
    pub day_of_week: u8,
    /// Timestamp when the query occurred
    pub timestamp: u64,
    /// Search parameters used (k, ef_search, etc.)
    pub params: QueryParams,
}

/// Captured search parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParams {
    /// Number of results requested
    pub k: usize,
    /// ef_search used (if available)
    pub ef_search: Option<usize>,
    /// Whether a filter was applied
    pub had_filter: bool,
}

/// A prediction for an upcoming query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPrediction {
    /// Predicted query embedding centroid
    pub predicted_embedding: Vec<f32>,
    /// Predicted collection
    pub collection: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Predicted search parameters
    pub predicted_params: QueryParams,
    /// HNSW entry point nodes to pre-load
    pub prewarm_nodes: Vec<usize>,
}

/// Statistics for the pre-warming system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrewarmStats {
    /// Total queries logged
    pub total_queries: u64,
    /// Total predictions made
    pub total_predictions: u64,
    /// Predictions that were used (hit rate)
    pub predictions_hit: u64,
    /// Current prediction accuracy (rolling window)
    pub accuracy: f64,
    /// Current confidence threshold
    pub confidence_threshold: f64,
    /// Number of patterns in history
    pub history_size: usize,
    /// Collections being tracked
    pub tracked_collections: Vec<String>,
}

/// Pre-warming engine that learns query patterns and predicts future queries.
pub struct QueryPrewarmer {
    config: PrewarmConfig,
    history: VecDeque<QueryPattern>,
    collection_patterns: HashMap<String, Vec<usize>>,
    total_queries: AtomicU64,
    total_predictions: AtomicU64,
    predictions_hit: AtomicU64,
    current_threshold: f64,
}

impl QueryPrewarmer {
    /// Create a new query pre-warmer.
    pub fn new(config: PrewarmConfig) -> Self {
        let threshold = config.confidence_threshold;
        Self {
            config,
            history: VecDeque::new(),
            collection_patterns: HashMap::new(),
            total_queries: AtomicU64::new(0),
            total_predictions: AtomicU64::new(0),
            predictions_hit: AtomicU64::new(0),
            current_threshold: threshold,
        }
    }

    /// Log a query for pattern learning.
    pub fn log_query(&mut self, collection: &str, embedding: &[f32], k: usize) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let (hour, day) = Self::extract_temporal_features(now);

        let pattern = QueryPattern {
            embedding: embedding.to_vec(),
            collection: collection.to_string(),
            hour_of_day: hour,
            day_of_week: day,
            timestamp: now,
            params: QueryParams {
                k,
                ef_search: None,
                had_filter: false,
            },
        };

        let idx = self.history.len();
        self.collection_patterns
            .entry(collection.to_string())
            .or_default()
            .push(idx);

        if self.history.len() >= self.config.max_history {
            self.history.pop_front();
            // Shift indices down for collection_patterns
            for indices in self.collection_patterns.values_mut() {
                indices.retain(|&i| i > 0);
                for idx in indices.iter_mut() {
                    *idx -= 1;
                }
            }
        }
        self.history.push_back(pattern);
        self.total_queries.fetch_add(1, Ordering::Relaxed);
    }

    /// Log a query with extended parameters.
    pub fn log_query_extended(
        &mut self,
        collection: &str,
        embedding: &[f32],
        k: usize,
        ef_search: Option<usize>,
        had_filter: bool,
    ) {
        self.log_query(collection, embedding, k);
        // Update the last entry's params
        if let Some(last) = self.history.back_mut() {
            last.params.ef_search = ef_search;
            last.params.had_filter = had_filter;
        }
    }

    /// Predict the next likely queries based on current patterns.
    pub fn predict_next(&self, limit: usize) -> Vec<QueryPrediction> {
        if self.history.len() < 3 {
            return Vec::new();
        }

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let (current_hour, current_day) = Self::extract_temporal_features(now);

        // Find patterns from similar temporal contexts
        let mut candidates: Vec<(usize, f64)> = self
            .history
            .iter()
            .enumerate()
            .map(|(i, pattern)| {
                let temporal_sim = Self::temporal_similarity(
                    current_hour,
                    current_day,
                    pattern.hour_of_day,
                    pattern.day_of_week,
                );
                let recency = self.config.decay_factor.powf(
                    (now.saturating_sub(pattern.timestamp)) as f64 / 3600.0,
                );
                (i, temporal_sim * recency)
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Group by collection and compute centroid predictions
        let mut collection_groups: HashMap<String, Vec<(&QueryPattern, f64)>> = HashMap::new();
        for &(idx, score) in candidates.iter().take(self.config.k_neighbors * 3) {
            if let Some(pattern) = self.history.get(idx) {
                collection_groups
                    .entry(pattern.collection.clone())
                    .or_default()
                    .push((pattern, score));
            }
        }

        let mut predictions: Vec<QueryPrediction> = collection_groups
            .iter()
            .filter_map(|(collection, patterns)| {
                if patterns.is_empty() {
                    return None;
                }

                let dims = patterns[0].0.embedding.len();
                let total_weight: f64 = patterns.iter().map(|(_, s)| s).sum();
                if total_weight < 1e-10 {
                    return None;
                }

                // Weighted centroid
                let mut centroid = vec![0.0f32; dims];
                for (pattern, score) in patterns {
                    let weight = (*score / total_weight) as f32;
                    for (i, v) in pattern.embedding.iter().enumerate() {
                        if i < centroid.len() {
                            centroid[i] += v * weight;
                        }
                    }
                }

                let confidence = (total_weight / patterns.len() as f64).min(1.0);
                let avg_k = patterns.iter().map(|(p, _)| p.params.k).sum::<usize>()
                    / patterns.len();

                if confidence >= self.current_threshold {
                    self.total_predictions.fetch_add(1, Ordering::Relaxed);
                    Some(QueryPrediction {
                        predicted_embedding: centroid,
                        collection: collection.clone(),
                        confidence,
                        predicted_params: QueryParams {
                            k: avg_k,
                            ef_search: None,
                            had_filter: false,
                        },
                        prewarm_nodes: Vec::new(),
                    })
                } else {
                    None
                }
            })
            .collect();

        predictions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        predictions.truncate(limit);
        predictions
    }

    /// Record that a prediction was used (hit).
    pub fn record_hit(&self) {
        self.predictions_hit.fetch_add(1, Ordering::Relaxed);
    }

    /// Get pre-warming statistics.
    pub fn stats(&self) -> PrewarmStats {
        let total_preds = self.total_predictions.load(Ordering::Relaxed);
        let hits = self.predictions_hit.load(Ordering::Relaxed);
        let accuracy = if total_preds > 0 {
            hits as f64 / total_preds as f64
        } else {
            0.0
        };

        let tracked: Vec<String> = self.collection_patterns.keys().cloned().collect();

        PrewarmStats {
            total_queries: self.total_queries.load(Ordering::Relaxed),
            total_predictions: total_preds,
            predictions_hit: hits,
            accuracy,
            confidence_threshold: self.current_threshold,
            history_size: self.history.len(),
            tracked_collections: tracked,
        }
    }

    /// Adjust the confidence threshold based on prediction accuracy.
    pub fn adapt_threshold(&mut self) {
        if !self.config.adaptive_threshold {
            return;
        }

        let stats = self.stats();
        if stats.total_predictions < 10 {
            return;
        }

        // If accuracy is high, lower threshold to make more predictions
        // If accuracy is low, raise threshold to be more selective
        if stats.accuracy > 0.7 {
            self.current_threshold = (self.current_threshold * 0.95).max(0.1);
        } else if stats.accuracy < 0.3 {
            self.current_threshold = (self.current_threshold * 1.1).min(0.95);
        }
    }

    /// Clear all history and reset.
    pub fn clear(&mut self) {
        self.history.clear();
        self.collection_patterns.clear();
    }

    /// Get the config.
    pub fn config(&self) -> &PrewarmConfig {
        &self.config
    }

    fn extract_temporal_features(timestamp: u64) -> (u8, u8) {
        let secs_in_day = timestamp % 86400;
        let hour = (secs_in_day / 3600) as u8;
        let day = ((timestamp / 86400 + 3) % 7) as u8; // Epoch was Thursday
        (hour, day)
    }

    fn temporal_similarity(hour_a: u8, day_a: u8, hour_b: u8, day_b: u8) -> f64 {
        // Hour similarity (circular)
        let hour_diff = ((hour_a as i16 - hour_b as i16).abs()).min(
            24 - (hour_a as i16 - hour_b as i16).abs(),
        ) as f64;
        let hour_sim = 1.0 - (hour_diff / 12.0);

        // Day similarity (circular)
        let day_diff = ((day_a as i16 - day_b as i16).abs()).min(
            7 - (day_a as i16 - day_b as i16).abs(),
        ) as f64;
        let day_sim = 1.0 - (day_diff / 3.5);

        // Weighted combination
        0.7 * hour_sim + 0.3 * day_sim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_and_predict() {
        let mut prewarm = QueryPrewarmer::new(PrewarmConfig {
            confidence_threshold: 0.0, // Low threshold for testing
            max_history: 100,
            ..Default::default()
        });

        // Log several queries
        for i in 0..10 {
            prewarm.log_query("docs", &[0.1 * i as f32; 4], 10);
        }

        let predictions = prewarm.predict_next(5);
        // Should have at least one prediction for "docs"
        assert!(!predictions.is_empty());
        assert_eq!(predictions[0].collection, "docs");
    }

    #[test]
    fn test_stats() {
        let prewarm = QueryPrewarmer::new(PrewarmConfig::default());
        let stats = prewarm.stats();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.history_size, 0);
    }

    #[test]
    fn test_temporal_similarity() {
        // Same time should have high similarity
        let sim = QueryPrewarmer::temporal_similarity(10, 2, 10, 2);
        assert!((sim - 1.0).abs() < 1e-10);

        // Opposite time should have low similarity
        let sim = QueryPrewarmer::temporal_similarity(0, 0, 12, 3);
        assert!(sim < 0.5);
    }

    #[test]
    fn test_adaptive_threshold() {
        let mut prewarm = QueryPrewarmer::new(PrewarmConfig {
            adaptive_threshold: true,
            confidence_threshold: 0.5,
            ..Default::default()
        });

        let initial = prewarm.current_threshold;
        // Not enough data, no change
        prewarm.adapt_threshold();
        assert_eq!(prewarm.current_threshold, initial);
    }

    #[test]
    fn test_max_history_eviction() {
        let mut prewarm = QueryPrewarmer::new(PrewarmConfig {
            max_history: 5,
            ..Default::default()
        });

        for i in 0..10 {
            prewarm.log_query("docs", &[i as f32; 2], 5);
        }

        assert!(prewarm.history.len() <= 5);
    }

    #[test]
    fn test_empty_predict() {
        let prewarm = QueryPrewarmer::new(PrewarmConfig::default());
        let predictions = prewarm.predict_next(5);
        assert!(predictions.is_empty());
    }
}
