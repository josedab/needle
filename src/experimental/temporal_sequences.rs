#![allow(clippy::unwrap_used)]
//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Temporal Vector Sequences
//!
//! First-class support for time-series vector data with trajectory search,
//! drift detection, and temporal aggregation functions.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::experimental::temporal_sequences::*;
//!
//! let mut seq = VectorSequence::new("user_behavior", 128);
//! seq.push(timestamp1, &embedding1);
//! seq.push(timestamp2, &embedding2);
//!
//! let drift = seq.detect_drift(DriftMethod::KlDivergence, 10);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A timestamped vector entry in a sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedVector {
    /// Timestamp (Unix epoch seconds)
    pub timestamp: u64,
    /// The vector data
    pub vector: Vec<f32>,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

/// A sequence of vectors over time, representing a trajectory or time-series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSequence {
    /// Sequence identifier
    pub id: String,
    /// Vector dimensions
    pub dimensions: usize,
    /// Timestamped vector entries, sorted by timestamp
    pub entries: Vec<TimestampedVector>,
}

impl VectorSequence {
    /// Create a new empty vector sequence.
    pub fn new(id: impl Into<String>, dimensions: usize) -> Self {
        Self {
            id: id.into(),
            dimensions,
            entries: Vec::new(),
        }
    }

    /// Add a vector to the sequence at the given timestamp.
    pub fn push(&mut self, timestamp: u64, vector: &[f32]) -> Result<(), String> {
        if vector.len() != self.dimensions {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            ));
        }
        self.entries.push(TimestampedVector {
            timestamp,
            vector: vector.to_vec(),
            metadata: None,
        });
        self.entries.sort_by_key(|e| e.timestamp);
        Ok(())
    }

    /// Add a vector with metadata.
    pub fn push_with_metadata(
        &mut self,
        timestamp: u64,
        vector: &[f32],
        metadata: serde_json::Value,
    ) -> Result<(), String> {
        if vector.len() != self.dimensions {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            ));
        }
        self.entries.push(TimestampedVector {
            timestamp,
            vector: vector.to_vec(),
            metadata: Some(metadata),
        });
        self.entries.sort_by_key(|e| e.timestamp);
        Ok(())
    }

    /// Get entries within a time range.
    pub fn range(&self, start: u64, end: u64) -> Vec<&TimestampedVector> {
        self.entries
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .collect()
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Compute the centroid of all vectors in the sequence.
    pub fn centroid(&self) -> Option<Vec<f32>> {
        if self.entries.is_empty() {
            return None;
        }
        let n = self.entries.len() as f32;
        let mut result = vec![0.0f32; self.dimensions];
        for entry in &self.entries {
            for (i, v) in entry.vector.iter().enumerate() {
                result[i] += v;
            }
        }
        for v in &mut result {
            *v /= n;
        }
        Some(result)
    }

    /// Compute a windowed centroid for a time window.
    pub fn window_centroid(&self, start: u64, end: u64) -> Option<Vec<f32>> {
        let entries = self.range(start, end);
        if entries.is_empty() {
            return None;
        }
        let n = entries.len() as f32;
        let mut result = vec![0.0f32; self.dimensions];
        for entry in &entries {
            for (i, v) in entry.vector.iter().enumerate() {
                result[i] += v;
            }
        }
        for v in &mut result {
            *v /= n;
        }
        Some(result)
    }

    /// Compute the trend direction vector (last centroid - first centroid).
    pub fn trend(&self, window_size: usize) -> Option<Vec<f32>> {
        if self.entries.len() < window_size * 2 {
            return None;
        }

        let first_window: Vec<&[f32]> = self.entries[..window_size]
            .iter()
            .map(|e| e.vector.as_slice())
            .collect();
        let last_window: Vec<&[f32]> = self.entries[self.entries.len() - window_size..]
            .iter()
            .map(|e| e.vector.as_slice())
            .collect();

        let first_centroid = compute_centroid(&first_window, self.dimensions);
        let last_centroid = compute_centroid(&last_window, self.dimensions);

        Some(
            first_centroid
                .iter()
                .zip(last_centroid.iter())
                .map(|(a, b)| b - a)
                .collect(),
        )
    }
}

fn compute_centroid(vectors: &[&[f32]], dims: usize) -> Vec<f32> {
    let n = vectors.len() as f32;
    let mut result = vec![0.0f32; dims];
    for v in vectors {
        for (i, val) in v.iter().enumerate() {
            result[i] += val;
        }
    }
    for v in &mut result {
        *v /= n;
    }
    result
}

/// Methods for trajectory distance computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrajectoryDistance {
    /// Dynamic Time Warping — elastic alignment of sequences
    Dtw,
    /// Discrete Fréchet distance — "dog walking" metric
    Frechet,
    /// Euclidean distance between sequence centroids
    CentroidDistance,
}

/// Compute the Dynamic Time Warping distance between two sequences.
pub fn dtw_distance(seq_a: &[Vec<f32>], seq_b: &[Vec<f32>]) -> f64 {
    let n = seq_a.len();
    let m = seq_b.len();
    if n == 0 || m == 0 {
        return f64::MAX;
    }

    let mut dp = vec![vec![f64::MAX; m + 1]; n + 1];
    dp[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            let cost = euclidean_dist(&seq_a[i - 1], &seq_b[j - 1]);
            dp[i][j] = cost + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
        }
    }

    dp[n][m]
}

/// Compute the discrete Fréchet distance between two sequences.
pub fn frechet_distance(seq_a: &[Vec<f32>], seq_b: &[Vec<f32>]) -> f64 {
    let n = seq_a.len();
    let m = seq_b.len();
    if n == 0 || m == 0 {
        return f64::MAX;
    }

    let mut dp = vec![vec![-1.0f64; m]; n];

    fn recurse(
        dp: &mut Vec<Vec<f64>>,
        seq_a: &[Vec<f32>],
        seq_b: &[Vec<f32>],
        i: usize,
        j: usize,
    ) -> f64 {
        if dp[i][j] >= 0.0 {
            return dp[i][j];
        }

        let d = euclidean_dist(&seq_a[i], &seq_b[j]);

        dp[i][j] = if i == 0 && j == 0 {
            d
        } else if i == 0 {
            d.max(recurse(dp, seq_a, seq_b, 0, j - 1))
        } else if j == 0 {
            d.max(recurse(dp, seq_a, seq_b, i - 1, 0))
        } else {
            let prev = recurse(dp, seq_a, seq_b, i - 1, j)
                .min(recurse(dp, seq_a, seq_b, i, j - 1))
                .min(recurse(dp, seq_a, seq_b, i - 1, j - 1));
            d.max(prev)
        };

        dp[i][j]
    }

    recurse(&mut dp, seq_a, seq_b, n - 1, m - 1)
}

fn euclidean_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x - *y) as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Drift detection methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftMethod {
    /// KL divergence between distribution windows
    KlDivergence,
    /// Maximum Mean Discrepancy
    Mmd,
    /// Cosine drift (angular change between centroids)
    CosineDrift,
}

/// Result of drift detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftResult {
    /// Drift score (higher = more drift)
    pub score: f64,
    /// Whether drift exceeds the threshold
    pub drift_detected: bool,
    /// Method used for detection
    pub method: DriftMethod,
    /// Threshold used
    pub threshold: f64,
    /// Window start timestamps
    pub window_a_start: u64,
    /// Window end timestamps
    pub window_b_end: u64,
}

/// Detect drift in a vector sequence using the specified method.
pub fn detect_drift(
    sequence: &VectorSequence,
    method: DriftMethod,
    window_size: usize,
    threshold: f64,
) -> Option<DriftResult> {
    if sequence.entries.len() < window_size * 2 {
        return None;
    }

    let mid = sequence.entries.len() / 2;
    let window_a: Vec<&[f32]> = sequence.entries[mid - window_size..mid]
        .iter()
        .map(|e| e.vector.as_slice())
        .collect();
    let window_b: Vec<&[f32]> = sequence.entries[mid..mid + window_size]
        .iter()
        .map(|e| e.vector.as_slice())
        .collect();

    let score = match method {
        DriftMethod::KlDivergence => {
            // Approximate KL divergence via centroid distance
            let ca = compute_centroid(&window_a, sequence.dimensions);
            let cb = compute_centroid(&window_b, sequence.dimensions);
            euclidean_dist(&ca, &cb)
        }
        DriftMethod::Mmd => {
            // Simplified MMD: ||mean(A) - mean(B)||^2
            let ca = compute_centroid(&window_a, sequence.dimensions);
            let cb = compute_centroid(&window_b, sequence.dimensions);
            let dist = euclidean_dist(&ca, &cb);
            dist * dist
        }
        DriftMethod::CosineDrift => {
            let ca = compute_centroid(&window_a, sequence.dimensions);
            let cb = compute_centroid(&window_b, sequence.dimensions);
            let dot: f64 = ca.iter().zip(cb.iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
            let mag_a: f64 = ca.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let mag_b: f64 = cb.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            if mag_a < 1e-10 || mag_b < 1e-10 {
                1.0
            } else {
                1.0 - (dot / (mag_a * mag_b))
            }
        }
    };

    let window_a_start = sequence.entries[mid - window_size].timestamp;
    let window_b_end = sequence.entries[mid + window_size - 1].timestamp;

    Some(DriftResult {
        score,
        drift_detected: score > threshold,
        method,
        threshold,
        window_a_start,
        window_b_end,
    })
}

/// Temporal aggregation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalAggregation {
    /// Compute centroid over a window
    WindowCentroid,
    /// Compute drift score over time
    Drift,
    /// Compute trend direction
    Trend,
    /// Compute spread (variance) over a window
    Spread,
}

/// Store for managing multiple vector sequences.
pub struct SequenceStore {
    sequences: HashMap<String, VectorSequence>,
}

impl SequenceStore {
    /// Create a new empty sequence store.
    pub fn new() -> Self {
        Self {
            sequences: HashMap::new(),
        }
    }

    /// Create or get a sequence.
    pub fn get_or_create(&mut self, id: &str, dimensions: usize) -> &mut VectorSequence {
        self.sequences
            .entry(id.to_string())
            .or_insert_with(|| VectorSequence::new(id, dimensions))
    }

    /// Get a sequence by ID.
    pub fn get(&self, id: &str) -> Option<&VectorSequence> {
        self.sequences.get(id)
    }

    /// List all sequence IDs.
    pub fn list(&self) -> Vec<&str> {
        self.sequences.keys().map(|s| s.as_str()).collect()
    }

    /// Remove a sequence.
    pub fn remove(&mut self, id: &str) -> Option<VectorSequence> {
        self.sequences.remove(id)
    }

    /// Search for sequences whose trajectory is similar to a query sequence.
    pub fn search_trajectory(
        &self,
        query: &VectorSequence,
        method: TrajectoryDistance,
        limit: usize,
    ) -> Vec<(String, f64)> {
        let query_vecs: Vec<Vec<f32>> = query.entries.iter().map(|e| e.vector.clone()).collect();

        let mut results: Vec<(String, f64)> = self
            .sequences
            .iter()
            .filter(|(id, _)| *id != &query.id)
            .map(|(id, seq)| {
                let seq_vecs: Vec<Vec<f32>> =
                    seq.entries.iter().map(|e| e.vector.clone()).collect();
                let distance = match method {
                    TrajectoryDistance::Dtw => dtw_distance(&query_vecs, &seq_vecs),
                    TrajectoryDistance::Frechet => frechet_distance(&query_vecs, &seq_vecs),
                    TrajectoryDistance::CentroidDistance => {
                        let qc = query.centroid().unwrap_or_default();
                        let sc = seq.centroid().unwrap_or_default();
                        euclidean_dist(&qc, &sc)
                    }
                };
                (id.clone(), distance)
            })
            .collect();

        results.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        results
    }

    /// Get the total number of sequences.
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
}

impl Default for SequenceStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_sequence_basic() {
        let mut seq = VectorSequence::new("test", 3);
        assert!(seq.push(100, &[1.0, 0.0, 0.0]).is_ok());
        assert!(seq.push(200, &[0.0, 1.0, 0.0]).is_ok());
        assert!(seq.push(300, &[0.0, 0.0, 1.0]).is_ok());

        assert_eq!(seq.len(), 3);

        // Dimension mismatch
        assert!(seq.push(400, &[1.0, 0.0]).is_err());
    }

    #[test]
    fn test_sequence_centroid() {
        let mut seq = VectorSequence::new("test", 2);
        seq.push(100, &[1.0, 0.0]).unwrap();
        seq.push(200, &[0.0, 1.0]).unwrap();

        let centroid = seq.centroid().unwrap();
        assert!((centroid[0] - 0.5).abs() < 1e-6);
        assert!((centroid[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_range() {
        let mut seq = VectorSequence::new("test", 2);
        seq.push(100, &[1.0, 0.0]).unwrap();
        seq.push(200, &[0.0, 1.0]).unwrap();
        seq.push(300, &[1.0, 1.0]).unwrap();

        let range = seq.range(150, 250);
        assert_eq!(range.len(), 1);
        assert_eq!(range[0].timestamp, 200);
    }

    #[test]
    fn test_dtw_distance() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let dist = dtw_distance(&a, &b);
        assert!(dist < 1e-10, "Identical sequences should have 0 DTW distance");
    }

    #[test]
    fn test_frechet_distance() {
        let a = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let b = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let dist = frechet_distance(&a, &b);
        assert!(dist < 1e-10, "Identical sequences should have 0 Fréchet distance");
    }

    #[test]
    fn test_drift_detection() {
        let mut seq = VectorSequence::new("test", 2);
        // First half: vectors near [1, 0]
        for i in 0..20 {
            seq.push(i * 10, &[1.0, 0.0]).unwrap();
        }
        // Second half: vectors near [0, 1] (drift!)
        for i in 20..40 {
            seq.push(i * 10, &[0.0, 1.0]).unwrap();
        }

        let result = detect_drift(&seq, DriftMethod::CosineDrift, 10, 0.5);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.drift_detected, "Should detect drift between opposite vectors");
    }

    #[test]
    fn test_sequence_store() {
        let mut store = SequenceStore::new();

        let seq = store.get_or_create("user1", 3);
        seq.push(100, &[1.0, 0.0, 0.0]).unwrap();
        seq.push(200, &[0.0, 1.0, 0.0]).unwrap();

        assert_eq!(store.len(), 1);
        assert!(store.get("user1").is_some());
        assert!(store.get("user2").is_none());
    }

    #[test]
    fn test_trajectory_search() {
        let mut store = SequenceStore::new();

        let s1 = store.get_or_create("s1", 2);
        s1.push(100, &[1.0, 0.0]).unwrap();
        s1.push(200, &[1.0, 0.1]).unwrap();

        let s2 = store.get_or_create("s2", 2);
        s2.push(100, &[0.0, 1.0]).unwrap();
        s2.push(200, &[0.1, 1.0]).unwrap();

        let mut query = VectorSequence::new("query", 2);
        query.push(100, &[1.0, 0.0]).unwrap();
        query.push(200, &[1.0, 0.1]).unwrap();

        let results = store.search_trajectory(&query, TrajectoryDistance::Dtw, 2);
        assert!(!results.is_empty());
        // s1 should be closest to query
        assert_eq!(results[0].0, "s1");
    }

    #[test]
    fn test_trend() {
        let mut seq = VectorSequence::new("test", 2);
        for i in 0..10 {
            seq.push(i * 100, &[i as f32, 0.0]).unwrap();
        }

        let trend = seq.trend(3);
        assert!(trend.is_some());
        let t = trend.unwrap();
        assert!(t[0] > 0.0, "Trend should show positive direction in dim 0");
    }
}
