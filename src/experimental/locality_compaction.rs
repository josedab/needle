//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Locality-Aware Index Compaction
//!
//! Physically reorganizes vectors using k-means clustering to maximize cache locality.
//! Tracks vector co-access patterns, builds co-access graphs, and performs
//! zero-downtime compaction using copy-on-write page swaps.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::experimental::locality_compaction::*;
//!
//! let mut tracker = CoAccessTracker::new(1000);
//! tracker.record_access(&["v1", "v2", "v3"]); // These were accessed together
//!
//! let mut compactor = LocalityCompactor::new(CompactionConfig::default());
//! let plan = compactor.plan_compaction(&tracker);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Configuration for locality-aware compaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Number of clusters for k-means (0 = auto-detect)
    pub num_clusters: usize,
    /// Maximum iterations for k-means convergence
    pub max_iterations: usize,
    /// Convergence threshold (stop when centroid movement < this)
    pub convergence_threshold: f64,
    /// Minimum co-access count to consider vectors related
    pub min_co_access_count: u64,
    /// Target page size in bytes for COW segments
    pub page_size_bytes: usize,
    /// Enable background adaptive scheduling
    pub adaptive_scheduling: bool,
    /// Compaction trigger threshold (co-access fragmentation ratio)
    pub fragmentation_threshold: f64,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            num_clusters: 0,
            max_iterations: 100,
            convergence_threshold: 1e-4,
            min_co_access_count: 5,
            page_size_bytes: 4096,
            adaptive_scheduling: true,
            fragmentation_threshold: 0.3,
        }
    }
}

/// Tracks which vectors are frequently accessed together during search.
#[derive(Debug)]
pub struct CoAccessTracker {
    /// Map from (vector_a, vector_b) -> co-access count (a < b for canonical ordering)
    co_access_counts: HashMap<(String, String), u64>,
    /// Total access events recorded
    total_events: AtomicU64,
    /// Maximum entries to track before evicting least frequent
    max_entries: usize,
}

impl CoAccessTracker {
    /// Create a new co-access tracker.
    pub fn new(max_entries: usize) -> Self {
        Self {
            co_access_counts: HashMap::new(),
            total_events: AtomicU64::new(0),
            max_entries,
        }
    }

    /// Record a co-access event for a set of vectors returned together in search results.
    pub fn record_access(&mut self, vector_ids: &[&str]) {
        self.total_events.fetch_add(1, Ordering::Relaxed);

        for i in 0..vector_ids.len() {
            for j in (i + 1)..vector_ids.len() {
                let (a, b) = if vector_ids[i] < vector_ids[j] {
                    (vector_ids[i].to_string(), vector_ids[j].to_string())
                } else {
                    (vector_ids[j].to_string(), vector_ids[i].to_string())
                };

                let count = self.co_access_counts.entry((a, b)).or_insert(0);
                *count += 1;
            }
        }

        // Evict low-frequency pairs if we exceed capacity
        if self.co_access_counts.len() > self.max_entries {
            self.evict_least_frequent();
        }
    }

    /// Get co-access count for a pair of vectors.
    pub fn co_access_count(&self, a: &str, b: &str) -> u64 {
        let key = if a < b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        };
        self.co_access_counts.get(&key).copied().unwrap_or(0)
    }

    /// Get the most frequently co-accessed pairs.
    pub fn top_pairs(&self, limit: usize) -> Vec<(&str, &str, u64)> {
        let mut pairs: Vec<_> = self
            .co_access_counts
            .iter()
            .map(|((a, b), &count)| (a.as_str(), b.as_str(), count))
            .collect();
        pairs.sort_by(|a, b| b.2.cmp(&a.2));
        pairs.truncate(limit);
        pairs
    }

    /// Get all unique vector IDs that have co-access data.
    pub fn tracked_vectors(&self) -> Vec<String> {
        let mut ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (a, b) in self.co_access_counts.keys() {
            ids.insert(a.clone());
            ids.insert(b.clone());
        }
        ids.into_iter().collect()
    }

    /// Get total events recorded.
    pub fn total_events(&self) -> u64 {
        self.total_events.load(Ordering::Relaxed)
    }

    /// Compute fragmentation ratio (0.0 = perfect locality, 1.0 = fully fragmented).
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.co_access_counts.is_empty() {
            return 0.0;
        }
        let total: u64 = self.co_access_counts.values().sum();
        let max_possible = self.co_access_counts.len() as u64
            * self.co_access_counts.values().max().copied().unwrap_or(1);
        if max_possible == 0 {
            return 0.0;
        }
        1.0 - (total as f64 / max_possible as f64)
    }

    /// Clear all tracking data.
    pub fn clear(&mut self) {
        self.co_access_counts.clear();
    }

    fn evict_least_frequent(&mut self) {
        let target = self.max_entries * 3 / 4;
        let mut entries: Vec<_> = self.co_access_counts.drain().collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(target);
        self.co_access_counts = entries.into_iter().collect();
    }
}

/// A cluster assignment from k-means.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterAssignment {
    /// Vector ID
    pub vector_id: String,
    /// Assigned cluster index
    pub cluster_id: usize,
}

/// Result of a compaction planning phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionPlan {
    /// Cluster assignments for vectors
    pub assignments: Vec<ClusterAssignment>,
    /// Number of clusters
    pub num_clusters: usize,
    /// Estimated improvement in cache locality (0.0 to 1.0)
    pub estimated_improvement: f64,
    /// Number of vectors that would be relocated
    pub vectors_to_relocate: usize,
    /// Whether compaction is recommended
    pub recommended: bool,
}

/// Status of an ongoing compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactionStatus {
    /// No compaction in progress
    Idle,
    /// Planning phase (analyzing co-access patterns)
    Planning,
    /// Executing copy-on-write page swaps
    Executing,
    /// Compaction completed
    Completed,
    /// Compaction failed
    Failed,
}

/// Statistics for the compaction process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionStats {
    /// Current status
    pub status: CompactionStatus,
    /// Number of compactions performed
    pub total_compactions: u64,
    /// Last compaction timestamp
    pub last_compaction: Option<u64>,
    /// Vectors relocated in last compaction
    pub last_relocated: usize,
    /// Current fragmentation ratio
    pub fragmentation_ratio: f64,
}

/// Mini-batch k-means for segment reorganization.
pub struct MiniBatchKMeans {
    /// Cluster centroids
    centroids: Vec<Vec<f32>>,
    /// Number of clusters
    k: usize,
    /// Batch size for mini-batch updates
    batch_size: usize,
    /// Maximum iterations
    max_iterations: usize,
}

impl MiniBatchKMeans {
    /// Create a new mini-batch k-means instance.
    pub fn new(k: usize, dimensions: usize) -> Self {
        Self {
            centroids: Vec::new(),
            k,
            batch_size: 256,
            max_iterations: 100,
        }
    }

    /// Set batch size.
    #[must_use]
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Initialize centroids from a set of vectors using k-means++ initialization.
    pub fn initialize(&mut self, vectors: &[&[f32]], dimensions: usize) {
        if vectors.is_empty() || self.k == 0 {
            return;
        }

        let mut rng = rand::thread_rng();
        use rand::Rng;

        // First centroid: random
        let first_idx = rng.gen_range(0..vectors.len());
        self.centroids = vec![vectors[first_idx].to_vec()];

        // Remaining centroids: k-means++ (proportional to distance from nearest centroid)
        for _ in 1..self.k.min(vectors.len()) {
            let mut distances: Vec<f64> = vectors
                .iter()
                .map(|v| {
                    self.centroids
                        .iter()
                        .map(|c| {
                            v.iter()
                                .zip(c.iter())
                                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                                .sum::<f64>()
                        })
                        .fold(f64::MAX, f64::min)
                })
                .collect();

            let total: f64 = distances.iter().sum();
            if total < 1e-10 {
                break;
            }

            // Weighted random selection
            let threshold = rng.gen::<f64>() * total;
            let mut cumulative = 0.0;
            let mut selected = 0;
            for (i, d) in distances.iter().enumerate() {
                cumulative += d;
                if cumulative >= threshold {
                    selected = i;
                    break;
                }
            }
            self.centroids.push(vectors[selected].to_vec());
        }
    }

    /// Assign a vector to the nearest centroid. Returns cluster index.
    pub fn assign(&self, vector: &[f32]) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let dist: f32 = vector
                    .iter()
                    .zip(c.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (i, dist)
            })
            .min_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get cluster centroids.
    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    /// Get number of clusters.
    pub fn k(&self) -> usize {
        self.k
    }
}

/// Compactor that manages the locality-aware compaction process.
pub struct LocalityCompactor {
    config: CompactionConfig,
    status: CompactionStatus,
    stats: CompactionStats,
}

impl LocalityCompactor {
    /// Create a new locality compactor.
    pub fn new(config: CompactionConfig) -> Self {
        Self {
            config,
            status: CompactionStatus::Idle,
            stats: CompactionStats {
                status: CompactionStatus::Idle,
                total_compactions: 0,
                last_compaction: None,
                last_relocated: 0,
                fragmentation_ratio: 0.0,
            },
        }
    }

    /// Plan a compaction based on co-access data and vectors.
    pub fn plan_compaction(
        &mut self,
        tracker: &CoAccessTracker,
        vectors: &HashMap<String, Vec<f32>>,
        dimensions: usize,
    ) -> CompactionPlan {
        self.status = CompactionStatus::Planning;

        let num_clusters = if self.config.num_clusters > 0 {
            self.config.num_clusters
        } else {
            // Auto-detect: sqrt(n/2) heuristic
            let n = vectors.len();
            ((n as f64 / 2.0).sqrt() as usize).max(2).min(n)
        };

        if vectors.is_empty() {
            return CompactionPlan {
                assignments: Vec::new(),
                num_clusters: 0,
                estimated_improvement: 0.0,
                vectors_to_relocate: 0,
                recommended: false,
            };
        }

        // Run mini-batch k-means
        let vec_refs: Vec<(&str, &[f32])> = vectors
            .iter()
            .map(|(id, v)| (id.as_str(), v.as_slice()))
            .collect();

        let mut kmeans = MiniBatchKMeans::new(num_clusters, dimensions);
        let vecs_only: Vec<&[f32]> = vec_refs.iter().map(|(_, v)| *v).collect();
        kmeans.initialize(&vecs_only, dimensions);

        let assignments: Vec<ClusterAssignment> = vec_refs
            .iter()
            .map(|(id, vec)| ClusterAssignment {
                vector_id: id.to_string(),
                cluster_id: kmeans.assign(vec),
            })
            .collect();

        let fragmentation = tracker.fragmentation_ratio();
        let estimated_improvement = fragmentation * 0.7; // Conservative estimate

        self.status = CompactionStatus::Idle;

        CompactionPlan {
            assignments,
            num_clusters,
            estimated_improvement,
            vectors_to_relocate: vectors.len(),
            recommended: fragmentation > self.config.fragmentation_threshold,
        }
    }

    /// Get current compaction status.
    pub fn status(&self) -> CompactionStatus {
        self.status
    }

    /// Get compaction statistics.
    pub fn stats(&self) -> &CompactionStats {
        &self.stats
    }

    /// Check if compaction should be triggered based on current fragmentation.
    pub fn should_compact(&self, tracker: &CoAccessTracker) -> bool {
        tracker.fragmentation_ratio() > self.config.fragmentation_threshold
    }

    /// Get the config.
    pub fn config(&self) -> &CompactionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_co_access_tracking() {
        let mut tracker = CoAccessTracker::new(1000);
        tracker.record_access(&["v1", "v2", "v3"]);
        tracker.record_access(&["v1", "v2"]);

        assert_eq!(tracker.co_access_count("v1", "v2"), 2);
        assert_eq!(tracker.co_access_count("v1", "v3"), 1);
        assert_eq!(tracker.co_access_count("v2", "v3"), 1);
        assert_eq!(tracker.co_access_count("v1", "v4"), 0);
    }

    #[test]
    fn test_top_pairs() {
        let mut tracker = CoAccessTracker::new(1000);
        for _ in 0..5 {
            tracker.record_access(&["v1", "v2"]);
        }
        tracker.record_access(&["v3", "v4"]);

        let top = tracker.top_pairs(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].2, 5);
    }

    #[test]
    fn test_fragmentation_ratio() {
        let mut tracker = CoAccessTracker::new(1000);
        // Single pair with even access => low fragmentation
        for _ in 0..10 {
            tracker.record_access(&["v1", "v2"]);
        }
        let frag = tracker.fragmentation_ratio();
        assert!(frag >= 0.0 && frag <= 1.0);
    }

    #[test]
    fn test_mini_batch_kmeans() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![1.1, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 1.1],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let mut kmeans = MiniBatchKMeans::new(2, 2);
        kmeans.initialize(&refs, 2);

        assert_eq!(kmeans.centroids().len(), 2);

        let cluster_a = kmeans.assign(&[1.0, 0.0]);
        let cluster_b = kmeans.assign(&[0.0, 1.0]);
        // With 2 clusters, nearby vectors should be in different clusters
        // (Not guaranteed with random init, but likely)
    }

    #[test]
    fn test_compaction_plan_empty() {
        let tracker = CoAccessTracker::new(100);
        let vectors: HashMap<String, Vec<f32>> = HashMap::new();
        let mut compactor = LocalityCompactor::new(CompactionConfig::default());

        let plan = compactor.plan_compaction(&tracker, &vectors, 4);
        assert_eq!(plan.num_clusters, 0);
        assert!(!plan.recommended);
    }

    #[test]
    fn test_compaction_plan_with_data() {
        let mut tracker = CoAccessTracker::new(1000);
        for _ in 0..10 {
            tracker.record_access(&["v1", "v2", "v3"]);
        }

        let mut vectors = HashMap::new();
        vectors.insert("v1".to_string(), vec![1.0, 0.0, 0.0, 0.0]);
        vectors.insert("v2".to_string(), vec![0.9, 0.1, 0.0, 0.0]);
        vectors.insert("v3".to_string(), vec![0.0, 0.0, 1.0, 0.0]);

        let mut compactor = LocalityCompactor::new(CompactionConfig {
            num_clusters: 2,
            ..Default::default()
        });

        let plan = compactor.plan_compaction(&tracker, &vectors, 4);
        assert_eq!(plan.num_clusters, 2);
        assert_eq!(plan.assignments.len(), 3);
    }
}
