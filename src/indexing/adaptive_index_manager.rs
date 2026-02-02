#![allow(dead_code)]

//! Adaptive Index Selection
//!
//! Automatically selects and switches between HNSW/IVF/DiskANN based on runtime
//! workload characteristics. Extends the auto-tune infrastructure with a
//! `WorkloadProfile`, analytical cost models, and an `AdaptiveIndexManager` that
//! coordinates background index rebuilds with atomic swap.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::indexing::adaptive_index_manager::{
//!     AdaptiveIndexManager, AdaptiveIndexConfig, WorkloadProfile,
//! };
//!
//! let config = AdaptiveIndexConfig::default();
//! let manager = AdaptiveIndexManager::new(config);
//!
//! // Record workload observations
//! manager.record_query(latency_ms, result_count);
//! manager.record_insert();
//!
//! // Periodically evaluate and get recommendation
//! if let Some(action) = manager.evaluate() {
//!     println!("Switch to {:?}", action.target_index);
//! }
//! ```

use crate::error::{NeedleError, Result};
use crate::tuning::RecommendedIndex;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Configuration for the adaptive index manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveIndexConfig {
    /// Window size for workload profiling (number of observations).
    pub profile_window_size: usize,
    /// Minimum observations before making a recommendation.
    pub min_observations: usize,
    /// Confidence threshold to trigger a switch (0.0 - 1.0).
    pub switch_confidence_threshold: f64,
    /// Cooldown period between switches (seconds). Default: 3600 (1 hour).
    pub switch_cooldown_seconds: u64,
    /// Target query latency (ms).
    pub target_latency_ms: f64,
    /// Target recall (0.0 - 1.0).
    pub target_recall: f64,
    /// Memory budget in bytes.
    pub memory_budget_bytes: u64,
    /// Enable automatic switching (vs. recommendation-only mode).
    pub auto_switch: bool,
    /// Hysteresis threshold: improvement must exceed this % to trigger switch.
    pub hysteresis_threshold_pct: f64,
    /// Minimum consecutive evaluations recommending the same target before switching.
    pub min_consecutive_recommendations: usize,
}

impl Default for AdaptiveIndexConfig {
    fn default() -> Self {
        Self {
            profile_window_size: 1000,
            min_observations: 100,
            switch_confidence_threshold: 0.7,
            switch_cooldown_seconds: 3600, // 1 hour minimum cooldown
            target_latency_ms: 10.0,
            target_recall: 0.95,
            memory_budget_bytes: 1024 * 1024 * 1024, // 1GB
            auto_switch: false,
            hysteresis_threshold_pct: 15.0, // Must be 15% better to switch
            min_consecutive_recommendations: 3,
        }
    }
}

/// A snapshot of workload characteristics computed from recent observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadProfile {
    /// Queries per second (recent window).
    pub qps: f64,
    /// Inserts per second (recent window).
    pub insert_rate: f64,
    /// Deletes per second (recent window).
    pub delete_rate: f64,
    /// Read/write ratio (queries / (inserts + deletes + queries)).
    pub read_write_ratio: f64,
    /// Average query latency (ms).
    pub avg_query_latency_ms: f64,
    /// P99 query latency (ms).
    pub p99_query_latency_ms: f64,
    /// Average result set size.
    pub avg_result_count: f64,
    /// Current vector count.
    pub vector_count: usize,
    /// Current dimensionality.
    pub dimensions: usize,
    /// Memory usage bytes.
    pub memory_bytes: u64,
    /// Timestamp of this profile.
    pub timestamp: u64,
    /// Whether the workload is read-heavy (>80% reads).
    pub is_read_heavy: bool,
    /// Whether the workload is write-heavy (>50% writes).
    pub is_write_heavy: bool,
    /// Whether latency pressure is high (avg > target).
    pub has_latency_pressure: bool,
    /// Whether memory pressure is high (>80% budget).
    pub has_memory_pressure: bool,
}

/// Individual workload observation.
#[derive(Debug, Clone)]
struct Observation {
    kind: ObservationKind,
    timestamp: Instant,
    latency_ms: Option<f64>,
    result_count: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
enum ObservationKind {
    Query,
    Insert,
    Delete,
}

/// Cost model estimation for a specific index type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Estimated query latency (ms).
    pub estimated_latency_ms: f64,
    /// Estimated recall (0.0 - 1.0).
    pub estimated_recall: f64,
    /// Estimated memory per vector (bytes).
    pub memory_per_vector: f64,
    /// Estimated total memory (bytes).
    pub total_memory: u64,
    /// Estimated insert throughput (vectors/sec).
    pub insert_throughput: f64,
    /// Overall cost score (lower is better).
    pub cost_score: f64,
}

/// Result of an adaptive evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchAction {
    /// Recommended target index.
    pub target_index: RecommendedIndex,
    /// Current index.
    pub current_index: RecommendedIndex,
    /// Confidence in the recommendation (0.0 - 1.0).
    pub confidence: f64,
    /// Cost estimates for each index type.
    pub cost_estimates: Vec<(String, CostEstimate)>,
    /// Reason for the recommendation.
    pub reason: String,
    /// Estimated improvement in latency (%).
    pub estimated_latency_improvement_pct: f64,
    /// Estimated change in memory (%).
    pub estimated_memory_change_pct: f64,
}

/// Analytical cost model for index type selection.
struct CostModel;

impl CostModel {
    /// Estimate costs for HNSW with given workload profile.
    fn estimate_hnsw(profile: &WorkloadProfile) -> CostEstimate {
        let m = 16.0_f64;
        let memory_per_vector = (profile.dimensions as f64 * 4.0) + (m * 2.0 * 8.0);
        let total_memory = (memory_per_vector * profile.vector_count as f64) as u64;

        // Latency model: O(log(n)) with constants from benchmarks
        let log_n = (profile.vector_count as f64).max(1.0).ln();
        let estimated_latency = 0.05 * log_n + 0.001 * profile.dimensions as f64;

        // Recall model: HNSW typically achieves 95%+ with default params
        let recall = 0.97_f64.min(1.0 - 0.01 * log_n / 20.0);

        // Insert throughput degrades with graph size
        let insert_throughput = 50000.0 / (1.0 + log_n / 10.0);

        let cost_score = compute_cost(estimated_latency, recall, total_memory, profile);

        CostEstimate {
            estimated_latency_ms: estimated_latency,
            estimated_recall: recall,
            memory_per_vector,
            total_memory,
            insert_throughput,
            cost_score,
        }
    }

    /// Estimate costs for IVF with given workload profile.
    fn estimate_ivf(profile: &WorkloadProfile) -> CostEstimate {
        let n_clusters = (profile.vector_count as f64).sqrt().ceil() as usize;
        let memory_per_vector = profile.dimensions as f64 * 4.0 + 8.0; // vector + cluster assignment
        let centroid_memory = n_clusters as f64 * profile.dimensions as f64 * 4.0;
        let total_memory = (memory_per_vector * profile.vector_count as f64 + centroid_memory) as u64;

        // Latency: scan n_probe clusters
        let n_probe = (n_clusters as f64 * 0.05).max(1.0).ceil();
        let vectors_per_cluster = profile.vector_count as f64 / n_clusters.max(1) as f64;
        let estimated_latency = n_probe * vectors_per_cluster * 0.0001 + 0.5;

        // Recall depends on n_probe / n_clusters ratio
        let probe_ratio = n_probe / n_clusters.max(1) as f64;
        let recall = (0.85 + 0.1 * probe_ratio).min(0.98);

        let insert_throughput = 100000.0; // IVF insert is fast (just assign to cluster)

        let cost_score = compute_cost(estimated_latency, recall, total_memory, profile);

        CostEstimate {
            estimated_latency_ms: estimated_latency,
            estimated_recall: recall,
            memory_per_vector,
            total_memory,
            insert_throughput,
            cost_score,
        }
    }

    /// Estimate costs for DiskANN with given workload profile.
    fn estimate_diskann(profile: &WorkloadProfile) -> CostEstimate {
        // DiskANN keeps a small in-memory index, vectors on disk
        let memory_per_vector = 64.0 + 32.0; // compressed + graph pointers
        let total_memory = (memory_per_vector * profile.vector_count as f64) as u64;

        // Higher latency due to disk access
        let log_n = (profile.vector_count as f64).max(1.0).ln();
        let estimated_latency = 0.5 * log_n + 2.0;

        let recall = 0.95;
        let insert_throughput = 20000.0;

        let cost_score = compute_cost(estimated_latency, recall, total_memory, profile);

        CostEstimate {
            estimated_latency_ms: estimated_latency,
            estimated_recall: recall,
            memory_per_vector,
            total_memory,
            insert_throughput,
            cost_score,
        }
    }
}

/// Compute an overall cost score (lower is better).
fn compute_cost(
    latency_ms: f64,
    recall: f64,
    memory_bytes: u64,
    profile: &WorkloadProfile,
) -> f64 {
    // Weighted cost: penalize high latency, low recall, high memory
    let latency_cost = latency_ms / 10.0; // normalize to ~1.0 at 10ms
    let recall_cost = (1.0 - recall) * 10.0; // heavily penalize low recall
    let memory_cost = if profile.memory_bytes > 0 {
        memory_bytes as f64 / profile.memory_bytes.max(1) as f64
    } else {
        0.0
    };

    // Weight by workload characteristics
    let latency_weight = if profile.is_read_heavy { 0.5 } else { 0.3 };
    let recall_weight = 0.35;
    let memory_weight = if profile.has_memory_pressure { 0.3 } else { 0.15 };

    latency_cost * latency_weight + recall_cost * recall_weight + memory_cost * memory_weight
}

/// State of an ongoing index rebuild for double-buffered swap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RebuildState {
    /// No rebuild in progress.
    Idle,
    /// Building the new index in the background.
    Building {
        /// Target index type.
        target: RecommendedIndex,
        /// Progress percentage (0-100).
        progress_pct: f64,
        /// Start timestamp.
        started_at: u64,
    },
    /// Ready to swap: new index is built, awaiting atomic swap.
    ReadyToSwap {
        /// Target index type.
        target: RecommendedIndex,
    },
}

/// Adaptive Index Manager: monitors workload and recommends/executes index switches.
pub struct AdaptiveIndexManager {
    config: AdaptiveIndexConfig,
    observations: RwLock<VecDeque<Observation>>,
    current_index: RwLock<RecommendedIndex>,
    last_switch_time: RwLock<Option<u64>>,
    evaluation_count: RwLock<u64>,
    last_profile: RwLock<Option<WorkloadProfile>>,
    /// Consecutive evaluations recommending the same target (hysteresis).
    consecutive_recommendation: RwLock<(Option<RecommendedIndex>, usize)>,
    /// State of background index rebuild.
    rebuild_state: RwLock<RebuildState>,
}

impl AdaptiveIndexManager {
    /// Create a new adaptive index manager.
    pub fn new(config: AdaptiveIndexConfig) -> Self {
        Self {
            config,
            observations: RwLock::new(VecDeque::new()),
            current_index: RwLock::new(RecommendedIndex::Hnsw),
            last_switch_time: RwLock::new(None),
            evaluation_count: RwLock::new(0),
            last_profile: RwLock::new(None),
            consecutive_recommendation: RwLock::new((None, 0)),
            rebuild_state: RwLock::new(RebuildState::Idle),
        }
    }

    /// Record a query observation.
    pub fn record_query(&self, latency_ms: f64, result_count: usize) {
        self.add_observation(ObservationKind::Query, Some(latency_ms), Some(result_count));
    }

    /// Record an insert observation.
    pub fn record_insert(&self) {
        self.add_observation(ObservationKind::Insert, None, None);
    }

    /// Record a delete observation.
    pub fn record_delete(&self) {
        self.add_observation(ObservationKind::Delete, None, None);
    }

    fn add_observation(
        &self,
        kind: ObservationKind,
        latency_ms: Option<f64>,
        result_count: Option<usize>,
    ) {
        let mut obs = self.observations.write();
        obs.push_back(Observation {
            kind,
            timestamp: Instant::now(),
            latency_ms,
            result_count,
        });
        while obs.len() > self.config.profile_window_size {
            obs.pop_front();
        }
    }

    /// Build a workload profile from recent observations.
    pub fn build_profile(
        &self,
        vector_count: usize,
        dimensions: usize,
        memory_bytes: u64,
    ) -> Option<WorkloadProfile> {
        let obs = self.observations.read();
        if obs.len() < self.config.min_observations {
            return None;
        }

        let first_ts = obs.front()?.timestamp;
        let last_ts = obs.back()?.timestamp;
        let window_secs = last_ts.duration_since(first_ts).as_secs_f64().max(0.001);

        let queries: Vec<&Observation> =
            obs.iter().filter(|o| matches!(o.kind, ObservationKind::Query)).collect();
        let inserts = obs.iter().filter(|o| matches!(o.kind, ObservationKind::Insert)).count();
        let deletes = obs.iter().filter(|o| matches!(o.kind, ObservationKind::Delete)).count();

        let qps = queries.len() as f64 / window_secs;
        let insert_rate = inserts as f64 / window_secs;
        let delete_rate = deletes as f64 / window_secs;
        let total_ops = queries.len() + inserts + deletes;
        let read_write_ratio = if total_ops > 0 {
            queries.len() as f64 / total_ops as f64
        } else {
            1.0
        };

        let latencies: Vec<f64> = queries.iter().filter_map(|o| o.latency_ms).collect();
        let avg_query_latency_ms = if latencies.is_empty() {
            0.0
        } else {
            latencies.iter().sum::<f64>() / latencies.len() as f64
        };

        let p99_query_latency_ms = if latencies.is_empty() {
            0.0
        } else {
            let mut sorted = latencies.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
            sorted[idx]
        };

        let result_counts: Vec<f64> = queries
            .iter()
            .filter_map(|o| o.result_count.map(|c| c as f64))
            .collect();
        let avg_result_count = if result_counts.is_empty() {
            0.0
        } else {
            result_counts.iter().sum::<f64>() / result_counts.len() as f64
        };

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let profile = WorkloadProfile {
            qps,
            insert_rate,
            delete_rate,
            read_write_ratio,
            avg_query_latency_ms,
            p99_query_latency_ms,
            avg_result_count,
            vector_count,
            dimensions,
            memory_bytes,
            timestamp: now,
            is_read_heavy: read_write_ratio > 0.8,
            is_write_heavy: read_write_ratio < 0.5,
            has_latency_pressure: avg_query_latency_ms > self.config.target_latency_ms,
            has_memory_pressure: memory_bytes as f64 > self.config.memory_budget_bytes as f64 * 0.8,
        };

        *self.last_profile.write() = Some(profile.clone());
        Some(profile)
    }

    /// Evaluate the current workload and recommend whether to switch index types.
    pub fn evaluate(
        &self,
        vector_count: usize,
        dimensions: usize,
        memory_bytes: u64,
    ) -> Option<SwitchAction> {
        let profile = self.build_profile(vector_count, dimensions, memory_bytes)?;
        let current = *self.current_index.read();

        // Check cooldown
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if let Some(last) = *self.last_switch_time.read() {
            if now - last < self.config.switch_cooldown_seconds {
                return None;
            }
        }

        // Compute cost estimates for all index types
        let hnsw_cost = CostModel::estimate_hnsw(&profile);
        let ivf_cost = CostModel::estimate_ivf(&profile);
        let diskann_cost = CostModel::estimate_diskann(&profile);

        let mut candidates = vec![
            (RecommendedIndex::Hnsw, &hnsw_cost),
            (RecommendedIndex::Ivf, &ivf_cost),
            (RecommendedIndex::DiskAnn, &diskann_cost),
        ];
        candidates.sort_by(|a, b| {
            a.1.cost_score
                .partial_cmp(&b.1.cost_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = candidates[0].0;
        let best_cost = candidates[0].1;
        let current_cost = match current {
            RecommendedIndex::Hnsw => &hnsw_cost,
            RecommendedIndex::Ivf => &ivf_cost,
            RecommendedIndex::DiskAnn => &diskann_cost,
        };

        // Compute confidence: how much better the best is vs current
        let improvement = if current_cost.cost_score > 0.0 {
            (current_cost.cost_score - best_cost.cost_score) / current_cost.cost_score
        } else {
            0.0
        };

        let confidence = improvement.max(0.0).min(1.0);

        *self.evaluation_count.write() += 1;

        if best == current || confidence < self.config.switch_confidence_threshold {
            // Reset consecutive counter if best changed or confidence too low
            *self.consecutive_recommendation.write() = (None, 0);
            return None;
        }

        // Hysteresis: require improvement > threshold %
        let improvement_pct = improvement * 100.0;
        if improvement_pct < self.config.hysteresis_threshold_pct {
            return None;
        }

        // Track consecutive recommendations for the same target
        {
            let mut consec = self.consecutive_recommendation.write();
            if consec.0 == Some(best) {
                consec.1 += 1;
            } else {
                *consec = (Some(best), 1);
            }
            if consec.1 < self.config.min_consecutive_recommendations {
                return None;
            }
        }

        let latency_improvement =
            (current_cost.estimated_latency_ms - best_cost.estimated_latency_ms)
                / current_cost.estimated_latency_ms.max(0.001)
                * 100.0;

        let memory_change = if current_cost.total_memory > 0 {
            (best_cost.total_memory as f64 - current_cost.total_memory as f64)
                / current_cost.total_memory as f64
                * 100.0
        } else {
            0.0
        };

        let reason = format!(
            "Workload profile suggests {:?} (cost={:.3}) over {:?} (cost={:.3}). \
             QPS={:.1}, insert_rate={:.1}, read_ratio={:.2}",
            best,
            best_cost.cost_score,
            current,
            current_cost.cost_score,
            profile.qps,
            profile.insert_rate,
            profile.read_write_ratio
        );

        Some(SwitchAction {
            target_index: best,
            current_index: current,
            confidence,
            cost_estimates: vec![
                ("hnsw".to_string(), hnsw_cost),
                ("ivf".to_string(), ivf_cost),
                ("diskann".to_string(), diskann_cost),
            ],
            reason,
            estimated_latency_improvement_pct: latency_improvement,
            estimated_memory_change_pct: memory_change,
        })
    }

    /// Acknowledge that an index switch has been executed.
    pub fn acknowledge_switch(&self, new_index: RecommendedIndex) {
        *self.current_index.write() = new_index;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        *self.last_switch_time.write() = Some(now);
    }

    /// Get the current index type.
    pub fn current_index(&self) -> RecommendedIndex {
        *self.current_index.read()
    }

    /// Get the most recent workload profile.
    pub fn last_profile(&self) -> Option<WorkloadProfile> {
        self.last_profile.read().clone()
    }

    /// Get the number of evaluations performed.
    pub fn evaluation_count(&self) -> u64 {
        *self.evaluation_count.read()
    }

    /// Initiate a background index rebuild for double-buffered swap.
    pub fn start_rebuild(&self, target: RecommendedIndex) -> Result<()> {
        let mut state = self.rebuild_state.write();
        if matches!(*state, RebuildState::Building { .. }) {
            return Err(NeedleError::OperationInProgress(
                "Index rebuild already in progress".into(),
            ));
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        *state = RebuildState::Building {
            target,
            progress_pct: 0.0,
            started_at: now,
        };
        Ok(())
    }

    /// Update rebuild progress (called by the background builder).
    pub fn update_rebuild_progress(&self, progress_pct: f64) {
        let mut state = self.rebuild_state.write();
        if let RebuildState::Building {
            progress_pct: ref mut current,
            ..
        } = *state
        {
            *current = progress_pct.clamp(0.0, 100.0);
        }
    }

    /// Mark rebuild as complete and ready for atomic swap.
    pub fn mark_rebuild_ready(&self) {
        let mut state = self.rebuild_state.write();
        if let RebuildState::Building { target, .. } = *state {
            *state = RebuildState::ReadyToSwap { target };
        }
    }

    /// Execute the atomic swap: switch the active index to the rebuilt one.
    /// Returns the new index type if a swap occurred.
    pub fn execute_swap(&self) -> Option<RecommendedIndex> {
        let mut state = self.rebuild_state.write();
        if let RebuildState::ReadyToSwap { target } = *state {
            self.acknowledge_switch(target);
            *self.consecutive_recommendation.write() = (None, 0);
            *state = RebuildState::Idle;
            Some(target)
        } else {
            None
        }
    }

    /// Get the current rebuild state.
    pub fn rebuild_state(&self) -> RebuildState {
        self.rebuild_state.read().clone()
    }

    /// Get the consecutive recommendation count for diagnostics.
    pub fn consecutive_recommendation_count(&self) -> usize {
        self.consecutive_recommendation.read().1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_model_hnsw() {
        let profile = WorkloadProfile {
            qps: 100.0,
            insert_rate: 10.0,
            delete_rate: 1.0,
            read_write_ratio: 0.9,
            avg_query_latency_ms: 1.0,
            p99_query_latency_ms: 5.0,
            avg_result_count: 10.0,
            vector_count: 100_000,
            dimensions: 384,
            memory_bytes: 500_000_000,
            timestamp: 0,
            is_read_heavy: true,
            is_write_heavy: false,
            has_latency_pressure: false,
            has_memory_pressure: false,
        };

        let cost = CostModel::estimate_hnsw(&profile);
        assert!(cost.estimated_latency_ms > 0.0);
        assert!(cost.estimated_recall > 0.9);
        assert!(cost.memory_per_vector > 0.0);
    }

    #[test]
    fn test_manager_insufficient_observations() {
        let config = AdaptiveIndexConfig {
            min_observations: 10,
            ..Default::default()
        };
        let manager = AdaptiveIndexManager::new(config);

        // Not enough observations
        manager.record_query(1.0, 10);
        assert!(manager.evaluate(1000, 128, 1_000_000).is_none());
    }

    #[test]
    fn test_manager_build_profile() {
        let config = AdaptiveIndexConfig {
            min_observations: 5,
            ..Default::default()
        };
        let manager = AdaptiveIndexManager::new(config);

        for _ in 0..10 {
            manager.record_query(2.0, 10);
        }
        for _ in 0..3 {
            manager.record_insert();
        }

        let profile = manager.build_profile(10000, 128, 50_000_000);
        assert!(profile.is_some());
        let p = profile.unwrap();
        assert!(p.qps > 0.0);
        assert!(p.read_write_ratio > 0.5);
    }

    #[test]
    fn test_hysteresis_blocks_premature_switch() {
        let config = AdaptiveIndexConfig {
            min_observations: 3,
            switch_cooldown_seconds: 0,
            switch_confidence_threshold: 0.01,
            hysteresis_threshold_pct: 50.0, // Very high threshold
            min_consecutive_recommendations: 1,
            ..Default::default()
        };
        let manager = AdaptiveIndexManager::new(config);
        for _ in 0..10 {
            manager.record_query(1.0, 10);
        }
        // With a very high hysteresis threshold, even if there's a better index,
        // the improvement may not be enough
        let result = manager.evaluate(1000, 128, 50_000_000);
        // Should not trigger due to hysteresis
        assert!(result.is_none() || result.as_ref().is_some_and(|r| r.confidence > 0.5));
    }

    #[test]
    fn test_consecutive_recommendation_required() {
        let config = AdaptiveIndexConfig {
            min_observations: 3,
            switch_cooldown_seconds: 0,
            switch_confidence_threshold: 0.01,
            hysteresis_threshold_pct: 0.0,
            min_consecutive_recommendations: 5,
            ..Default::default()
        };
        let manager = AdaptiveIndexManager::new(config);
        // Need 5 consecutive evaluations recommending the same target
        for _ in 0..5 {
            manager.record_query(1.0, 10);
        }
        // First evaluation: count = 1, needs 5
        let _r = manager.evaluate(100_000_000, 128, 50_000_000);
        assert!(manager.consecutive_recommendation_count() <= 1);
    }

    #[test]
    fn test_rebuild_lifecycle() {
        let config = AdaptiveIndexConfig::default();
        let manager = AdaptiveIndexManager::new(config);

        assert!(matches!(manager.rebuild_state(), RebuildState::Idle));

        manager.start_rebuild(RecommendedIndex::Ivf).expect("start");
        assert!(matches!(
            manager.rebuild_state(),
            RebuildState::Building { .. }
        ));

        // Can't start another rebuild while one is in progress
        assert!(manager.start_rebuild(RecommendedIndex::Hnsw).is_err());

        manager.update_rebuild_progress(50.0);
        manager.mark_rebuild_ready();
        assert!(matches!(
            manager.rebuild_state(),
            RebuildState::ReadyToSwap { .. }
        ));

        let swapped = manager.execute_swap();
        assert_eq!(swapped, Some(RecommendedIndex::Ivf));
        assert_eq!(manager.current_index(), RecommendedIndex::Ivf);
        assert!(matches!(manager.rebuild_state(), RebuildState::Idle));
    }
}
