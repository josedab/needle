use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::config::MergeStrategy;

/// Search result from a federated query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedSearchResult {
    /// Vector ID
    pub id: String,
    /// Distance to query
    pub distance: f32,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
    /// Source instance
    pub source_instance: String,
    /// Collection name
    pub collection: String,
}

/// Aggregated results from federated search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedSearchResponse {
    /// Merged results
    pub results: Vec<FederatedSearchResult>,
    /// Total results before limiting
    pub total_found: usize,
    /// Instances that responded
    pub instances_responded: Vec<String>,
    /// Instances that failed
    pub instances_failed: Vec<String>,
    /// Query execution time
    pub execution_time_ms: f64,
    /// Per-instance latencies
    pub instance_latencies: HashMap<String, f64>,
    /// Whether results are partial
    pub is_partial: bool,
}

/// Merges results from multiple instances
pub struct ResultMerger {
    strategy: MergeStrategy,
}

impl ResultMerger {
    /// Create a new merger
    pub fn new(strategy: MergeStrategy) -> Self {
        Self { strategy }
    }

    /// Merge results from multiple instances
    pub fn merge(
        &self,
        instance_results: Vec<(String, Vec<FederatedSearchResult>)>,
        k: usize,
    ) -> Vec<FederatedSearchResult> {
        match self.strategy {
            MergeStrategy::DistanceBased => Self::merge_by_distance(instance_results, k),
            MergeStrategy::ReciprocalRankFusion => Self::merge_rrf(instance_results, k),
            MergeStrategy::FirstResponse => Self::merge_first(instance_results, k),
            MergeStrategy::PriorityWeighted => Self::merge_by_distance(instance_results, k), // Simplified
            MergeStrategy::Consensus => Self::merge_consensus(instance_results, k),
        }
    }

    fn merge_by_distance(
        instance_results: Vec<(String, Vec<FederatedSearchResult>)>,
        k: usize,
    ) -> Vec<FederatedSearchResult> {
        let mut all_results: Vec<FederatedSearchResult> = instance_results
            .into_iter()
            .flat_map(|(_, results)| results)
            .collect();

        // Sort by distance
        all_results.sort_by_key(|r| OrderedFloat(r.distance));

        // Deduplicate by ID (keep lowest distance)
        let mut seen = std::collections::HashSet::new();
        all_results.retain(|r| seen.insert(r.id.clone()));

        // Take top k
        all_results.truncate(k);
        all_results
    }

    fn merge_rrf(
        instance_results: Vec<(String, Vec<FederatedSearchResult>)>,
        k: usize,
    ) -> Vec<FederatedSearchResult> {
        let rrf_k = 60.0; // RRF constant

        // Calculate RRF scores
        let mut scores: HashMap<String, (f64, FederatedSearchResult)> = HashMap::new();

        for (_, results) in instance_results {
            for (rank, result) in results.into_iter().enumerate() {
                let rrf_score = 1.0 / (rrf_k + rank as f64 + 1.0);

                scores
                    .entry(result.id.clone())
                    .and_modify(|(score, _)| *score += rrf_score)
                    .or_insert((rrf_score, result));
            }
        }

        // Sort by RRF score (descending)
        let mut sorted: Vec<_> = scores.into_values().collect();
        sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        sorted.into_iter().take(k).map(|(_, r)| r).collect()
    }

    fn merge_first(
        instance_results: Vec<(String, Vec<FederatedSearchResult>)>,
        k: usize,
    ) -> Vec<FederatedSearchResult> {
        // Take results from first non-empty response
        instance_results
            .into_iter()
            .find(|(_, results)| !results.is_empty())
            .map(|(_, mut results)| {
                results.truncate(k);
                results
            })
            .unwrap_or_default()
    }

    fn merge_consensus(
        instance_results: Vec<(String, Vec<FederatedSearchResult>)>,
        k: usize,
    ) -> Vec<FederatedSearchResult> {
        // Count occurrences and average distances
        let mut consensus: HashMap<String, (usize, f32, FederatedSearchResult)> = HashMap::new();

        for (_, results) in instance_results {
            for result in results {
                consensus
                    .entry(result.id.clone())
                    .and_modify(|(count, total_dist, _)| {
                        *count += 1;
                        *total_dist += result.distance;
                    })
                    .or_insert((1, result.distance, result));
            }
        }

        // Sort by count (desc) then distance (asc)
        let mut sorted: Vec<_> = consensus.into_values().collect();
        sorted.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| OrderedFloat(a.1 / a.0 as f32).cmp(&OrderedFloat(b.1 / b.0 as f32)))
        });

        sorted.into_iter().take(k).map(|(_, _, r)| r).collect()
    }
}

#[cfg(test)]
mod tests {}
