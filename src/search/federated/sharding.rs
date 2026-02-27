use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use super::instance::{HealthStatus, InstanceInfo, InstanceRegistry};
use super::merger::FederatedSearchResult;
use super::FederationError;

/// Deduplicates search results that appear from multiple instances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DedupStrategy {
    /// Keep the result with the smallest distance.
    BestDistance,
    /// Keep the result from the first instance that returned it.
    FirstSeen,
    /// Average the distances across instances.
    AverageDistance,
}

impl Default for DedupStrategy {
    fn default() -> Self {
        Self::BestDistance
    }
}

/// Deduplicates and merges results from multiple instances.
pub struct CrossInstanceDedup {
    strategy: DedupStrategy,
}

impl CrossInstanceDedup {
    pub fn new(strategy: DedupStrategy) -> Self {
        Self { strategy }
    }

    /// Deduplicate results. Each inner Vec is from one instance.
    pub fn dedup(
        &self,
        results: &[Vec<FederatedSearchResult>],
        k: usize,
    ) -> Vec<FederatedSearchResult> {
        let mut seen: HashMap<String, (FederatedSearchResult, usize)> = HashMap::new();

        for instance_results in results {
            for r in instance_results {
                match seen.get_mut(&r.id) {
                    Some((existing, count)) => match self.strategy {
                        DedupStrategy::BestDistance => {
                            if r.distance < existing.distance {
                                *existing = r.clone();
                            }
                        }
                        DedupStrategy::FirstSeen => {
                            // keep existing
                        }
                        DedupStrategy::AverageDistance => {
                            *count += 1;
                            existing.distance = (existing.distance * (*count - 1) as f32
                                + r.distance)
                                / *count as f32;
                        }
                    },
                    None => {
                        seen.insert(r.id.clone(), (r.clone(), 1));
                    }
                }
            }
        }

        let mut merged: Vec<FederatedSearchResult> = seen.into_values().map(|(r, _)| r).collect();
        merged.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(k);
        merged
    }
}

// ---------------------------------------------------------------------------
// Query Consistency Controls
// ---------------------------------------------------------------------------

/// Consistency level for federated queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Return results from any single healthy instance.
    One,
    /// Require results from a majority of instances.
    Quorum,
    /// Require results from all instances.
    All,
    /// Best-effort: return whatever is available before timeout.
    BestEffort,
}

impl Default for ConsistencyLevel {
    fn default() -> Self {
        Self::BestEffort
    }
}

/// A federated query plan describing which instances to query and how.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    pub target_instances: Vec<String>,
    pub consistency: ConsistencyLevel,
    pub dedup: DedupStrategy,
    pub k: usize,
    pub timeout: Duration,
}

/// Plans and validates federated queries before execution.
pub struct QueryPlanner {
    registry: Arc<InstanceRegistry>,
}

impl QueryPlanner {
    pub fn new(registry: Arc<InstanceRegistry>) -> Self {
        Self { registry }
    }

    /// Create a query plan for the given collection and consistency level.
    pub fn plan(
        &self,
        collection: &str,
        k: usize,
        consistency: ConsistencyLevel,
        timeout: Duration,
    ) -> Result<QueryPlan, FederationError> {
        let candidates = self.registry.instances_with_collection(collection);
        let healthy: Vec<InstanceInfo> = candidates
            .into_iter()
            .filter(|i| i.status == HealthStatus::Healthy)
            .collect();

        if healthy.is_empty() {
            // Fall back to all healthy instances
            let all_healthy = self.registry.healthy_instances();
            if all_healthy.is_empty() {
                return Err(FederationError::NoHealthyInstances);
            }
            return Ok(QueryPlan {
                target_instances: all_healthy.iter().map(|i| i.config.id.clone()).collect(),
                consistency,
                dedup: DedupStrategy::BestDistance,
                k,
                timeout,
            });
        }

        let required = match consistency {
            ConsistencyLevel::One => 1,
            ConsistencyLevel::Quorum => (healthy.len() / 2) + 1,
            ConsistencyLevel::All => healthy.len(),
            ConsistencyLevel::BestEffort => healthy.len(),
        };

        if healthy.len() < required {
            return Err(FederationError::QuorumNotReached {
                required,
                available: healthy.len(),
            });
        }

        let targets: Vec<String> = healthy
            .iter()
            .take(required)
            .map(|i| i.config.id.clone())
            .collect();

        Ok(QueryPlan {
            target_instances: targets,
            consistency,
            dedup: DedupStrategy::BestDistance,
            k,
            timeout,
        })
    }

    /// Validate whether a query plan is still executable.
    pub fn validate(&self, plan: &QueryPlan) -> bool {
        for id in &plan.target_instances {
            match self.registry.get(id) {
                Some(info) if info.status == HealthStatus::Healthy => {}
                _ => return false,
            }
        }
        true
    }
}

// ============================================================================
// Hash-Ring Consistent Hashing for Shard Assignment
// ============================================================================

/// Number of virtual nodes per physical instance on the hash ring.
const VIRTUAL_NODES_PER_INSTANCE: usize = 128;

/// Consistent hash ring for mapping collection keys to instances.
/// Uses virtual nodes for balanced distribution and supports shard rebalancing.
pub struct HashRing {
    /// Sorted list of (hash, instance_id) virtual node entries.
    ring: Vec<(u64, String)>,
    /// Replica factor: how many distinct instances each key maps to.
    replica_factor: usize,
}

impl HashRing {
    /// Create a new hash ring with the given replica factor.
    pub fn new(replica_factor: usize) -> Self {
        Self {
            ring: Vec::new(),
            replica_factor: replica_factor.max(1),
        }
    }

    /// Add an instance to the ring with virtual nodes.
    pub fn add_instance(&mut self, instance_id: &str) {
        for vn in 0..VIRTUAL_NODES_PER_INSTANCE {
            let key = format!("{instance_id}:vn{vn}");
            let hash = Self::hash_key(&key);
            self.ring.push((hash, instance_id.to_string()));
        }
        self.ring.sort_by_key(|(h, _)| *h);
    }

    /// Remove an instance from the ring.
    pub fn remove_instance(&mut self, instance_id: &str) {
        self.ring.retain(|(_, id)| id != instance_id);
    }

    /// Find the primary instance responsible for a given key.
    pub fn get_instance(&self, key: &str) -> Option<&str> {
        if self.ring.is_empty() {
            return None;
        }
        let hash = Self::hash_key(key);
        let idx = match self.ring.binary_search_by_key(&hash, |(h, _)| *h) {
            Ok(i) => i,
            Err(i) => i % self.ring.len(),
        };
        Some(&self.ring[idx].1)
    }

    /// Find multiple distinct instances for a key (for replication).
    pub fn get_instances(&self, key: &str) -> Vec<&str> {
        if self.ring.is_empty() {
            return Vec::new();
        }
        let hash = Self::hash_key(key);
        let start = match self.ring.binary_search_by_key(&hash, |(h, _)| *h) {
            Ok(i) => i,
            Err(i) => i % self.ring.len(),
        };

        let mut result = Vec::with_capacity(self.replica_factor);
        let mut seen = std::collections::HashSet::new();
        let n = self.ring.len();

        for offset in 0..n {
            let idx = (start + offset) % n;
            let id = &self.ring[idx].1;
            if seen.insert(id.as_str()) {
                result.push(id.as_str());
                if result.len() >= self.replica_factor {
                    break;
                }
            }
        }
        result
    }

    /// Number of instances on the ring.
    pub fn instance_count(&self) -> usize {
        let mut ids: Vec<&str> = self.ring.iter().map(|(_, id)| id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    }

    /// Check distribution balance: returns (min_keys, max_keys) for a sample.
    pub fn check_balance(&self, sample_keys: &[&str]) -> (usize, usize) {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for key in sample_keys {
            if let Some(instance) = self.get_instance(key) {
                *counts.entry(instance).or_default() += 1;
            }
        }
        if counts.is_empty() {
            return (0, 0);
        }
        let min = *counts.values().min().unwrap_or(&0);
        let max = *counts.values().max().unwrap_or(&0);
        (min, max)
    }

    fn hash_key(key: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

/// Shard assignment and rebalancing coordinator.
pub struct ShardManager {
    ring: HashRing,
    /// Map from shard key to assigned instances.
    shard_assignments: HashMap<String, Vec<String>>,
    /// Rebalancing threshold: trigger if imbalance ratio exceeds this.
    rebalance_threshold: f64,
}

impl ShardManager {
    /// Create a new shard manager.
    pub fn new(replica_factor: usize, rebalance_threshold: f64) -> Self {
        Self {
            ring: HashRing::new(replica_factor),
            shard_assignments: HashMap::new(),
            rebalance_threshold: rebalance_threshold.max(1.0),
        }
    }

    /// Add an instance and trigger rebalancing if needed.
    pub fn add_instance(&mut self, instance_id: &str) -> RebalanceResult {
        self.ring.add_instance(instance_id);
        self.rebalance()
    }

    /// Remove an instance and trigger rebalancing.
    pub fn remove_instance(&mut self, instance_id: &str) -> RebalanceResult {
        self.ring.remove_instance(instance_id);
        // Remove from all shard assignments
        for assignments in self.shard_assignments.values_mut() {
            assignments.retain(|id| id != instance_id);
        }
        self.rebalance()
    }

    /// Assign a shard key to instances based on the hash ring.
    pub fn assign_shard(&mut self, shard_key: &str) -> Vec<String> {
        let instances: Vec<String> = self
            .ring
            .get_instances(shard_key)
            .iter()
            .map(|s| s.to_string())
            .collect();
        self.shard_assignments
            .insert(shard_key.to_string(), instances.clone());
        instances
    }

    /// Get current assignment for a shard key.
    pub fn get_assignment(&self, shard_key: &str) -> Option<&Vec<String>> {
        self.shard_assignments.get(shard_key)
    }

    /// Check if rebalancing is needed and perform it.
    pub fn rebalance(&mut self) -> RebalanceResult {
        let keys: Vec<String> = self.shard_assignments.keys().cloned().collect();
        let mut moves = Vec::new();

        for key in &keys {
            let new_assignment: Vec<String> = self
                .ring
                .get_instances(key)
                .iter()
                .map(|s| s.to_string())
                .collect();
            if let Some(old) = self.shard_assignments.get(key) {
                if *old != new_assignment {
                    moves.push(ShardMove {
                        shard_key: key.clone(),
                        from: old.clone(),
                        to: new_assignment.clone(),
                    });
                }
            }
            self.shard_assignments
                .insert(key.clone(), new_assignment);
        }

        RebalanceResult {
            moves_needed: moves.len(),
            shard_moves: moves,
            total_shards: self.shard_assignments.len(),
        }
    }

    /// Check the current load balance across instances.
    pub fn load_distribution(&self) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for assignments in self.shard_assignments.values() {
            for instance in assignments {
                *counts.entry(instance.clone()).or_default() += 1;
            }
        }
        counts
    }

    /// Returns true if the shard distribution is imbalanced beyond the threshold.
    pub fn is_imbalanced(&self) -> bool {
        let dist = self.load_distribution();
        if dist.is_empty() {
            return false;
        }
        let min = *dist.values().min().unwrap_or(&0);
        let max = *dist.values().max().unwrap_or(&0);
        if min == 0 {
            return max > 0;
        }
        (max as f64 / min as f64) > self.rebalance_threshold
    }
}

/// Record of a shard being moved between instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMove {
    pub shard_key: String,
    pub from: Vec<String>,
    pub to: Vec<String>,
}

/// Result of a rebalancing operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceResult {
    pub moves_needed: usize,
    pub shard_moves: Vec<ShardMove>,
    pub total_shards: usize,
}

#[cfg(test)]
mod tests {}
