#![allow(dead_code)]

//! Snapshot-Based Replication Manager
//!
//! Lightweight read-replica support via periodic snapshot shipping. Simpler than
//! Raft consensus, providing eventual consistency with targets of <30s replica lag
//! and <5s failover.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐         snapshot         ┌──────────────┐
//! │   Leader     │ ─────────────────────►   │  Follower 1  │
//! │  (primary)   │         + WAL delta      │  (replica)   │
//! │              │ ─────────────────────►   │              │
//! └─────────────┘                          └──────────────┘
//!       │                                         │
//!       │         snapshot + WAL delta             │
//!       └──────────────────────────────►   ┌──────────────┐
//!                                         │  Follower 2  │
//!                                         │  (replica)   │
//!                                         └──────────────┘
//! ```
//!
//! # Phases
//!
//! 1. **Snapshot Protocol**: Incremental snapshot generation with checksums
//! 2. **Replica Manager**: Follower nodes poll leader, apply snapshots atomically
//! 3. **Health & Failover**: Monitoring with manual failover support

use crate::error::{NeedleError, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Configuration for the replication manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaManagerConfig {
    /// Polling interval for followers (seconds).
    pub poll_interval_secs: u64,
    /// Maximum acceptable replica lag (seconds).
    pub max_lag_secs: u64,
    /// Snapshot shipping interval (seconds).
    pub snapshot_interval_secs: u64,
    /// Maximum number of replicas.
    pub max_replicas: usize,
    /// Health check interval (seconds).
    pub health_check_interval_secs: u64,
    /// Failover timeout (seconds). If leader is unresponsive for this long, trigger failover.
    pub failover_timeout_secs: u64,
    /// Enable incremental snapshots (WAL-based delta shipping).
    pub enable_incremental: bool,
    /// Maximum WAL entries to ship per poll cycle.
    pub max_wal_entries_per_poll: usize,
}

impl Default for ReplicaManagerConfig {
    fn default() -> Self {
        Self {
            poll_interval_secs: 5,
            max_lag_secs: 30,
            snapshot_interval_secs: 300,
            max_replicas: 8,
            health_check_interval_secs: 10,
            failover_timeout_secs: 30,
            enable_incremental: true,
            max_wal_entries_per_poll: 10_000,
        }
    }
}

/// Incremental snapshot with checksums.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalSnapshot {
    /// Snapshot ID.
    pub id: String,
    /// Base snapshot ID (None for full snapshot).
    pub base_snapshot_id: Option<String>,
    /// LSN (Log Sequence Number) at snapshot time.
    pub lsn: u64,
    /// Timestamp.
    pub timestamp: u64,
    /// SHA-256 checksum of the snapshot data.
    pub checksum: String,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Number of vectors in this snapshot.
    pub vector_count: usize,
    /// Number of collections.
    pub collection_count: usize,
    /// Whether this is a full or incremental snapshot.
    pub is_full: bool,
    /// WAL entries included (for incremental snapshots).
    pub wal_entries_count: usize,
}

/// Replica node state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaState {
    /// Replica node ID.
    pub node_id: String,
    /// Current applied LSN.
    pub applied_lsn: u64,
    /// Last snapshot applied.
    pub last_snapshot_id: Option<String>,
    /// Last heartbeat timestamp.
    pub last_heartbeat: u64,
    /// Current health status.
    pub health: ReplicaHealth,
    /// Replication lag (seconds).
    pub lag_secs: f64,
    /// Number of snapshots applied.
    pub snapshots_applied: u64,
    /// Number of WAL entries applied.
    pub wal_entries_applied: u64,
    /// Average apply latency (ms).
    pub avg_apply_latency_ms: f64,
    /// Endpoint URL.
    pub endpoint: String,
}

/// Replica health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicaHealth {
    /// Replica is healthy and in sync.
    Healthy,
    /// Replica is lagging behind the leader.
    Lagging,
    /// Replica is catching up after a long disconnect.
    CatchingUp,
    /// Replica is not responding.
    Unresponsive,
    /// Replica is in an error state.
    Error,
    /// Replica has been manually paused.
    Paused,
}

/// Failover action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverAction {
    /// Previous leader node ID.
    pub previous_leader: String,
    /// New leader node ID.
    pub new_leader: String,
    /// Timestamp of failover.
    pub timestamp: u64,
    /// LSN at failover.
    pub lsn: u64,
    /// Whether this was a manual failover.
    pub is_manual: bool,
    /// Reason for failover.
    pub reason: String,
}

/// Result of a health check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Leader node ID.
    pub leader_id: String,
    /// Leader LSN.
    pub leader_lsn: u64,
    /// Status of each replica.
    pub replicas: Vec<ReplicaState>,
    /// Overall health status.
    pub overall_health: ClusterHealth,
    /// Number of healthy replicas.
    pub healthy_count: usize,
    /// Number of unhealthy replicas.
    pub unhealthy_count: usize,
    /// Maximum lag across all replicas (seconds).
    pub max_lag_secs: f64,
    /// Average lag across all replicas (seconds).
    pub avg_lag_secs: f64,
    /// Timestamp.
    pub timestamp: u64,
}

/// Overall cluster health.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusterHealth {
    /// All replicas healthy and in sync.
    Healthy,
    /// Some replicas degraded but cluster operational.
    Degraded,
    /// Critical: majority of replicas down.
    Critical,
    /// No replicas available.
    NoReplicas,
}

/// Manages the replication lifecycle for a set of replica nodes.
pub struct ReplicaManager {
    config: ReplicaManagerConfig,
    /// Leader node ID.
    leader_id: String,
    /// Current leader LSN.
    leader_lsn: RwLock<u64>,
    /// Known replicas.
    replicas: RwLock<HashMap<String, ReplicaState>>,
    /// Snapshot history.
    snapshots: RwLock<Vec<IncrementalSnapshot>>,
    /// Failover history.
    failover_history: RwLock<Vec<FailoverAction>>,
    /// Last health check.
    last_health_check: RwLock<Option<HealthCheckResult>>,
}

impl ReplicaManager {
    /// Create a new replica manager.
    pub fn new(leader_id: &str, config: ReplicaManagerConfig) -> Self {
        Self {
            config,
            leader_id: leader_id.to_string(),
            leader_lsn: RwLock::new(0),
            replicas: RwLock::new(HashMap::new()),
            snapshots: RwLock::new(Vec::new()),
            failover_history: RwLock::new(Vec::new()),
            last_health_check: RwLock::new(None),
        }
    }

    /// Register a new replica node.
    pub fn add_replica(&self, node_id: &str, endpoint: &str) -> Result<()> {
        let replicas = self.replicas.read();
        if replicas.len() >= self.config.max_replicas {
            return Err(NeedleError::CapacityExceeded(format!(
                "Maximum replicas ({}) reached",
                self.config.max_replicas
            )));
        }
        drop(replicas);

        let now = Self::now();
        self.replicas.write().insert(
            node_id.to_string(),
            ReplicaState {
                node_id: node_id.to_string(),
                applied_lsn: 0,
                last_snapshot_id: None,
                last_heartbeat: now,
                health: ReplicaHealth::CatchingUp,
                lag_secs: 0.0,
                snapshots_applied: 0,
                wal_entries_applied: 0,
                avg_apply_latency_ms: 0.0,
                endpoint: endpoint.to_string(),
            },
        );

        Ok(())
    }

    /// Remove a replica node.
    pub fn remove_replica(&self, node_id: &str) -> Result<()> {
        self.replicas
            .write()
            .remove(node_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Replica '{}' not found", node_id)))?;
        Ok(())
    }

    /// Record a leader LSN advance (after a write operation).
    pub fn advance_leader_lsn(&self, lsn: u64) {
        *self.leader_lsn.write() = lsn;
    }

    /// Record a heartbeat from a replica.
    pub fn record_heartbeat(&self, node_id: &str, applied_lsn: u64) {
        let now = Self::now();
        let leader_lsn = *self.leader_lsn.read();

        if let Some(replica) = self.replicas.write().get_mut(node_id) {
            replica.applied_lsn = applied_lsn;
            replica.last_heartbeat = now;
            replica.lag_secs = if leader_lsn > applied_lsn {
                // Estimate lag based on LSN difference
                (leader_lsn - applied_lsn) as f64 * 0.001 // rough estimate
            } else {
                0.0
            };

            replica.health = if replica.lag_secs <= self.config.max_lag_secs as f64 {
                ReplicaHealth::Healthy
            } else {
                ReplicaHealth::Lagging
            };
        }
    }

    /// Record that a replica has applied a snapshot.
    pub fn record_snapshot_applied(&self, node_id: &str, snapshot_id: &str, apply_latency_ms: f64) {
        if let Some(replica) = self.replicas.write().get_mut(node_id) {
            replica.last_snapshot_id = Some(snapshot_id.to_string());
            replica.snapshots_applied += 1;
            // Exponential moving average for apply latency
            replica.avg_apply_latency_ms =
                replica.avg_apply_latency_ms * 0.9 + apply_latency_ms * 0.1;
        }
    }

    /// Record that a replica has applied WAL entries.
    pub fn record_wal_applied(&self, node_id: &str, entries_count: u64) {
        if let Some(replica) = self.replicas.write().get_mut(node_id) {
            replica.wal_entries_applied += entries_count;
        }
    }

    /// Create a snapshot record (actual data transfer is handled by transport layer).
    pub fn create_snapshot(
        &self,
        vector_count: usize,
        collection_count: usize,
        size_bytes: u64,
        checksum: &str,
    ) -> IncrementalSnapshot {
        let leader_lsn = *self.leader_lsn.read();
        let now = Self::now();

        let base_id = self
            .snapshots
            .read()
            .last()
            .map(|s| s.id.clone());

        let is_full = base_id.is_none();

        let snapshot = IncrementalSnapshot {
            id: format!("snap_{}_{}", self.leader_id, leader_lsn),
            base_snapshot_id: base_id,
            lsn: leader_lsn,
            timestamp: now,
            checksum: checksum.to_string(),
            size_bytes,
            vector_count,
            collection_count,
            is_full,
            wal_entries_count: 0,
        };

        self.snapshots.write().push(snapshot.clone());
        snapshot
    }

    /// Perform a health check across all replicas.
    pub fn health_check(&self) -> HealthCheckResult {
        let now = Self::now();
        let leader_lsn = *self.leader_lsn.read();
        let replicas = self.replicas.read();

        let mut replica_states: Vec<ReplicaState> = Vec::new();
        let mut healthy_count = 0;
        let mut unhealthy_count = 0;
        let mut max_lag = 0.0_f64;
        let mut total_lag = 0.0_f64;

        for replica in replicas.values() {
            let mut state = replica.clone();

            // Check for unresponsive replicas
            let time_since_heartbeat = now.saturating_sub(replica.last_heartbeat);
            if time_since_heartbeat > self.config.failover_timeout_secs {
                state.health = ReplicaHealth::Unresponsive;
            }

            match state.health {
                ReplicaHealth::Healthy => healthy_count += 1,
                _ => unhealthy_count += 1,
            }

            max_lag = max_lag.max(state.lag_secs);
            total_lag += state.lag_secs;

            replica_states.push(state);
        }

        let avg_lag = if replica_states.is_empty() {
            0.0
        } else {
            total_lag / replica_states.len() as f64
        };

        let overall_health = if replica_states.is_empty() {
            ClusterHealth::NoReplicas
        } else if unhealthy_count == 0 {
            ClusterHealth::Healthy
        } else if healthy_count > unhealthy_count {
            ClusterHealth::Degraded
        } else {
            ClusterHealth::Critical
        };

        let result = HealthCheckResult {
            leader_id: self.leader_id.clone(),
            leader_lsn,
            replicas: replica_states,
            overall_health,
            healthy_count,
            unhealthy_count,
            max_lag_secs: max_lag,
            avg_lag_secs: avg_lag,
            timestamp: now,
        };

        *self.last_health_check.write() = Some(result.clone());
        result
    }

    /// Initiate a manual failover to a specific replica.
    pub fn manual_failover(&self, new_leader_id: &str) -> Result<FailoverAction> {
        let replicas = self.replicas.read();
        let replica = replicas
            .get(new_leader_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Replica '{}' not found", new_leader_id)))?;

        if replica.health == ReplicaHealth::Unresponsive || replica.health == ReplicaHealth::Error {
            return Err(NeedleError::InvalidOperation(format!(
                "Cannot failover to unhealthy replica '{}'",
                new_leader_id
            )));
        }

        let now = Self::now();
        let leader_lsn = *self.leader_lsn.read();

        let action = FailoverAction {
            previous_leader: self.leader_id.clone(),
            new_leader: new_leader_id.to_string(),
            timestamp: now,
            lsn: leader_lsn,
            is_manual: true,
            reason: "Manual failover initiated".to_string(),
        };

        self.failover_history.write().push(action.clone());
        Ok(action)
    }

    /// Pause replication for a specific replica.
    pub fn pause_replica(&self, node_id: &str) -> Result<()> {
        let mut replicas = self.replicas.write();
        let replica = replicas
            .get_mut(node_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Replica '{}' not found", node_id)))?;
        replica.health = ReplicaHealth::Paused;
        Ok(())
    }

    /// Resume replication for a paused replica.
    pub fn resume_replica(&self, node_id: &str) -> Result<()> {
        let mut replicas = self.replicas.write();
        let replica = replicas
            .get_mut(node_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Replica '{}' not found", node_id)))?;
        if replica.health == ReplicaHealth::Paused {
            replica.health = ReplicaHealth::CatchingUp;
        }
        Ok(())
    }

    /// Get all replica states.
    pub fn list_replicas(&self) -> Vec<ReplicaState> {
        self.replicas.read().values().cloned().collect()
    }

    /// Get snapshot history.
    pub fn list_snapshots(&self) -> Vec<IncrementalSnapshot> {
        self.snapshots.read().clone()
    }

    /// Get failover history.
    pub fn failover_history(&self) -> Vec<FailoverAction> {
        self.failover_history.read().clone()
    }

    /// Get the most recent health check result.
    pub fn last_health_check(&self) -> Option<HealthCheckResult> {
        self.last_health_check.read().clone()
    }

    /// Get the leader ID.
    pub fn leader_id(&self) -> &str {
        &self.leader_id
    }

    /// Get the current leader LSN.
    pub fn leader_lsn(&self) -> u64 {
        *self.leader_lsn.read()
    }

    /// Automatic failover: select the best replica (lowest lag, highest applied LSN).
    /// Returns the failover action if a suitable replica was found.
    pub fn auto_failover(&self, reason: &str) -> Result<FailoverAction> {
        let replicas = self.replicas.read();
        let candidate = replicas
            .values()
            .filter(|r| {
                r.health == ReplicaHealth::Healthy || r.health == ReplicaHealth::Lagging
            })
            .max_by_key(|r| r.applied_lsn);

        let candidate = candidate.ok_or_else(|| {
            NeedleError::InvalidOperation(
                "No healthy replica available for automatic failover".to_string(),
            )
        })?;

        let new_leader_id = candidate.node_id.clone();
        let leader_lsn = *self.leader_lsn.read();
        let now = Self::now();

        let action = FailoverAction {
            previous_leader: self.leader_id.clone(),
            new_leader: new_leader_id,
            timestamp: now,
            lsn: leader_lsn,
            is_manual: false,
            reason: reason.to_string(),
        };

        self.failover_history.write().push(action.clone());
        Ok(action)
    }

    /// Get WAL entries that need to be shipped to a specific replica.
    /// Returns entries between the replica's applied LSN and the leader LSN.
    pub fn pending_wal_entries_for(&self, node_id: &str) -> Result<WalShipment> {
        let replicas = self.replicas.read();
        let replica = replicas
            .get(node_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Replica '{node_id}' not found")))?;

        let leader_lsn = *self.leader_lsn.read();
        let from_lsn = replica.applied_lsn;

        Ok(WalShipment {
            target_node_id: node_id.to_string(),
            from_lsn,
            to_lsn: leader_lsn,
            entries_pending: leader_lsn.saturating_sub(from_lsn),
            lag_secs: replica.lag_secs,
            max_entries: self.config.max_wal_entries_per_poll as u64,
        })
    }

    /// Check if a replica's lag is within the configured tolerance.
    pub fn is_within_lag_tolerance(&self, node_id: &str) -> bool {
        let replicas = self.replicas.read();
        replicas
            .get(node_id)
            .is_some_and(|r| r.lag_secs <= self.config.max_lag_secs as f64)
    }

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

/// WAL shipment descriptor for a replica.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalShipment {
    /// Target replica node ID.
    pub target_node_id: String,
    /// Starting LSN (exclusive).
    pub from_lsn: u64,
    /// Ending LSN (inclusive).
    pub to_lsn: u64,
    /// Number of entries pending.
    pub entries_pending: u64,
    /// Current replica lag in seconds.
    pub lag_secs: f64,
    /// Maximum entries to ship in one batch.
    pub max_entries: u64,
}

/// A read-only database view backed by replicated data.
/// Accepts queries only if the replica lag is within the configured tolerance.
pub struct ReplicaDatabase {
    node_id: String,
    applied_lsn: u64,
    lag_tolerance_secs: f64,
    current_lag_secs: f64,
    read_only: bool,
    collections: HashMap<String, ReplicaCollectionState>,
}

/// Replica-side state for a single collection.
#[derive(Debug, Clone)]
pub struct ReplicaCollectionState {
    /// Collection name.
    pub name: String,
    /// Number of vectors replicated.
    pub vector_count: usize,
    /// Dimensions.
    pub dimensions: usize,
    /// Last applied LSN for this collection.
    pub applied_lsn: u64,
}

impl ReplicaDatabase {
    /// Create a new read-only replica database.
    pub fn new(node_id: &str, lag_tolerance_secs: f64) -> Self {
        Self {
            node_id: node_id.to_string(),
            applied_lsn: 0,
            lag_tolerance_secs,
            current_lag_secs: 0.0,
            read_only: true,
            collections: HashMap::new(),
        }
    }

    /// Check if the replica is ready to serve queries.
    pub fn is_query_ready(&self) -> bool {
        self.current_lag_secs <= self.lag_tolerance_secs
    }

    /// Get the current replica lag.
    pub fn lag_secs(&self) -> f64 {
        self.current_lag_secs
    }

    /// Update the replica lag estimate.
    pub fn update_lag(&mut self, lag_secs: f64) {
        self.current_lag_secs = lag_secs;
    }

    /// Apply a WAL entry to the replica state.
    pub fn apply_wal_entry(&mut self, lsn: u64, entry: &crate::wal::WalEntry) -> Result<()> {
        if !self.read_only {
            return Err(NeedleError::InvalidOperation(
                "Cannot apply WAL to non-replica database".to_string(),
            ));
        }

        match entry {
            crate::wal::WalEntry::Insert { collection, .. }
            | crate::wal::WalEntry::Update { collection, .. } => {
                let state = self.collections.entry(collection.clone()).or_insert(
                    ReplicaCollectionState {
                        name: collection.clone(),
                        vector_count: 0,
                        dimensions: 0,
                        applied_lsn: 0,
                    },
                );
                state.vector_count += 1;
                state.applied_lsn = lsn;
            }
            crate::wal::WalEntry::Delete { collection, .. } => {
                if let Some(state) = self.collections.get_mut(collection) {
                    state.vector_count = state.vector_count.saturating_sub(1);
                    state.applied_lsn = lsn;
                }
            }
            _ => {}
        }

        self.applied_lsn = lsn;
        Ok(())
    }

    /// Validate that a read query is allowed on this replica.
    pub fn validate_read(&self) -> Result<()> {
        if !self.is_query_ready() {
            return Err(NeedleError::InvalidOperation(format!(
                "Replica '{}' lag ({:.1}s) exceeds tolerance ({:.1}s)",
                self.node_id, self.current_lag_secs, self.lag_tolerance_secs
            )));
        }
        Ok(())
    }

    /// Get the applied LSN.
    pub fn applied_lsn(&self) -> u64 {
        self.applied_lsn
    }

    /// List replicated collections.
    pub fn collections(&self) -> Vec<&ReplicaCollectionState> {
        self.collections.values().collect()
    }

    /// Get replica node ID.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replica_manager_lifecycle() {
        let config = ReplicaManagerConfig::default();
        let manager = ReplicaManager::new("leader-1", config);

        // Add replicas
        manager.add_replica("replica-1", "http://replica1:8080").unwrap();
        manager.add_replica("replica-2", "http://replica2:8080").unwrap();

        assert_eq!(manager.list_replicas().len(), 2);

        // Advance leader
        manager.advance_leader_lsn(100);

        // Record heartbeats
        manager.record_heartbeat("replica-1", 95);
        manager.record_heartbeat("replica-2", 50);

        // Health check
        let health = manager.health_check();
        assert_eq!(health.replicas.len(), 2);
        assert_eq!(health.leader_lsn, 100);
    }

    #[test]
    fn test_create_snapshot() {
        let manager = ReplicaManager::new("leader-1", ReplicaManagerConfig::default());
        manager.advance_leader_lsn(50);

        let snap = manager.create_snapshot(1000, 3, 500_000, "abc123");
        assert!(snap.id.contains("leader-1"));
        assert_eq!(snap.vector_count, 1000);
        assert_eq!(snap.lsn, 50);
        assert!(snap.is_full); // First snapshot is always full
    }

    #[test]
    fn test_manual_failover() {
        let manager = ReplicaManager::new("leader-1", ReplicaManagerConfig::default());
        manager.add_replica("replica-1", "http://replica1:8080").unwrap();
        manager.record_heartbeat("replica-1", 0);

        let action = manager.manual_failover("replica-1").unwrap();
        assert_eq!(action.previous_leader, "leader-1");
        assert_eq!(action.new_leader, "replica-1");
        assert!(action.is_manual);
    }

    #[test]
    fn test_failover_to_nonexistent() {
        let manager = ReplicaManager::new("leader-1", ReplicaManagerConfig::default());
        assert!(manager.manual_failover("nonexistent").is_err());
    }

    #[test]
    fn test_pause_resume() {
        let manager = ReplicaManager::new("leader-1", ReplicaManagerConfig::default());
        manager.add_replica("replica-1", "http://replica1:8080").unwrap();

        manager.pause_replica("replica-1").unwrap();
        let replicas = manager.list_replicas();
        assert_eq!(replicas[0].health, ReplicaHealth::Paused);

        manager.resume_replica("replica-1").unwrap();
        let replicas = manager.list_replicas();
        assert_eq!(replicas[0].health, ReplicaHealth::CatchingUp);
    }

    #[test]
    fn test_max_replicas() {
        let config = ReplicaManagerConfig {
            max_replicas: 2,
            ..Default::default()
        };
        let manager = ReplicaManager::new("leader-1", config);

        manager.add_replica("r1", "http://r1:8080").unwrap();
        manager.add_replica("r2", "http://r2:8080").unwrap();
        assert!(manager.add_replica("r3", "http://r3:8080").is_err());
    }

    #[test]
    fn test_health_check_no_replicas() {
        let manager = ReplicaManager::new("leader-1", ReplicaManagerConfig::default());
        let health = manager.health_check();
        assert_eq!(health.overall_health, ClusterHealth::NoReplicas);
    }

    #[test]
    fn test_remove_replica() {
        let manager = ReplicaManager::new("leader-1", ReplicaManagerConfig::default());
        manager.add_replica("r1", "http://r1:8080").unwrap();

        manager.remove_replica("r1").unwrap();
        assert_eq!(manager.list_replicas().len(), 0);
        assert!(manager.remove_replica("r1").is_err());
    }

    #[test]
    fn test_auto_failover() {
        let manager = ReplicaManager::new("leader-1", ReplicaManagerConfig::default());
        manager.add_replica("r1", "http://r1:8080").unwrap();
        manager.add_replica("r2", "http://r2:8080").unwrap();
        manager.advance_leader_lsn(100);
        manager.record_heartbeat("r1", 90);
        manager.record_heartbeat("r2", 95);

        let action = manager.auto_failover("leader unresponsive").unwrap();
        // r2 has higher applied_lsn so should be selected
        assert_eq!(action.new_leader, "r2");
        assert!(!action.is_manual);
    }

    #[test]
    fn test_auto_failover_no_healthy() {
        let manager = ReplicaManager::new("leader-1", ReplicaManagerConfig::default());
        // No replicas registered
        assert!(manager.auto_failover("test").is_err());
    }

    #[test]
    fn test_wal_shipment() {
        let manager = ReplicaManager::new("leader-1", ReplicaManagerConfig::default());
        manager.add_replica("r1", "http://r1:8080").unwrap();
        manager.advance_leader_lsn(100);
        manager.record_heartbeat("r1", 50);

        let shipment = manager.pending_wal_entries_for("r1").unwrap();
        assert_eq!(shipment.from_lsn, 50);
        assert_eq!(shipment.to_lsn, 100);
        assert_eq!(shipment.entries_pending, 50);
    }

    #[test]
    fn test_lag_tolerance() {
        let config = ReplicaManagerConfig {
            max_lag_secs: 10,
            ..Default::default()
        };
        let manager = ReplicaManager::new("leader-1", config);
        manager.add_replica("r1", "http://r1:8080").unwrap();
        manager.advance_leader_lsn(100);
        manager.record_heartbeat("r1", 99);

        assert!(manager.is_within_lag_tolerance("r1"));
    }

    #[test]
    fn test_replica_database() {
        let mut replica = ReplicaDatabase::new("r1", 5.0);
        assert!(replica.is_query_ready());
        assert!(replica.validate_read().is_ok());

        replica.update_lag(10.0);
        assert!(!replica.is_query_ready());
        assert!(replica.validate_read().is_err());
    }

    #[test]
    fn test_replica_database_wal_apply() {
        let mut replica = ReplicaDatabase::new("r1", 5.0);

        let entry = crate::wal::WalEntry::Insert {
            collection: "docs".to_string(),
            id: "doc1".to_string(),
            vector: vec![0.1; 4],
            metadata: None,
        };
        replica.apply_wal_entry(1, &entry).unwrap();
        assert_eq!(replica.applied_lsn(), 1);
        assert_eq!(replica.collections().len(), 1);
    }
}
