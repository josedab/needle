//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Edge Sync - Offline-first CRDT-based synchronization for vector databases.
//!
//! Enables conflict-free replication between edge devices and central servers
//! using Conflict-free Replicated Data Types (CRDTs).
//!
//! # Features
//!
//! - **Offline-first**: Work without connectivity, sync when available
//! - **Conflict-free**: Automatic conflict resolution without coordination
//! - **Vector CRDTs**: Specialized CRDTs for vector embeddings
//! - **Delta sync**: Efficient incremental synchronization
//! - **Causal ordering**: Preserve operation ordering
//!
//! # Example
//!
//! ```ignore
//! use needle::crdt::{VectorCRDT, ReplicaId, SyncManager};
//!
//! let replica_id = ReplicaId::new();
//! let mut crdt = VectorCRDT::new(replica_id);
//!
//! // Add vectors (works offline)
//! crdt.add("vec1", &embedding, metadata)?;
//!
//! // Get delta for sync
//! let delta = crdt.delta_since(last_sync);
//!
//! // Merge remote changes
//! crdt.merge(remote_delta)?;
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BTreeMap};
use std::time::{SystemTime, UNIX_EPOCH};

/// Unique replica identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplicaId(pub u64);

impl ReplicaId {
    /// Generate a new replica ID.
    pub fn new() -> Self {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();

        let mut hasher = DefaultHasher::new();
        now.hash(&mut hasher);
        ReplicaId(hasher.finish())
    }

    /// Create from specific value.
    pub fn from(id: u64) -> Self {
        ReplicaId(id)
    }
}

impl Default for ReplicaId {
    fn default() -> Self {
        Self::new()
    }
}

/// Hybrid Logical Clock for causal ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HLC {
    /// Physical timestamp.
    pub physical: u64,
    /// Logical counter.
    pub logical: u32,
    /// Replica ID.
    pub replica: u64,
}

impl HLC {
    /// Create new HLC.
    pub fn new(replica: ReplicaId) -> Self {
        Self {
            physical: Self::now(),
            logical: 0,
            replica: replica.0,
        }
    }

    /// Get current time.
    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_millis() as u64
    }

    /// Tick the clock for local event.
    pub fn tick(&mut self) -> HLC {
        let now = Self::now();
        if now > self.physical {
            self.physical = now;
            self.logical = 0;
        } else {
            self.logical += 1;
        }
        *self
    }

    /// Update clock on receiving remote timestamp.
    pub fn receive(&mut self, remote: HLC) -> HLC {
        let now = Self::now();
        if now > self.physical && now > remote.physical {
            self.physical = now;
            self.logical = 0;
        } else if self.physical > remote.physical {
            self.logical += 1;
        } else if remote.physical > self.physical {
            self.physical = remote.physical;
            self.logical = remote.logical + 1;
        } else {
            self.logical = self.logical.max(remote.logical) + 1;
        }
        *self
    }
}

/// Operation types for the CRDT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    /// Add a vector.
    Add {
        id: String,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    },
    /// Update a vector.
    Update {
        id: String,
        vector: Vec<f32>,
    },
    /// Update metadata.
    UpdateMetadata {
        id: String,
        key: String,
        value: Option<String>,
    },
    /// Delete a vector.
    Delete {
        id: String,
    },
}

/// An operation with its timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedOp {
    /// The operation.
    pub op: Operation,
    /// HLC timestamp.
    pub timestamp: HLC,
    /// Origin replica.
    pub origin: ReplicaId,
}

impl TimestampedOp {
    /// Create new timestamped operation.
    pub fn new(op: Operation, timestamp: HLC, origin: ReplicaId) -> Self {
        Self { op, timestamp, origin }
    }
}

/// Vector entry in the CRDT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CRDTVector {
    /// Vector ID.
    pub id: String,
    /// Vector data.
    pub vector: Vec<f32>,
    /// Metadata (LWW-Register per key).
    pub metadata: HashMap<String, (String, HLC)>,
    /// Creation timestamp.
    pub created_at: HLC,
    /// Last update timestamp.
    pub updated_at: HLC,
    /// Tombstone (for deletion).
    pub deleted: Option<HLC>,
}

impl CRDTVector {
    /// Create new vector entry.
    pub fn new(id: &str, vector: Vec<f32>, metadata: HashMap<String, String>, timestamp: HLC) -> Self {
        let meta_with_ts: HashMap<String, (String, HLC)> = metadata
            .into_iter()
            .map(|(k, v)| (k, (v, timestamp)))
            .collect();

        Self {
            id: id.to_string(),
            vector,
            metadata: meta_with_ts,
            created_at: timestamp,
            updated_at: timestamp,
            deleted: None,
        }
    }

    /// Get metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|(v, _)| v.as_str())
    }

    /// Get all metadata.
    pub fn get_all_metadata(&self) -> HashMap<String, String> {
        self.metadata.iter()
            .map(|(k, (v, _))| (k.clone(), v.clone()))
            .collect()
    }

    /// Check if deleted.
    pub fn is_deleted(&self) -> bool {
        self.deleted.is_some()
    }
}

/// Delta for synchronization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    /// Operations in this delta.
    pub operations: Vec<TimestampedOp>,
    /// Minimum timestamp in delta.
    pub from_timestamp: Option<HLC>,
    /// Maximum timestamp in delta.
    pub to_timestamp: Option<HLC>,
    /// Origin replica.
    pub origin: ReplicaId,
}

impl Delta {
    /// Create empty delta.
    pub fn empty(origin: ReplicaId) -> Self {
        Self {
            operations: Vec::new(),
            from_timestamp: None,
            to_timestamp: None,
            origin,
        }
    }

    /// Check if delta is empty.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Number of operations.
    pub fn len(&self) -> usize {
        self.operations.len()
    }
}

/// CRDT-based vector store.
pub struct VectorCRDT {
    /// This replica's ID.
    replica_id: ReplicaId,
    /// Hybrid logical clock.
    clock: HLC,
    /// Vector entries.
    vectors: HashMap<String, CRDTVector>,
    /// Operation log for delta sync.
    operation_log: BTreeMap<HLC, TimestampedOp>,
    /// Peers we know about.
    known_peers: HashSet<ReplicaId>,
    /// Last sync timestamp per peer.
    peer_sync_state: HashMap<ReplicaId, HLC>,
}

impl VectorCRDT {
    /// Create new CRDT.
    pub fn new(replica_id: ReplicaId) -> Self {
        Self {
            replica_id,
            clock: HLC::new(replica_id),
            vectors: HashMap::new(),
            operation_log: BTreeMap::new(),
            known_peers: HashSet::new(),
            peer_sync_state: HashMap::new(),
        }
    }

    /// Get replica ID.
    pub fn replica_id(&self) -> ReplicaId {
        self.replica_id
    }

    /// Add a vector.
    pub fn add(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: HashMap<String, String>,
    ) -> Result<HLC> {
        let timestamp = self.clock.tick();

        let op = Operation::Add {
            id: id.to_string(),
            vector: vector.to_vec(),
            metadata: metadata.clone(),
        };

        self.apply_operation(op.clone(), timestamp)?;

        let timestamped = TimestampedOp::new(op, timestamp, self.replica_id);
        self.operation_log.insert(timestamp, timestamped);

        Ok(timestamp)
    }

    /// Update a vector.
    pub fn update(&mut self, id: &str, vector: &[f32]) -> Result<HLC> {
        if !self.vectors.contains_key(id) {
            return Err(NeedleError::NotFound(format!("Vector '{}' not found", id)));
        }

        let timestamp = self.clock.tick();

        let op = Operation::Update {
            id: id.to_string(),
            vector: vector.to_vec(),
        };

        self.apply_operation(op.clone(), timestamp)?;

        let timestamped = TimestampedOp::new(op, timestamp, self.replica_id);
        self.operation_log.insert(timestamp, timestamped);

        Ok(timestamp)
    }

    /// Update metadata.
    pub fn update_metadata(&mut self, id: &str, key: &str, value: Option<&str>) -> Result<HLC> {
        if !self.vectors.contains_key(id) {
            return Err(NeedleError::NotFound(format!("Vector '{}' not found", id)));
        }

        let timestamp = self.clock.tick();

        let op = Operation::UpdateMetadata {
            id: id.to_string(),
            key: key.to_string(),
            value: value.map(|s| s.to_string()),
        };

        self.apply_operation(op.clone(), timestamp)?;

        let timestamped = TimestampedOp::new(op, timestamp, self.replica_id);
        self.operation_log.insert(timestamp, timestamped);

        Ok(timestamp)
    }

    /// Delete a vector.
    pub fn delete(&mut self, id: &str) -> Result<HLC> {
        if !self.vectors.contains_key(id) {
            return Err(NeedleError::NotFound(format!("Vector '{}' not found", id)));
        }

        let timestamp = self.clock.tick();

        let op = Operation::Delete {
            id: id.to_string(),
        };

        self.apply_operation(op.clone(), timestamp)?;

        let timestamped = TimestampedOp::new(op, timestamp, self.replica_id);
        self.operation_log.insert(timestamp, timestamped);

        Ok(timestamp)
    }

    /// Apply an operation.
    fn apply_operation(&mut self, op: Operation, timestamp: HLC) -> Result<()> {
        match op {
            Operation::Add { id, vector, metadata } => {
                if let Some(existing) = self.vectors.get(&id) {
                    // LWW: only apply if newer
                    if timestamp > existing.updated_at {
                        let entry = CRDTVector::new(&id, vector, metadata, timestamp);
                        self.vectors.insert(id, entry);
                    }
                } else {
                    let entry = CRDTVector::new(&id, vector, metadata, timestamp);
                    self.vectors.insert(id, entry);
                }
            }
            Operation::Update { id, vector } => {
                if let Some(entry) = self.vectors.get_mut(&id) {
                    if timestamp > entry.updated_at && entry.deleted.is_none() {
                        entry.vector = vector;
                        entry.updated_at = timestamp;
                    }
                }
            }
            Operation::UpdateMetadata { id, key, value } => {
                if let Some(entry) = self.vectors.get_mut(&id) {
                    if entry.deleted.is_none() {
                        let should_update = entry.metadata
                            .get(&key)
                            .map(|(_, ts)| timestamp > *ts)
                            .unwrap_or(true);

                        if should_update {
                            if let Some(v) = value {
                                entry.metadata.insert(key, (v, timestamp));
                            } else {
                                entry.metadata.remove(&key);
                            }
                            entry.updated_at = timestamp;
                        }
                    }
                }
            }
            Operation::Delete { id } => {
                if let Some(entry) = self.vectors.get_mut(&id) {
                    // Only delete if timestamp is newer
                    if entry.deleted.map(|d| timestamp > d).unwrap_or(true)
                        && timestamp > entry.updated_at
                    {
                        entry.deleted = Some(timestamp);
                    }
                }
            }
        }
        Ok(())
    }

    /// Get a vector.
    pub fn get(&self, id: &str) -> Option<&CRDTVector> {
        self.vectors.get(id).filter(|v| !v.is_deleted())
    }

    /// List all vectors (excluding deleted).
    pub fn list(&self) -> Vec<&CRDTVector> {
        self.vectors.values().filter(|v| !v.is_deleted()).collect()
    }

    /// Get delta since a timestamp.
    pub fn delta_since(&self, since: Option<HLC>) -> Delta {
        let operations: Vec<TimestampedOp> = match since {
            Some(ts) => self.operation_log
                .range(ts..)
                .filter(|(t, _)| **t > ts)
                .map(|(_, op)| op.clone())
                .collect(),
            None => self.operation_log.values().cloned().collect(),
        };

        let from_timestamp = operations.first().map(|op| op.timestamp);
        let to_timestamp = operations.last().map(|op| op.timestamp);

        Delta {
            operations,
            from_timestamp,
            to_timestamp,
            origin: self.replica_id,
        }
    }

    /// Get delta for a specific peer.
    pub fn delta_for_peer(&self, peer: ReplicaId) -> Delta {
        let last_sync = self.peer_sync_state.get(&peer).copied();
        self.delta_since(last_sync)
    }

    /// Merge a delta from another replica.
    pub fn merge(&mut self, delta: Delta) -> Result<MergeResult> {
        if delta.is_empty() {
            return Ok(MergeResult::default());
        }

        let mut applied = 0;
        let mut skipped = 0;
        let mut conflicts = 0;

        // Update clock
        if let Some(ts) = delta.to_timestamp {
            self.clock.receive(ts);
        }

        // Track peer
        self.known_peers.insert(delta.origin);

        for timestamped in delta.operations {
            // Skip our own operations
            if timestamped.origin == self.replica_id {
                skipped += 1;
                continue;
            }

            // Check if we already have this operation
            if self.operation_log.contains_key(&timestamped.timestamp) {
                skipped += 1;
                continue;
            }

            // Check for conflicts (same key, concurrent updates)
            let has_conflict = match &timestamped.op {
                Operation::Update { id, .. } | Operation::Delete { id } => {
                    if let Some(existing) = self.vectors.get(id) {
                        existing.updated_at.replica != timestamped.timestamp.replica
                            && existing.updated_at.physical == timestamped.timestamp.physical
                    } else {
                        false
                    }
                }
                _ => false,
            };

            if has_conflict {
                conflicts += 1;
            }

            // Apply operation
            self.apply_operation(timestamped.op.clone(), timestamped.timestamp)?;
            self.operation_log.insert(timestamped.timestamp, timestamped);
            applied += 1;
        }

        // Update sync state
        if let Some(ts) = delta.to_timestamp {
            self.peer_sync_state.insert(delta.origin, ts);
        }

        Ok(MergeResult {
            applied,
            skipped,
            conflicts,
        })
    }

    /// Compact the operation log.
    pub fn compact(&mut self, keep_after: HLC) {
        self.operation_log.retain(|ts, _| *ts >= keep_after);
    }

    /// Get current clock value.
    pub fn current_clock(&self) -> HLC {
        self.clock
    }

    /// Get number of vectors (excluding deleted).
    pub fn len(&self) -> usize {
        self.vectors.values().filter(|v| !v.is_deleted()).count()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get operation log size.
    pub fn log_size(&self) -> usize {
        self.operation_log.len()
    }

    /// Get sync state.
    pub fn sync_state(&self) -> SyncState {
        SyncState {
            replica_id: self.replica_id,
            clock: self.clock,
            vector_count: self.len(),
            log_size: self.log_size(),
            known_peers: self.known_peers.iter().copied().collect(),
        }
    }
}

/// Result of a merge operation.
#[derive(Debug, Clone, Default)]
pub struct MergeResult {
    /// Operations applied.
    pub applied: usize,
    /// Operations skipped (duplicates).
    pub skipped: usize,
    /// Conflicts resolved.
    pub conflicts: usize,
}

/// Sync state information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncState {
    /// This replica's ID.
    pub replica_id: ReplicaId,
    /// Current clock.
    pub clock: HLC,
    /// Number of vectors.
    pub vector_count: usize,
    /// Operation log size.
    pub log_size: usize,
    /// Known peers.
    pub known_peers: Vec<ReplicaId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_crdt() {
        let replica = ReplicaId::new();
        let crdt = VectorCRDT::new(replica);

        assert!(crdt.is_empty());
        assert_eq!(crdt.replica_id(), replica);
    }

    #[test]
    fn test_add_vector() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0, 2.0, 3.0], HashMap::new()).unwrap();

        assert_eq!(crdt.len(), 1);
        assert!(crdt.get("vec1").is_some());
    }

    #[test]
    fn test_update_vector() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0, 2.0, 3.0], HashMap::new()).unwrap();
        crdt.update("vec1", &[4.0, 5.0, 6.0]).unwrap();

        let vec = crdt.get("vec1").unwrap();
        assert_eq!(vec.vector, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_delete_vector() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0, 2.0, 3.0], HashMap::new()).unwrap();
        crdt.delete("vec1").unwrap();

        assert!(crdt.get("vec1").is_none());
        assert_eq!(crdt.len(), 0);
    }

    #[test]
    fn test_metadata() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        let mut meta = HashMap::new();
        meta.insert("key".to_string(), "value".to_string());

        crdt.add("vec1", &[1.0], meta).unwrap();

        let vec = crdt.get("vec1").unwrap();
        assert_eq!(vec.get_metadata("key"), Some("value"));
    }

    #[test]
    fn test_update_metadata() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0], HashMap::new()).unwrap();
        crdt.update_metadata("vec1", "key", Some("value")).unwrap();

        let vec = crdt.get("vec1").unwrap();
        assert_eq!(vec.get_metadata("key"), Some("value"));
    }

    #[test]
    fn test_delta_sync() {
        let replica1 = ReplicaId::from(1);
        let replica2 = ReplicaId::from(2);

        let mut crdt1 = VectorCRDT::new(replica1);
        let mut crdt2 = VectorCRDT::new(replica2);

        // Add on replica 1
        crdt1.add("vec1", &[1.0, 2.0], HashMap::new()).unwrap();
        crdt1.add("vec2", &[3.0, 4.0], HashMap::new()).unwrap();

        // Get delta and merge to replica 2
        let delta = crdt1.delta_since(None);
        let result = crdt2.merge(delta).unwrap();

        assert_eq!(result.applied, 2);
        assert_eq!(crdt2.len(), 2);
    }

    #[test]
    fn test_concurrent_updates() {
        let replica1 = ReplicaId::from(1);
        let replica2 = ReplicaId::from(2);

        let mut crdt1 = VectorCRDT::new(replica1);
        let mut crdt2 = VectorCRDT::new(replica2);

        // Both add same ID concurrently
        crdt1.add("vec1", &[1.0], HashMap::new()).unwrap();
        crdt2.add("vec1", &[2.0], HashMap::new()).unwrap();

        // Merge in both directions
        let delta1 = crdt1.delta_since(None);
        let delta2 = crdt2.delta_since(None);

        crdt1.merge(delta2).unwrap();
        crdt2.merge(delta1).unwrap();

        // Both should converge to same value (LWW)
        let vec1 = crdt1.get("vec1").unwrap().vector.clone();
        let vec2 = crdt2.get("vec1").unwrap().vector.clone();
        assert_eq!(vec1, vec2);
    }

    #[test]
    fn test_incremental_delta() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0], HashMap::new()).unwrap();
        let checkpoint = crdt.current_clock();

        crdt.add("vec2", &[2.0], HashMap::new()).unwrap();
        crdt.add("vec3", &[3.0], HashMap::new()).unwrap();

        let delta = crdt.delta_since(Some(checkpoint));
        assert_eq!(delta.len(), 2); // Only vec2 and vec3
    }

    #[test]
    fn test_hlc_ordering() {
        let replica = ReplicaId::from(1);
        let mut clock = HLC::new(replica);

        let t1 = clock.tick();
        let t2 = clock.tick();
        let t3 = clock.tick();

        assert!(t1 < t2);
        assert!(t2 < t3);
    }

    #[test]
    fn test_hlc_receive() {
        let replica1 = ReplicaId::from(1);
        let replica2 = ReplicaId::from(2);

        let mut clock1 = HLC::new(replica1);
        let mut clock2 = HLC::new(replica2);

        let t1 = clock1.tick();
        clock2.receive(t1);

        let t2 = clock2.tick();
        assert!(t2 > t1);
    }

    #[test]
    fn test_compact() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0], HashMap::new()).unwrap();
        crdt.add("vec2", &[2.0], HashMap::new()).unwrap();
        let checkpoint = crdt.current_clock(); // After vec2

        assert_eq!(crdt.log_size(), 2);

        // Compact keeps only entries >= checkpoint (which is vec2's timestamp)
        crdt.compact(checkpoint);
        assert_eq!(crdt.log_size(), 1);
    }

    #[test]
    fn test_sync_state() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0], HashMap::new()).unwrap();

        let state = crdt.sync_state();
        assert_eq!(state.vector_count, 1);
        assert_eq!(state.log_size, 1);
    }

    #[test]
    fn test_delta_for_peer() {
        let replica1 = ReplicaId::from(1);
        let replica2 = ReplicaId::from(2);

        let mut crdt1 = VectorCRDT::new(replica1);

        crdt1.add("vec1", &[1.0], HashMap::new()).unwrap();

        // First sync
        let delta = crdt1.delta_for_peer(replica2);
        assert_eq!(delta.len(), 1);

        // Simulate sync complete
        crdt1.peer_sync_state.insert(replica2, crdt1.current_clock());

        crdt1.add("vec2", &[2.0], HashMap::new()).unwrap();

        // Second sync should only have new ops
        let delta2 = crdt1.delta_for_peer(replica2);
        assert_eq!(delta2.len(), 1);
    }

    #[test]
    fn test_list_vectors() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0], HashMap::new()).unwrap();
        crdt.add("vec2", &[2.0], HashMap::new()).unwrap();
        crdt.add("vec3", &[3.0], HashMap::new()).unwrap();
        crdt.delete("vec2").unwrap();

        let list = crdt.list();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_skip_own_operations() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0], HashMap::new()).unwrap();

        let delta = crdt.delta_since(None);
        let result = crdt.merge(delta).unwrap();

        assert_eq!(result.applied, 0);
        assert_eq!(result.skipped, 1);
    }
}
