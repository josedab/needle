//! Replicated Database — Raft-based distributed consensus
//!
//! ⚠️ **Enterprise Feature**: This module wraps a `Database` with Raft consensus
//! to replicate all write operations across a cluster.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tracing::warn;

use crate::collection::CollectionConfig;
use crate::database::{CollectionRef, Database};
use crate::error::{NeedleError, Result};
use crate::raft::{
    Command as RaftCommand, NodeId, RaftConfig, RaftError, RaftMessage, RaftNode, RaftState,
};

/// A database command that can be replicated across the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicatedCommand {
    /// Create a new collection.
    CreateCollection {
        name: String,
        dimensions: usize,
        config: Option<CollectionConfig>,
    },
    /// Drop a collection.
    DropCollection { name: String },
    /// Insert a vector into a collection.
    Insert {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    /// Update a vector in a collection.
    Update {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    /// Delete a vector from a collection.
    Delete { collection: String, id: String },
    /// Compact a collection.
    Compact { collection: String },
    /// Clear a collection.
    Clear { collection: String },
    /// No-op command for Raft leader establishment.
    Noop,
}

impl ReplicatedCommand {
    /// Convert to a Raft command for serialization.
    fn to_raft_command(&self) -> RaftCommand {
        let serialized = serde_json::to_string(self).unwrap_or_default();
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("cmd".to_string(), serialized);

        RaftCommand::Insert {
            id: Self::command_id(),
            vector: Vec::new(),
            metadata,
        }
    }

    /// Generate a unique command ID for deduplication.
    fn command_id() -> String {
        let mut hasher = DefaultHasher::new();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        now.hash(&mut hasher);
        format!("cmd_{:016x}", hasher.finish())
    }

    /// Parse a replicated command from a Raft command.
    fn from_raft_command(cmd: &RaftCommand) -> Option<Self> {
        match cmd {
            RaftCommand::Insert { metadata, .. } => metadata
                .get("cmd")
                .and_then(|s| serde_json::from_str(s).ok()),
            RaftCommand::Noop => Some(ReplicatedCommand::Noop),
            _ => None,
        }
    }
}

/// Configuration for replicated database.
#[derive(Debug, Clone)]
pub struct ReplicatedDatabaseConfig {
    /// Node ID for this replica.
    pub node_id: NodeId,
    /// Raft configuration.
    pub raft_config: RaftConfig,
    /// Peer nodes in the cluster.
    pub peers: Vec<NodeId>,
    /// Allow reads from followers (eventually consistent).
    pub allow_follower_reads: bool,
}

impl Default for ReplicatedDatabaseConfig {
    fn default() -> Self {
        Self {
            node_id: NodeId(1),
            raft_config: RaftConfig::default(),
            peers: Vec::new(),
            allow_follower_reads: true,
        }
    }
}

impl ReplicatedDatabaseConfig {
    /// Create a new config with the given node ID.
    pub fn new(node_id: u64) -> Self {
        Self {
            node_id: NodeId(node_id),
            ..Default::default()
        }
    }

    /// Add peer nodes to the cluster.
    pub fn with_peers(mut self, peers: Vec<u64>) -> Self {
        self.peers = peers.into_iter().map(NodeId).collect();
        self
    }

    /// Set Raft configuration.
    pub fn with_raft_config(mut self, config: RaftConfig) -> Self {
        self.raft_config = config;
        self
    }

    /// Enable or disable follower reads.
    pub fn allow_follower_reads(mut self, allow: bool) -> Self {
        self.allow_follower_reads = allow;
        self
    }
}

/// A replicated database that uses Raft for consensus.
///
/// This wraps a `Database` and ensures all write operations are
/// replicated across the cluster before being applied locally.
///
/// # Example
///
/// ```ignore
/// use needle::enterprise::replicated_database::{ReplicatedDatabase, ReplicatedDatabaseConfig};
/// use needle::Database;
///
/// let db = Database::in_memory();
/// let config = ReplicatedDatabaseConfig::new(1)
///     .with_peers(vec![2, 3]);
///
/// let mut replicated_db = ReplicatedDatabase::new(db, config);
/// replicated_db.propose_create_collection("documents", 384)?;
/// ```
pub struct ReplicatedDatabase {
    /// The underlying database.
    db: Database,
    /// Raft node for consensus.
    raft: RaftNode,
    /// Configuration.
    config: ReplicatedDatabaseConfig,
    /// Message handler for network communication.
    message_handler: Option<Box<dyn MessageHandler>>,
}

/// Trait for handling Raft messages (network communication).
pub trait MessageHandler: Send + Sync {
    /// Send a message to another node.
    fn send(&self, to: NodeId, message: RaftMessage) -> Result<()>;

    /// Receive messages from other nodes.
    fn receive(&self) -> Vec<(NodeId, RaftMessage)>;
}

impl ReplicatedDatabase {
    /// Create a new replicated database.
    pub fn new(db: Database, config: ReplicatedDatabaseConfig) -> Self {
        let mut raft = RaftNode::new(config.node_id, config.raft_config.clone());
        raft.initialize(config.peers.clone());

        Self {
            db,
            raft,
            config,
            message_handler: None,
        }
    }

    /// Set the message handler for network communication.
    pub fn with_message_handler<H: MessageHandler + 'static>(mut self, handler: H) -> Self {
        self.message_handler = Some(Box::new(handler));
        self
    }

    /// Get the underlying database reference.
    pub fn database(&self) -> &Database {
        &self.db
    }

    /// Check if this node is the leader.
    pub fn is_leader(&self) -> bool {
        self.raft.is_leader()
    }

    /// Get the current leader's node ID.
    pub fn leader(&self) -> Option<NodeId> {
        self.raft.leader()
    }

    /// Get the current Raft state.
    pub fn state(&self) -> RaftState {
        self.raft.state()
    }

    /// Get the node ID.
    pub fn node_id(&self) -> NodeId {
        self.raft.id()
    }

    /// Tick the Raft node (should be called periodically).
    ///
    /// This handles election timeouts and heartbeats.
    pub fn tick(&mut self) {
        self.raft.tick();

        // Send outgoing messages
        if let Some(handler) = &self.message_handler {
            for msg in self.raft.take_messages() {
                if let Err(e) = handler.send(msg.to, msg.message) {
                    warn!(target = ?msg.to, error = %e, "Failed to send Raft message");
                }
            }
        }

        // Receive and process incoming messages
        if let Some(handler) = &self.message_handler {
            for (from, message) in handler.receive() {
                self.raft.handle_message(from, message);
            }
        }

        // Apply committed commands
        self.apply_committed();
    }

    /// Apply committed commands to the database.
    fn apply_committed(&mut self) {
        for cmd in self.raft.take_committed() {
            if let Some(replicated_cmd) = ReplicatedCommand::from_raft_command(&cmd) {
                if let Err(e) = self.apply_command(&replicated_cmd) {
                    warn!(error = %e, "Failed to apply committed command");
                }
            }
        }
    }

    /// Apply a single command to the database.
    fn apply_command(&self, cmd: &ReplicatedCommand) -> Result<()> {
        match cmd {
            ReplicatedCommand::CreateCollection {
                name,
                dimensions,
                config,
            } => {
                if let Some(cfg) = config {
                    self.db.create_collection_with_config(cfg.clone())
                } else {
                    self.db.create_collection(name, *dimensions)
                }
            }
            ReplicatedCommand::DropCollection { name } => self.db.drop_collection(name).map(|_| ()),
            ReplicatedCommand::Insert {
                collection,
                id,
                vector,
                metadata,
            } => {
                let coll = self.db.collection(collection)?;
                coll.insert(id, vector, metadata.clone())
            }
            ReplicatedCommand::Update {
                collection,
                id,
                vector,
                metadata,
            } => {
                let coll = self.db.collection(collection)?;
                coll.update(id, vector, metadata.clone())
            }
            ReplicatedCommand::Delete { collection, id } => {
                let coll = self.db.collection(collection)?;
                coll.delete(id).map(|_| ())
            }
            ReplicatedCommand::Compact { collection } => {
                let coll = self.db.collection(collection)?;
                coll.compact().map(|_| ())
            }
            ReplicatedCommand::Clear { collection } => {
                let coll = self.db.collection(collection)?;
                let ids = coll.ids()?;
                for id in ids {
                    coll.delete(&id)?;
                }
                Ok(())
            }
            ReplicatedCommand::Noop => Ok(()),
        }
    }

    /// Propose creating a new collection.
    pub fn propose_create_collection(
        &mut self,
        name: &str,
        dimensions: usize,
    ) -> std::result::Result<(), ReplicatedDatabaseError> {
        self.propose_command(ReplicatedCommand::CreateCollection {
            name: name.to_string(),
            dimensions,
            config: None,
        })
    }

    /// Propose creating a collection with custom configuration.
    pub fn propose_create_collection_with_config(
        &mut self,
        config: CollectionConfig,
    ) -> std::result::Result<(), ReplicatedDatabaseError> {
        self.propose_command(ReplicatedCommand::CreateCollection {
            name: config.name.clone(),
            dimensions: config.dimensions,
            config: Some(config),
        })
    }

    /// Propose dropping a collection.
    pub fn propose_drop_collection(
        &mut self,
        name: &str,
    ) -> std::result::Result<(), ReplicatedDatabaseError> {
        self.propose_command(ReplicatedCommand::DropCollection {
            name: name.to_string(),
        })
    }

    /// Propose inserting a vector.
    pub fn propose_insert(
        &mut self,
        collection: &str,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> std::result::Result<(), ReplicatedDatabaseError> {
        self.propose_command(ReplicatedCommand::Insert {
            collection: collection.to_string(),
            id: id.to_string(),
            vector: vector.to_vec(),
            metadata,
        })
    }

    /// Propose updating a vector.
    pub fn propose_update(
        &mut self,
        collection: &str,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> std::result::Result<(), ReplicatedDatabaseError> {
        self.propose_command(ReplicatedCommand::Update {
            collection: collection.to_string(),
            id: id.to_string(),
            vector: vector.to_vec(),
            metadata,
        })
    }

    /// Propose deleting a vector.
    pub fn propose_delete(
        &mut self,
        collection: &str,
        id: &str,
    ) -> std::result::Result<(), ReplicatedDatabaseError> {
        self.propose_command(ReplicatedCommand::Delete {
            collection: collection.to_string(),
            id: id.to_string(),
        })
    }

    /// Propose a command to the Raft cluster.
    fn propose_command(
        &mut self,
        cmd: ReplicatedCommand,
    ) -> std::result::Result<(), ReplicatedDatabaseError> {
        if !self.is_leader() {
            return Err(ReplicatedDatabaseError::NotLeader(self.leader()));
        }

        let raft_cmd = cmd.to_raft_command();
        self.raft.propose(raft_cmd).map_err(|e| match e {
            RaftError::NotLeader(leader) => ReplicatedDatabaseError::NotLeader(leader),
            RaftError::InvalidOperation(msg) => ReplicatedDatabaseError::InvalidOperation(msg),
        })?;

        Ok(())
    }

    /// Read from the local database.
    ///
    /// If `allow_follower_reads` is false and this node is not the leader,
    /// this will return an error.
    pub fn read_collection(
        &self,
        name: &str,
    ) -> std::result::Result<CollectionRef<'_>, ReplicatedDatabaseError> {
        if !self.config.allow_follower_reads && !self.is_leader() {
            return Err(ReplicatedDatabaseError::NotLeader(self.leader()));
        }

        self.db
            .collection(name)
            .map_err(ReplicatedDatabaseError::Database)
    }

    /// List all collections (local read).
    pub fn list_collections(&self) -> std::result::Result<Vec<String>, ReplicatedDatabaseError> {
        if !self.config.allow_follower_reads && !self.is_leader() {
            return Err(ReplicatedDatabaseError::NotLeader(self.leader()));
        }

        Ok(self.db.list_collections())
    }

    /// Check if a collection exists (local read).
    pub fn has_collection(&self, name: &str) -> std::result::Result<bool, ReplicatedDatabaseError> {
        if !self.config.allow_follower_reads && !self.is_leader() {
            return Err(ReplicatedDatabaseError::NotLeader(self.leader()));
        }

        Ok(self.db.has_collection(name))
    }

    /// Get cluster status.
    pub fn status(&self) -> ReplicatedDatabaseStatus {
        let raft_status = self.raft.status();
        ReplicatedDatabaseStatus {
            node_id: raft_status.id,
            state: raft_status.state,
            term: raft_status.term,
            leader_id: raft_status.leader_id,
            commit_index: raft_status.commit_index,
            last_applied: raft_status.last_applied,
            cluster_size: raft_status.cluster_size,
            collections: self.db.list_collections(),
            total_vectors: self.db.total_vectors(),
        }
    }
}

/// Errors from replicated database operations.
#[derive(Debug)]
pub enum ReplicatedDatabaseError {
    /// Not the leader; redirect to the leader.
    NotLeader(Option<NodeId>),
    /// Invalid operation.
    InvalidOperation(String),
    /// Database error.
    Database(NeedleError),
}

impl std::fmt::Display for ReplicatedDatabaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplicatedDatabaseError::NotLeader(Some(leader)) => {
                write!(f, "Not the leader; redirect to node {:?}", leader)
            }
            ReplicatedDatabaseError::NotLeader(None) => {
                write!(f, "Not the leader; leader unknown")
            }
            ReplicatedDatabaseError::InvalidOperation(msg) => {
                write!(f, "Invalid operation: {}", msg)
            }
            ReplicatedDatabaseError::Database(e) => {
                write!(f, "Database error: {}", e)
            }
        }
    }
}

impl std::error::Error for ReplicatedDatabaseError {}

/// Status of a replicated database node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicatedDatabaseStatus {
    /// This node's ID.
    pub node_id: NodeId,
    /// Current Raft state.
    pub state: RaftState,
    /// Current term.
    pub term: u64,
    /// Known leader ID.
    pub leader_id: Option<NodeId>,
    /// Commit index.
    pub commit_index: u64,
    /// Last applied index.
    pub last_applied: u64,
    /// Number of nodes in cluster.
    pub cluster_size: usize,
    /// List of collections.
    pub collections: Vec<String>,
    /// Total vectors across all collections.
    pub total_vectors: usize,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn make_replicated_db() -> ReplicatedDatabase {
        let db = Database::in_memory();
        let config = ReplicatedDatabaseConfig::new(1).with_peers(vec![2, 3]);
        ReplicatedDatabase::new(db, config)
    }

    // ── Config ──────────────────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let config = ReplicatedDatabaseConfig::default();
        assert_eq!(config.node_id, NodeId(1));
        assert!(config.peers.is_empty());
        assert!(config.allow_follower_reads);
    }

    #[test]
    fn test_config_with_peers() {
        let config = ReplicatedDatabaseConfig::new(5).with_peers(vec![1, 2, 3]);
        assert_eq!(config.node_id, NodeId(5));
        assert_eq!(config.peers.len(), 3);
    }

    #[test]
    fn test_config_follower_reads_disabled() {
        let config = ReplicatedDatabaseConfig::new(1).allow_follower_reads(false);
        assert!(!config.allow_follower_reads);
    }

    // ── ReplicatedDatabase creation ─────────────────────────────────────

    #[test]
    fn test_new_replicated_db() {
        let rdb = make_replicated_db();
        assert_eq!(rdb.node_id(), NodeId(1));
    }

    #[test]
    fn test_database_accessor() {
        let rdb = make_replicated_db();
        let db = rdb.database();
        assert!(db.list_collections().is_empty());
    }

    #[test]
    fn test_initial_state() {
        let rdb = make_replicated_db();
        // Freshly created node starts as follower
        let state = rdb.state();
        assert!(state == RaftState::Follower || state == RaftState::Candidate);
    }

    // ── Status ──────────────────────────────────────────────────────────

    #[test]
    fn test_status_initial() {
        let rdb = make_replicated_db();
        let status = rdb.status();
        assert_eq!(status.node_id, NodeId(1));
        assert!(status.collections.is_empty());
        assert_eq!(status.total_vectors, 0);
    }

    // ── Propose commands when not leader ─────────────────────────────────

    #[test]
    fn test_propose_create_collection_not_leader() {
        let mut rdb = make_replicated_db();
        let result = rdb.propose_create_collection("test", 128);
        // Freshly created node is not leader, should fail
        assert!(result.is_err());
    }

    #[test]
    fn test_propose_insert_not_leader() {
        let mut rdb = make_replicated_db();
        let result = rdb.propose_insert("test", "v1", &[1.0, 0.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_propose_delete_not_leader() {
        let mut rdb = make_replicated_db();
        let result = rdb.propose_delete("test", "v1");
        assert!(result.is_err());
    }

    #[test]
    fn test_propose_update_not_leader() {
        let mut rdb = make_replicated_db();
        let result = rdb.propose_update("test", "v1", &[1.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_propose_drop_collection_not_leader() {
        let mut rdb = make_replicated_db();
        let result = rdb.propose_drop_collection("test");
        assert!(result.is_err());
    }

    // ── Read operations ─────────────────────────────────────────────────

    #[test]
    fn test_list_collections_follower_reads_allowed() {
        let rdb = make_replicated_db();
        let collections = rdb.list_collections().unwrap();
        assert!(collections.is_empty());
    }

    #[test]
    fn test_list_collections_follower_reads_disabled() {
        let db = Database::in_memory();
        let config = ReplicatedDatabaseConfig::new(1)
            .with_peers(vec![2, 3])
            .allow_follower_reads(false);
        let rdb = ReplicatedDatabase::new(db, config);
        let result = rdb.list_collections();
        // Not leader and follower reads disabled
        assert!(result.is_err());
    }

    #[test]
    fn test_has_collection_empty() {
        let rdb = make_replicated_db();
        let has = rdb.has_collection("nonexistent").unwrap();
        assert!(!has);
    }

    #[test]
    fn test_read_collection_not_found() {
        let rdb = make_replicated_db();
        let result = rdb.read_collection("nonexistent");
        assert!(result.is_err());
    }

    // ── ReplicatedCommand serialization ─────────────────────────────────

    #[test]
    fn test_command_roundtrip() {
        let cmd = ReplicatedCommand::CreateCollection {
            name: "test".into(),
            dimensions: 128,
            config: None,
        };
        let raft_cmd = cmd.to_raft_command();
        let parsed = ReplicatedCommand::from_raft_command(&raft_cmd);
        assert!(parsed.is_some());
        if let Some(ReplicatedCommand::CreateCollection { name, dimensions, .. }) = parsed {
            assert_eq!(name, "test");
            assert_eq!(dimensions, 128);
        }
    }

    #[test]
    fn test_command_noop_roundtrip() {
        let cmd = ReplicatedCommand::Noop;
        let raft_cmd = cmd.to_raft_command();
        let parsed = ReplicatedCommand::from_raft_command(&raft_cmd);
        assert!(parsed.is_some());
    }

    #[test]
    fn test_noop_from_raft_noop() {
        let parsed = ReplicatedCommand::from_raft_command(&RaftCommand::Noop);
        assert!(matches!(parsed, Some(ReplicatedCommand::Noop)));
    }

    // ── Error display ───────────────────────────────────────────────────

    #[test]
    fn test_error_display_not_leader_known() {
        let err = ReplicatedDatabaseError::NotLeader(Some(NodeId(3)));
        let msg = err.to_string();
        assert!(msg.contains("Not the leader"));
    }

    #[test]
    fn test_error_display_not_leader_unknown() {
        let err = ReplicatedDatabaseError::NotLeader(None);
        assert!(err.to_string().contains("leader unknown"));
    }

    #[test]
    fn test_error_display_invalid_operation() {
        let err = ReplicatedDatabaseError::InvalidOperation("bad".into());
        assert!(err.to_string().contains("bad"));
    }

    #[test]
    fn test_error_display_database() {
        let err = ReplicatedDatabaseError::Database(
            NeedleError::CollectionNotFound("x".into()),
        );
        assert!(err.to_string().contains("Database error"));
    }

    // ── Tick ────────────────────────────────────────────────────────────

    #[test]
    fn test_tick_no_crash() {
        let mut rdb = make_replicated_db();
        // Should not panic
        for _ in 0..10 {
            rdb.tick();
        }
    }

    // ── Apply command directly ──────────────────────────────────────────

    #[test]
    fn test_apply_noop() {
        let rdb = make_replicated_db();
        let result = rdb.apply_command(&ReplicatedCommand::Noop);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_create_collection() {
        let rdb = make_replicated_db();
        let result = rdb.apply_command(&ReplicatedCommand::CreateCollection {
            name: "docs".into(),
            dimensions: 64,
            config: None,
        });
        assert!(result.is_ok());
        assert!(rdb.database().has_collection("docs"));
    }

    #[test]
    fn test_apply_insert_nonexistent_collection() {
        let rdb = make_replicated_db();
        let result = rdb.apply_command(&ReplicatedCommand::Insert {
            collection: "nonexistent".into(),
            id: "v1".into(),
            vector: vec![1.0],
            metadata: None,
        });
        assert!(result.is_err());
    }
}
