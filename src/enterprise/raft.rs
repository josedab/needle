// Allow dead_code for this public API module - types are exported for library users
#![allow(dead_code)]
#![allow(clippy::redundant_closure)]

//! Raft Consensus - Distributed high-availability for vector databases.
//!
//! Implements the Raft consensus algorithm for leader election and log replication,
//! enabling fault-tolerant distributed vector storage.
//!
//! # Features
//!
//! - **Leader election**: Automatic leader failover
//! - **Log replication**: Consistent updates across nodes
//! - **Membership changes**: Dynamic cluster reconfiguration
//! - **Snapshot support**: Efficient state transfer
//! - **Persistent storage**: Durable log and state persistence
//! - **Read scalability**: Follower reads with linearizability
//!
//! # Example
//!
//! ```ignore
//! use needle::raft::{RaftNode, RaftConfig, NodeId, FileStorage};
//!
//! // Create persistent storage
//! let storage = FileStorage::open("/path/to/raft")?;
//!
//! // Create node with persistent storage
//! let config = RaftConfig::default();
//! let mut node = RaftNode::with_storage(NodeId(1), config, Box::new(storage))?;
//!
//! // Start the node
//! node.start(peers)?;
//!
//! // Propose a command
//! node.propose(Command::Insert { id, vector })?;
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Node identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

/// Term number.
pub type Term = u64;

/// Log index.
pub type LogIndex = u64;

/// Raft node state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RaftState {
    /// Follower state.
    Follower,
    /// Candidate state.
    Candidate,
    /// Leader state.
    Leader,
}

/// Configuration for Raft.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftConfig {
    /// Election timeout range (min).
    pub election_timeout_min: Duration,
    /// Election timeout range (max).
    pub election_timeout_max: Duration,
    /// Heartbeat interval.
    pub heartbeat_interval: Duration,
    /// Maximum entries per append.
    pub max_entries_per_append: usize,
    /// Snapshot threshold.
    pub snapshot_threshold: usize,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            election_timeout_min: Duration::from_millis(150),
            election_timeout_max: Duration::from_millis(300),
            heartbeat_interval: Duration::from_millis(50),
            max_entries_per_append: 100,
            snapshot_threshold: 10000,
        }
    }
}

/// A log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Term when entry was received.
    pub term: Term,
    /// Log index.
    pub index: LogIndex,
    /// Command to apply.
    pub command: Command,
}

/// Commands that can be proposed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    /// Insert a vector.
    Insert {
        id: String,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    },
    /// Update a vector.
    Update { id: String, vector: Vec<f32> },
    /// Delete a vector.
    Delete { id: String },
    /// No-op (for leader establishment).
    Noop,
    /// Cluster configuration change.
    ConfigChange {
        add: Option<NodeId>,
        remove: Option<NodeId>,
    },
}

/// Request for votes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVote {
    /// Candidate's term.
    pub term: Term,
    /// Candidate requesting vote.
    pub candidate_id: NodeId,
    /// Index of candidate's last log entry.
    pub last_log_index: LogIndex,
    /// Term of candidate's last log entry.
    pub last_log_term: Term,
}

/// Response to vote request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteResponse {
    /// Current term.
    pub term: Term,
    /// True if vote granted.
    pub vote_granted: bool,
}

/// Append entries request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntries {
    /// Leader's term.
    pub term: Term,
    /// Leader ID.
    pub leader_id: NodeId,
    /// Index of log entry preceding new ones.
    pub prev_log_index: LogIndex,
    /// Term of prev_log_index entry.
    pub prev_log_term: Term,
    /// Log entries to store (empty for heartbeat).
    pub entries: Vec<LogEntry>,
    /// Leader's commit index.
    pub leader_commit: LogIndex,
}

/// Response to append entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponse {
    /// Current term.
    pub term: Term,
    /// True if follower contained matching entry.
    pub success: bool,
    /// Follower's last log index (for optimization).
    pub match_index: LogIndex,
}

/// Snapshot metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// Last included index.
    pub last_included_index: LogIndex,
    /// Last included term.
    pub last_included_term: Term,
    /// Cluster configuration.
    pub config: Vec<NodeId>,
}

/// Message types for Raft protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaftMessage {
    /// Request vote.
    RequestVote(RequestVote),
    /// Vote response.
    RequestVoteResponse(RequestVoteResponse),
    /// Append entries.
    AppendEntries(AppendEntries),
    /// Append response.
    AppendEntriesResponse(AppendEntriesResponse),
    /// Install snapshot.
    InstallSnapshot {
        term: Term,
        leader_id: NodeId,
        metadata: SnapshotMetadata,
        data: Vec<u8>,
    },
    /// Snapshot response.
    InstallSnapshotResponse { term: Term, success: bool },
}

/// Outgoing message.
#[derive(Debug, Clone)]
pub struct OutgoingMessage {
    /// Target node.
    pub to: NodeId,
    /// Message content.
    pub message: RaftMessage,
}

/// Raft node.
pub struct RaftNode {
    /// This node's ID.
    id: NodeId,
    /// Configuration.
    config: RaftConfig,
    /// Current state.
    state: RaftState,
    /// Current term.
    current_term: Term,
    /// Candidate that received vote in current term.
    voted_for: Option<NodeId>,
    /// Log entries.
    log: Vec<LogEntry>,
    /// Index of highest log entry known to be committed.
    commit_index: LogIndex,
    /// Index of highest log entry applied to state machine.
    last_applied: LogIndex,
    /// Current leader (if known).
    leader_id: Option<NodeId>,
    /// Cluster members.
    cluster: HashSet<NodeId>,
    /// For leader: next index to send to each follower.
    next_index: HashMap<NodeId, LogIndex>,
    /// For leader: highest index known to be replicated on each follower.
    match_index: HashMap<NodeId, LogIndex>,
    /// Votes received in current election.
    votes_received: HashSet<NodeId>,
    /// Last heartbeat time.
    last_heartbeat: Instant,
    /// Election timeout.
    election_timeout: Duration,
    /// Pending messages to send.
    outbox: VecDeque<OutgoingMessage>,
    /// Commands waiting to be committed.
    pending_commands: Vec<(LogIndex, Command)>,
}

impl RaftNode {
    /// Create a new Raft node.
    pub fn new(id: NodeId, config: RaftConfig) -> Self {
        let election_timeout = Self::random_election_timeout(&config);

        Self {
            id,
            config,
            state: RaftState::Follower,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            leader_id: None,
            cluster: HashSet::new(),
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            votes_received: HashSet::new(),
            last_heartbeat: Instant::now(),
            election_timeout,
            outbox: VecDeque::new(),
            pending_commands: Vec::new(),
        }
    }

    /// Generate random election timeout.
    fn random_election_timeout(config: &RaftConfig) -> Duration {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        let mut hasher = DefaultHasher::new();
        now.hash(&mut hasher);
        let rand = hasher.finish() as f64 / u64::MAX as f64;

        let min = config.election_timeout_min.as_millis() as f64;
        let max = config.election_timeout_max.as_millis() as f64;
        let timeout_ms = min + rand * (max - min);

        Duration::from_millis(timeout_ms as u64)
    }

    /// Initialize cluster with peers.
    pub fn initialize(&mut self, peers: Vec<NodeId>) {
        self.cluster.insert(self.id);
        for peer in peers {
            self.cluster.insert(peer);
        }
    }

    /// Get node ID.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Get current state.
    pub fn state(&self) -> RaftState {
        self.state
    }

    /// Get current term.
    pub fn term(&self) -> Term {
        self.current_term
    }

    /// Get current leader.
    pub fn leader(&self) -> Option<NodeId> {
        self.leader_id
    }

    /// Check if this node is the leader.
    pub fn is_leader(&self) -> bool {
        self.state == RaftState::Leader
    }

    /// Propose a command (leader only).
    pub fn propose(&mut self, command: Command) -> std::result::Result<LogIndex, RaftError> {
        if self.state != RaftState::Leader {
            return Err(RaftError::NotLeader(self.leader_id));
        }

        let index = self.last_log_index() + 1;
        let entry = LogEntry {
            term: self.current_term,
            index,
            command: command.clone(),
        };

        self.log.push(entry);
        self.pending_commands.push((index, command));

        // Send to all followers
        self.send_append_entries_to_all();

        Ok(index)
    }

    /// Tick the node (call periodically).
    pub fn tick(&mut self) {
        let elapsed = self.last_heartbeat.elapsed();

        match self.state {
            RaftState::Follower | RaftState::Candidate => {
                if elapsed >= self.election_timeout {
                    self.start_election();
                }
            }
            RaftState::Leader => {
                if elapsed >= self.config.heartbeat_interval {
                    self.send_heartbeats();
                    self.last_heartbeat = Instant::now();
                }
            }
        }
    }

    /// Start an election.
    fn start_election(&mut self) {
        self.state = RaftState::Candidate;
        self.current_term += 1;
        self.voted_for = Some(self.id);
        self.votes_received.clear();
        self.votes_received.insert(self.id);
        self.election_timeout = Self::random_election_timeout(&self.config);
        self.last_heartbeat = Instant::now();

        // Request votes from all peers
        let request = RequestVote {
            term: self.current_term,
            candidate_id: self.id,
            last_log_index: self.last_log_index(),
            last_log_term: self.last_log_term(),
        };

        for &peer in &self.cluster {
            if peer != self.id {
                self.outbox.push_back(OutgoingMessage {
                    to: peer,
                    message: RaftMessage::RequestVote(request.clone()),
                });
            }
        }
    }

    /// Send heartbeats to all followers.
    fn send_heartbeats(&mut self) {
        self.send_append_entries_to_all();
    }

    /// Send append entries to all followers.
    fn send_append_entries_to_all(&mut self) {
        let peers: Vec<NodeId> = self
            .cluster
            .iter()
            .filter(|&&peer| peer != self.id)
            .copied()
            .collect();
        for peer in peers {
            self.send_append_entries_to(peer);
        }
    }

    /// Send append entries to a specific follower.
    fn send_append_entries_to(&mut self, peer: NodeId) {
        let next_idx = *self.next_index.get(&peer).unwrap_or(&1);
        let prev_log_index = next_idx.saturating_sub(1);
        let prev_log_term = self.log_term_at(prev_log_index);

        let entries: Vec<LogEntry> = self
            .log
            .iter()
            .filter(|e| e.index >= next_idx)
            .take(self.config.max_entries_per_append)
            .cloned()
            .collect();

        let append = AppendEntries {
            term: self.current_term,
            leader_id: self.id,
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit: self.commit_index,
        };

        self.outbox.push_back(OutgoingMessage {
            to: peer,
            message: RaftMessage::AppendEntries(append),
        });
    }

    /// Handle incoming message.
    pub fn handle_message(&mut self, from: NodeId, message: RaftMessage) {
        match message {
            RaftMessage::RequestVote(req) => self.handle_request_vote(from, req),
            RaftMessage::RequestVoteResponse(resp) => self.handle_vote_response(from, resp),
            RaftMessage::AppendEntries(append) => self.handle_append_entries(from, append),
            RaftMessage::AppendEntriesResponse(resp) => self.handle_append_response(from, resp),
            RaftMessage::InstallSnapshot {
                term,
                leader_id,
                metadata,
                data,
            } => {
                self.handle_install_snapshot(term, leader_id, metadata, data);
            }
            RaftMessage::InstallSnapshotResponse { term, success } => {
                self.handle_snapshot_response(from, term, success);
            }
        }
    }

    /// Handle vote request.
    fn handle_request_vote(&mut self, from: NodeId, req: RequestVote) {
        // Update term if needed
        if req.term > self.current_term {
            self.become_follower(req.term);
        }

        let vote_granted = req.term >= self.current_term
            && (self.voted_for.is_none() || self.voted_for == Some(req.candidate_id))
            && self.log_is_up_to_date(req.last_log_index, req.last_log_term);

        if vote_granted {
            self.voted_for = Some(req.candidate_id);
            self.last_heartbeat = Instant::now();
        }

        let response = RequestVoteResponse {
            term: self.current_term,
            vote_granted,
        };

        self.outbox.push_back(OutgoingMessage {
            to: from,
            message: RaftMessage::RequestVoteResponse(response),
        });
    }

    /// Handle vote response.
    fn handle_vote_response(&mut self, _from: NodeId, resp: RequestVoteResponse) {
        if resp.term > self.current_term {
            self.become_follower(resp.term);
            return;
        }

        if self.state != RaftState::Candidate || resp.term != self.current_term {
            return;
        }

        if resp.vote_granted {
            self.votes_received.insert(_from);

            // Check if we have majority
            if self.votes_received.len() > self.cluster.len() / 2 {
                self.become_leader();
            }
        }
    }

    /// Handle append entries.
    fn handle_append_entries(&mut self, from: NodeId, append: AppendEntries) {
        if append.term > self.current_term {
            self.become_follower(append.term);
        }

        if append.term < self.current_term {
            let response = AppendEntriesResponse {
                term: self.current_term,
                success: false,
                match_index: 0,
            };
            self.outbox.push_back(OutgoingMessage {
                to: from,
                message: RaftMessage::AppendEntriesResponse(response),
            });
            return;
        }

        // Valid leader
        self.leader_id = Some(append.leader_id);
        self.last_heartbeat = Instant::now();

        // Check log consistency
        let log_ok = append.prev_log_index == 0
            || (append.prev_log_index <= self.last_log_index()
                && self.log_term_at(append.prev_log_index) == append.prev_log_term);

        if !log_ok {
            let response = AppendEntriesResponse {
                term: self.current_term,
                success: false,
                match_index: self.last_log_index(),
            };
            self.outbox.push_back(OutgoingMessage {
                to: from,
                message: RaftMessage::AppendEntriesResponse(response),
            });
            return;
        }

        // Append entries
        for entry in append.entries {
            if entry.index <= self.last_log_index() {
                // Check for conflict
                if self.log_term_at(entry.index) != entry.term {
                    // Delete conflicting entries
                    self.log.truncate((entry.index - 1) as usize);
                    self.log.push(entry);
                }
            } else {
                self.log.push(entry);
            }
        }

        // Update commit index
        if append.leader_commit > self.commit_index {
            self.commit_index = append.leader_commit.min(self.last_log_index());
        }

        let response = AppendEntriesResponse {
            term: self.current_term,
            success: true,
            match_index: self.last_log_index(),
        };

        self.outbox.push_back(OutgoingMessage {
            to: from,
            message: RaftMessage::AppendEntriesResponse(response),
        });
    }

    /// Handle append entries response.
    fn handle_append_response(&mut self, from: NodeId, resp: AppendEntriesResponse) {
        if resp.term > self.current_term {
            self.become_follower(resp.term);
            return;
        }

        if self.state != RaftState::Leader || resp.term != self.current_term {
            return;
        }

        if resp.success {
            self.match_index.insert(from, resp.match_index);
            self.next_index.insert(from, resp.match_index + 1);

            // Update commit index
            self.update_commit_index();
        } else {
            // Decrement next_index and retry
            let next = self.next_index.entry(from).or_insert(1);
            *next = next.saturating_sub(1).max(1);
            self.send_append_entries_to(from);
        }
    }

    /// Handle install snapshot.
    fn handle_install_snapshot(
        &mut self,
        term: Term,
        leader_id: NodeId,
        metadata: SnapshotMetadata,
        _data: Vec<u8>,
    ) {
        if term > self.current_term {
            self.become_follower(term);
        }

        if term < self.current_term {
            return;
        }

        self.leader_id = Some(leader_id);
        self.last_heartbeat = Instant::now();

        // Apply snapshot (simplified)
        self.log.retain(|e| e.index > metadata.last_included_index);
        self.commit_index = metadata.last_included_index;
        self.last_applied = metadata.last_included_index;

        // Update cluster config
        self.cluster.clear();
        for node in metadata.config {
            self.cluster.insert(node);
        }

        let response = RaftMessage::InstallSnapshotResponse {
            term: self.current_term,
            success: true,
        };

        self.outbox.push_back(OutgoingMessage {
            to: leader_id,
            message: response,
        });
    }

    /// Handle snapshot response.
    fn handle_snapshot_response(&mut self, from: NodeId, term: Term, success: bool) {
        if term > self.current_term {
            self.become_follower(term);
            return;
        }

        if success && self.state == RaftState::Leader {
            // Update indices for follower
            self.match_index.insert(from, self.commit_index);
            self.next_index.insert(from, self.commit_index + 1);
        }
    }

    /// Become a follower.
    fn become_follower(&mut self, term: Term) {
        self.state = RaftState::Follower;
        self.current_term = term;
        self.voted_for = None;
        self.leader_id = None;
        self.election_timeout = Self::random_election_timeout(&self.config);
    }

    /// Become the leader.
    fn become_leader(&mut self) {
        self.state = RaftState::Leader;
        self.leader_id = Some(self.id);

        // Initialize leader state
        let last_index = self.last_log_index();
        for &peer in &self.cluster {
            if peer != self.id {
                self.next_index.insert(peer, last_index + 1);
                self.match_index.insert(peer, 0);
            }
        }

        // Append no-op entry
        let _ = self.propose(Command::Noop);

        // Send initial heartbeats
        self.send_heartbeats();
        self.last_heartbeat = Instant::now();
    }

    /// Update commit index based on replication.
    fn update_commit_index(&mut self) {
        let mut indices: Vec<LogIndex> = self.match_index.values().copied().collect();
        indices.push(self.last_log_index()); // Include leader
        indices.sort_unstable();

        let majority_idx = indices.len() / 2;
        let new_commit = indices[majority_idx];

        if new_commit > self.commit_index && self.log_term_at(new_commit) == self.current_term {
            self.commit_index = new_commit;
        }
    }

    /// Get last log index.
    fn last_log_index(&self) -> LogIndex {
        self.log.last().map(|e| e.index).unwrap_or(0)
    }

    /// Get last log term.
    fn last_log_term(&self) -> Term {
        self.log.last().map(|e| e.term).unwrap_or(0)
    }

    /// Get term at specific index.
    fn log_term_at(&self, index: LogIndex) -> Term {
        if index == 0 {
            return 0;
        }
        self.log
            .iter()
            .find(|e| e.index == index)
            .map(|e| e.term)
            .unwrap_or(0)
    }

    /// Check if candidate's log is at least as up-to-date.
    fn log_is_up_to_date(&self, last_index: LogIndex, last_term: Term) -> bool {
        let my_term = self.last_log_term();
        let my_index = self.last_log_index();

        last_term > my_term || (last_term == my_term && last_index >= my_index)
    }

    /// Get pending outgoing messages.
    pub fn take_messages(&mut self) -> Vec<OutgoingMessage> {
        self.outbox.drain(..).collect()
    }

    /// Get committed entries to apply.
    pub fn take_committed(&mut self) -> Vec<Command> {
        let mut commands = Vec::new();
        while self.last_applied < self.commit_index {
            self.last_applied += 1;
            if let Some(entry) = self.log.iter().find(|e| e.index == self.last_applied) {
                commands.push(entry.command.clone());
            }
        }
        commands
    }

    /// Get node status.
    pub fn status(&self) -> NodeStatus {
        NodeStatus {
            id: self.id,
            state: self.state,
            term: self.current_term,
            leader_id: self.leader_id,
            commit_index: self.commit_index,
            last_applied: self.last_applied,
            log_length: self.log.len(),
            cluster_size: self.cluster.len(),
        }
    }
}

/// Raft error types.
#[derive(Debug, Clone)]
pub enum RaftError {
    /// Not the leader.
    NotLeader(Option<NodeId>),
    /// Invalid operation.
    InvalidOperation(String),
}

/// Node status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    /// Node ID.
    pub id: NodeId,
    /// Current state.
    pub state: RaftState,
    /// Current term.
    pub term: Term,
    /// Known leader.
    pub leader_id: Option<NodeId>,
    /// Commit index.
    pub commit_index: LogIndex,
    /// Last applied.
    pub last_applied: LogIndex,
    /// Log length.
    pub log_length: usize,
    /// Cluster size.
    pub cluster_size: usize,
}

// ============================================================================
// Persistent Storage
// ============================================================================

/// Persistent state that must survive restarts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PersistentState {
    /// Current term.
    pub current_term: Term,
    /// Candidate voted for in current term.
    pub voted_for: Option<NodeId>,
    /// Cluster configuration.
    pub cluster: Vec<NodeId>,
}

/// Trait for persistent storage backends.
pub trait RaftStorage: Send + Sync {
    /// Load persistent state.
    fn load_state(&self) -> Result<Option<PersistentState>>;

    /// Save persistent state.
    fn save_state(&self, state: &PersistentState) -> Result<()>;

    /// Append log entries.
    fn append_entries(&self, entries: &[LogEntry]) -> Result<()>;

    /// Load log entries from a given index.
    fn load_entries(&self, from_index: LogIndex) -> Result<Vec<LogEntry>>;

    /// Truncate log from a given index.
    fn truncate_log(&self, from_index: LogIndex) -> Result<()>;

    /// Get the last log index.
    fn last_log_index(&self) -> Result<LogIndex>;

    /// Save a snapshot.
    fn save_snapshot(&self, snapshot: &Snapshot) -> Result<()>;

    /// Load the latest snapshot.
    fn load_snapshot(&self) -> Result<Option<Snapshot>>;

    /// Compact log up to the snapshot index.
    fn compact_log(&self, up_to_index: LogIndex) -> Result<()>;

    /// Sync all pending writes to disk.
    fn sync(&self) -> Result<()>;
}

/// A snapshot of the state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Metadata about the snapshot.
    pub metadata: SnapshotMetadata,
    /// Serialized state machine data.
    pub data: Vec<u8>,
}

/// In-memory storage for testing.
pub struct MemoryStorage {
    state: std::sync::RwLock<PersistentState>,
    log: std::sync::RwLock<Vec<LogEntry>>,
    snapshot: std::sync::RwLock<Option<Snapshot>>,
}

impl MemoryStorage {
    /// Create a new in-memory storage.
    pub fn new() -> Self {
        Self {
            state: std::sync::RwLock::new(PersistentState::default()),
            log: std::sync::RwLock::new(Vec::new()),
            snapshot: std::sync::RwLock::new(None),
        }
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl RaftStorage for MemoryStorage {
    fn load_state(&self) -> Result<Option<PersistentState>> {
        let state = self.state.read().map_err(|_| NeedleError::LockError)?;
        Ok(Some(state.clone()))
    }

    fn save_state(&self, state: &PersistentState) -> Result<()> {
        let mut current = self.state.write().map_err(|_| NeedleError::LockError)?;
        *current = state.clone();
        Ok(())
    }

    fn append_entries(&self, entries: &[LogEntry]) -> Result<()> {
        let mut log = self.log.write().map_err(|_| NeedleError::LockError)?;
        log.extend(entries.iter().cloned());
        Ok(())
    }

    fn load_entries(&self, from_index: LogIndex) -> Result<Vec<LogEntry>> {
        let log = self.log.read().map_err(|_| NeedleError::LockError)?;
        Ok(log
            .iter()
            .filter(|e| e.index >= from_index)
            .cloned()
            .collect())
    }

    fn truncate_log(&self, from_index: LogIndex) -> Result<()> {
        let mut log = self.log.write().map_err(|_| NeedleError::LockError)?;
        log.retain(|e| e.index < from_index);
        Ok(())
    }

    fn last_log_index(&self) -> Result<LogIndex> {
        let log = self.log.read().map_err(|_| NeedleError::LockError)?;
        Ok(log.last().map(|e| e.index).unwrap_or(0))
    }

    fn save_snapshot(&self, snapshot: &Snapshot) -> Result<()> {
        let mut current = self.snapshot.write().map_err(|_| NeedleError::LockError)?;
        *current = Some(snapshot.clone());
        Ok(())
    }

    fn load_snapshot(&self) -> Result<Option<Snapshot>> {
        let snapshot = self.snapshot.read().map_err(|_| NeedleError::LockError)?;
        Ok(snapshot.clone())
    }

    fn compact_log(&self, up_to_index: LogIndex) -> Result<()> {
        let mut log = self.log.write().map_err(|_| NeedleError::LockError)?;
        log.retain(|e| e.index > up_to_index);
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        Ok(()) // No-op for memory storage
    }
}

/// File-based persistent storage.
#[allow(dead_code)]
pub struct FileStorage {
    /// Base directory for storage (kept for path derivations).
    dir: PathBuf,
    /// State file path.
    state_path: PathBuf,
    /// Log directory path.
    log_dir: PathBuf,
    /// Snapshot directory path.
    snapshot_dir: PathBuf,
    /// Current log segment file.
    current_segment: std::sync::Mutex<Option<BufWriter<File>>>,
    /// Entries in current segment.
    segment_entries: std::sync::RwLock<Vec<LogEntry>>,
    /// Maximum entries per segment.
    max_segment_entries: usize,
    /// Current segment number.
    current_segment_num: std::sync::atomic::AtomicU64,
}

impl FileStorage {
    /// Open or create file storage at the given path.
    pub fn open<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let state_path = dir.join("state.json");
        let log_dir = dir.join("log");
        let snapshot_dir = dir.join("snapshots");

        fs::create_dir_all(&log_dir).map_err(|e| NeedleError::Io(e))?;
        fs::create_dir_all(&snapshot_dir).map_err(|e| NeedleError::Io(e))?;

        let storage = Self {
            dir,
            state_path,
            log_dir,
            snapshot_dir,
            current_segment: std::sync::Mutex::new(None),
            segment_entries: std::sync::RwLock::new(Vec::new()),
            max_segment_entries: 10000,
            current_segment_num: std::sync::atomic::AtomicU64::new(0),
        };

        // Find the latest segment
        storage.find_latest_segment()?;

        Ok(storage)
    }

    fn find_latest_segment(&self) -> Result<()> {
        let mut max_segment = 0u64;

        if let Ok(entries) = fs::read_dir(&self.log_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Some(num_str) = name.strip_prefix("segment_") {
                        if let Ok(num) = num_str.parse::<u64>() {
                            max_segment = max_segment.max(num);
                        }
                    }
                }
            }
        }

        self.current_segment_num
            .store(max_segment, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    fn segment_path(&self, num: u64) -> PathBuf {
        self.log_dir.join(format!("segment_{:08}.log", num))
    }

    fn current_segment_path(&self) -> PathBuf {
        let num = self
            .current_segment_num
            .load(std::sync::atomic::Ordering::SeqCst);
        self.segment_path(num)
    }

    fn rotate_segment(&self) -> Result<()> {
        // Close current segment
        {
            let mut segment = self
                .current_segment
                .lock()
                .map_err(|_| NeedleError::LockError)?;
            if let Some(ref mut writer) = *segment {
                writer.flush().map_err(|e| NeedleError::Io(e))?;
            }
            *segment = None;
        }

        // Clear segment entries
        {
            let mut entries = self
                .segment_entries
                .write()
                .map_err(|_| NeedleError::LockError)?;
            entries.clear();
        }

        // Increment segment number
        self.current_segment_num
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Ok(())
    }

    fn ensure_segment_open(&self) -> Result<()> {
        let mut segment = self
            .current_segment
            .lock()
            .map_err(|_| NeedleError::LockError)?;

        if segment.is_none() {
            let path = self.current_segment_path();
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| NeedleError::Io(e))?;
            *segment = Some(BufWriter::new(file));
        }

        Ok(())
    }

    fn read_segment(&self, path: &Path) -> Result<Vec<LogEntry>> {
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(path).map_err(|e| NeedleError::Io(e))?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();

        loop {
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(NeedleError::Io(e)),
            }

            let len = u32::from_le_bytes(len_buf) as usize;
            let mut data = vec![0u8; len];
            reader
                .read_exact(&mut data)
                .map_err(|e| NeedleError::Io(e))?;

            let entry: LogEntry =
                serde_json::from_slice(&data).map_err(|e| NeedleError::Serialization(e))?;
            entries.push(entry);
        }

        Ok(entries)
    }

    fn all_segment_paths(&self) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();

        if let Ok(entries) = fs::read_dir(&self.log_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().map(|e| e == "log").unwrap_or(false) {
                    paths.push(path);
                }
            }
        }

        paths.sort();
        Ok(paths)
    }
}

impl RaftStorage for FileStorage {
    fn load_state(&self) -> Result<Option<PersistentState>> {
        if !self.state_path.exists() {
            return Ok(None);
        }

        let file = File::open(&self.state_path).map_err(|e| NeedleError::Io(e))?;
        let reader = BufReader::new(file);
        let state: PersistentState =
            serde_json::from_reader(reader).map_err(|e| NeedleError::Serialization(e))?;
        Ok(Some(state))
    }

    fn save_state(&self, state: &PersistentState) -> Result<()> {
        let temp_path = self.state_path.with_extension("tmp");
        let file = File::create(&temp_path).map_err(|e| NeedleError::Io(e))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, state).map_err(|e| NeedleError::Serialization(e))?;

        // Atomic rename
        fs::rename(&temp_path, &self.state_path).map_err(|e| NeedleError::Io(e))?;
        Ok(())
    }

    fn append_entries(&self, entries: &[LogEntry]) -> Result<()> {
        self.ensure_segment_open()?;

        let mut segment = self
            .current_segment
            .lock()
            .map_err(|_| NeedleError::LockError)?;
        let mut segment_entries = self
            .segment_entries
            .write()
            .map_err(|_| NeedleError::LockError)?;

        if let Some(ref mut writer) = *segment {
            for entry in entries {
                let data = serde_json::to_vec(entry).map_err(|e| NeedleError::Serialization(e))?;
                let len = (data.len() as u32).to_le_bytes();
                writer.write_all(&len).map_err(|e| NeedleError::Io(e))?;
                writer.write_all(&data).map_err(|e| NeedleError::Io(e))?;
                segment_entries.push(entry.clone());
            }
            writer.flush().map_err(|e| NeedleError::Io(e))?;
        }

        // Check if we need to rotate
        if segment_entries.len() >= self.max_segment_entries {
            drop(segment);
            drop(segment_entries);
            self.rotate_segment()?;
        }

        Ok(())
    }

    fn load_entries(&self, from_index: LogIndex) -> Result<Vec<LogEntry>> {
        let mut all_entries = Vec::new();
        let paths = self.all_segment_paths()?;

        for path in paths {
            let entries = self.read_segment(&path)?;
            all_entries.extend(entries);
        }

        Ok(all_entries
            .into_iter()
            .filter(|e| e.index >= from_index)
            .collect())
    }

    fn truncate_log(&self, from_index: LogIndex) -> Result<()> {
        // Read all entries
        let entries = self.load_entries(1)?;

        // Filter entries to keep
        let keep: Vec<LogEntry> = entries
            .into_iter()
            .filter(|e| e.index < from_index)
            .collect();

        // Delete all segments
        for path in self.all_segment_paths()? {
            fs::remove_file(&path).map_err(|e| NeedleError::Io(e))?;
        }

        // Reset segment
        self.current_segment_num
            .store(0, std::sync::atomic::Ordering::SeqCst);
        {
            let mut segment = self
                .current_segment
                .lock()
                .map_err(|_| NeedleError::LockError)?;
            *segment = None;
        }
        {
            let mut segment_entries = self
                .segment_entries
                .write()
                .map_err(|_| NeedleError::LockError)?;
            segment_entries.clear();
        }

        // Re-append kept entries
        if !keep.is_empty() {
            self.append_entries(&keep)?;
        }

        Ok(())
    }

    fn last_log_index(&self) -> Result<LogIndex> {
        // First check in-memory entries
        {
            let segment_entries = self
                .segment_entries
                .read()
                .map_err(|_| NeedleError::LockError)?;
            if let Some(last) = segment_entries.last() {
                return Ok(last.index);
            }
        }

        // Otherwise read from files
        let paths = self.all_segment_paths()?;
        if let Some(last_path) = paths.last() {
            let entries = self.read_segment(last_path)?;
            if let Some(last) = entries.last() {
                return Ok(last.index);
            }
        }

        Ok(0)
    }

    fn save_snapshot(&self, snapshot: &Snapshot) -> Result<()> {
        let snapshot_path = self.snapshot_dir.join(format!(
            "snapshot_{:08}.snap",
            snapshot.metadata.last_included_index
        ));
        let temp_path = snapshot_path.with_extension("tmp");

        let file = File::create(&temp_path).map_err(|e| NeedleError::Io(e))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, snapshot).map_err(|e| NeedleError::Serialization(e))?;

        fs::rename(&temp_path, &snapshot_path).map_err(|e| NeedleError::Io(e))?;
        Ok(())
    }

    fn load_snapshot(&self) -> Result<Option<Snapshot>> {
        let mut snapshots: Vec<PathBuf> = Vec::new();

        if let Ok(entries) = fs::read_dir(&self.snapshot_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().map(|e| e == "snap").unwrap_or(false) {
                    snapshots.push(path);
                }
            }
        }

        snapshots.sort();

        if let Some(latest) = snapshots.last() {
            let file = File::open(latest).map_err(|e| NeedleError::Io(e))?;
            let reader = BufReader::new(file);
            let snapshot: Snapshot =
                serde_json::from_reader(reader).map_err(|e| NeedleError::Serialization(e))?;
            return Ok(Some(snapshot));
        }

        Ok(None)
    }

    fn compact_log(&self, up_to_index: LogIndex) -> Result<()> {
        // Read all entries
        let entries = self.load_entries(1)?;

        // Keep only entries after up_to_index
        let keep: Vec<LogEntry> = entries
            .into_iter()
            .filter(|e| e.index > up_to_index)
            .collect();

        // Delete old segments
        for path in self.all_segment_paths()? {
            fs::remove_file(&path).map_err(|e| NeedleError::Io(e))?;
        }

        // Reset
        self.current_segment_num
            .store(0, std::sync::atomic::Ordering::SeqCst);
        {
            let mut segment = self
                .current_segment
                .lock()
                .map_err(|_| NeedleError::LockError)?;
            *segment = None;
        }
        {
            let mut segment_entries = self
                .segment_entries
                .write()
                .map_err(|_| NeedleError::LockError)?;
            segment_entries.clear();
        }

        // Re-append kept entries
        if !keep.is_empty() {
            self.append_entries(&keep)?;
        }

        // Clean up old snapshots (keep only the latest)
        let mut snapshots: Vec<PathBuf> = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.snapshot_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().map(|e| e == "snap").unwrap_or(false) {
                    snapshots.push(path);
                }
            }
        }
        snapshots.sort();

        // Remove all but the latest snapshot
        if snapshots.len() > 1 {
            for path in &snapshots[..snapshots.len() - 1] {
                let _ = fs::remove_file(path);
            }
        }

        Ok(())
    }

    fn sync(&self) -> Result<()> {
        let segment = self
            .current_segment
            .lock()
            .map_err(|_| NeedleError::LockError)?;
        if let Some(ref writer) = *segment {
            writer
                .get_ref()
                .sync_all()
                .map_err(|e| NeedleError::Io(e))?;
        }
        Ok(())
    }
}

// ============================================================================
// Snapshot Builder
// ============================================================================

/// Builder for creating snapshots.
pub struct SnapshotBuilder {
    last_included_index: LogIndex,
    last_included_term: Term,
    config: Vec<NodeId>,
    data: Vec<u8>,
}

impl SnapshotBuilder {
    /// Create a new snapshot builder.
    pub fn new(last_included_index: LogIndex, last_included_term: Term) -> Self {
        Self {
            last_included_index,
            last_included_term,
            config: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Set the cluster configuration.
    pub fn config(mut self, config: Vec<NodeId>) -> Self {
        self.config = config;
        self
    }

    /// Set the snapshot data.
    pub fn data(mut self, data: Vec<u8>) -> Self {
        self.data = data;
        self
    }

    /// Build the snapshot.
    pub fn build(self) -> Snapshot {
        Snapshot {
            metadata: SnapshotMetadata {
                last_included_index: self.last_included_index,
                last_included_term: self.last_included_term,
                config: self.config,
            },
            data: self.data,
        }
    }
}

// ============================================================================
// Extended RaftNode with Persistence
// ============================================================================

impl RaftNode {
    /// Create a new Raft node with persistent storage.
    pub fn with_storage(
        id: NodeId,
        config: RaftConfig,
        storage: Box<dyn RaftStorage>,
    ) -> Result<Self> {
        let election_timeout = Self::random_election_timeout(&config);

        let mut node = Self {
            id,
            config,
            state: RaftState::Follower,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            leader_id: None,
            cluster: HashSet::new(),
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            votes_received: HashSet::new(),
            last_heartbeat: Instant::now(),
            election_timeout,
            outbox: VecDeque::new(),
            pending_commands: Vec::new(),
        };

        // Restore from storage
        if let Some(state) = storage.load_state()? {
            node.current_term = state.current_term;
            node.voted_for = state.voted_for;
            for peer in state.cluster {
                node.cluster.insert(peer);
            }
        }

        // Load snapshot if available
        if let Some(snapshot) = storage.load_snapshot()? {
            node.commit_index = snapshot.metadata.last_included_index;
            node.last_applied = snapshot.metadata.last_included_index;
            for peer in snapshot.metadata.config {
                node.cluster.insert(peer);
            }
        }

        // Load log entries
        let entries = storage.load_entries(node.last_applied + 1)?;
        node.log = entries;

        Ok(node)
    }

    /// Create a snapshot of the current state.
    pub fn create_snapshot<F>(&mut self, serialize_state: F) -> Result<Snapshot>
    where
        F: FnOnce() -> Vec<u8>,
    {
        let snapshot = SnapshotBuilder::new(self.last_applied, self.log_term_at(self.last_applied))
            .config(self.cluster.iter().copied().collect())
            .data(serialize_state())
            .build();

        Ok(snapshot)
    }

    /// Check if a snapshot should be taken.
    pub fn should_snapshot(&self) -> bool {
        self.log.len() >= self.config.snapshot_threshold
    }

    /// Get persistent state for saving.
    pub fn persistent_state(&self) -> PersistentState {
        PersistentState {
            current_term: self.current_term,
            voted_for: self.voted_for,
            cluster: self.cluster.iter().copied().collect(),
        }
    }

    /// Get log entries for persistence.
    pub fn log_entries(&self) -> &[LogEntry] {
        &self.log
    }

    /// Apply a snapshot.
    pub fn apply_snapshot<F>(&mut self, snapshot: &Snapshot, restore_state: F) -> Result<()>
    where
        F: FnOnce(&[u8]) -> Result<()>,
    {
        // Apply snapshot data
        restore_state(&snapshot.data)?;

        // Update indices
        self.log
            .retain(|e| e.index > snapshot.metadata.last_included_index);
        self.commit_index = snapshot.metadata.last_included_index;
        self.last_applied = snapshot.metadata.last_included_index;

        // Update cluster config
        self.cluster.clear();
        for node in &snapshot.metadata.config {
            self.cluster.insert(*node);
        }

        Ok(())
    }

    /// Send snapshot to a lagging follower.
    pub fn send_snapshot(&mut self, to: NodeId, snapshot: &Snapshot) {
        self.outbox.push_back(OutgoingMessage {
            to,
            message: RaftMessage::InstallSnapshot {
                term: self.current_term,
                leader_id: self.id,
                metadata: snapshot.metadata.clone(),
                data: snapshot.data.clone(),
            },
        });
    }

    /// Check if a follower needs a snapshot (is too far behind).
    pub fn follower_needs_snapshot(&self, peer: NodeId, snapshot_index: LogIndex) -> bool {
        if let Some(&next_idx) = self.next_index.get(&peer) {
            // If the follower needs entries we don't have (compacted), send snapshot
            let first_log_idx = self.log.first().map(|e| e.index).unwrap_or(1);
            next_idx < first_log_idx && snapshot_index >= next_idx
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_cluster(size: usize) -> Vec<RaftNode> {
        let config = RaftConfig {
            election_timeout_min: Duration::from_millis(50),
            election_timeout_max: Duration::from_millis(100),
            heartbeat_interval: Duration::from_millis(25),
            ..Default::default()
        };

        let nodes: Vec<NodeId> = (0..size).map(|i| NodeId(i as u64)).collect();

        nodes
            .iter()
            .map(|&id| {
                let mut node = RaftNode::new(id, config.clone());
                node.initialize(nodes.clone());
                node
            })
            .collect()
    }

    #[test]
    fn test_create_node() {
        let node = RaftNode::new(NodeId(1), RaftConfig::default());
        assert_eq!(node.id(), NodeId(1));
        assert_eq!(node.state(), RaftState::Follower);
    }

    #[test]
    fn test_initial_state() {
        let cluster = create_cluster(3);
        for node in &cluster {
            assert_eq!(node.state(), RaftState::Follower);
            assert_eq!(node.term(), 0);
        }
    }

    #[test]
    fn test_election_timeout() {
        let mut node = RaftNode::new(
            NodeId(1),
            RaftConfig {
                election_timeout_min: Duration::from_millis(1),
                election_timeout_max: Duration::from_millis(2),
                ..Default::default()
            },
        );
        node.initialize(vec![NodeId(2), NodeId(3)]);

        std::thread::sleep(Duration::from_millis(10));
        node.tick();

        assert_eq!(node.state(), RaftState::Candidate);
        assert_eq!(node.term(), 1);
    }

    #[test]
    fn test_request_vote() {
        let mut node = RaftNode::new(NodeId(1), RaftConfig::default());
        node.initialize(vec![NodeId(2)]);

        let request = RequestVote {
            term: 1,
            candidate_id: NodeId(2),
            last_log_index: 0,
            last_log_term: 0,
        };

        node.handle_message(NodeId(2), RaftMessage::RequestVote(request));

        let messages = node.take_messages();
        assert_eq!(messages.len(), 1);

        if let RaftMessage::RequestVoteResponse(resp) = &messages[0].message {
            assert!(resp.vote_granted);
        } else {
            panic!("Expected RequestVoteResponse");
        }
    }

    #[test]
    fn test_become_leader() {
        let config = RaftConfig::default();
        let mut leader = RaftNode::new(NodeId(0), config);
        leader.initialize(vec![NodeId(1), NodeId(2)]);

        // Manually become candidate and get votes
        leader.current_term = 1;
        leader.state = RaftState::Candidate;
        leader.votes_received.insert(NodeId(0));
        leader.votes_received.insert(NodeId(1));
        leader.votes_received.insert(NodeId(2));

        // Have majority, become leader
        leader.become_leader();

        assert!(leader.is_leader());
        assert_eq!(leader.leader(), Some(NodeId(0)));
    }

    #[test]
    fn test_propose_command() {
        let mut leader = RaftNode::new(NodeId(0), RaftConfig::default());
        leader.initialize(vec![NodeId(1)]);
        leader.state = RaftState::Leader;
        leader.leader_id = Some(NodeId(0));

        let command = Command::Insert {
            id: "vec1".to_string(),
            vector: vec![1.0, 2.0],
            metadata: HashMap::new(),
        };

        let index = leader.propose(command).unwrap();
        assert_eq!(index, 1);
        assert_eq!(leader.log.len(), 1);
    }

    #[test]
    fn test_propose_not_leader() {
        let mut node = RaftNode::new(NodeId(1), RaftConfig::default());

        let result = node.propose(Command::Noop);
        assert!(matches!(result, Err(RaftError::NotLeader(_))));
    }

    #[test]
    fn test_append_entries() {
        let mut follower = RaftNode::new(NodeId(1), RaftConfig::default());
        follower.initialize(vec![NodeId(0)]);

        let append = AppendEntries {
            term: 1,
            leader_id: NodeId(0),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![LogEntry {
                term: 1,
                index: 1,
                command: Command::Noop,
            }],
            leader_commit: 1,
        };

        follower.handle_message(NodeId(0), RaftMessage::AppendEntries(append));

        assert_eq!(follower.log.len(), 1);
        assert_eq!(follower.commit_index, 1);
    }

    #[test]
    fn test_higher_term_conversion() {
        let mut node = RaftNode::new(NodeId(1), RaftConfig::default());
        node.current_term = 1;
        node.state = RaftState::Leader;

        let append = AppendEntries {
            term: 5,
            leader_id: NodeId(0),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        };

        node.handle_message(NodeId(0), RaftMessage::AppendEntries(append));

        assert_eq!(node.state(), RaftState::Follower);
        assert_eq!(node.term(), 5);
    }

    #[test]
    fn test_status() {
        let node = RaftNode::new(NodeId(1), RaftConfig::default());
        let status = node.status();

        assert_eq!(status.id, NodeId(1));
        assert_eq!(status.state, RaftState::Follower);
        assert_eq!(status.term, 0);
    }

    #[test]
    fn test_take_committed() {
        let mut node = RaftNode::new(NodeId(0), RaftConfig::default());
        node.state = RaftState::Leader;
        node.initialize(vec![]);

        node.log.push(LogEntry {
            term: 1,
            index: 1,
            command: Command::Noop,
        });
        node.commit_index = 1;

        let commands = node.take_committed();
        assert_eq!(commands.len(), 1);
        assert_eq!(node.last_applied, 1);
    }

    // ================== Storage Tests ==================

    #[test]
    fn test_memory_storage_state() {
        let storage = MemoryStorage::new();

        // Initial state should be empty/default
        let state = storage.load_state().unwrap();
        assert!(state.is_some());

        // Save new state
        let new_state = PersistentState {
            current_term: 5,
            voted_for: Some(NodeId(1)),
            cluster: vec![NodeId(1), NodeId(2), NodeId(3)],
        };
        storage.save_state(&new_state).unwrap();

        // Load and verify
        let loaded = storage.load_state().unwrap().unwrap();
        assert_eq!(loaded.current_term, 5);
        assert_eq!(loaded.voted_for, Some(NodeId(1)));
        assert_eq!(loaded.cluster.len(), 3);
    }

    #[test]
    fn test_memory_storage_log_entries() {
        let storage = MemoryStorage::new();

        // Append entries
        let entries = vec![
            LogEntry {
                term: 1,
                index: 1,
                command: Command::Noop,
            },
            LogEntry {
                term: 1,
                index: 2,
                command: Command::Noop,
            },
            LogEntry {
                term: 2,
                index: 3,
                command: Command::Noop,
            },
        ];
        storage.append_entries(&entries).unwrap();

        // Load entries
        let loaded = storage.load_entries(1).unwrap();
        assert_eq!(loaded.len(), 3);

        // Load from middle
        let partial = storage.load_entries(2).unwrap();
        assert_eq!(partial.len(), 2);

        // Last log index
        assert_eq!(storage.last_log_index().unwrap(), 3);
    }

    #[test]
    fn test_memory_storage_truncate() {
        let storage = MemoryStorage::new();

        let entries = vec![
            LogEntry {
                term: 1,
                index: 1,
                command: Command::Noop,
            },
            LogEntry {
                term: 1,
                index: 2,
                command: Command::Noop,
            },
            LogEntry {
                term: 2,
                index: 3,
                command: Command::Noop,
            },
        ];
        storage.append_entries(&entries).unwrap();

        // Truncate from index 2
        storage.truncate_log(2).unwrap();

        let remaining = storage.load_entries(1).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].index, 1);
    }

    #[test]
    fn test_memory_storage_snapshot() {
        let storage = MemoryStorage::new();

        // Initially no snapshot
        assert!(storage.load_snapshot().unwrap().is_none());

        // Save snapshot
        let snapshot = Snapshot {
            metadata: SnapshotMetadata {
                last_included_index: 10,
                last_included_term: 2,
                config: vec![NodeId(1), NodeId(2)],
            },
            data: b"snapshot data".to_vec(),
        };
        storage.save_snapshot(&snapshot).unwrap();

        // Load snapshot
        let loaded = storage.load_snapshot().unwrap().unwrap();
        assert_eq!(loaded.metadata.last_included_index, 10);
        assert_eq!(loaded.metadata.last_included_term, 2);
        assert_eq!(loaded.data, b"snapshot data");
    }

    #[test]
    fn test_memory_storage_compact() {
        let storage = MemoryStorage::new();

        let entries = vec![
            LogEntry {
                term: 1,
                index: 1,
                command: Command::Noop,
            },
            LogEntry {
                term: 1,
                index: 2,
                command: Command::Noop,
            },
            LogEntry {
                term: 2,
                index: 3,
                command: Command::Noop,
            },
            LogEntry {
                term: 2,
                index: 4,
                command: Command::Noop,
            },
        ];
        storage.append_entries(&entries).unwrap();

        // Compact up to index 2 (keep entries > 2)
        storage.compact_log(2).unwrap();

        let remaining = storage.load_entries(1).unwrap();
        assert_eq!(remaining.len(), 2);
        assert!(remaining.iter().all(|e| e.index > 2));
    }

    #[test]
    fn test_memory_storage_sync() {
        let storage = MemoryStorage::new();
        // Sync should succeed (no-op for memory)
        assert!(storage.sync().is_ok());
    }

    #[test]
    fn test_file_storage_basic() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage = FileStorage::open(temp_dir.path()).unwrap();

        // Save state
        let state = PersistentState {
            current_term: 3,
            voted_for: Some(NodeId(5)),
            cluster: vec![NodeId(5), NodeId(6)],
        };
        storage.save_state(&state).unwrap();

        // Load state
        let loaded = storage.load_state().unwrap().unwrap();
        assert_eq!(loaded.current_term, 3);
        assert_eq!(loaded.voted_for, Some(NodeId(5)));
    }

    #[test]
    fn test_file_storage_log_entries() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage = FileStorage::open(temp_dir.path()).unwrap();

        // Append entries
        let entries = vec![
            LogEntry {
                term: 1,
                index: 1,
                command: Command::Noop,
            },
            LogEntry {
                term: 1,
                index: 2,
                command: Command::Insert {
                    id: "vec1".to_string(),
                    vector: vec![1.0, 2.0, 3.0],
                    metadata: HashMap::new(),
                },
            },
        ];
        storage.append_entries(&entries).unwrap();

        // Load entries
        let loaded = storage.load_entries(1).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(storage.last_log_index().unwrap(), 2);
    }

    #[test]
    fn test_file_storage_snapshot() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage = FileStorage::open(temp_dir.path()).unwrap();

        let snapshot = Snapshot {
            metadata: SnapshotMetadata {
                last_included_index: 100,
                last_included_term: 5,
                config: vec![NodeId(1)],
            },
            data: vec![1, 2, 3, 4, 5],
        };
        storage.save_snapshot(&snapshot).unwrap();

        let loaded = storage.load_snapshot().unwrap().unwrap();
        assert_eq!(loaded.metadata.last_included_index, 100);
        assert_eq!(loaded.data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_snapshot_builder() {
        let snapshot = SnapshotBuilder::new(50, 3)
            .config(vec![NodeId(1), NodeId(2)])
            .data(b"state machine data".to_vec())
            .build();

        assert_eq!(snapshot.metadata.last_included_index, 50);
        assert_eq!(snapshot.metadata.last_included_term, 3);
        assert_eq!(snapshot.metadata.config.len(), 2);
        assert_eq!(snapshot.data, b"state machine data");
    }

    #[test]
    fn test_persistent_state_default() {
        let state = PersistentState::default();
        assert_eq!(state.current_term, 0);
        assert!(state.voted_for.is_none());
        assert!(state.cluster.is_empty());
    }

    #[test]
    fn test_raft_config_default() {
        let config = RaftConfig::default();
        assert_eq!(config.election_timeout_min, Duration::from_millis(150));
        assert_eq!(config.election_timeout_max, Duration::from_millis(300));
        assert_eq!(config.heartbeat_interval, Duration::from_millis(50));
        assert_eq!(config.max_entries_per_append, 100);
        assert_eq!(config.snapshot_threshold, 10000);
    }

    #[test]
    fn test_node_with_storage() {
        let storage = MemoryStorage::new();

        // Pre-populate storage
        let state = PersistentState {
            current_term: 10,
            voted_for: Some(NodeId(2)),
            cluster: vec![NodeId(1), NodeId(2), NodeId(3)],
        };
        storage.save_state(&state).unwrap();

        // Create node with storage
        let node =
            RaftNode::with_storage(NodeId(1), RaftConfig::default(), Box::new(storage)).unwrap();

        // Node should have restored state
        assert_eq!(node.term(), 10);
    }

    #[test]
    fn test_should_snapshot() {
        let config = RaftConfig {
            snapshot_threshold: 5,
            ..Default::default()
        };
        let mut node = RaftNode::new(NodeId(1), config);

        // Initially no snapshot needed
        assert!(!node.should_snapshot());

        // Add entries to exceed threshold
        for i in 1..=6 {
            node.log.push(LogEntry {
                term: 1,
                index: i,
                command: Command::Noop,
            });
        }

        assert!(node.should_snapshot());
    }

    #[test]
    fn test_persistent_state_roundtrip() {
        let node = RaftNode::new(NodeId(5), RaftConfig::default());
        let state = node.persistent_state();

        assert_eq!(state.current_term, 0);
        assert!(state.voted_for.is_none());
    }

    #[test]
    fn test_log_entries_accessor() {
        let mut node = RaftNode::new(NodeId(1), RaftConfig::default());
        node.log.push(LogEntry {
            term: 1,
            index: 1,
            command: Command::Noop,
        });

        let entries = node.log_entries();
        assert_eq!(entries.len(), 1);
    }
}
