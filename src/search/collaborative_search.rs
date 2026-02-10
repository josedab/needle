//! Real-Time Collaborative Search
//!
//! Enables multiple users and applications to collaborate on vector search
//! in real-time with automatic conflict resolution and live updates.
//!
//! # Features
//!
//! - **CRDT-based Collections**: Conflict-free replicated vector collections
//! - **Live Query Subscriptions**: Real-time search result updates
//! - **Presence Tracking**: See who's searching in real-time
//! - **Shared Search Sessions**: Collaborative search workspaces
//! - **Annotation Sharing**: Share insights on search results
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    Collaborative Search Layer                        │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
//! │  │  Presence   │  │    Live     │  │   Shared    │                 │
//! │  │  Tracking   │  │  Queries    │  │  Sessions   │                 │
//! │  └─────────────┘  └─────────────┘  └─────────────┘                 │
//! │         │                │                │                         │
//! │  ┌──────┴────────────────┴────────────────┴──────┐                 │
//! │  │              CRDT Collection Layer             │                 │
//! │  │    (VectorCRDT + CollectionCRDT + Sync)       │                 │
//! │  └────────────────────────────────────────────────┘                 │
//! │                           │                                         │
//! │  ┌────────────────────────┴────────────────────────┐               │
//! │  │              Event Broadcasting                  │               │
//! │  │         (WebSocket / SSE / Polling)             │               │
//! │  └──────────────────────────────────────────────────┘               │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::collaborative_search::{
//!     CollaborativeCollection, SessionManager, LiveQuery, Presence,
//! };
//!
//! // Create a collaborative collection
//! let collection = CollaborativeCollection::new("shared_docs", 384);
//!
//! // Join a session
//! let session = SessionManager::join_or_create("team-search-session");
//! session.set_presence(Presence::active("user_123", "Searching for ML papers"));
//!
//! // Create a live query that updates automatically
//! let mut live_query = collection.live_search(&query_vector, 10);
//! live_query.on_update(|results| {
//!     println!("New results: {:?}", results);
//! });
//!
//! // Add annotation to a result
//! session.annotate("doc_456", "Highly relevant for our project");
//!
//! // Sync changes across replicas
//! collection.sync().await?;
//! ```

use crate::crdt::{Delta, HLC, MergeResult, ReplicaId, VectorCRDT};
use crate::error::{NeedleError, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ============================================================================
// Constants
// ============================================================================

/// Default maximum results for live queries
const DEFAULT_MAX_RESULTS: usize = 100;

/// Default presence timeout in seconds
const PRESENCE_TIMEOUT_SECS: u64 = 60;

/// Maximum annotations per document
const MAX_ANNOTATIONS_PER_DOC: usize = 100;

/// Maximum session participants
#[allow(dead_code)]
const MAX_SESSION_PARTICIPANTS: usize = 100;

/// Event buffer size for subscriptions
const EVENT_BUFFER_SIZE: usize = 1024;

// ============================================================================
// Collaborative Collection CRDT
// ============================================================================

/// A collection-level CRDT that wraps VectorCRDT with additional collaborative features.
pub struct CollaborativeCollection {
    /// Name of the collection
    name: String,
    /// Vector dimension
    dimension: usize,
    /// Underlying CRDT for vectors
    vector_crdt: RwLock<VectorCRDT>,
    /// Annotations CRDT (document_id -> annotations)
    annotations: RwLock<AnnotationStore>,
    /// Active live queries
    live_queries: RwLock<HashMap<u64, LiveQueryState>>,
    /// Query ID counter
    next_query_id: AtomicU64,
    /// Event subscribers
    subscribers: RwLock<Vec<EventSubscriber>>,
    /// Collection metadata
    metadata: RwLock<CollectionMetadata>,
}

/// Collection metadata with CRDT semantics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMetadata {
    /// Creation timestamp
    pub created_at: HLC,
    /// Last modified timestamp
    pub modified_at: HLC,
    /// Description (LWW)
    pub description: Option<(String, HLC)>,
    /// Tags (Add-wins set)
    pub tags: HashMap<String, HLC>,
    /// Custom properties (LWW per key)
    pub properties: HashMap<String, (String, HLC)>,
}

impl CollectionMetadata {
    fn new(timestamp: HLC) -> Self {
        Self {
            created_at: timestamp,
            modified_at: timestamp,
            description: None,
            tags: HashMap::new(),
            properties: HashMap::new(),
        }
    }

    fn merge(&mut self, other: &CollectionMetadata) {
        if other.modified_at > self.modified_at {
            self.modified_at = other.modified_at;
        }

        // LWW for description
        match (&self.description, &other.description) {
            (None, Some(d)) => self.description = Some(d.clone()),
            (Some((_, ts1)), Some((_, ts2))) if ts2 > ts1 => {
                self.description = other.description.clone();
            }
            _ => {}
        }

        // Add-wins for tags
        for (tag, ts) in &other.tags {
            self.tags
                .entry(tag.clone())
                .and_modify(|existing_ts| {
                    if ts > existing_ts {
                        *existing_ts = *ts;
                    }
                })
                .or_insert(*ts);
        }

        // LWW per key for properties
        for (key, (value, ts)) in &other.properties {
            self.properties
                .entry(key.clone())
                .and_modify(|(existing_value, existing_ts)| {
                    if ts > existing_ts {
                        *existing_value = value.clone();
                        *existing_ts = *ts;
                    }
                })
                .or_insert((value.clone(), *ts));
        }
    }
}

impl CollaborativeCollection {
    /// Create a new collaborative collection.
    pub fn new(name: &str, dimension: usize) -> Self {
        let replica_id = ReplicaId::new();
        let vector_crdt = VectorCRDT::new(replica_id);
        let clock = vector_crdt.current_clock();

        Self {
            name: name.to_string(),
            dimension,
            vector_crdt: RwLock::new(vector_crdt),
            annotations: RwLock::new(AnnotationStore::new(replica_id)),
            live_queries: RwLock::new(HashMap::new()),
            next_query_id: AtomicU64::new(1),
            subscribers: RwLock::new(Vec::new()),
            metadata: RwLock::new(CollectionMetadata::new(clock)),
        }
    }

    /// Create with a specific replica ID.
    pub fn with_replica_id(name: &str, dimension: usize, replica_id: ReplicaId) -> Self {
        let vector_crdt = VectorCRDT::new(replica_id);
        let clock = vector_crdt.current_clock();

        Self {
            name: name.to_string(),
            dimension,
            vector_crdt: RwLock::new(vector_crdt),
            annotations: RwLock::new(AnnotationStore::new(replica_id)),
            live_queries: RwLock::new(HashMap::new()),
            next_query_id: AtomicU64::new(1),
            subscribers: RwLock::new(Vec::new()),
            metadata: RwLock::new(CollectionMetadata::new(clock)),
        }
    }

    /// Get collection name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get replica ID.
    pub fn replica_id(&self) -> ReplicaId {
        self.vector_crdt.read().replica_id()
    }

    /// Insert a vector with collaborative tracking.
    pub fn insert(
        &self,
        id: &str,
        vector: &[f32],
        metadata: HashMap<String, String>,
    ) -> Result<HLC> {
        if vector.len() != self.dimension {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let timestamp = self.vector_crdt.write().add(id, vector, metadata)?;

        // Notify subscribers
        self.notify_subscribers(CollaborativeEvent::VectorInserted {
            id: id.to_string(),
            timestamp,
        });

        // Update live queries
        self.invalidate_live_queries();

        Ok(timestamp)
    }

    /// Update a vector.
    pub fn update(&self, id: &str, vector: &[f32]) -> Result<HLC> {
        if vector.len() != self.dimension {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let timestamp = self.vector_crdt.write().update(id, vector)?;

        self.notify_subscribers(CollaborativeEvent::VectorUpdated {
            id: id.to_string(),
            timestamp,
        });

        self.invalidate_live_queries();

        Ok(timestamp)
    }

    /// Delete a vector.
    pub fn delete(&self, id: &str) -> Result<HLC> {
        let timestamp = self.vector_crdt.write().delete(id)?;

        self.notify_subscribers(CollaborativeEvent::VectorDeleted {
            id: id.to_string(),
            timestamp,
        });

        self.invalidate_live_queries();

        Ok(timestamp)
    }

    /// Get a vector by ID.
    pub fn get(&self, id: &str) -> Option<CollaborativeVector> {
        let crdt = self.vector_crdt.read();
        let vec = crdt.get(id)?;
        let annotations = self.annotations.read().get_for_document(id);

        Some(CollaborativeVector {
            id: vec.id.clone(),
            vector: vec.vector.clone(),
            metadata: vec.get_all_metadata(),
            created_at: vec.created_at,
            updated_at: vec.updated_at,
            annotations,
        })
    }

    /// List all vectors.
    pub fn list(&self) -> Vec<CollaborativeVector> {
        let crdt = self.vector_crdt.read();
        let annotations = self.annotations.read();

        crdt.list()
            .into_iter()
            .map(|vec| {
                let doc_annotations = annotations.get_for_document(&vec.id);
                CollaborativeVector {
                    id: vec.id.clone(),
                    vector: vec.vector.clone(),
                    metadata: vec.get_all_metadata(),
                    created_at: vec.created_at,
                    updated_at: vec.updated_at,
                    annotations: doc_annotations,
                }
            })
            .collect()
    }

    /// Search for similar vectors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<CollaborativeSearchResult>> {
        if query.len() != self.dimension {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }

        let crdt = self.vector_crdt.read();
        let annotations = self.annotations.read();
        let vectors = crdt.list();

        // Simple brute-force search (would be replaced with HNSW in production)
        let mut results: Vec<_> = vectors
            .iter()
            .map(|vec| {
                let distance = cosine_distance(query, &vec.vector);
                (vec, distance)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results
            .into_iter()
            .take(k)
            .map(|(vec, distance)| {
                let doc_annotations = annotations.get_for_document(&vec.id);
                CollaborativeSearchResult {
                    id: vec.id.clone(),
                    distance,
                    metadata: vec.get_all_metadata(),
                    annotations: doc_annotations,
                    version: vec.updated_at,
                }
            })
            .collect())
    }

    /// Create a live query that updates automatically.
    pub fn live_search(&self, query: Vec<f32>, k: usize) -> LiveQuery {
        let query_id = self.next_query_id.fetch_add(1, Ordering::SeqCst);

        let state = LiveQueryState {
            query: query.clone(),
            k,
            last_results: Vec::new(),
            last_update: Instant::now(),
            subscribers: Vec::new(),
        };

        self.live_queries.write().insert(query_id, state);

        LiveQuery {
            id: query_id,
            query,
            k,
            collection_name: self.name.clone(),
        }
    }

    /// Get delta for synchronization.
    pub fn delta_since(&self, since: Option<HLC>) -> CollaborativeDelta {
        let vector_delta = self.vector_crdt.read().delta_since(since);
        let annotation_delta = self.annotations.read().delta_since(since);
        let metadata = self.metadata.read().clone();

        CollaborativeDelta {
            vector_delta,
            annotation_delta,
            metadata,
            collection_name: self.name.clone(),
        }
    }

    /// Merge a delta from another replica.
    pub fn merge(&self, delta: CollaborativeDelta) -> Result<CollaborativeMergeResult> {
        let vector_result = self.vector_crdt.write().merge(delta.vector_delta)?;
        let annotation_result = self.annotations.write().merge(delta.annotation_delta);

        self.metadata.write().merge(&delta.metadata);

        // Notify subscribers of sync
        self.notify_subscribers(CollaborativeEvent::Synchronized {
            vectors_applied: vector_result.applied,
            annotations_applied: annotation_result.applied,
        });

        // Update live queries after sync
        self.invalidate_live_queries();

        Ok(CollaborativeMergeResult {
            vector_result,
            annotation_result,
        })
    }

    /// Add an annotation to a document.
    pub fn annotate(&self, document_id: &str, user_id: &str, content: &str) -> Result<HLC> {
        let timestamp = self.annotations.write().add(document_id, user_id, content)?;

        self.notify_subscribers(CollaborativeEvent::AnnotationAdded {
            document_id: document_id.to_string(),
            user_id: user_id.to_string(),
            timestamp,
        });

        Ok(timestamp)
    }

    /// Subscribe to collection events.
    pub fn subscribe(&self) -> EventReceiver {
        let (sender, receiver) = EventChannel::create(EVENT_BUFFER_SIZE);
        self.subscribers.write().push(sender);
        receiver
    }

    /// Get collection statistics.
    pub fn stats(&self) -> CollaborativeCollectionStats {
        let crdt = self.vector_crdt.read();
        let annotations = self.annotations.read();
        let live_queries = self.live_queries.read();

        CollaborativeCollectionStats {
            name: self.name.clone(),
            dimension: self.dimension,
            vector_count: crdt.len(),
            annotation_count: annotations.len(),
            live_query_count: live_queries.len(),
            log_size: crdt.log_size(),
            replica_id: crdt.replica_id(),
        }
    }

    fn notify_subscribers(&self, event: CollaborativeEvent) {
        let mut subscribers = self.subscribers.write();
        subscribers.retain(|sub| sub.send(event.clone()).is_ok());
    }

    fn invalidate_live_queries(&self) {
        let mut queries = self.live_queries.write();
        for state in queries.values_mut() {
            state.last_update = Instant::now() - Duration::from_secs(3600); // Force refresh
        }
    }
}

/// A vector with collaborative metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeVector {
    /// Vector ID
    pub id: String,
    /// Vector data
    pub vector: Vec<f32>,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Creation timestamp
    pub created_at: HLC,
    /// Last update timestamp
    pub updated_at: HLC,
    /// Annotations from users
    pub annotations: Vec<Annotation>,
}

/// Search result with collaborative metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeSearchResult {
    /// Document ID
    pub id: String,
    /// Distance from query
    pub distance: f32,
    /// Document metadata
    pub metadata: HashMap<String, String>,
    /// Annotations
    pub annotations: Vec<Annotation>,
    /// Current version
    pub version: HLC,
}

/// Delta for collaborative collection sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeDelta {
    /// Vector operations
    pub vector_delta: Delta,
    /// Annotation operations
    pub annotation_delta: AnnotationDelta,
    /// Collection metadata
    pub metadata: CollectionMetadata,
    /// Collection name
    pub collection_name: String,
}

/// Result of merging a collaborative delta.
#[derive(Debug, Clone)]
pub struct CollaborativeMergeResult {
    /// Vector merge result
    pub vector_result: MergeResult,
    /// Annotation merge result
    pub annotation_result: AnnotationMergeResult,
}

/// Collection statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeCollectionStats {
    /// Collection name
    pub name: String,
    /// Vector dimension
    pub dimension: usize,
    /// Number of vectors
    pub vector_count: usize,
    /// Number of annotations
    pub annotation_count: usize,
    /// Number of active live queries
    pub live_query_count: usize,
    /// Operation log size
    pub log_size: usize,
    /// This replica's ID
    pub replica_id: ReplicaId,
}

// ============================================================================
// Annotation CRDT
// ============================================================================

/// Store for annotations with CRDT semantics.
pub struct AnnotationStore {
    /// Replica ID
    replica_id: ReplicaId,
    /// HLC clock
    clock: HLC,
    /// Annotations by document ID
    annotations: HashMap<String, Vec<Annotation>>,
    /// Operation log
    operation_log: BTreeMap<HLC, TimestampedAnnotationOp>,
}

/// An annotation on a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    /// Unique annotation ID
    pub id: String,
    /// Document being annotated
    pub document_id: String,
    /// User who created the annotation
    pub user_id: String,
    /// Annotation content
    pub content: String,
    /// Creation timestamp
    pub created_at: HLC,
    /// Whether deleted
    pub deleted: bool,
}

/// Annotation operation for CRDT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationOp {
    Add {
        id: String,
        document_id: String,
        user_id: String,
        content: String,
    },
    Delete {
        id: String,
    },
}

/// Timestamped annotation operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedAnnotationOp {
    pub op: AnnotationOp,
    pub timestamp: HLC,
    pub origin: ReplicaId,
}

/// Delta for annotation sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationDelta {
    pub operations: Vec<TimestampedAnnotationOp>,
    pub from_timestamp: Option<HLC>,
    pub to_timestamp: Option<HLC>,
    pub origin: ReplicaId,
}

impl AnnotationDelta {
    #[allow(dead_code)]
    fn empty(origin: ReplicaId) -> Self {
        Self {
            operations: Vec::new(),
            from_timestamp: None,
            to_timestamp: None,
            origin,
        }
    }
}

/// Result of annotation merge.
#[derive(Debug, Clone, Default)]
pub struct AnnotationMergeResult {
    pub applied: usize,
    pub skipped: usize,
}

impl AnnotationStore {
    fn new(replica_id: ReplicaId) -> Self {
        Self {
            replica_id,
            clock: HLC::new(replica_id),
            annotations: HashMap::new(),
            operation_log: BTreeMap::new(),
        }
    }

    fn add(&mut self, document_id: &str, user_id: &str, content: &str) -> Result<HLC> {
        let annotations = self.annotations.entry(document_id.to_string()).or_default();
        if annotations.len() >= MAX_ANNOTATIONS_PER_DOC {
            return Err(NeedleError::InvalidOperation(
                "Maximum annotations per document exceeded".to_string(),
            ));
        }

        let timestamp = self.clock.tick();
        let id = format!("{}:{}", timestamp.physical, timestamp.logical);

        let annotation = Annotation {
            id: id.clone(),
            document_id: document_id.to_string(),
            user_id: user_id.to_string(),
            content: content.to_string(),
            created_at: timestamp,
            deleted: false,
        };

        annotations.push(annotation);

        let op = AnnotationOp::Add {
            id,
            document_id: document_id.to_string(),
            user_id: user_id.to_string(),
            content: content.to_string(),
        };

        self.operation_log.insert(
            timestamp,
            TimestampedAnnotationOp {
                op,
                timestamp,
                origin: self.replica_id,
            },
        );

        Ok(timestamp)
    }

    fn get_for_document(&self, document_id: &str) -> Vec<Annotation> {
        self.annotations
            .get(document_id)
            .map(|anns| anns.iter().filter(|a| !a.deleted).cloned().collect())
            .unwrap_or_default()
    }

    fn len(&self) -> usize {
        self.annotations
            .values()
            .map(|anns| anns.iter().filter(|a| !a.deleted).count())
            .sum()
    }

    fn delta_since(&self, since: Option<HLC>) -> AnnotationDelta {
        let operations: Vec<_> = match since {
            Some(ts) => self
                .operation_log
                .range(ts..)
                .filter(|(t, _)| **t > ts)
                .map(|(_, op)| op.clone())
                .collect(),
            None => self.operation_log.values().cloned().collect(),
        };

        let from_timestamp = operations.first().map(|op| op.timestamp);
        let to_timestamp = operations.last().map(|op| op.timestamp);

        AnnotationDelta {
            operations,
            from_timestamp,
            to_timestamp,
            origin: self.replica_id,
        }
    }

    fn merge(&mut self, delta: AnnotationDelta) -> AnnotationMergeResult {
        let mut applied = 0;
        let mut skipped = 0;

        if let Some(ts) = delta.to_timestamp {
            self.clock.receive(ts);
        }

        for timestamped in delta.operations {
            if timestamped.origin == self.replica_id {
                skipped += 1;
                continue;
            }

            if self.operation_log.contains_key(&timestamped.timestamp) {
                skipped += 1;
                continue;
            }

            match &timestamped.op {
                AnnotationOp::Add {
                    id,
                    document_id,
                    user_id,
                    content,
                } => {
                    let annotations = self.annotations.entry(document_id.clone()).or_default();
                    if annotations.len() < MAX_ANNOTATIONS_PER_DOC {
                        annotations.push(Annotation {
                            id: id.clone(),
                            document_id: document_id.clone(),
                            user_id: user_id.clone(),
                            content: content.clone(),
                            created_at: timestamped.timestamp,
                            deleted: false,
                        });
                        applied += 1;
                    }
                }
                AnnotationOp::Delete { id } => {
                    for annotations in self.annotations.values_mut() {
                        for ann in annotations.iter_mut() {
                            if ann.id == *id && !ann.deleted {
                                ann.deleted = true;
                                applied += 1;
                            }
                        }
                    }
                }
            }

            self.operation_log.insert(timestamped.timestamp, timestamped);
        }

        AnnotationMergeResult { applied, skipped }
    }
}

// ============================================================================
// Live Queries
// ============================================================================

/// State for a live query.
#[allow(dead_code)]
struct LiveQueryState {
    query: Vec<f32>,
    k: usize,
    last_results: Vec<CollaborativeSearchResult>,
    last_update: Instant,
    subscribers: Vec<LiveQuerySubscriber>,
}

/// Subscriber to live query updates.
type LiveQuerySubscriber = Box<dyn Fn(&[CollaborativeSearchResult]) + Send + Sync>;

/// A live query that automatically updates when the collection changes.
#[derive(Debug)]
pub struct LiveQuery {
    /// Query ID
    pub id: u64,
    /// Query vector
    pub query: Vec<f32>,
    /// Number of results
    pub k: usize,
    /// Collection name
    pub collection_name: String,
}

impl LiveQuery {
    /// Get current query ID.
    pub fn query_id(&self) -> u64 {
        self.id
    }

    /// Get the query vector.
    pub fn query_vector(&self) -> &[f32] {
        &self.query
    }

    /// Get result count.
    pub fn result_count(&self) -> usize {
        self.k
    }
}

/// Configuration for live queries.
#[derive(Debug, Clone)]
pub struct LiveQueryConfig {
    /// Maximum results
    pub max_results: usize,
    /// Debounce interval for updates
    pub debounce_ms: u64,
    /// Include annotations in results
    pub include_annotations: bool,
    /// Minimum distance change to trigger update
    pub min_distance_change: f32,
}

impl Default for LiveQueryConfig {
    fn default() -> Self {
        Self {
            max_results: DEFAULT_MAX_RESULTS,
            debounce_ms: 100,
            include_annotations: true,
            min_distance_change: 0.001,
        }
    }
}

// ============================================================================
// Presence Tracking
// ============================================================================

/// User presence in a collaborative session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Presence {
    /// User ID
    pub user_id: String,
    /// User display name
    pub display_name: Option<String>,
    /// Current status
    pub status: PresenceStatus,
    /// What the user is doing
    pub activity: Option<String>,
    /// Current query (if searching)
    pub current_query: Option<Vec<f32>>,
    /// Last update timestamp
    pub last_seen: u64,
    /// Custom data
    pub custom_data: HashMap<String, String>,
}

/// Presence status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresenceStatus {
    /// User is actively engaged
    Active,
    /// User is idle
    Idle,
    /// User is away
    Away,
    /// User is offline
    Offline,
}

impl Presence {
    /// Create an active presence.
    pub fn active(user_id: &str, activity: &str) -> Self {
        Self {
            user_id: user_id.to_string(),
            display_name: None,
            status: PresenceStatus::Active,
            activity: Some(activity.to_string()),
            current_query: None,
            last_seen: current_timestamp_millis(),
            custom_data: HashMap::new(),
        }
    }

    /// Create an idle presence.
    pub fn idle(user_id: &str) -> Self {
        Self {
            user_id: user_id.to_string(),
            display_name: None,
            status: PresenceStatus::Idle,
            activity: None,
            current_query: None,
            last_seen: current_timestamp_millis(),
            custom_data: HashMap::new(),
        }
    }

    /// Set display name.
    pub fn with_display_name(mut self, name: &str) -> Self {
        self.display_name = Some(name.to_string());
        self
    }

    /// Set current query.
    pub fn with_query(mut self, query: Vec<f32>) -> Self {
        self.current_query = Some(query);
        self
    }

    /// Add custom data.
    pub fn with_custom_data(mut self, key: &str, value: &str) -> Self {
        self.custom_data.insert(key.to_string(), value.to_string());
        self
    }

    /// Check if presence is stale.
    pub fn is_stale(&self) -> bool {
        let now = current_timestamp_millis();
        now - self.last_seen > PRESENCE_TIMEOUT_SECS * 1000
    }

    /// Update last seen time.
    pub fn touch(&mut self) {
        self.last_seen = current_timestamp_millis();
    }
}

/// Presence tracker for a session.
pub struct PresenceTracker {
    /// All presences
    presences: RwLock<HashMap<String, Presence>>,
    /// Event subscribers
    subscribers: RwLock<Vec<PresenceEventSender>>,
}

type PresenceEventSender = Box<dyn Fn(PresenceEvent) + Send + Sync>;

/// Presence events.
#[derive(Debug, Clone)]
pub enum PresenceEvent {
    /// User joined
    Joined(Presence),
    /// User updated presence
    Updated(Presence),
    /// User left
    Left(String),
}

impl PresenceTracker {
    /// Create a new presence tracker.
    pub fn new() -> Self {
        Self {
            presences: RwLock::new(HashMap::new()),
            subscribers: RwLock::new(Vec::new()),
        }
    }

    /// Set user presence.
    pub fn set(&self, presence: Presence) {
        let user_id = presence.user_id.clone();
        let is_new = !self.presences.read().contains_key(&user_id);

        self.presences.write().insert(user_id.clone(), presence.clone());

        let event = if is_new {
            PresenceEvent::Joined(presence)
        } else {
            PresenceEvent::Updated(presence)
        };

        self.notify(event);
    }

    /// Remove user presence.
    pub fn remove(&self, user_id: &str) {
        self.presences.write().remove(user_id);
        self.notify(PresenceEvent::Left(user_id.to_string()));
    }

    /// Get user presence.
    pub fn get(&self, user_id: &str) -> Option<Presence> {
        self.presences.read().get(user_id).cloned()
    }

    /// List all active presences.
    pub fn list_active(&self) -> Vec<Presence> {
        self.presences
            .read()
            .values()
            .filter(|p| !p.is_stale() && p.status != PresenceStatus::Offline)
            .cloned()
            .collect()
    }

    /// Clean up stale presences.
    pub fn cleanup_stale(&self) -> Vec<String> {
        let mut removed = Vec::new();
        let mut presences = self.presences.write();

        presences.retain(|user_id, presence| {
            if presence.is_stale() {
                removed.push(user_id.clone());
                false
            } else {
                true
            }
        });

        drop(presences);

        for user_id in &removed {
            self.notify(PresenceEvent::Left(user_id.clone()));
        }

        removed
    }

    /// Subscribe to presence events.
    pub fn subscribe<F>(&self, callback: F)
    where
        F: Fn(PresenceEvent) + Send + Sync + 'static,
    {
        self.subscribers.write().push(Box::new(callback));
    }

    fn notify(&self, event: PresenceEvent) {
        for subscriber in self.subscribers.read().iter() {
            subscriber(event.clone());
        }
    }
}

impl Default for PresenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Collaborative Sessions
// ============================================================================

/// A collaborative search session.
pub struct Session {
    /// Session ID
    pub id: String,
    /// Session name
    pub name: String,
    /// Creator user ID
    pub created_by: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Presence tracker
    pub presence: PresenceTracker,
    /// Shared search history
    search_history: RwLock<VecDeque<SharedSearch>>,
    /// Session metadata
    metadata: RwLock<HashMap<String, String>>,
    /// Access control
    access: RwLock<SessionAccess>,
}

/// A shared search in a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedSearch {
    /// Search ID
    pub id: String,
    /// User who performed the search
    pub user_id: String,
    /// Query vector
    pub query: Vec<f32>,
    /// Query text (if available)
    pub query_text: Option<String>,
    /// Top results
    pub results: Vec<String>,
    /// Timestamp
    pub timestamp: u64,
    /// Notes from the user
    pub notes: Option<String>,
}

/// Session access control.
#[derive(Debug, Clone)]
pub struct SessionAccess {
    /// Users with full access
    pub owners: HashSet<String>,
    /// Users with write access
    pub editors: HashSet<String>,
    /// Users with read access
    pub viewers: HashSet<String>,
    /// Whether session is public
    pub is_public: bool,
}

impl SessionAccess {
    fn new(creator: &str) -> Self {
        let mut owners = HashSet::new();
        owners.insert(creator.to_string());
        Self {
            owners,
            editors: HashSet::new(),
            viewers: HashSet::new(),
            is_public: false,
        }
    }

    /// Check if user can read.
    pub fn can_read(&self, user_id: &str) -> bool {
        self.is_public
            || self.owners.contains(user_id)
            || self.editors.contains(user_id)
            || self.viewers.contains(user_id)
    }

    /// Check if user can write.
    pub fn can_write(&self, user_id: &str) -> bool {
        self.owners.contains(user_id) || self.editors.contains(user_id)
    }

    /// Check if user is owner.
    pub fn is_owner(&self, user_id: &str) -> bool {
        self.owners.contains(user_id)
    }
}

impl Session {
    /// Create a new session.
    pub fn new(id: &str, name: &str, created_by: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            created_by: created_by.to_string(),
            created_at: current_timestamp_millis(),
            presence: PresenceTracker::new(),
            search_history: RwLock::new(VecDeque::with_capacity(100)),
            metadata: RwLock::new(HashMap::new()),
            access: RwLock::new(SessionAccess::new(created_by)),
        }
    }

    /// Join the session.
    pub fn join(&self, user_id: &str) -> Result<()> {
        let access = self.access.read();
        if !access.can_read(user_id) {
            return Err(NeedleError::Unauthorized(
                "User not authorized to join session".to_string(),
            ));
        }
        drop(access);

        self.presence.set(Presence::active(user_id, "Joined session"));
        Ok(())
    }

    /// Leave the session.
    pub fn leave(&self, user_id: &str) {
        self.presence.remove(user_id);
    }

    /// Share a search.
    pub fn share_search(&self, search: SharedSearch) -> Result<()> {
        let access = self.access.read();
        if !access.can_write(&search.user_id) {
            return Err(NeedleError::Unauthorized(
                "User not authorized to share searches".to_string(),
            ));
        }
        drop(access);

        let mut history = self.search_history.write();
        if history.len() >= 100 {
            history.pop_front();
        }
        history.push_back(search);
        Ok(())
    }

    /// Get recent searches.
    pub fn recent_searches(&self, limit: usize) -> Vec<SharedSearch> {
        self.search_history
            .read()
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get active participants.
    pub fn participants(&self) -> Vec<Presence> {
        self.presence.list_active()
    }

    /// Add editor access.
    pub fn add_editor(&self, user_id: &str) {
        self.access.write().editors.insert(user_id.to_string());
    }

    /// Add viewer access.
    pub fn add_viewer(&self, user_id: &str) {
        self.access.write().viewers.insert(user_id.to_string());
    }

    /// Set public access.
    pub fn set_public(&self, is_public: bool) {
        self.access.write().is_public = is_public;
    }

    /// Set metadata.
    pub fn set_metadata(&self, key: &str, value: &str) {
        self.metadata.write().insert(key.to_string(), value.to_string());
    }

    /// Get metadata.
    pub fn get_metadata(&self, key: &str) -> Option<String> {
        self.metadata.read().get(key).cloned()
    }
}

/// Manager for collaborative sessions.
pub struct SessionManager {
    /// Active sessions
    sessions: RwLock<HashMap<String, Arc<Session>>>,
}

impl SessionManager {
    /// Create a new session manager.
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new session.
    pub fn create(&self, name: &str, created_by: &str) -> Arc<Session> {
        let id = generate_session_id();
        let session = Arc::new(Session::new(&id, name, created_by));
        self.sessions.write().insert(id, session.clone());
        // Creator automatically joins the session
        session.presence.set(Presence::active(created_by, "Created session"));
        session
    }

    /// Get a session by ID.
    pub fn get(&self, id: &str) -> Option<Arc<Session>> {
        self.sessions.read().get(id).cloned()
    }

    /// Join or create a session.
    pub fn join_or_create(&self, name: &str, user_id: &str) -> Arc<Session> {
        // Look for existing session by name
        {
            let sessions = self.sessions.read();
            for session in sessions.values() {
                if session.name == name {
                    let _ = session.join(user_id);
                    return session.clone();
                }
            }
        }

        // Create new session
        let session = self.create(name, user_id);
        let _ = session.join(user_id);
        session
    }

    /// List all sessions.
    pub fn list(&self) -> Vec<SessionInfo> {
        self.sessions
            .read()
            .values()
            .map(|s| SessionInfo {
                id: s.id.clone(),
                name: s.name.clone(),
                created_by: s.created_by.clone(),
                created_at: s.created_at,
                participant_count: s.participants().len(),
            })
            .collect()
    }

    /// Delete a session.
    pub fn delete(&self, id: &str) -> bool {
        self.sessions.write().remove(id).is_some()
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary information about a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    /// Session ID
    pub id: String,
    /// Session name
    pub name: String,
    /// Creator
    pub created_by: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Number of active participants
    pub participant_count: usize,
}

// ============================================================================
// Event System
// ============================================================================

/// Events emitted by collaborative collections.
#[derive(Debug, Clone)]
pub enum CollaborativeEvent {
    /// Vector was inserted
    VectorInserted { id: String, timestamp: HLC },
    /// Vector was updated
    VectorUpdated { id: String, timestamp: HLC },
    /// Vector was deleted
    VectorDeleted { id: String, timestamp: HLC },
    /// Annotation was added
    AnnotationAdded {
        document_id: String,
        user_id: String,
        timestamp: HLC,
    },
    /// Collection was synchronized
    Synchronized {
        vectors_applied: usize,
        annotations_applied: usize,
    },
}

/// Event channel for subscriptions.
struct EventChannel;

impl EventChannel {
    fn create(capacity: usize) -> (EventSubscriber, EventReceiver) {
        let buffer = Arc::new(RwLock::new(VecDeque::with_capacity(capacity)));
        let closed = Arc::new(std::sync::atomic::AtomicBool::new(false));

        (
            EventSubscriber {
                buffer: buffer.clone(),
                closed: closed.clone(),
                capacity,
            },
            EventReceiver {
                buffer,
                closed,
                position: 0,
            },
        )
    }
}

/// Sender side of event channel.
struct EventSubscriber {
    buffer: Arc<RwLock<VecDeque<CollaborativeEvent>>>,
    closed: Arc<std::sync::atomic::AtomicBool>,
    capacity: usize,
}

impl EventSubscriber {
    fn send(&self, event: CollaborativeEvent) -> Result<()> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(NeedleError::InvalidOperation("Channel closed".to_string()));
        }

        let mut buffer = self.buffer.write();
        if buffer.len() >= self.capacity {
            buffer.pop_front();
        }
        buffer.push_back(event);
        Ok(())
    }
}

/// Receiver side of event channel.
pub struct EventReceiver {
    buffer: Arc<RwLock<VecDeque<CollaborativeEvent>>>,
    closed: Arc<std::sync::atomic::AtomicBool>,
    position: usize,
}

impl EventReceiver {
    /// Try to receive the next event.
    pub fn try_recv(&mut self) -> Option<CollaborativeEvent> {
        let buffer = self.buffer.read();
        if self.position < buffer.len() {
            let event = buffer.get(self.position)?.clone();
            self.position += 1;
            Some(event)
        } else {
            None
        }
    }

    /// Receive all pending events.
    pub fn recv_all(&mut self) -> Vec<CollaborativeEvent> {
        let buffer = self.buffer.read();
        let events: Vec<_> = buffer.iter().skip(self.position).cloned().collect();
        self.position = buffer.len();
        events
    }

    /// Close the receiver.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Relaxed);
    }
}

// ============================================================================
// Sync Manager
// ============================================================================

/// Manager for synchronizing collaborative collections.
pub struct SyncManager {
    /// Collections being managed
    collections: RwLock<HashMap<String, Arc<CollaborativeCollection>>>,
    /// Pending syncs
    pending_syncs: RwLock<VecDeque<PendingSync>>,
    /// Sync statistics
    stats: RwLock<SyncStats>,
}

/// A pending sync operation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PendingSync {
    collection_name: String,
    peer_id: ReplicaId,
    delta: CollaborativeDelta,
    created_at: Instant,
}

/// Synchronization statistics.
#[derive(Debug, Clone, Default)]
pub struct SyncStats {
    /// Total syncs performed
    pub total_syncs: u64,
    /// Successful syncs
    pub successful_syncs: u64,
    /// Failed syncs
    pub failed_syncs: u64,
    /// Total vectors synchronized
    pub vectors_synced: u64,
    /// Total annotations synchronized
    pub annotations_synced: u64,
}

impl SyncManager {
    /// Create a new sync manager.
    pub fn new() -> Self {
        Self {
            collections: RwLock::new(HashMap::new()),
            pending_syncs: RwLock::new(VecDeque::new()),
            stats: RwLock::new(SyncStats::default()),
        }
    }

    /// Register a collection for sync.
    pub fn register(&self, collection: Arc<CollaborativeCollection>) {
        self.collections
            .write()
            .insert(collection.name().to_string(), collection);
    }

    /// Unregister a collection.
    pub fn unregister(&self, name: &str) {
        self.collections.write().remove(name);
    }

    /// Get delta for a collection.
    pub fn get_delta(&self, collection_name: &str, since: Option<HLC>) -> Option<CollaborativeDelta> {
        let collections = self.collections.read();
        collections.get(collection_name).map(|c| c.delta_since(since))
    }

    /// Apply a delta to a collection.
    pub fn apply_delta(&self, delta: CollaborativeDelta) -> Result<CollaborativeMergeResult> {
        let collections = self.collections.read();
        let collection = collections
            .get(&delta.collection_name)
            .ok_or_else(|| NeedleError::CollectionNotFound(delta.collection_name.clone()))?;

        let result = collection.merge(delta)?;

        let mut stats = self.stats.write();
        stats.total_syncs += 1;
        stats.successful_syncs += 1;
        stats.vectors_synced += result.vector_result.applied as u64;
        stats.annotations_synced += result.annotation_result.applied as u64;

        Ok(result)
    }

    /// Queue a sync for later processing.
    pub fn queue_sync(&self, collection_name: &str, peer_id: ReplicaId, delta: CollaborativeDelta) {
        self.pending_syncs.write().push_back(PendingSync {
            collection_name: collection_name.to_string(),
            peer_id,
            delta,
            created_at: Instant::now(),
        });
    }

    /// Process pending syncs.
    pub fn process_pending(&self) -> Vec<Result<CollaborativeMergeResult>> {
        let pending: Vec<_> = self.pending_syncs.write().drain(..).collect();

        pending
            .into_iter()
            .map(|sync| self.apply_delta(sync.delta))
            .collect()
    }

    /// Get sync statistics.
    pub fn stats(&self) -> SyncStats {
        self.stats.read().clone()
    }

    /// List registered collections.
    pub fn collections(&self) -> Vec<String> {
        self.collections.read().keys().cloned().collect()
    }
}

impl Default for SyncManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Utilities
// ============================================================================

/// Get current timestamp in milliseconds.
fn current_timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after Unix epoch")
        .as_millis() as u64
}

/// Compute cosine distance between two vectors.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let similarity = dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-10);
    1.0 - similarity
}

/// Generate a unique session ID.
fn generate_session_id() -> String {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let now = current_timestamp_millis();
    let mut hasher = DefaultHasher::new();
    now.hash(&mut hasher);
    format!("session_{:016x}", hasher.finish())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_collaborative_collection_create() {
        let collection = CollaborativeCollection::new("test", 128);
        assert_eq!(collection.name(), "test");
        assert_eq!(collection.dimension(), 128);
    }

    #[test]
    fn test_insert_and_get() {
        let collection = CollaborativeCollection::new("test", 4);
        let vector = vec![1.0, 2.0, 3.0, 4.0];

        collection
            .insert("vec1", &vector, HashMap::new())
            .unwrap();

        let retrieved = collection.get("vec1").unwrap();
        assert_eq!(retrieved.id, "vec1");
        assert_eq!(retrieved.vector, vector);
    }

    #[test]
    fn test_search() {
        let collection = CollaborativeCollection::new("test", 4);

        collection
            .insert("vec1", &[1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        collection
            .insert("vec2", &[0.0, 1.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        collection
            .insert("vec3", &[0.9, 0.1, 0.0, 0.0], HashMap::new())
            .unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = collection.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "vec1");
    }

    #[test]
    fn test_annotations() {
        let collection = CollaborativeCollection::new("test", 4);
        collection
            .insert("vec1", &[1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();

        collection
            .annotate("vec1", "user1", "This is relevant")
            .unwrap();
        collection
            .annotate("vec1", "user2", "I agree")
            .unwrap();

        let vec = collection.get("vec1").unwrap();
        assert_eq!(vec.annotations.len(), 2);
    }

    #[test]
    fn test_sync_between_replicas() {
        let replica1 = ReplicaId::from(1);
        let replica2 = ReplicaId::from(2);

        let collection1 = CollaborativeCollection::with_replica_id("test", 4, replica1);
        let collection2 = CollaborativeCollection::with_replica_id("test", 4, replica2);

        // Insert on replica 1
        collection1
            .insert("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();

        // Get delta and merge to replica 2
        let delta = collection1.delta_since(None);
        collection2.merge(delta).unwrap();

        // Verify sync
        let vec = collection2.get("vec1").unwrap();
        assert_eq!(vec.vector, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_presence_tracking() {
        let tracker = PresenceTracker::new();

        tracker.set(Presence::active("user1", "Searching"));
        tracker.set(Presence::active("user2", "Reviewing"));

        let active = tracker.list_active();
        assert_eq!(active.len(), 2);

        tracker.remove("user1");
        let active = tracker.list_active();
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn test_session_management() {
        let manager = SessionManager::new();

        let session = manager.create("Team Search", "user1");
        assert_eq!(session.name, "Team Search");

        // Make session public so user2 can join
        session.set_public(true);
        session.join("user2").unwrap();
        assert_eq!(session.participants().len(), 2);

        // Share a search
        let search = SharedSearch {
            id: "search1".to_string(),
            user_id: "user1".to_string(),
            query: vec![1.0, 2.0, 3.0, 4.0],
            query_text: Some("test query".to_string()),
            results: vec!["doc1".to_string(), "doc2".to_string()],
            timestamp: current_timestamp_millis(),
            notes: None,
        };
        session.share_search(search).unwrap();

        let recent = session.recent_searches(10);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_sync_manager() {
        let manager = SyncManager::new();

        let collection = Arc::new(CollaborativeCollection::new("test", 4));
        collection
            .insert("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();

        manager.register(collection.clone());

        let delta = manager.get_delta("test", None).unwrap();
        assert!(!delta.vector_delta.is_empty());

        let stats = manager.stats();
        assert_eq!(stats.total_syncs, 0);
    }

    #[test]
    fn test_live_query() {
        let collection = CollaborativeCollection::new("test", 4);

        collection
            .insert("vec1", &[1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let live_query = collection.live_search(query.clone(), 10);

        assert_eq!(live_query.query_vector(), &query);
        assert_eq!(live_query.result_count(), 10);
    }

    #[test]
    fn test_event_subscription() {
        let collection = CollaborativeCollection::new("test", 4);
        let mut receiver = collection.subscribe();

        collection
            .insert("vec1", &[1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();

        let events = receiver.recv_all();
        assert_eq!(events.len(), 1);

        match &events[0] {
            CollaborativeEvent::VectorInserted { id, .. } => {
                assert_eq!(id, "vec1");
            }
            _ => panic!("Expected VectorInserted event"),
        }
    }

    #[test]
    fn test_collection_stats() {
        let collection = CollaborativeCollection::new("test", 4);

        collection
            .insert("vec1", &[1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        collection
            .insert("vec2", &[0.0, 1.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        collection.annotate("vec1", "user1", "Note").unwrap();

        let stats = collection.stats();
        assert_eq!(stats.vector_count, 2);
        assert_eq!(stats.annotation_count, 1);
    }

    #[test]
    fn test_dimension_mismatch() {
        let collection = CollaborativeCollection::new("test", 4);

        let result = collection.insert("vec1", &[1.0, 2.0], HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_vector() {
        let collection = CollaborativeCollection::new("test", 4);

        collection
            .insert("vec1", &[1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        assert!(collection.get("vec1").is_some());

        collection.delete("vec1").unwrap();
        assert!(collection.get("vec1").is_none());
    }

    #[test]
    fn test_update_vector() {
        let collection = CollaborativeCollection::new("test", 4);

        collection
            .insert("vec1", &[1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        collection.update("vec1", &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let vec = collection.get("vec1").unwrap();
        assert_eq!(vec.vector, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_metadata_merge() {
        let replica1 = ReplicaId::from(1);
        let clock = HLC::new(replica1);

        let mut meta1 = CollectionMetadata::new(clock);
        meta1.tags.insert("tag1".to_string(), clock);

        let replica2 = ReplicaId::from(2);
        let clock2 = HLC::new(replica2);
        let mut meta2 = CollectionMetadata::new(clock2);
        meta2.tags.insert("tag2".to_string(), clock2);

        meta1.merge(&meta2);

        assert!(meta1.tags.contains_key("tag1"));
        assert!(meta1.tags.contains_key("tag2"));
    }

    #[test]
    fn test_session_access_control() {
        let manager = SessionManager::new();
        let session = manager.create("Private Session", "owner");

        // Owner can always access
        assert!(session.join("owner").is_ok());

        // Non-member cannot join
        assert!(session.join("stranger").is_err());

        // Add viewer
        session.add_viewer("viewer1");
        assert!(session.join("viewer1").is_ok());

        // Viewer cannot share searches
        let search = SharedSearch {
            id: "s1".to_string(),
            user_id: "viewer1".to_string(),
            query: vec![1.0],
            query_text: None,
            results: vec![],
            timestamp: 0,
            notes: None,
        };
        assert!(session.share_search(search).is_err());

        // Add editor
        session.add_editor("editor1");
        let search = SharedSearch {
            id: "s2".to_string(),
            user_id: "editor1".to_string(),
            query: vec![1.0],
            query_text: None,
            results: vec![],
            timestamp: 0,
            notes: None,
        };
        assert!(session.share_search(search).is_ok());
    }

    #[test]
    fn test_concurrent_sync() {
        let replica1 = ReplicaId::from(1);
        let replica2 = ReplicaId::from(2);

        let collection1 = CollaborativeCollection::with_replica_id("test", 4, replica1);
        let collection2 = CollaborativeCollection::with_replica_id("test", 4, replica2);

        // Both insert same ID
        collection1
            .insert("vec1", &[1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        collection2
            .insert("vec1", &[0.0, 1.0, 0.0, 0.0], HashMap::new())
            .unwrap();

        // Sync both ways
        let delta1 = collection1.delta_since(None);
        let delta2 = collection2.delta_since(None);

        collection1.merge(delta2).unwrap();
        collection2.merge(delta1).unwrap();

        // Both should converge (LWW)
        let vec1 = collection1.get("vec1").unwrap().vector.clone();
        let vec2 = collection2.get("vec1").unwrap().vector.clone();
        assert_eq!(vec1, vec2);
    }
}
