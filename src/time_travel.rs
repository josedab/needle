//! Semantic Time-Travel Module
//!
//! Provides point-in-time vector queries with MVCC (Multi-Version Concurrency Control)
//! support and natural language time expressions.
//!
//! # Features
//!
//! - **MVCC**: Multi-version concurrency control for vectors
//! - **Point-in-Time Queries**: Search vectors as they existed at any timestamp
//! - **Natural Language Time**: "last Tuesday", "a week ago", "yesterday at 3pm"
//! - **Snapshot Management**: Efficient snapshot creation and GC
//! - **Time Range Queries**: Search within specific time windows
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::time_travel::{TimeTravelIndex, MvccConfig, TimeExpression};
//!
//! // Create time-travel enabled index
//! let mut index = TimeTravelIndex::new(db, "documents", MvccConfig::default());
//!
//! // Insert vectors with automatic versioning
//! index.insert("doc1", &embedding, metadata)?;
//!
//! // Query at a specific point in time
//! let results = index.search_at(
//!     &query,
//!     10,
//!     TimeExpression::parse("last Tuesday")?,
//! )?;
//!
//! // Query with natural language time
//! let results = index.search_since(
//!     &query,
//!     10,
//!     TimeExpression::parse("2 weeks ago")?,
//! )?;
//!
//! // Create a named snapshot
//! let snapshot_id = index.create_snapshot("production-v1")?;
//!
//! // Query at snapshot
//! let results = index.search_at_snapshot(&query, 10, &snapshot_id)?;
//! ```

use crate::database::Database;
use crate::error::{NeedleError, Result};
use crate::metadata::Filter;
use crate::SearchResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// MVCC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MvccConfig {
    /// Maximum versions to keep per vector
    pub max_versions_per_vector: usize,
    /// Enable automatic snapshot creation
    pub auto_snapshot: bool,
    /// Interval for auto snapshots (in seconds)
    pub snapshot_interval_seconds: u64,
    /// Maximum snapshots to retain
    pub max_snapshots: usize,
    /// Enable garbage collection
    pub enable_gc: bool,
    /// Minimum age before GC (in seconds)
    pub gc_min_age_seconds: u64,
    /// Timestamp field name in metadata
    pub timestamp_field: String,
    /// Version field name in metadata
    pub version_field: String,
}

impl Default for MvccConfig {
    fn default() -> Self {
        Self {
            max_versions_per_vector: 100,
            auto_snapshot: true,
            snapshot_interval_seconds: 3600, // 1 hour
            max_snapshots: 168, // 1 week of hourly snapshots
            enable_gc: true,
            gc_min_age_seconds: 604800, // 1 week
            timestamp_field: "_timestamp".to_string(),
            version_field: "_version".to_string(),
        }
    }
}

/// A vector version in MVCC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorVersion {
    /// Version number (transaction ID)
    pub txn_id: u64,
    /// Creation timestamp
    pub created_at: u64,
    /// Vector data
    pub vector: Vec<f32>,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
    /// Whether this version is a deletion marker
    pub is_tombstone: bool,
    /// Transaction that deleted this version (if applicable)
    pub deleted_at_txn: Option<u64>,
}

/// A named snapshot of the database state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Snapshot ID
    pub id: String,
    /// Human-readable name
    pub name: Option<String>,
    /// Transaction ID at snapshot time
    pub txn_id: u64,
    /// Timestamp when snapshot was created
    pub timestamp: u64,
    /// Description
    pub description: Option<String>,
    /// Whether this is an auto-created snapshot
    pub is_auto: bool,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Natural language time expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeExpression {
    /// Absolute Unix timestamp
    Timestamp(u64),
    /// Relative time in seconds from now
    RelativeSeconds(i64),
    /// Named time like "yesterday", "last week"
    Named(NamedTime),
    /// Specific date/time components
    DateTime(DateTimeComponents),
    /// At a specific snapshot
    AtSnapshot(String),
}

/// Named time expressions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NamedTime {
    Now,
    Yesterday,
    LastWeek,
    LastMonth,
    LastYear,
    StartOfDay,
    StartOfWeek,
    StartOfMonth,
    StartOfYear,
}

/// Date/time components for specific times
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DateTimeComponents {
    pub year: Option<i32>,
    pub month: Option<u32>,
    pub day: Option<u32>,
    pub hour: Option<u32>,
    pub minute: Option<u32>,
    pub second: Option<u32>,
    /// Day of week (0 = Sunday, 6 = Saturday)
    pub day_of_week: Option<u32>,
    /// "last" modifier (e.g., "last Tuesday")
    pub last_occurrence: bool,
}

impl TimeExpression {
    /// Parse a natural language time expression
    pub fn parse(input: &str) -> Result<Self> {
        let input = input.trim().to_lowercase();

        // Check for named times
        match input.as_str() {
            "now" => return Ok(Self::Named(NamedTime::Now)),
            "yesterday" => return Ok(Self::Named(NamedTime::Yesterday)),
            "last week" => return Ok(Self::Named(NamedTime::LastWeek)),
            "last month" => return Ok(Self::Named(NamedTime::LastMonth)),
            "last year" => return Ok(Self::Named(NamedTime::LastYear)),
            "today" | "start of day" => return Ok(Self::Named(NamedTime::StartOfDay)),
            "this week" | "start of week" => return Ok(Self::Named(NamedTime::StartOfWeek)),
            "this month" | "start of month" => return Ok(Self::Named(NamedTime::StartOfMonth)),
            "this year" | "start of year" => return Ok(Self::Named(NamedTime::StartOfYear)),
            _ => {}
        }

        // Check for "X ago" pattern
        if input.ends_with(" ago") {
            let duration_str = input.trim_end_matches(" ago");
            if let Some(seconds) = Self::parse_duration(duration_str) {
                return Ok(Self::RelativeSeconds(-seconds));
            }
        }

        // Check for "last <day of week>" pattern
        if input.starts_with("last ") {
            let day_str = input.trim_start_matches("last ");
            if let Some(day_of_week) = Self::parse_day_of_week(day_str) {
                return Ok(Self::DateTime(DateTimeComponents {
                    day_of_week: Some(day_of_week),
                    last_occurrence: true,
                    ..Default::default()
                }));
            }
        }

        // Check for "in X" (future) pattern
        if input.starts_with("in ") {
            let duration_str = input.trim_start_matches("in ");
            if let Some(seconds) = Self::parse_duration(duration_str) {
                return Ok(Self::RelativeSeconds(seconds));
            }
        }

        // Try to parse as Unix timestamp
        if let Ok(ts) = input.parse::<u64>() {
            return Ok(Self::Timestamp(ts));
        }

        // Check for snapshot reference
        if input.starts_with("snapshot:") || input.starts_with("@") {
            let snapshot_id = input
                .trim_start_matches("snapshot:")
                .trim_start_matches("@")
                .to_string();
            return Ok(Self::AtSnapshot(snapshot_id));
        }

        Err(NeedleError::InvalidInput(format!(
            "Could not parse time expression: '{}'",
            input
        )))
    }

    /// Parse a duration string like "2 hours", "3 days", "1 week"
    fn parse_duration(input: &str) -> Option<i64> {
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.len() < 2 {
            // Try single word like "hour", "day", "week"
            return match input.trim() {
                "second" => Some(1),
                "minute" => Some(60),
                "hour" => Some(3600),
                "day" => Some(86400),
                "week" => Some(604800),
                "month" => Some(2592000), // 30 days
                "year" => Some(31536000), // 365 days
                _ => None,
            };
        }

        let amount: i64 = parts[0].parse().ok()?;
        let unit = parts[1].trim_end_matches('s'); // Remove plural 's'

        let multiplier = match unit {
            "second" => 1,
            "minute" => 60,
            "hour" => 3600,
            "day" => 86400,
            "week" => 604800,
            "month" => 2592000,
            "year" => 31536000,
            _ => return None,
        };

        Some(amount * multiplier)
    }

    /// Parse day of week name
    fn parse_day_of_week(input: &str) -> Option<u32> {
        match input.trim() {
            "sunday" | "sun" => Some(0),
            "monday" | "mon" => Some(1),
            "tuesday" | "tue" | "tues" => Some(2),
            "wednesday" | "wed" => Some(3),
            "thursday" | "thu" | "thur" | "thurs" => Some(4),
            "friday" | "fri" => Some(5),
            "saturday" | "sat" => Some(6),
            _ => None,
        }
    }

    /// Resolve to a Unix timestamp
    pub fn resolve(&self) -> Result<u64> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        match self {
            TimeExpression::Timestamp(ts) => Ok(*ts),
            TimeExpression::RelativeSeconds(offset) => {
                if *offset >= 0 {
                    Ok(now + *offset as u64)
                } else {
                    Ok(now.saturating_sub((-offset) as u64))
                }
            }
            TimeExpression::Named(named) => Self::resolve_named(*named, now),
            TimeExpression::DateTime(dt) => Self::resolve_datetime(dt, now),
            TimeExpression::AtSnapshot(_) => Err(NeedleError::InvalidInput(
                "Cannot resolve snapshot reference without index".to_string(),
            )),
        }
    }

    fn resolve_named(named: NamedTime, now: u64) -> Result<u64> {
        Ok(match named {
            NamedTime::Now => now,
            NamedTime::Yesterday => now - 86400,
            NamedTime::LastWeek => now - 604800,
            NamedTime::LastMonth => now - 2592000,
            NamedTime::LastYear => now - 31536000,
            NamedTime::StartOfDay => now - (now % 86400),
            NamedTime::StartOfWeek => {
                // Find last Sunday
                let day_of_week = (now / 86400 + 4) % 7; // Unix epoch was Thursday
                now - (now % 86400) - (day_of_week * 86400)
            }
            NamedTime::StartOfMonth => {
                // Approximate: go back to start of current 30-day period
                now - (now % 2592000)
            }
            NamedTime::StartOfYear => {
                // Approximate: go back to start of current 365-day period
                now - (now % 31536000)
            }
        })
    }

    fn resolve_datetime(dt: &DateTimeComponents, now: u64) -> Result<u64> {
        // Handle "last <day of week>"
        if let Some(target_day) = dt.day_of_week {
            if dt.last_occurrence {
                let current_day_of_week = ((now / 86400) + 4) % 7; // Unix epoch was Thursday
                let days_back = if current_day_of_week >= target_day as u64 {
                    current_day_of_week - target_day as u64
                } else {
                    7 - (target_day as u64 - current_day_of_week)
                };
                // Make sure we go back at least 1 day
                let days_back = if days_back == 0 { 7 } else { days_back };
                let target_timestamp = now - (days_back * 86400);
                // Round to start of that day
                return Ok(target_timestamp - (target_timestamp % 86400));
            }
        }

        // For other datetime components, we'd need more complex calendar math
        // For now, return approximate calculation
        Err(NeedleError::InvalidInput(
            "Complex datetime parsing not yet implemented".to_string(),
        ))
    }
}

/// Time-Travel enabled vector index with MVCC
pub struct TimeTravelIndex {
    db: Arc<Database>,
    collection_name: String,
    config: MvccConfig,
    /// Version history: id -> versions (ordered by txn_id)
    versions: HashMap<String, VecDeque<VectorVersion>>,
    /// Transaction ID counter
    current_txn: u64,
    /// Named snapshots
    snapshots: HashMap<String, Snapshot>,
    /// Snapshot index by timestamp for range queries
    snapshot_timeline: BTreeMap<u64, String>,
    /// Last auto-snapshot timestamp
    last_auto_snapshot: u64,
    /// Statistics
    stats: TimeTravelStats,
}

/// Statistics for the time-travel index
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimeTravelStats {
    /// Total vectors (current version)
    pub total_vectors: usize,
    /// Total versions across all vectors
    pub total_versions: usize,
    /// Total snapshots
    pub total_snapshots: usize,
    /// Oldest version timestamp
    pub oldest_version_timestamp: Option<u64>,
    /// Newest version timestamp
    pub newest_version_timestamp: Option<u64>,
    /// Total tombstones (deleted versions)
    pub total_tombstones: usize,
    /// GC runs performed
    pub gc_runs: usize,
    /// Versions collected by GC
    pub gc_versions_collected: usize,
}

/// Result from a time-travel search
#[derive(Debug, Clone)]
pub struct TimeTravelSearchResult {
    /// Base search result
    pub result: SearchResult,
    /// Version of this result
    pub version: u64,
    /// Timestamp of this version
    pub timestamp: u64,
    /// Query timestamp (point in time)
    pub query_timestamp: u64,
    /// Whether this is from a snapshot
    pub from_snapshot: Option<String>,
}

impl TimeTravelIndex {
    /// Create a new time-travel index
    pub fn new(db: Arc<Database>, collection_name: &str, config: MvccConfig) -> Self {
        Self {
            db,
            collection_name: collection_name.to_string(),
            config,
            versions: HashMap::new(),
            current_txn: 0,
            snapshots: HashMap::new(),
            snapshot_timeline: BTreeMap::new(),
            last_auto_snapshot: 0,
            stats: TimeTravelStats::default(),
        }
    }

    /// Get current timestamp
    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Start a new transaction
    fn next_txn(&mut self) -> u64 {
        self.current_txn += 1;
        self.current_txn
    }

    /// Insert a vector with automatic versioning
    pub fn insert(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: Option<serde_json::Value>,
    ) -> Result<u64> {
        let txn_id = self.next_txn();
        let timestamp = Self::now();

        // Add system metadata
        let mut meta = metadata.unwrap_or(serde_json::json!({}));
        if let serde_json::Value::Object(ref mut map) = meta {
            map.insert(self.config.timestamp_field.clone(), serde_json::json!(timestamp));
            map.insert(self.config.version_field.clone(), serde_json::json!(txn_id));
        }

        // Create version
        let version = VectorVersion {
            txn_id,
            created_at: timestamp,
            vector: vector.to_vec(),
            metadata: Some(meta.clone()),
            is_tombstone: false,
            deleted_at_txn: None,
        };

        // Insert into collection (delete first if exists)
        let collection = self.db.collection(&self.collection_name)?;
        let _ = collection.delete(id); // Ignore error if doesn't exist
        collection.insert(id, vector, Some(meta))?;

        // Store version history
        let versions = self.versions.entry(id.to_string()).or_default();
        versions.push_back(version);

        // Enforce max versions
        while versions.len() > self.config.max_versions_per_vector {
            versions.pop_front();
        }

        // Update stats
        self.stats.total_vectors = self.versions.len();
        self.stats.total_versions = self.versions.values().map(|v| v.len()).sum();
        self.stats.newest_version_timestamp = Some(timestamp);
        if self.stats.oldest_version_timestamp.is_none() {
            self.stats.oldest_version_timestamp = Some(timestamp);
        }

        // Check for auto-snapshot
        self.maybe_auto_snapshot()?;

        Ok(txn_id)
    }

    /// Update a vector (creates new version)
    pub fn update(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: Option<serde_json::Value>,
    ) -> Result<u64> {
        if !self.versions.contains_key(id) {
            return Err(NeedleError::VectorNotFound(id.to_string()));
        }

        // Insert creates a new version
        self.insert(id, vector, metadata)
    }

    /// Delete a vector (creates tombstone version)
    pub fn delete(&mut self, id: &str) -> Result<u64> {
        if !self.versions.contains_key(id) {
            return Err(NeedleError::VectorNotFound(id.to_string()));
        }

        let txn_id = self.next_txn();
        let timestamp = Self::now();

        // Create tombstone version
        let version = VectorVersion {
            txn_id,
            created_at: timestamp,
            vector: Vec::new(),
            metadata: None,
            is_tombstone: true,
            deleted_at_txn: None,
        };

        // Mark previous version as deleted
        if let Some(versions) = self.versions.get_mut(id) {
            if let Some(last) = versions.back_mut() {
                last.deleted_at_txn = Some(txn_id);
            }
            versions.push_back(version);
        }

        // Delete from collection
        let collection = self.db.collection(&self.collection_name)?;
        let _ = collection.delete(id);

        self.stats.total_tombstones += 1;

        Ok(txn_id)
    }

    /// Get the visible version of a vector at a specific time
    fn get_visible_version(&self, id: &str, as_of_txn: u64) -> Option<&VectorVersion> {
        let versions = self.versions.get(id)?;

        // Find the latest version that was created before or at as_of_txn
        // and was not deleted before as_of_txn
        // Use rev().find() instead of filter().last() for efficiency
        versions.iter().rev().find(|v| {
            v.txn_id <= as_of_txn
                && !v.is_tombstone
                && v.deleted_at_txn.is_none_or(|del| del > as_of_txn)
        })
    }

    /// Search at a specific point in time
    pub fn search_at(
        &self,
        query: &[f32],
        k: usize,
        time: TimeExpression,
    ) -> Result<Vec<TimeTravelSearchResult>> {
        let timestamp = match &time {
            TimeExpression::AtSnapshot(snapshot_id) => {
                let snapshot = self.snapshots.get(snapshot_id).ok_or_else(|| {
                    NeedleError::NotFound(format!("Snapshot '{}' not found", snapshot_id))
                })?;
                snapshot.timestamp
            }
            _ => time.resolve()?,
        };

        // Find the transaction ID at this timestamp
        let as_of_txn = self.txn_at_timestamp(timestamp);

        let snapshot_name = match &time {
            TimeExpression::AtSnapshot(name) => Some(name.clone()),
            _ => None,
        };
        self.search_at_txn(query, k, as_of_txn, timestamp, snapshot_name)
    }

    /// Search at a specific transaction ID
    fn search_at_txn(
        &self,
        query: &[f32],
        k: usize,
        as_of_txn: u64,
        query_timestamp: u64,
        from_snapshot: Option<String>,
    ) -> Result<Vec<TimeTravelSearchResult>> {
        // Get IDs visible at this transaction
        let visible_ids: Vec<String> = self
            .versions
            .iter()
            .filter_map(|(id, _)| {
                self.get_visible_version(id, as_of_txn).map(|_| id.clone())
            })
            .collect();

        if visible_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Build a temporary in-memory index for historical vectors
        // In production, this would use a more efficient approach
        let mut historical_vectors: Vec<(String, Vec<f32>, &VectorVersion)> = visible_ids
            .iter()
            .filter_map(|id| {
                let version = self.get_visible_version(id, as_of_txn)?;
                Some((id.clone(), version.vector.clone(), version))
            })
            .collect();

        // Sort by similarity to query
        historical_vectors.sort_by(|a, b| {
            let sim_a = cosine_similarity(query, &a.1);
            let sim_b = cosine_similarity(query, &b.1);
            sim_b.partial_cmp(&sim_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top k
        let results = historical_vectors
            .into_iter()
            .take(k)
            .map(|(id, vector, version)| {
                let distance = cosine_distance(query, &vector);
                TimeTravelSearchResult {
                    result: SearchResult {
                        id,
                        distance,
                        metadata: version.metadata.clone(),
                    },
                    version: version.txn_id,
                    timestamp: version.created_at,
                    query_timestamp,
                    from_snapshot: from_snapshot.clone(),
                }
            })
            .collect();

        Ok(results)
    }

    /// Search for vectors added/modified since a specific time
    pub fn search_since(
        &self,
        query: &[f32],
        k: usize,
        since: TimeExpression,
    ) -> Result<Vec<TimeTravelSearchResult>> {
        let since_timestamp = since.resolve()?;
        let since_txn = self.txn_at_timestamp(since_timestamp);

        // Get vectors that have versions after since_txn
        let recent_ids: Vec<String> = self
            .versions
            .iter()
            .filter(|(_, versions)| versions.iter().any(|v| v.txn_id > since_txn && !v.is_tombstone))
            .map(|(id, _)| id.clone())
            .collect();

        if recent_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Search in current collection with filter
        let collection = self.db.collection(&self.collection_name)?;

        // Search current state
        let results = collection.search(query, k * 2)?;

        // Filter to only recent IDs and convert
        let now = Self::now();
        let filtered: Vec<TimeTravelSearchResult> = results
            .into_iter()
            .filter(|r| recent_ids.contains(&r.id))
            .take(k)
            .map(|r| {
                let version = self
                    .versions
                    .get(&r.id)
                    .and_then(|v| v.back())
                    .map(|v| v.txn_id)
                    .unwrap_or(0);
                let timestamp = self
                    .versions
                    .get(&r.id)
                    .and_then(|v| v.back())
                    .map(|v| v.created_at)
                    .unwrap_or(0);

                TimeTravelSearchResult {
                    result: r,
                    version,
                    timestamp,
                    query_timestamp: now,
                    from_snapshot: None,
                }
            })
            .collect();

        Ok(filtered)
    }

    /// Search within a time range
    pub fn search_in_range(
        &self,
        query: &[f32],
        k: usize,
        start: TimeExpression,
        end: TimeExpression,
    ) -> Result<Vec<TimeTravelSearchResult>> {
        let start_ts = start.resolve()?;
        let end_ts = end.resolve()?;

        // Use the end timestamp for the point-in-time query
        let end_txn = self.txn_at_timestamp(end_ts);

        // Get IDs with versions in the range
        let ids_in_range: Vec<String> = self
            .versions
            .iter()
            .filter(|(_, versions)| {
                versions.iter().any(|v| {
                    v.created_at >= start_ts && v.created_at <= end_ts && !v.is_tombstone
                })
            })
            .map(|(id, _)| id.clone())
            .collect();

        // Search at end time, filter to range
        let all_results = self.search_at_txn(query, k * 2, end_txn, end_ts, None)?;

        let filtered: Vec<TimeTravelSearchResult> = all_results
            .into_iter()
            .filter(|r| ids_in_range.contains(&r.result.id))
            .take(k)
            .collect();

        Ok(filtered)
    }

    /// Create a named snapshot
    pub fn create_snapshot(&mut self, name: &str) -> Result<String> {
        self.create_snapshot_with_options(name, None, false, Vec::new())
    }

    /// Create a snapshot with full options
    pub fn create_snapshot_with_options(
        &mut self,
        name: &str,
        description: Option<String>,
        is_auto: bool,
        tags: Vec<String>,
    ) -> Result<String> {
        let timestamp = Self::now();
        let snapshot_id = format!("snap_{}", self.current_txn);

        let snapshot = Snapshot {
            id: snapshot_id.clone(),
            name: Some(name.to_string()),
            txn_id: self.current_txn,
            timestamp,
            description,
            is_auto,
            tags,
        };

        self.snapshots.insert(snapshot_id.clone(), snapshot);
        self.snapshot_timeline.insert(timestamp, snapshot_id.clone());

        // Enforce max snapshots
        while self.snapshots.len() > self.config.max_snapshots {
            // Remove oldest auto-snapshot
            let oldest_auto = self
                .snapshots
                .iter()
                .filter(|(_, s)| s.is_auto)
                .min_by_key(|(_, s)| s.timestamp);

            if let Some((id, _)) = oldest_auto {
                let id = id.clone();
                if let Some(snap) = self.snapshots.remove(&id) {
                    self.snapshot_timeline.remove(&snap.timestamp);
                }
            } else {
                break;
            }
        }

        self.stats.total_snapshots = self.snapshots.len();

        Ok(snapshot_id)
    }

    /// List all snapshots
    pub fn list_snapshots(&self) -> Vec<&Snapshot> {
        let mut snapshots: Vec<_> = self.snapshots.values().collect();
        snapshots.sort_by_key(|s| std::cmp::Reverse(s.timestamp));
        snapshots
    }

    /// Get a snapshot by ID
    pub fn get_snapshot(&self, id: &str) -> Option<&Snapshot> {
        self.snapshots.get(id)
    }

    /// Delete a snapshot
    pub fn delete_snapshot(&mut self, id: &str) -> Result<()> {
        let snapshot = self.snapshots.remove(id).ok_or_else(|| {
            NeedleError::NotFound(format!("Snapshot '{}' not found", id))
        })?;
        self.snapshot_timeline.remove(&snapshot.timestamp);
        self.stats.total_snapshots = self.snapshots.len();
        Ok(())
    }

    /// Search at a specific snapshot
    pub fn search_at_snapshot(
        &self,
        query: &[f32],
        k: usize,
        snapshot_id: &str,
    ) -> Result<Vec<TimeTravelSearchResult>> {
        let snapshot = self.snapshots.get(snapshot_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Snapshot '{}' not found", snapshot_id))
        })?;

        self.search_at_txn(
            query,
            k,
            snapshot.txn_id,
            snapshot.timestamp,
            Some(snapshot_id.to_string()),
        )
    }

    /// Get version history for a vector
    pub fn get_history(&self, id: &str) -> Option<Vec<&VectorVersion>> {
        self.versions.get(id).map(|v| v.iter().collect())
    }

    /// Get vector at a specific version
    pub fn get_at_version(&self, id: &str, version: u64) -> Option<&VectorVersion> {
        self.versions
            .get(id)?
            .iter()
            .find(|v| v.txn_id == version)
    }

    /// Compare two versions of a vector
    pub fn diff(&self, id: &str, v1: u64, v2: u64) -> Option<VectorDiff> {
        let ver1 = self.get_at_version(id, v1)?;
        let ver2 = self.get_at_version(id, v2)?;

        let similarity = if !ver1.is_tombstone && !ver2.is_tombstone {
            Some(cosine_similarity(&ver1.vector, &ver2.vector))
        } else {
            None
        };

        let metadata_diff = if ver1.metadata != ver2.metadata {
            Some(MetadataDiff {
                added: diff_metadata_added(&ver1.metadata, &ver2.metadata),
                removed: diff_metadata_removed(&ver1.metadata, &ver2.metadata),
                changed: diff_metadata_changed(&ver1.metadata, &ver2.metadata),
            })
        } else {
            None
        };

        Some(VectorDiff {
            id: id.to_string(),
            from_version: v1,
            to_version: v2,
            from_timestamp: ver1.created_at,
            to_timestamp: ver2.created_at,
            vector_similarity: similarity,
            metadata_diff,
            from_tombstone: ver1.is_tombstone,
            to_tombstone: ver2.is_tombstone,
        })
    }

    /// Run garbage collection
    pub fn gc(&mut self) -> Result<GcResult> {
        if !self.config.enable_gc {
            return Ok(GcResult::default());
        }

        let now = Self::now();
        let min_age = self.config.gc_min_age_seconds;
        let cutoff = now.saturating_sub(min_age);

        let mut versions_removed = 0;
        let mut tombstones_removed = 0;

        for versions in self.versions.values_mut() {
            // Keep at least the latest version and any versions newer than cutoff
            let min_keep = versions.len().saturating_sub(1);
            let mut to_remove = Vec::new();

            for (i, v) in versions.iter().enumerate() {
                if i < min_keep && v.created_at < cutoff {
                    to_remove.push(i);
                }
            }

            // Remove in reverse order to maintain indices
            for i in to_remove.into_iter().rev() {
                if let Some(v) = versions.remove(i) {
                    if v.is_tombstone {
                        tombstones_removed += 1;
                    } else {
                        versions_removed += 1;
                    }
                }
            }
        }

        // Remove empty version lists
        self.versions.retain(|_, v| !v.is_empty());

        self.stats.gc_runs += 1;
        self.stats.gc_versions_collected += versions_removed + tombstones_removed;
        self.stats.total_versions = self.versions.values().map(|v| v.len()).sum();
        self.stats.total_vectors = self.versions.len();
        self.stats.total_tombstones = self
            .versions
            .values()
            .flat_map(|v| v.iter())
            .filter(|v| v.is_tombstone)
            .count();

        Ok(GcResult {
            versions_removed,
            tombstones_removed,
            vectors_remaining: self.versions.len(),
            versions_remaining: self.stats.total_versions,
        })
    }

    /// Get statistics
    pub fn stats(&self) -> &TimeTravelStats {
        &self.stats
    }

    /// Check and maybe create auto-snapshot
    fn maybe_auto_snapshot(&mut self) -> Result<()> {
        if !self.config.auto_snapshot {
            return Ok(());
        }

        let now = Self::now();
        if now - self.last_auto_snapshot >= self.config.snapshot_interval_seconds {
            let name = format!("auto_{}", now);
            self.create_snapshot_with_options(&name, Some("Automatic snapshot".to_string()), true, Vec::new())?;
            self.last_auto_snapshot = now;
        }

        Ok(())
    }

    /// Find transaction ID at a given timestamp
    fn txn_at_timestamp(&self, timestamp: u64) -> u64 {
        // Find the highest txn_id with created_at <= timestamp
        let mut max_txn = 0;
        for versions in self.versions.values() {
            for v in versions {
                if v.created_at <= timestamp && v.txn_id > max_txn {
                    max_txn = v.txn_id;
                }
            }
        }
        max_txn
    }
}

/// Difference between two vector versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDiff {
    pub id: String,
    pub from_version: u64,
    pub to_version: u64,
    pub from_timestamp: u64,
    pub to_timestamp: u64,
    pub vector_similarity: Option<f32>,
    pub metadata_diff: Option<MetadataDiff>,
    pub from_tombstone: bool,
    pub to_tombstone: bool,
}

/// Metadata differences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataDiff {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub changed: Vec<String>,
}

/// Result of garbage collection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GcResult {
    pub versions_removed: usize,
    pub tombstones_removed: usize,
    pub vectors_remaining: usize,
    pub versions_remaining: usize,
}

// Helper functions
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

fn diff_metadata_added(
    old: &Option<serde_json::Value>,
    new: &Option<serde_json::Value>,
) -> Vec<String> {
    match (old, new) {
        (_, Some(serde_json::Value::Object(new_map))) => {
            let old_keys: std::collections::HashSet<_> = old
                .as_ref()
                .and_then(|v| v.as_object())
                .map(|m| m.keys().collect())
                .unwrap_or_default();
            new_map
                .keys()
                .filter(|k| !old_keys.contains(*k))
                .cloned()
                .collect()
        }
        _ => Vec::new(),
    }
}

fn diff_metadata_removed(
    old: &Option<serde_json::Value>,
    new: &Option<serde_json::Value>,
) -> Vec<String> {
    diff_metadata_added(new, old)
}

fn diff_metadata_changed(
    old: &Option<serde_json::Value>,
    new: &Option<serde_json::Value>,
) -> Vec<String> {
    match (old, new) {
        (Some(serde_json::Value::Object(old_map)), Some(serde_json::Value::Object(new_map))) => {
            old_map
                .keys()
                .filter(|k| {
                    new_map.contains_key(*k) && old_map.get(*k) != new_map.get(*k)
                })
                .cloned()
                .collect()
        }
        _ => Vec::new(),
    }
}

/// Builder for time-travel queries
pub struct TimeTravelQueryBuilder<'a> {
    index: &'a TimeTravelIndex,
    query: Vec<f32>,
    k: usize,
    at: Option<TimeExpression>,
    since: Option<TimeExpression>,
    until: Option<TimeExpression>,
    filter: Option<Filter>,
}

impl<'a> TimeTravelQueryBuilder<'a> {
    pub fn new(index: &'a TimeTravelIndex, query: Vec<f32>) -> Self {
        Self {
            index,
            query,
            k: 10,
            at: None,
            since: None,
            until: None,
            filter: None,
        }
    }

    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    pub fn at(mut self, time: TimeExpression) -> Self {
        self.at = Some(time);
        self
    }

    pub fn at_str(self, time: &str) -> Result<Self> {
        Ok(self.at(TimeExpression::parse(time)?))
    }

    pub fn since(mut self, time: TimeExpression) -> Self {
        self.since = Some(time);
        self
    }

    pub fn since_str(self, time: &str) -> Result<Self> {
        Ok(self.since(TimeExpression::parse(time)?))
    }

    pub fn until(mut self, time: TimeExpression) -> Self {
        self.until = Some(time);
        self
    }

    pub fn until_str(self, time: &str) -> Result<Self> {
        Ok(self.until(TimeExpression::parse(time)?))
    }

    pub fn last_hours(self, hours: u64) -> Self {
        self.since(TimeExpression::RelativeSeconds(-(hours as i64 * 3600)))
    }

    pub fn last_days(self, days: u64) -> Self {
        self.since(TimeExpression::RelativeSeconds(-(days as i64 * 86400)))
    }

    pub fn with_filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn execute(self) -> Result<Vec<TimeTravelSearchResult>> {
        if let Some(at) = self.at {
            return self.index.search_at(&self.query, self.k, at);
        }

        match (self.since, self.until) {
            (Some(since), Some(until)) => {
                return self.index.search_in_range(&self.query, self.k, since, until);
            }
            (Some(since), None) => {
                return self.index.search_since(&self.query, self.k, since);
            }
            _ => {}
        }

        // Default: search current state
        let collection = self.index.db.collection(&self.index.collection_name)?;
        let results = if let Some(filter) = self.filter {
            collection.search_with_filter(&self.query, self.k, &filter)?
        } else {
            collection.search(&self.query, self.k)?
        };

        let now = TimeTravelIndex::now();
        Ok(results
            .into_iter()
            .map(|r| TimeTravelSearchResult {
                version: self.index.current_txn,
                timestamp: now,
                query_timestamp: now,
                from_snapshot: None,
                result: r,
            })
            .collect())
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    fn create_test_index() -> TimeTravelIndex {
        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 8).unwrap();
        TimeTravelIndex::new(db, "test", MvccConfig::default())
    }

    fn random_vec(seed: u64) -> Vec<f32> {
        let mut rng = seed;
        (0..8)
            .map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                ((rng >> 16) as f32 / 32768.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn test_time_expression_parse() {
        assert!(TimeExpression::parse("now").is_ok());
        assert!(TimeExpression::parse("yesterday").is_ok());
        assert!(TimeExpression::parse("last week").is_ok());
        assert!(TimeExpression::parse("2 hours ago").is_ok());
        assert!(TimeExpression::parse("3 days ago").is_ok());
        assert!(TimeExpression::parse("last tuesday").is_ok());
        assert!(TimeExpression::parse("@my_snapshot").is_ok());
    }

    #[test]
    fn test_time_expression_resolve() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let expr = TimeExpression::parse("now").unwrap();
        let resolved = expr.resolve().unwrap();
        assert!((resolved as i64 - now as i64).abs() < 2);

        let expr = TimeExpression::parse("yesterday").unwrap();
        let resolved = expr.resolve().unwrap();
        assert!((resolved as i64 - (now as i64 - 86400)).abs() < 2);

        let expr = TimeExpression::parse("2 hours ago").unwrap();
        let resolved = expr.resolve().unwrap();
        assert!((resolved as i64 - (now as i64 - 7200)).abs() < 2);
    }

    #[test]
    fn test_insert_creates_version() {
        let mut index = create_test_index();
        let vector = random_vec(1);

        let txn = index.insert("doc1", &vector, None).unwrap();
        assert_eq!(txn, 1);

        let history = index.get_history("doc1").unwrap();
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_update_creates_new_version() {
        let mut index = create_test_index();
        let v1 = random_vec(1);
        let v2 = random_vec(2);

        index.insert("doc1", &v1, None).unwrap();
        index.update("doc1", &v2, None).unwrap();

        let history = index.get_history("doc1").unwrap();
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_delete_creates_tombstone() {
        let mut index = create_test_index();
        let vector = random_vec(1);

        index.insert("doc1", &vector, None).unwrap();
        index.delete("doc1").unwrap();

        let history = index.get_history("doc1").unwrap();
        assert_eq!(history.len(), 2);
        assert!(history.last().unwrap().is_tombstone);
    }

    #[test]
    fn test_snapshot_creation() {
        let mut index = create_test_index();

        index.insert("doc1", &random_vec(1), None).unwrap();
        let snap_id = index.create_snapshot("test-snap").unwrap();

        let snapshot = index.get_snapshot(&snap_id).unwrap();
        assert_eq!(snapshot.name, Some("test-snap".to_string()));
    }

    #[test]
    fn test_diff_versions() {
        let mut index = create_test_index();
        let v1 = random_vec(1);
        let v2 = random_vec(2);

        let txn1 = index.insert("doc1", &v1, None).unwrap();
        let txn2 = index.update("doc1", &v2, None).unwrap();

        let diff = index.diff("doc1", txn1, txn2).unwrap();
        assert!(diff.vector_similarity.is_some());
        assert!(diff.vector_similarity.unwrap() < 1.0); // Different vectors
    }

    #[test]
    fn test_gc() {
        let mut config = MvccConfig::default();
        config.gc_min_age_seconds = 0; // GC immediately

        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 8).unwrap();
        let mut index = TimeTravelIndex::new(db, "test", config);

        // Insert many versions
        for i in 0..10 {
            index.insert("doc1", &random_vec(i), None).unwrap();
        }

        let result = index.gc().unwrap();
        // Should have cleaned up old versions
        assert!(result.versions_removed > 0 || index.get_history("doc1").unwrap().len() <= 10);
    }

    #[test]
    fn test_query_builder() {
        let mut index = create_test_index();

        index.insert("doc1", &random_vec(1), None).unwrap();
        index.insert("doc2", &random_vec(2), None).unwrap();

        let query = random_vec(1);
        let results = TimeTravelQueryBuilder::new(&index, query)
            .k(5)
            .execute()
            .unwrap();

        assert!(!results.is_empty());
    }
}
