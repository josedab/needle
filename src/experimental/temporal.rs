//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Temporal Search
//!
//! Time-aware vector search with:
//! - Time-decay scoring
//! - Version history
//! - Point-in-time queries
//! - Temporal range filters
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::temporal::{TemporalIndex, TimeDecayConfig};
//!
//! let index = TemporalIndex::new(collection, TimeDecayConfig::default());
//!
//! // Insert with timestamp
//! index.insert("doc1", &vector, timestamp, metadata)?;
//!
//! // Search with time decay (newer = higher score)
//! let results = index.search_with_decay(&query, 10, DecayFunction::Exponential)?;
//!
//! // Point-in-time query (as of specific timestamp)
//! let historical = index.search_at(&query, 10, past_timestamp)?;
//! ```

use crate::database::Database;
use crate::error::{NeedleError, Result};
use crate::metadata::Filter;
use crate::SearchResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Time decay function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecayFunction {
    /// Linear decay: score *= 1 - (age / max_age)
    Linear { max_age_seconds: u64 },
    /// Exponential decay: score *= exp(-lambda * age)
    Exponential { half_life_seconds: u64 },
    /// Gaussian decay: score *= exp(-(age/scale)^2)
    Gaussian { scale_seconds: u64 },
    /// Step function: full score within window, zero outside
    Step { window_seconds: u64 },
    /// No decay
    None,
}

impl Default for DecayFunction {
    fn default() -> Self {
        Self::Exponential {
            half_life_seconds: 604800, // 1 week
        }
    }
}

/// Configuration for temporal index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    /// Time decay function
    pub decay: DecayFunction,
    /// Keep version history
    pub keep_history: bool,
    /// Maximum versions per vector
    pub max_versions: usize,
    /// Timestamp field name in metadata
    pub timestamp_field: String,
    /// Enable point-in-time queries
    pub enable_pit: bool,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            decay: DecayFunction::default(),
            keep_history: true,
            max_versions: 100,
            timestamp_field: "timestamp".to_string(),
            enable_pit: true,
        }
    }
}

/// Vector version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorVersion {
    /// Version number (auto-incrementing)
    pub version: u64,
    /// Timestamp of this version
    pub timestamp: u64,
    /// Vector data
    pub vector: Vec<f32>,
    /// Metadata at this version
    pub metadata: Option<serde_json::Value>,
    /// Change type
    pub change_type: ChangeType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChangeType {
    Insert,
    Update,
    Delete,
}

/// Temporal search result
#[derive(Debug, Clone)]
pub struct TemporalSearchResult {
    /// Base search result
    pub result: SearchResult,
    /// Timestamp of the vector
    pub timestamp: u64,
    /// Age in seconds
    pub age_seconds: u64,
    /// Original score (without decay)
    pub original_score: f32,
    /// Score after decay
    pub decayed_score: f32,
    /// Decay factor applied
    pub decay_factor: f32,
}

/// Temporal index for time-aware search
pub struct TemporalIndex {
    db: Arc<Database>,
    collection_name: String,
    config: TemporalConfig,
    /// Version history: id -> versions (ordered by timestamp)
    versions: HashMap<String, Vec<VectorVersion>>,
    /// Timestamp index: timestamp -> vector ids
    time_index: BTreeMap<u64, Vec<String>>,
    /// Current version counter
    version_counter: u64,
}

impl TemporalIndex {
    /// Create a new temporal index
    pub fn new(db: Arc<Database>, collection_name: &str, config: TemporalConfig) -> Self {
        Self {
            db,
            collection_name: collection_name.to_string(),
            config,
            versions: HashMap::new(),
            time_index: BTreeMap::new(),
            version_counter: 0,
        }
    }

    /// Insert a vector with timestamp
    pub fn insert(
        &mut self,
        id: &str,
        vector: &[f32],
        timestamp: u64,
        metadata: Option<serde_json::Value>,
    ) -> Result<u64> {
        let collection = self.db.collection(&self.collection_name)?;

        // Add timestamp to metadata
        let mut meta = metadata.unwrap_or(serde_json::json!({}));
        if let serde_json::Value::Object(ref mut map) = meta {
            map.insert(self.config.timestamp_field.clone(), serde_json::json!(timestamp));
        }

        // Insert into collection
        collection.insert(id, vector, Some(meta.clone()))?;

        // Record version
        self.version_counter += 1;
        let version = VectorVersion {
            version: self.version_counter,
            timestamp,
            vector: vector.to_vec(),
            metadata: Some(meta),
            change_type: ChangeType::Insert,
        };

        // Store version history
        self.versions
            .entry(id.to_string())
            .or_default()
            .push(version);

        // Prune old versions if needed
        if let Some(versions) = self.versions.get_mut(id) {
            if versions.len() > self.config.max_versions {
                versions.remove(0);
            }
        }

        // Update time index
        self.time_index
            .entry(timestamp)
            .or_default()
            .push(id.to_string());

        Ok(self.version_counter)
    }

    /// Update a vector with new timestamp
    pub fn update(
        &mut self,
        id: &str,
        vector: &[f32],
        timestamp: u64,
        metadata: Option<serde_json::Value>,
    ) -> Result<u64> {
        // Check if exists
        if !self.versions.contains_key(id) {
            return Err(NeedleError::VectorNotFound(id.to_string()));
        }

        let collection = self.db.collection(&self.collection_name)?;

        // Add timestamp to metadata
        let mut meta = metadata.unwrap_or(serde_json::json!({}));
        if let serde_json::Value::Object(ref mut map) = meta {
            map.insert(self.config.timestamp_field.clone(), serde_json::json!(timestamp));
        }

        // Update in collection (delete and re-insert)
        let _ = collection.delete(id); // Ignore if doesn't exist
        collection.insert(id, vector, Some(meta.clone()))?;

        // Record version
        self.version_counter += 1;
        let version = VectorVersion {
            version: self.version_counter,
            timestamp,
            vector: vector.to_vec(),
            metadata: Some(meta),
            change_type: ChangeType::Update,
        };

        self.versions.get_mut(id).expect("version history exists after contains_key check").push(version);

        // Prune old versions
        if let Some(versions) = self.versions.get_mut(id) {
            if versions.len() > self.config.max_versions {
                versions.remove(0);
            }
        }

        // Update time index
        self.time_index
            .entry(timestamp)
            .or_default()
            .push(id.to_string());

        Ok(self.version_counter)
    }

    /// Search with time decay
    pub fn search_with_decay(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<TemporalSearchResult>> {
        self.search_with_decay_and_filter(query, k, None)
    }

    /// Search with time decay and filter
    pub fn search_with_decay_and_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<TemporalSearchResult>> {
        let collection = self.db.collection(&self.collection_name)?;

        // Fetch more candidates to account for decay reordering
        let candidates = k * 3;

        let results = if let Some(f) = filter {
            collection.search_with_filter(query, candidates, f)?
        } else {
            collection.search(query, candidates)?
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before UNIX epoch")
            .as_secs();

        // Apply decay and convert to temporal results
        let mut temporal_results: Vec<TemporalSearchResult> = results
            .into_iter()
            .filter_map(|result| {
                let timestamp = self.extract_timestamp(&result)?;
                let age_seconds = now.saturating_sub(timestamp);
                let decay_factor = self.calculate_decay(age_seconds);
                let original_score = 1.0 - result.distance; // Convert distance to similarity
                let decayed_score = original_score * decay_factor;

                Some(TemporalSearchResult {
                    result,
                    timestamp,
                    age_seconds,
                    original_score,
                    decayed_score,
                    decay_factor,
                })
            })
            .collect();

        // Sort by decayed score
        temporal_results.sort_by(|a, b| {
            b.decayed_score
                .partial_cmp(&a.decayed_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top k
        temporal_results.truncate(k);

        Ok(temporal_results)
    }

    /// Point-in-time search (as of specific timestamp)
    pub fn search_at(
        &self,
        query: &[f32],
        k: usize,
        as_of_timestamp: u64,
    ) -> Result<Vec<SearchResult>> {
        if !self.config.enable_pit {
            return Err(NeedleError::InvalidConfig(
                "Point-in-time queries not enabled".to_string(),
            ));
        }

        // Get vector ids that existed at that timestamp
        let valid_ids: Vec<&String> = self
            .versions
            .iter()
            .filter_map(|(id, versions)| {
                // Find the version that was current at as_of_timestamp
                let valid_version = versions
                    .iter()
                    .rfind(|v| v.timestamp <= as_of_timestamp && v.change_type != ChangeType::Delete);

                valid_version.map(|_| id)
            })
            .collect();

        if valid_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Create filter for valid ids
        let id_filter = Filter::is_in(
            "id".to_string(),
            valid_ids.iter().map(|id| serde_json::json!(id.as_str())).collect(),
        );

        let collection = self.db.collection(&self.collection_name)?;
        let results = collection.search_with_filter(query, k, &id_filter)?;

        Ok(results)
    }

    /// Search within time range
    pub fn search_in_range(
        &self,
        query: &[f32],
        k: usize,
        start_timestamp: u64,
        end_timestamp: u64,
    ) -> Result<Vec<TemporalSearchResult>> {
        // Create temporal filter
        let filter = Filter::And(vec![
            Filter::gte(self.config.timestamp_field.clone(), serde_json::json!(start_timestamp)),
            Filter::lte(self.config.timestamp_field.clone(), serde_json::json!(end_timestamp)),
        ]);

        self.search_with_decay_and_filter(query, k, Some(&filter))
    }

    /// Get version history for a vector
    pub fn get_history(&self, id: &str) -> Option<&Vec<VectorVersion>> {
        self.versions.get(id)
    }

    /// Get vector at specific version
    pub fn get_at_version(&self, id: &str, version: u64) -> Option<&VectorVersion> {
        self.versions
            .get(id)?
            .iter()
            .find(|v| v.version == version)
    }

    /// Get latest version of a vector
    pub fn get_latest(&self, id: &str) -> Option<&VectorVersion> {
        self.versions.get(id)?.last()
    }

    /// Compare two versions of a vector
    pub fn diff_versions(&self, id: &str, v1: u64, v2: u64) -> Option<VersionDiff> {
        let ver1 = self.get_at_version(id, v1)?;
        let ver2 = self.get_at_version(id, v2)?;

        // Calculate vector similarity
        let similarity = cosine_similarity(&ver1.vector, &ver2.vector);

        // Calculate metadata changes
        let metadata_changed = ver1.metadata != ver2.metadata;

        Some(VersionDiff {
            id: id.to_string(),
            from_version: v1,
            to_version: v2,
            vector_similarity: similarity,
            metadata_changed,
            time_diff_seconds: ver2.timestamp.saturating_sub(ver1.timestamp),
        })
    }

    /// Get vectors added/updated in time range
    pub fn get_changes_in_range(
        &self,
        start: u64,
        end: u64,
    ) -> Vec<(String, &VectorVersion)> {
        let mut changes = Vec::new();

        for (id, versions) in &self.versions {
            for version in versions {
                if version.timestamp >= start && version.timestamp <= end {
                    changes.push((id.clone(), version));
                }
            }
        }

        // Sort by timestamp
        changes.sort_by_key(|(_, v)| v.timestamp);

        changes
    }

    /// Calculate decay factor for age
    fn calculate_decay(&self, age_seconds: u64) -> f32 {
        match &self.config.decay {
            DecayFunction::None => 1.0,
            DecayFunction::Linear { max_age_seconds } => {
                if age_seconds >= *max_age_seconds {
                    0.0
                } else {
                    1.0 - (age_seconds as f32 / *max_age_seconds as f32)
                }
            }
            DecayFunction::Exponential { half_life_seconds } => {
                let lambda = (2.0_f32).ln() / *half_life_seconds as f32;
                (-lambda * age_seconds as f32).exp()
            }
            DecayFunction::Gaussian { scale_seconds } => {
                let x = age_seconds as f32 / *scale_seconds as f32;
                (-x * x).exp()
            }
            DecayFunction::Step { window_seconds } => {
                if age_seconds <= *window_seconds {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    fn extract_timestamp(&self, result: &SearchResult) -> Option<u64> {
        result
            .metadata
            .as_ref()?
            .get(&self.config.timestamp_field)?
            .as_u64()
    }

    /// Get statistics about temporal data
    pub fn stats(&self) -> TemporalStats {
        let total_vectors = self.versions.len();
        let total_versions: usize = self.versions.values().map(|v| v.len()).sum();

        let timestamps: Vec<u64> = self
            .versions
            .values()
            .filter_map(|v| v.last().map(|ver| ver.timestamp))
            .collect();

        let (oldest, newest) = if timestamps.is_empty() {
            (0, 0)
        } else {
            (
                *timestamps.iter().min().expect("timestamps is non-empty"),
                *timestamps.iter().max().expect("timestamps is non-empty"),
            )
        };

        TemporalStats {
            total_vectors,
            total_versions,
            avg_versions_per_vector: if total_vectors > 0 {
                total_versions as f64 / total_vectors as f64
            } else {
                0.0
            },
            oldest_timestamp: oldest,
            newest_timestamp: newest,
            time_span_seconds: newest.saturating_sub(oldest),
        }
    }
}

/// Version difference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    pub id: String,
    pub from_version: u64,
    pub to_version: u64,
    pub vector_similarity: f32,
    pub metadata_changed: bool,
    pub time_diff_seconds: u64,
}

/// Temporal statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalStats {
    pub total_vectors: usize,
    pub total_versions: usize,
    pub avg_versions_per_vector: f64,
    pub oldest_timestamp: u64,
    pub newest_timestamp: u64,
    pub time_span_seconds: u64,
}

/// Helper function for cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Builder for temporal queries
pub struct TemporalQueryBuilder<'a> {
    index: &'a TemporalIndex,
    query: Vec<f32>,
    k: usize,
    decay: Option<DecayFunction>,
    time_range: Option<(u64, u64)>,
    filter: Option<Filter>,
    as_of: Option<u64>,
}

impl<'a> TemporalQueryBuilder<'a> {
    pub fn new(index: &'a TemporalIndex, query: Vec<f32>) -> Self {
        Self {
            index,
            query,
            k: 10,
            decay: None,
            time_range: None,
            filter: None,
            as_of: None,
        }
    }

    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    pub fn with_decay(mut self, decay: DecayFunction) -> Self {
        self.decay = Some(decay);
        self
    }

    pub fn in_range(mut self, start: u64, end: u64) -> Self {
        self.time_range = Some((start, end));
        self
    }

    pub fn last_hours(self, hours: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before UNIX epoch")
            .as_secs();
        self.in_range(now - hours * 3600, now)
    }

    pub fn last_days(self, days: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before UNIX epoch")
            .as_secs();
        self.in_range(now - days * 86400, now)
    }

    pub fn as_of(mut self, timestamp: u64) -> Self {
        self.as_of = Some(timestamp);
        self
    }

    pub fn with_filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn execute(self) -> Result<Vec<TemporalSearchResult>> {
        if let Some(timestamp) = self.as_of {
            // Point-in-time query
            let results = self.index.search_at(&self.query, self.k, timestamp)?;
            return Ok(results
                .into_iter()
                .map(|r| TemporalSearchResult {
                    result: r,
                    timestamp,
                    age_seconds: 0,
                    original_score: 1.0,
                    decayed_score: 1.0,
                    decay_factor: 1.0,
                })
                .collect());
        }

        if let Some((start, end)) = self.time_range {
            return self.index.search_in_range(&self.query, self.k, start, end);
        }

        self.index.search_with_decay_and_filter(&self.query, self.k, self.filter.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_index() -> TemporalIndex {
        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 8).unwrap();
        TemporalIndex::new(db, "test", TemporalConfig::default())
    }

    #[test]
    fn test_insert_with_timestamp() {
        let mut index = create_test_index();
        let vector = vec![1.0; 8];

        let version = index.insert("doc1", &vector, 1000, None).unwrap();
        assert_eq!(version, 1);

        let history = index.get_history("doc1").unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].timestamp, 1000);
    }

    #[test]
    fn test_update_creates_version() {
        let mut index = create_test_index();
        let vector1 = vec![1.0; 8];
        let vector2 = vec![2.0; 8];

        index.insert("doc1", &vector1, 1000, None).unwrap();
        index.update("doc1", &vector2, 2000, None).unwrap();

        let history = index.get_history("doc1").unwrap();
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_decay_calculation() {
        let index = create_test_index();

        // Exponential decay with 1 week half-life
        let decay_at_0 = index.calculate_decay(0);
        assert!((decay_at_0 - 1.0).abs() < 0.001);

        let decay_at_halflife = index.calculate_decay(604800);
        assert!((decay_at_halflife - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_linear_decay() {
        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 8).unwrap();

        let config = TemporalConfig {
            decay: DecayFunction::Linear { max_age_seconds: 100 },
            ..Default::default()
        };

        let index = TemporalIndex::new(db, "test", config);

        assert!((index.calculate_decay(0) - 1.0).abs() < 0.001);
        assert!((index.calculate_decay(50) - 0.5).abs() < 0.001);
        assert!((index.calculate_decay(100) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_version_diff() {
        let mut index = create_test_index();

        let vector1 = vec![1.0; 8];
        let vector2 = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        index.insert("doc1", &vector1, 1000, None).unwrap();
        index.update("doc1", &vector2, 2000, None).unwrap();

        let diff = index.diff_versions("doc1", 1, 2).unwrap();
        assert!(diff.vector_similarity < 1.0);
        assert_eq!(diff.time_diff_seconds, 1000);
    }

    #[test]
    fn test_temporal_stats() {
        let mut index = create_test_index();

        for i in 0..10 {
            let vector = vec![i as f32; 8];
            index.insert(&format!("doc{}", i), &vector, 1000 + i * 100, None).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.total_vectors, 10);
        assert_eq!(stats.total_versions, 10);
        assert_eq!(stats.oldest_timestamp, 1000);
        assert_eq!(stats.newest_timestamp, 1900);
    }

    #[test]
    fn test_changes_in_range() {
        let mut index = create_test_index();

        index.insert("doc1", &[1.0; 8], 1000, None).unwrap();
        index.insert("doc2", &[2.0; 8], 2000, None).unwrap();
        index.insert("doc3", &[3.0; 8], 3000, None).unwrap();

        let changes = index.get_changes_in_range(1500, 2500);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].0, "doc2");
    }
}
