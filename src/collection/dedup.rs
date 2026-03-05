//! Semantic deduplication for collections.
//!
//! Provides near-duplicate detection during insert via 1-NN search.
//! When a new vector is within `distance_threshold` of an existing vector,
//! the configured [`DedupPolicy`](super::config::DedupPolicy) is applied.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

use super::config::DedupPolicy;
use super::Collection;
use crate::error::{NeedleError, Result};
use serde_json::Value;

// ── Counters ─────────────────────────────────────────────────────────────────

/// Global counter: inserts rejected due to dedup.
pub static DEDUP_REJECTED_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Global counter: inserts where metadata was merged into existing vector.
pub static DEDUP_MERGED_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Global counter: inserts stored as versioned copies.
pub static DEDUP_VERSIONED_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Global counter: total 1-NN dedup checks performed.
pub static DEDUP_CHECKED_TOTAL: AtomicU64 = AtomicU64::new(0);

// ── Types ────────────────────────────────────────────────────────────────────

/// Result of a dedup scan showing identified duplicates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupScanResult {
    /// Total vectors scanned.
    pub vectors_scanned: usize,
    /// Number of duplicate groups found.
    pub duplicate_groups: usize,
    /// Total duplicate vectors (excluding the "canonical" one in each group).
    pub duplicate_count: usize,
    /// Details of each duplicate group.
    pub groups: Vec<DedupGroup>,
}

/// A group of near-duplicate vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupGroup {
    /// The canonical (first/best) vector ID in the group.
    pub canonical_id: String,
    /// IDs of duplicate vectors.
    pub duplicate_ids: Vec<String>,
    /// Distances of duplicates to the canonical vector.
    pub distances: Vec<f32>,
}

/// Result of attempting to insert a vector with dedup enabled.
#[derive(Debug, Clone)]
pub enum DedupInsertResult {
    /// Vector was inserted normally (no duplicate found).
    Inserted,
    /// Vector was rejected as a duplicate.
    Rejected {
        existing_id: String,
        distance: f32,
    },
    /// Metadata was merged with existing vector.
    MetadataMerged {
        existing_id: String,
        distance: f32,
    },
    /// Vector was stored as a new version.
    Versioned {
        versioned_id: String,
        existing_id: String,
        distance: f32,
    },
}

// ── Collection integration ───────────────────────────────────────────────────

impl Collection {
    /// Check whether `vector` is a near-duplicate of an existing vector.
    ///
    /// Returns `None` if dedup is disabled or the collection is empty.
    /// Returns `Some((existing_id, distance))` if a duplicate is found.
    pub(crate) fn check_dedup(&self, vector: &[f32]) -> Option<(String, f32)> {
        let cfg = match self.config.dedup {
            Some(ref c) if c.enabled => c,
            _ => return None,
        };

        if self.len() == 0 {
            return None;
        }

        DEDUP_CHECKED_TOTAL.fetch_add(1, Ordering::Relaxed);

        // Perform 1-NN search against existing vectors
        let results = self.search(vector, 1).ok()?;
        if let Some(nearest) = results.first() {
            if nearest.distance < cfg.distance_threshold {
                return Some((nearest.id.clone(), nearest.distance));
            }
        }
        None
    }

    /// Apply the configured dedup policy given a detected duplicate.
    pub(crate) fn apply_dedup_policy(
        &mut self,
        new_id: &str,
        vector: Vec<f32>,
        metadata: Option<Value>,
        existing_id: &str,
        distance: f32,
        policy: DedupPolicy,
        ttl_seconds: Option<u64>,
    ) -> Result<DedupInsertResult> {
        match policy {
            DedupPolicy::Reject => {
                DEDUP_REJECTED_TOTAL.fetch_add(1, Ordering::Relaxed);
                Err(NeedleError::InvalidOperation(format!(
                    "Semantic dedup: vector '{}' is a near-duplicate of '{}' \
                     (distance={:.6}, threshold={})",
                    new_id,
                    existing_id,
                    distance,
                    self.config
                        .dedup
                        .as_ref()
                        .map_or(0.02, |c| c.distance_threshold)
                )))
            }
            DedupPolicy::MergeMetadata => {
                if let Some(new_meta) = metadata {
                    if let Some(internal_id) = self.metadata.get_internal_id(existing_id) {
                        if let Some(entry) = self.metadata.get(internal_id) {
                            let existing_meta =
                                entry.data.clone().unwrap_or(Value::Object(Default::default()));
                            let merged = merge_json_values(existing_meta, new_meta);
                            let _ = self.metadata.update_data(internal_id, Some(merged));
                        }
                    }
                }
                DEDUP_MERGED_TOTAL.fetch_add(1, Ordering::Relaxed);
                Ok(DedupInsertResult::MetadataMerged {
                    existing_id: existing_id.to_string(),
                    distance,
                })
            }
            DedupPolicy::Version => {
                let versioned_id = format!("{}-v{}", new_id, self.len());
                // Use the normal insert path (dedup won't re-trigger for versioned ID
                // because the vector will have a different ID)
                self.insert_vec_with_ttl(versioned_id.clone(), vector, metadata, ttl_seconds)?;
                DEDUP_VERSIONED_TOTAL.fetch_add(1, Ordering::Relaxed);
                Ok(DedupInsertResult::Versioned {
                    versioned_id,
                    existing_id: existing_id.to_string(),
                    distance,
                })
            }
        }
    }

    /// Scan the entire collection for near-duplicate groups.
    pub fn dedup_scan(&self, threshold_override: Option<f32>) -> DedupScanResult {
        let threshold = threshold_override.unwrap_or_else(|| {
            self.config
                .dedup
                .as_ref()
                .map_or(0.02, |c| c.distance_threshold)
        });

        let all_ids = self.metadata.all_external_ids();
        let n = all_ids.len();
        let mut visited = vec![false; n];
        let mut groups = Vec::new();

        for i in 0..n {
            if visited[i] {
                continue;
            }
            let vector_i = match self.get(&all_ids[i]) {
                Some((v, _)) => v.to_vec(),
                None => continue,
            };

            let k = (n / 2).clamp(10, 200);
            let results = match self.search(&vector_i, k) {
                Ok(r) => r,
                Err(_) => continue,
            };

            let mut duplicate_ids = Vec::new();
            let mut distances = Vec::new();

            for result in &results {
                if result.id == all_ids[i] {
                    continue;
                }
                if result.distance < threshold {
                    if let Some(j) = all_ids.iter().position(|id| id == &result.id) {
                        if !visited[j] {
                            visited[j] = true;
                            duplicate_ids.push(result.id.clone());
                            distances.push(result.distance);
                        }
                    }
                }
            }

            if !duplicate_ids.is_empty() {
                groups.push(DedupGroup {
                    canonical_id: all_ids[i].clone(),
                    duplicate_ids,
                    distances,
                });
            }
        }

        let duplicate_count: usize = groups.iter().map(|g| g.duplicate_ids.len()).sum();

        DedupScanResult {
            vectors_scanned: n,
            duplicate_groups: groups.len(),
            duplicate_count,
            groups,
        }
    }
}

/// Merge two JSON values, preferring fields from `new` over `existing`.
fn merge_json_values(existing: Value, new: Value) -> Value {
    match (existing, new) {
        (Value::Object(mut base), Value::Object(overlay)) => {
            for (k, v) in overlay {
                base.insert(k, v);
            }
            Value::Object(base)
        }
        (_, new) => new,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collection::config::SemanticDedupConfig;

    #[test]
    fn test_dedup_scan_no_duplicates() {
        let mut coll = Collection::with_dimensions("test", 4);
        coll.insert("a", &[1.0, 0.0, 0.0, 0.0], None).expect("ok");
        coll.insert("b", &[0.0, 1.0, 0.0, 0.0], None).expect("ok");

        let result = coll.dedup_scan(Some(0.01));
        assert_eq!(result.duplicate_count, 0);
        assert_eq!(result.vectors_scanned, 2);
    }

    #[test]
    fn test_dedup_scan_finds_duplicates() {
        let mut coll = Collection::with_dimensions("test", 4);
        coll.insert("a", &[1.0, 0.0, 0.0, 0.0], None).expect("ok");
        coll.insert("b", &[0.999, 0.001, 0.0, 0.0], None).expect("ok");
        coll.insert("c", &[0.0, 1.0, 0.0, 0.0], None).expect("ok");

        let result = coll.dedup_scan(Some(0.1));
        assert!(result.duplicate_count >= 1);
    }

    #[test]
    fn test_check_dedup_empty_collection() {
        let cfg = crate::CollectionConfig::new("test", 4)
            .with_dedup(SemanticDedupConfig::strict());
        let coll = Collection::new(cfg);
        assert!(coll.check_dedup(&[1.0, 0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn test_check_dedup_no_config() {
        let mut coll = Collection::with_dimensions("test", 4);
        coll.insert("a", &[1.0, 0.0, 0.0, 0.0], None).expect("ok");
        assert!(coll.check_dedup(&[1.0, 0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn test_merge_json_values() {
        use serde_json::json;
        let existing = json!({"a": 1, "b": 2});
        let new = json!({"b": 3, "c": 4});
        let merged = merge_json_values(existing, new);
        assert_eq!(merged, json!({"a": 1, "b": 3, "c": 4}));
    }
}
