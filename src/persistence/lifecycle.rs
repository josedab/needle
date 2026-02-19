//! Vector Lifecycle Policies
//!
//! Configurable per-collection policies for automatic vector lifecycle management:
//! TTL-based expiration, age-based archival, and retention windows.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Actions that can be taken on vectors matching a lifecycle rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleAction {
    /// Delete the vector permanently.
    Delete,
    /// Move to archive (cold/external storage).
    Archive,
    /// Compress the vector data in-place.
    Compress,
    /// No action (rule is informational only).
    None,
}

/// A single lifecycle rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleRule {
    /// Rule name for identification.
    pub name: String,
    /// Action to take when the rule matches.
    pub action: LifecycleAction,
    /// Age in seconds after which this rule applies.
    pub age_seconds: u64,
    /// Optional: only apply to vectors with matching metadata keys.
    pub metadata_filter: Option<HashMap<String, String>>,
    /// Whether this rule is enabled.
    pub enabled: bool,
}

impl LifecycleRule {
    /// Create a TTL rule (delete after N days).
    pub fn ttl_days(name: &str, days: u64) -> Self {
        Self {
            name: name.to_string(),
            action: LifecycleAction::Delete,
            age_seconds: days * 86400,
            metadata_filter: None,
            enabled: true,
        }
    }

    /// Create an archive rule (archive after N days).
    pub fn archive_days(name: &str, days: u64) -> Self {
        Self {
            name: name.to_string(),
            action: LifecycleAction::Archive,
            age_seconds: days * 86400,
            metadata_filter: None,
            enabled: true,
        }
    }

    /// Create a compress rule.
    pub fn compress_days(name: &str, days: u64) -> Self {
        Self {
            name: name.to_string(),
            action: LifecycleAction::Compress,
            age_seconds: days * 86400,
            metadata_filter: None,
            enabled: true,
        }
    }
}

/// Per-collection lifecycle policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecyclePolicy {
    /// Policy name.
    pub name: String,
    /// Collection this policy applies to (empty = all collections).
    pub collection: Option<String>,
    /// Ordered list of rules (evaluated in order, first match wins).
    pub rules: Vec<LifecycleRule>,
    /// Whether this policy is enabled.
    pub enabled: bool,
}

impl LifecyclePolicy {
    /// Create a new empty policy.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            collection: None,
            rules: Vec::new(),
            enabled: true,
        }
    }

    /// Bind to a specific collection.
    #[must_use]
    pub fn for_collection(mut self, collection: &str) -> Self {
        self.collection = Some(collection.to_string());
        self
    }

    /// Add a rule.
    #[must_use]
    pub fn with_rule(mut self, rule: LifecycleRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Create a standard TTL-only policy (delete after N days).
    pub fn ttl_only(name: &str, days: u64) -> Self {
        Self::new(name).with_rule(LifecycleRule::ttl_days("ttl", days))
    }

    /// Create a tiered policy: compress after N days, archive after M days, delete after P days.
    pub fn tiered(name: &str, compress_days: u64, archive_days: u64, delete_days: u64) -> Self {
        Self::new(name)
            .with_rule(LifecycleRule::compress_days("compress", compress_days))
            .with_rule(LifecycleRule::archive_days("archive", archive_days))
            .with_rule(LifecycleRule::ttl_days("delete", delete_days))
    }

    /// Evaluate which action should be taken for a vector with the given age.
    pub fn evaluate(
        &self,
        age_seconds: u64,
        metadata: Option<&HashMap<String, String>>,
    ) -> LifecycleAction {
        if !self.enabled {
            return LifecycleAction::None;
        }

        // Rules are ordered by age (ascending). Find the first rule whose
        // age threshold is exceeded.
        let mut matched_action = LifecycleAction::None;
        for rule in &self.rules {
            if !rule.enabled || age_seconds < rule.age_seconds {
                continue;
            }
            // Check metadata filter
            if let Some(ref filter) = rule.metadata_filter {
                if let Some(meta) = metadata {
                    if !filter.iter().all(|(k, v)| meta.get(k) == Some(v)) {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            matched_action = rule.action;
        }
        matched_action
    }
}

/// Manages lifecycle policies and evaluates vectors against them.
pub struct LifecyclePolicyEngine {
    /// Registered policies.
    policies: Vec<LifecyclePolicy>,
    /// Total evaluations performed.
    evaluations: u64,
    /// Actions taken by type.
    actions_taken: HashMap<String, u64>,
}

impl LifecyclePolicyEngine {
    /// Create a new policy engine.
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
            evaluations: 0,
            actions_taken: HashMap::new(),
        }
    }

    /// Add a policy.
    pub fn add_policy(&mut self, policy: LifecyclePolicy) {
        self.policies.push(policy);
    }

    /// Remove a policy by name.
    pub fn remove_policy(&mut self, name: &str) -> bool {
        let before = self.policies.len();
        self.policies.retain(|p| p.name != name);
        self.policies.len() < before
    }

    /// Evaluate a vector against all applicable policies.
    pub fn evaluate(
        &mut self,
        collection: &str,
        created_at_secs: u64,
        metadata: Option<&HashMap<String, String>>,
    ) -> LifecycleAction {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let age = now.saturating_sub(created_at_secs);

        self.evaluations += 1;

        let mut result = LifecycleAction::None;
        for policy in &self.policies {
            if !policy.enabled {
                continue;
            }
            if let Some(ref col) = policy.collection {
                if col != collection {
                    continue;
                }
            }
            let action = policy.evaluate(age, metadata);
            if action != LifecycleAction::None {
                result = action;
                break;
            }
        }

        if result != LifecycleAction::None {
            let key = format!("{result:?}");
            *self.actions_taken.entry(key).or_default() += 1;
        }

        result
    }

    /// List all policies.
    pub fn policies(&self) -> &[LifecyclePolicy] {
        &self.policies
    }

    /// Get statistics.
    pub fn stats(&self) -> LifecycleStats {
        LifecycleStats {
            policy_count: self.policies.len(),
            evaluations: self.evaluations,
            actions_taken: self.actions_taken.clone(),
            expired_count: *self.actions_taken.get("Delete").unwrap_or(&0),
            archived_count: *self.actions_taken.get("Archive").unwrap_or(&0),
            compressed_count: *self.actions_taken.get("Compress").unwrap_or(&0),
        }
    }
}

impl Default for LifecyclePolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Lifecycle engine statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleStats {
    /// Number of registered policies.
    pub policy_count: usize,
    /// Total evaluations.
    pub evaluations: u64,
    /// Actions taken by type.
    pub actions_taken: HashMap<String, u64>,
    /// Vectors expired (deleted by TTL).
    pub expired_count: u64,
    /// Vectors archived.
    pub archived_count: u64,
    /// Vectors compressed.
    pub compressed_count: u64,
}

/// Result of running lifecycle maintenance on a set of vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleMaintenanceResult {
    /// Vectors evaluated.
    pub evaluated: usize,
    /// Vectors deleted by TTL.
    pub deleted: usize,
    /// Vectors marked for archival.
    pub archived: usize,
    /// Vectors compressed.
    pub compressed: usize,
    /// Vectors with no action needed.
    pub skipped: usize,
    /// IDs of vectors that should be deleted.
    pub delete_ids: Vec<String>,
    /// IDs of vectors that should be archived.
    pub archive_ids: Vec<String>,
}

impl LifecyclePolicyEngine {
    /// Evaluate a batch of vectors and return maintenance actions.
    ///
    /// Does not modify any data — returns a `LifecycleMaintenanceResult`
    /// that the caller uses to apply the actions.
    pub fn evaluate_batch(
        &mut self,
        collection: &str,
        vectors: &[(String, u64, Option<HashMap<String, String>>)],
    ) -> LifecycleMaintenanceResult {
        let mut result = LifecycleMaintenanceResult {
            evaluated: vectors.len(),
            deleted: 0,
            archived: 0,
            compressed: 0,
            skipped: 0,
            delete_ids: Vec::new(),
            archive_ids: Vec::new(),
        };

        for (id, created_at, meta) in vectors {
            let action = self.evaluate(collection, *created_at, meta.as_ref());
            match action {
                LifecycleAction::Delete => {
                    result.deleted += 1;
                    result.delete_ids.push(id.clone());
                }
                LifecycleAction::Archive => {
                    result.archived += 1;
                    result.archive_ids.push(id.clone());
                }
                LifecycleAction::Compress => {
                    result.compressed += 1;
                }
                LifecycleAction::None => {
                    result.skipped += 1;
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ttl_policy() {
        let policy = LifecyclePolicy::ttl_only("test-ttl", 30);
        // Vector is 31 days old → should be deleted
        let action = policy.evaluate(31 * 86400, None);
        assert_eq!(action, LifecycleAction::Delete);
        // Vector is 1 day old → no action
        let action = policy.evaluate(86400, None);
        assert_eq!(action, LifecycleAction::None);
    }

    #[test]
    fn test_tiered_policy() {
        let policy = LifecyclePolicy::tiered("test-tiered", 7, 30, 90);
        // 5 days → no action
        assert_eq!(policy.evaluate(5 * 86400, None), LifecycleAction::None);
        // 10 days → compress
        assert_eq!(
            policy.evaluate(10 * 86400, None),
            LifecycleAction::Compress
        );
        // 45 days → archive (overrides compress)
        assert_eq!(
            policy.evaluate(45 * 86400, None),
            LifecycleAction::Archive
        );
        // 100 days → delete (overrides archive)
        assert_eq!(
            policy.evaluate(100 * 86400, None),
            LifecycleAction::Delete
        );
    }

    #[test]
    fn test_metadata_filter() {
        let mut rule = LifecycleRule::ttl_days("filtered", 1);
        let mut filter = HashMap::new();
        filter.insert("tier".to_string(), "temp".to_string());
        rule.metadata_filter = Some(filter);

        let policy = LifecyclePolicy::new("test").with_rule(rule);

        // No metadata → no match
        assert_eq!(policy.evaluate(2 * 86400, None), LifecycleAction::None);

        // Wrong metadata → no match
        let mut meta = HashMap::new();
        meta.insert("tier".to_string(), "permanent".to_string());
        assert_eq!(
            policy.evaluate(2 * 86400, Some(&meta)),
            LifecycleAction::None
        );

        // Matching metadata → delete
        let mut meta = HashMap::new();
        meta.insert("tier".to_string(), "temp".to_string());
        assert_eq!(
            policy.evaluate(2 * 86400, Some(&meta)),
            LifecycleAction::Delete
        );
    }

    #[test]
    fn test_engine_evaluate() {
        let mut engine = LifecyclePolicyEngine::new();
        engine.add_policy(LifecyclePolicy::ttl_only("docs-ttl", 30).for_collection("docs"));

        // Very old vector in "docs" collection
        let action = engine.evaluate("docs", 0, None);
        assert_eq!(action, LifecycleAction::Delete);

        // Same age in different collection → no action
        let action = engine.evaluate("other", 0, None);
        assert_eq!(action, LifecycleAction::None);
    }

    #[test]
    fn test_engine_stats() {
        let mut engine = LifecyclePolicyEngine::new();
        engine.add_policy(LifecyclePolicy::ttl_only("test", 1));
        engine.evaluate("col", 0, None);
        engine.evaluate("col", 0, None);

        let stats = engine.stats();
        assert_eq!(stats.evaluations, 2);
        assert_eq!(stats.policy_count, 1);
    }

    #[test]
    fn test_disabled_policy() {
        let mut policy = LifecyclePolicy::ttl_only("disabled", 1);
        policy.enabled = false;
        assert_eq!(policy.evaluate(2 * 86400, None), LifecycleAction::None);
    }

    #[test]
    fn test_batch_evaluate() {
        let mut engine = LifecyclePolicyEngine::new();
        engine.add_policy(LifecyclePolicy::ttl_only("ttl-30d", 30).for_collection("docs"));

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let vectors = vec![
            ("old-1".to_string(), now - 40 * 86400, None),   // 40 days old → delete
            ("old-2".to_string(), now - 35 * 86400, None),   // 35 days old → delete
            ("recent".to_string(), now - 5 * 86400, None),   // 5 days old → skip
        ];

        let result = engine.evaluate_batch("docs", &vectors);
        assert_eq!(result.evaluated, 3);
        assert_eq!(result.deleted, 2);
        assert_eq!(result.skipped, 1);
        assert_eq!(result.delete_ids.len(), 2);
        assert!(result.delete_ids.contains(&"old-1".to_string()));
        assert!(result.delete_ids.contains(&"old-2".to_string()));
    }
}
