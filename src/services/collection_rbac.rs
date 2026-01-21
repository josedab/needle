//! Collection-Level RBAC Policies
//!
//! Declarative JSON policies per-collection with field-level filtering,
//! row-level security based on metadata predicates, and token-scoped access.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::collection_rbac::{
//!     RbacPolicy, PolicyRule, Permission, PolicyEngine, AccessToken,
//! };
//!
//! let mut engine = PolicyEngine::new();
//!
//! // Create a policy: "readers" can only read, filtered by department
//! engine.add_policy(RbacPolicy::new("docs")
//!     .add_rule(PolicyRule::new("readers")
//!         .allow(Permission::Read)
//!         .with_row_filter("department", "engineering")));
//!
//! // Check access
//! let token = AccessToken::new("user1", vec!["readers".into()]);
//! assert!(engine.can_read("docs", &token));
//! assert!(!engine.can_write("docs", &token));
//! ```

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{NeedleError, Result};

// ── Permissions ──────────────────────────────────────────────────────────────

/// Access permission types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    /// Read vectors and metadata.
    Read,
    /// Insert new vectors.
    Write,
    /// Update existing vectors.
    Update,
    /// Delete vectors.
    Delete,
    /// Search the collection.
    Search,
    /// Compact or maintain the collection.
    Admin,
}

impl std::fmt::Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "read"),
            Self::Write => write!(f, "write"),
            Self::Update => write!(f, "update"),
            Self::Delete => write!(f, "delete"),
            Self::Search => write!(f, "search"),
            Self::Admin => write!(f, "admin"),
        }
    }
}

// ── Row Filter ───────────────────────────────────────────────────────────────

/// A metadata-based row filter for row-level security.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowFilter {
    /// Metadata field to filter on.
    pub field: String,
    /// Required value.
    pub value: Value,
}

impl RowFilter {
    /// Check if a metadata value matches this filter.
    pub fn matches(&self, metadata: &Value) -> bool {
        metadata.get(&self.field).map_or(false, |v| v == &self.value)
    }
}

// ── Policy Rule ──────────────────────────────────────────────────────────────

/// A rule within a policy granting permissions to a role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    /// Role name this rule applies to.
    pub role: String,
    /// Granted permissions.
    pub permissions: Vec<Permission>,
    /// Row-level filters (all must match for access).
    pub row_filters: Vec<RowFilter>,
    /// Field-level read restrictions (if non-empty, only these fields visible).
    pub visible_fields: Vec<String>,
}

impl PolicyRule {
    /// Create a new rule for a role.
    pub fn new(role: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            permissions: Vec::new(),
            row_filters: Vec::new(),
            visible_fields: Vec::new(),
        }
    }

    /// Grant a permission.
    #[must_use]
    pub fn allow(mut self, perm: Permission) -> Self {
        self.permissions.push(perm);
        self
    }

    /// Add a row filter.
    #[must_use]
    pub fn with_row_filter(mut self, field: &str, value: impl Into<Value>) -> Self {
        self.row_filters.push(RowFilter {
            field: field.into(),
            value: value.into(),
        });
        self
    }

    /// Restrict visible metadata fields.
    #[must_use]
    pub fn with_visible_fields(mut self, fields: Vec<String>) -> Self {
        self.visible_fields = fields;
        self
    }
}

// ── RBAC Policy ──────────────────────────────────────────────────────────────

/// A collection-level RBAC policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbacPolicy {
    /// Collection this policy applies to.
    pub collection: String,
    /// Rules in this policy.
    pub rules: Vec<PolicyRule>,
    /// Whether this policy is active.
    pub enabled: bool,
}

impl RbacPolicy {
    /// Create a new policy for a collection.
    pub fn new(collection: impl Into<String>) -> Self {
        Self {
            collection: collection.into(),
            rules: Vec::new(),
            enabled: true,
        }
    }

    /// Add a rule.
    #[must_use]
    pub fn add_rule(mut self, rule: PolicyRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Disable this policy.
    #[must_use]
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

// ── Access Token ─────────────────────────────────────────────────────────────

/// A scoped access token with roles and expiry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessToken {
    /// Token subject (user/service ID).
    pub subject: String,
    /// Assigned roles.
    pub roles: Vec<String>,
    /// Token expiry (epoch seconds).
    pub expires_at: Option<u64>,
    /// Optional metadata claims.
    pub claims: HashMap<String, Value>,
}

impl AccessToken {
    /// Create a new token.
    pub fn new(subject: impl Into<String>, roles: Vec<String>) -> Self {
        Self {
            subject: subject.into(),
            roles,
            expires_at: None,
            claims: HashMap::new(),
        }
    }

    /// Set expiry.
    #[must_use]
    pub fn with_expiry(mut self, duration: Duration) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        self.expires_at = Some(now + duration.as_secs());
        self
    }

    /// Check if the token has expired.
    pub fn is_expired(&self) -> bool {
        if let Some(exp) = self.expires_at {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
            now > exp
        } else {
            false
        }
    }

    /// Check if the token has a specific role.
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }
}

// ── Audit Entry ──────────────────────────────────────────────────────────────

/// Audit log entry for policy evaluations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Subject.
    pub subject: String,
    /// Collection.
    pub collection: String,
    /// Permission requested.
    pub permission: Permission,
    /// Whether access was granted.
    pub granted: bool,
    /// Timestamp.
    pub timestamp: u64,
    /// Matched rule (if any).
    pub matched_rule: Option<String>,
}

// ── Policy Engine ────────────────────────────────────────────────────────────

/// RBAC policy evaluation engine.
pub struct PolicyEngine {
    policies: HashMap<String, RbacPolicy>,
    audit_log: Vec<AuditEntry>,
    max_audit: usize,
}

impl PolicyEngine {
    /// Create a new policy engine.
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
            audit_log: Vec::new(),
            max_audit: 10_000,
        }
    }

    /// Add or replace a policy for a collection.
    pub fn add_policy(&mut self, policy: RbacPolicy) {
        self.policies.insert(policy.collection.clone(), policy);
    }

    /// Remove a policy.
    pub fn remove_policy(&mut self, collection: &str) -> bool {
        self.policies.remove(collection).is_some()
    }

    /// Check if a token has read access.
    pub fn can_read(&mut self, collection: &str, token: &AccessToken) -> bool {
        self.check(collection, token, Permission::Read)
    }

    /// Check if a token has write access.
    pub fn can_write(&mut self, collection: &str, token: &AccessToken) -> bool {
        self.check(collection, token, Permission::Write)
    }

    /// Check if a token has search access.
    pub fn can_search(&mut self, collection: &str, token: &AccessToken) -> bool {
        self.check(collection, token, Permission::Search)
    }

    /// Check if a token has a specific permission.
    pub fn check(&mut self, collection: &str, token: &AccessToken, perm: Permission) -> bool {
        if token.is_expired() {
            self.log_audit(token, collection, perm, false, None);
            return false;
        }

        let policy = match self.policies.get(collection) {
            Some(p) if p.enabled => p,
            Some(_) => {
                self.log_audit(token, collection, perm, true, Some("policy-disabled"));
                return true;
            }
            None => {
                self.log_audit(token, collection, perm, true, Some("no-policy"));
                return true;
            }
        };

        // Clone the matching rule name to release the borrow on self.policies
        let matched_role = policy.rules.iter()
            .find(|rule| token.has_role(&rule.role) && rule.permissions.contains(&perm))
            .map(|rule| rule.role.clone());

        if let Some(role) = matched_role {
            self.log_audit(token, collection, perm, true, Some(&role));
            true
        } else {
            self.log_audit(token, collection, perm, false, None);
            false
        }
    }

    /// Get row filters for a token on a collection.
    pub fn row_filters(&self, collection: &str, token: &AccessToken) -> Vec<&RowFilter> {
        let policy = match self.policies.get(collection) {
            Some(p) if p.enabled => p,
            _ => return Vec::new(),
        };

        let mut filters = Vec::new();
        for rule in &policy.rules {
            if token.has_role(&rule.role) {
                filters.extend(rule.row_filters.iter());
            }
        }
        filters
    }

    /// Get the audit log.
    pub fn audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }

    /// Policy count.
    pub fn policy_count(&self) -> usize {
        self.policies.len()
    }

    fn log_audit(&mut self, token: &AccessToken, collection: &str, perm: Permission, granted: bool, rule: Option<&str>) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        self.audit_log.push(AuditEntry {
            subject: token.subject.clone(),
            collection: collection.into(),
            permission: perm,
            granted,
            timestamp: now,
            matched_rule: rule.map(String::from),
        });
        if self.audit_log.len() > self.max_audit {
            self.audit_log.remove(0);
        }
    }
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_rbac() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(
            RbacPolicy::new("docs")
                .add_rule(PolicyRule::new("reader").allow(Permission::Read).allow(Permission::Search))
                .add_rule(PolicyRule::new("writer").allow(Permission::Read).allow(Permission::Write)),
        );

        let reader = AccessToken::new("user1", vec!["reader".into()]);
        assert!(engine.can_read("docs", &reader));
        assert!(engine.can_search("docs", &reader));
        assert!(!engine.can_write("docs", &reader));

        let writer = AccessToken::new("user2", vec!["writer".into()]);
        assert!(engine.can_write("docs", &writer));
    }

    #[test]
    fn test_no_policy_allows() {
        let mut engine = PolicyEngine::new();
        let token = AccessToken::new("u", vec![]);
        assert!(engine.can_read("anything", &token));
    }

    #[test]
    fn test_expired_token() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(
            RbacPolicy::new("docs")
                .add_rule(PolicyRule::new("admin").allow(Permission::Admin)),
        );
        let token = AccessToken {
            subject: "u".into(),
            roles: vec!["admin".into()],
            expires_at: Some(0), // already expired
            claims: HashMap::new(),
        };
        assert!(!engine.check("docs", &token, Permission::Admin));
    }

    #[test]
    fn test_row_filters() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(
            RbacPolicy::new("docs")
                .add_rule(PolicyRule::new("eng")
                    .allow(Permission::Read)
                    .with_row_filter("dept", "engineering")),
        );

        let token = AccessToken::new("u", vec!["eng".into()]);
        let filters = engine.row_filters("docs", &token);
        assert_eq!(filters.len(), 1);
        assert!(filters[0].matches(&serde_json::json!({"dept": "engineering"})));
        assert!(!filters[0].matches(&serde_json::json!({"dept": "sales"})));
    }

    #[test]
    fn test_audit_log() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(
            RbacPolicy::new("docs")
                .add_rule(PolicyRule::new("r").allow(Permission::Read)),
        );
        let token = AccessToken::new("u1", vec!["r".into()]);
        engine.can_read("docs", &token);
        engine.can_write("docs", &token);

        assert_eq!(engine.audit_log().len(), 2);
        assert!(engine.audit_log()[0].granted);
        assert!(!engine.audit_log()[1].granted);
    }

    #[test]
    fn test_disabled_policy() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(RbacPolicy::new("docs").disabled());
        let token = AccessToken::new("u", vec![]);
        assert!(engine.can_read("docs", &token));
    }

    #[test]
    fn test_remove_policy() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(RbacPolicy::new("docs"));
        assert!(engine.remove_policy("docs"));
        assert_eq!(engine.policy_count(), 0);
    }

    #[test]
    fn test_multiple_roles() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(
            RbacPolicy::new("docs")
                .add_rule(PolicyRule::new("reader").allow(Permission::Read))
                .add_rule(PolicyRule::new("writer").allow(Permission::Write)),
        );

        let token = AccessToken::new("u", vec!["reader".into(), "writer".into()]);
        assert!(engine.can_read("docs", &token));
        assert!(engine.can_write("docs", &token));
    }
}
