//! Vector Access Control Lists (ACLs)
//!
//! Per-vector and per-metadata-field access control beyond collection-level RBAC.
//! Supports row-level security policies for multi-tenant SaaS applications.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐     ┌──────────────┐     ┌─────────────────┐
//! │ User/Token   │────►│ Policy Engine │────►│ Query Rewriting │
//! └──────────────┘     └──────────────┘     └─────────────────┘
//!                             │
//!                      ┌──────┴──────┐
//!                      │ Audit Trail │
//!                      └─────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::enterprise::vector_acl::*;
//!
//! let mut engine = AclEngine::new();
//!
//! // Define a tenant isolation policy
//! engine.add_policy(AclPolicy {
//!     id: "tenant-isolation".into(),
//!     principal: PrincipalMatcher::Attribute("tenant_id".into(), "acme".into()),
//!     resource: ResourceMatcher::MetadataField("tenant_id".into(), "acme".into()),
//!     effect: Effect::Allow,
//!     actions: vec![AclAction::Read, AclAction::Search],
//! });
//!
//! // Check access
//! let ctx = RequestContext::new("user1", vec![("tenant_id".into(), "acme".into())]);
//! let decision = engine.evaluate(&ctx, &AclAction::Read, "doc1", &metadata)?;
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Policy Types
// ============================================================================

/// Actions that can be controlled by ACLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AclAction {
    /// Read vector data.
    Read,
    /// Write/update vector data.
    Write,
    /// Delete vectors.
    Delete,
    /// Search (vector appears in results).
    Search,
    /// Read specific metadata fields.
    ReadMetadata,
}

/// Effect of an ACL policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Effect {
    /// Explicitly allow the action.
    Allow,
    /// Explicitly deny the action (takes precedence over Allow).
    Deny,
}

/// Matches a principal (user/token) against criteria.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrincipalMatcher {
    /// Match any principal.
    Any,
    /// Match a specific user ID.
    UserId(String),
    /// Match a specific role.
    Role(String),
    /// Match a principal attribute key=value (e.g., tenant_id=acme).
    Attribute(String, String),
}

/// Matches a resource (vector/metadata) against criteria.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceMatcher {
    /// Match any resource.
    Any,
    /// Match a specific vector by ID pattern (exact or prefix with *).
    VectorId(String),
    /// Match vectors where metadata field equals value.
    MetadataField(String, String),
    /// Match a specific metadata field name for field-level ACL.
    FieldName(String),
}

/// An ACL policy definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AclPolicy {
    /// Unique policy ID.
    pub id: String,
    /// Who this policy applies to.
    pub principal: PrincipalMatcher,
    /// What resources this policy covers.
    pub resource: ResourceMatcher,
    /// Allow or deny.
    pub effect: Effect,
    /// Which actions are covered.
    pub actions: Vec<AclAction>,
    /// Priority (higher = evaluated first). Default 0.
    #[serde(default)]
    pub priority: i32,
    /// Optional description.
    #[serde(default)]
    pub description: String,
}

// ============================================================================
// Request Context
// ============================================================================

/// Context for an access control evaluation.
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// User or service ID.
    pub user_id: String,
    /// Roles assigned to this principal.
    pub roles: Vec<String>,
    /// Arbitrary attributes (e.g., tenant_id, department).
    pub attributes: HashMap<String, String>,
}

impl RequestContext {
    /// Create a new request context.
    pub fn new(user_id: impl Into<String>, attributes: Vec<(String, String)>) -> Self {
        Self {
            user_id: user_id.into(),
            roles: Vec::new(),
            attributes: attributes.into_iter().collect(),
        }
    }

    /// Add a role.
    #[must_use]
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.roles.push(role.into());
        self
    }
}

// ============================================================================
// Access Decision
// ============================================================================

/// Result of an ACL evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessDecision {
    /// Whether access is allowed.
    pub allowed: bool,
    /// ID of the policy that made the decision (if any).
    pub policy_id: Option<String>,
    /// Reason for the decision.
    pub reason: String,
    /// Metadata fields that should be redacted (for field-level ACL).
    pub redacted_fields: Vec<String>,
}

// ============================================================================
// Audit Log
// ============================================================================

/// An audit log entry for access control decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AclAuditEntry {
    /// Timestamp (epoch seconds).
    pub timestamp: u64,
    /// User/principal ID.
    pub user_id: String,
    /// Action attempted.
    pub action: AclAction,
    /// Vector ID.
    pub vector_id: String,
    /// Whether access was allowed.
    pub allowed: bool,
    /// Policy that decided (if any).
    pub policy_id: Option<String>,
}

// ============================================================================
// ACL Engine
// ============================================================================

/// Engine that evaluates vector-level access control policies.
pub struct AclEngine {
    /// Policies ordered by priority (descending).
    policies: Vec<AclPolicy>,
    /// Audit log (bounded).
    audit_log: Vec<AclAuditEntry>,
    /// Max audit entries to retain.
    max_audit_entries: usize,
    /// Default effect when no policy matches.
    default_effect: Effect,
}

impl AclEngine {
    /// Create a new ACL engine with deny-by-default.
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
            audit_log: Vec::new(),
            max_audit_entries: 10_000,
            default_effect: Effect::Deny,
        }
    }

    /// Create an engine that allows by default (opt-in deny).
    pub fn allow_by_default() -> Self {
        Self {
            default_effect: Effect::Allow,
            ..Self::new()
        }
    }

    /// Add a policy.
    pub fn add_policy(&mut self, policy: AclPolicy) -> Result<()> {
        if policy.id.is_empty() {
            return Err(NeedleError::InvalidInput(
                "Policy ID cannot be empty".to_string(),
            ));
        }
        if self.policies.iter().any(|p| p.id == policy.id) {
            return Err(NeedleError::InvalidInput(format!(
                "Duplicate policy ID: '{}'",
                policy.id
            )));
        }
        self.policies.push(policy);
        self.policies.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(())
    }

    /// Remove a policy by ID.
    pub fn remove_policy(&mut self, id: &str) -> bool {
        let len = self.policies.len();
        self.policies.retain(|p| p.id != id);
        self.policies.len() < len
    }

    /// List all policies.
    pub fn list_policies(&self) -> &[AclPolicy] {
        &self.policies
    }

    /// Evaluate access for a request.
    pub fn evaluate(
        &mut self,
        ctx: &RequestContext,
        action: &AclAction,
        vector_id: &str,
        metadata: &Value,
    ) -> AccessDecision {
        let mut decision = AccessDecision {
            allowed: self.default_effect == Effect::Allow,
            policy_id: None,
            reason: if self.default_effect == Effect::Allow {
                "Default allow".to_string()
            } else {
                "Default deny — no matching policy".to_string()
            },
            redacted_fields: Vec::new(),
        };

        // Collect field-level redactions
        let mut field_denies: Vec<String> = Vec::new();

        for policy in &self.policies {
            if !policy.actions.contains(action) {
                continue;
            }
            if !Self::matches_principal(&policy.principal, ctx) {
                continue;
            }
            if !Self::matches_resource(&policy.resource, vector_id, metadata) {
                continue;
            }

            // Check for field-level deny
            if let ResourceMatcher::FieldName(ref field) = policy.resource {
                if policy.effect == Effect::Deny {
                    field_denies.push(field.clone());
                    continue;
                }
            }

            // First matching policy wins (sorted by priority)
            match policy.effect {
                Effect::Allow => {
                    decision.allowed = true;
                    decision.policy_id = Some(policy.id.clone());
                    decision.reason = format!("Allowed by policy '{}'", policy.id);
                }
                Effect::Deny => {
                    decision.allowed = false;
                    decision.policy_id = Some(policy.id.clone());
                    decision.reason = format!("Denied by policy '{}'", policy.id);
                }
            }
            break;
        }

        decision.redacted_fields = field_denies;

        // Audit
        let audit_entry = AclAuditEntry {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            user_id: ctx.user_id.clone(),
            action: *action,
            vector_id: vector_id.to_string(),
            allowed: decision.allowed,
            policy_id: decision.policy_id.clone(),
        };
        self.audit_log.push(audit_entry);
        if self.audit_log.len() > self.max_audit_entries {
            self.audit_log.remove(0);
        }

        decision
    }

    /// Rewrite a metadata filter to enforce ACL policies.
    ///
    /// Returns additional filter conditions that should be AND-ed with the
    /// user's query to enforce row-level security.
    pub fn row_level_filters(&self, ctx: &RequestContext) -> Vec<(String, String)> {
        let mut filters = Vec::new();
        for policy in &self.policies {
            if policy.effect != Effect::Allow {
                continue;
            }
            if !Self::matches_principal(&policy.principal, ctx) {
                continue;
            }
            if let ResourceMatcher::MetadataField(ref field, ref value) = policy.resource {
                filters.push((field.clone(), value.clone()));
            }
        }
        filters
    }

    /// Get the audit log.
    pub fn audit_log(&self) -> &[AclAuditEntry] {
        &self.audit_log
    }

    /// Clear the audit log.
    pub fn clear_audit_log(&mut self) {
        self.audit_log.clear();
    }

    // -- Matching helpers --

    fn matches_principal(matcher: &PrincipalMatcher, ctx: &RequestContext) -> bool {
        match matcher {
            PrincipalMatcher::Any => true,
            PrincipalMatcher::UserId(id) => ctx.user_id == *id,
            PrincipalMatcher::Role(role) => ctx.roles.contains(role),
            PrincipalMatcher::Attribute(key, value) => {
                ctx.attributes.get(key).is_some_and(|v| v == value)
            }
        }
    }

    fn matches_resource(matcher: &ResourceMatcher, vector_id: &str, metadata: &Value) -> bool {
        match matcher {
            ResourceMatcher::Any => true,
            ResourceMatcher::VectorId(pattern) => {
                if pattern.ends_with('*') {
                    vector_id.starts_with(&pattern[..pattern.len() - 1])
                } else {
                    vector_id == pattern
                }
            }
            ResourceMatcher::MetadataField(field, value) => metadata
                .get(field)
                .and_then(|v| v.as_str())
                .is_some_and(|v| v == value),
            ResourceMatcher::FieldName(_) => true, // field-level: always matches at row level
        }
    }
}

impl Default for AclEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tenant_policy(tenant: &str) -> AclPolicy {
        AclPolicy {
            id: format!("tenant-{}", tenant),
            principal: PrincipalMatcher::Attribute("tenant_id".into(), tenant.into()),
            resource: ResourceMatcher::MetadataField("tenant_id".into(), tenant.into()),
            effect: Effect::Allow,
            actions: vec![AclAction::Read, AclAction::Search],
            priority: 0,
            description: format!("Allow tenant {} to read their own data", tenant),
        }
    }

    #[test]
    fn test_basic_allow_deny() {
        let mut engine = AclEngine::new(); // deny by default
        engine
            .add_policy(AclPolicy {
                id: "allow-admin".into(),
                principal: PrincipalMatcher::Role("admin".into()),
                resource: ResourceMatcher::Any,
                effect: Effect::Allow,
                actions: vec![AclAction::Read, AclAction::Write, AclAction::Delete],
                priority: 10,
                description: String::new(),
            })
            .expect("add policy");

        let admin = RequestContext::new("alice", vec![]).with_role("admin");
        let user = RequestContext::new("bob", vec![]);

        let meta = json!({"title": "test"});
        let dec = engine.evaluate(&admin, &AclAction::Read, "doc1", &meta);
        assert!(dec.allowed);

        let dec = engine.evaluate(&user, &AclAction::Read, "doc1", &meta);
        assert!(!dec.allowed); // default deny
    }

    #[test]
    fn test_tenant_isolation() {
        let mut engine = AclEngine::new();
        engine.add_policy(tenant_policy("acme")).expect("add");
        engine.add_policy(tenant_policy("globex")).expect("add");

        let acme_user = RequestContext::new("u1", vec![("tenant_id".into(), "acme".into())]);
        let globex_user = RequestContext::new("u2", vec![("tenant_id".into(), "globex".into())]);

        let acme_doc = json!({"tenant_id": "acme", "title": "Secret"});
        let globex_doc = json!({"tenant_id": "globex", "title": "Other"});

        // acme user can read acme docs
        let dec = engine.evaluate(&acme_user, &AclAction::Read, "d1", &acme_doc);
        assert!(dec.allowed);

        // acme user cannot read globex docs
        let dec = engine.evaluate(&acme_user, &AclAction::Read, "d2", &globex_doc);
        assert!(!dec.allowed);

        // globex user can read globex docs
        let dec = engine.evaluate(&globex_user, &AclAction::Read, "d2", &globex_doc);
        assert!(dec.allowed);
    }

    #[test]
    fn test_deny_overrides_allow() {
        let mut engine = AclEngine::allow_by_default();
        engine
            .add_policy(AclPolicy {
                id: "deny-sensitive".into(),
                principal: PrincipalMatcher::Any,
                resource: ResourceMatcher::VectorId("sensitive-*".into()),
                effect: Effect::Deny,
                actions: vec![AclAction::Read],
                priority: 100,
                description: String::new(),
            })
            .expect("add");

        let ctx = RequestContext::new("anyone", vec![]);
        let meta = json!({});

        // Normal doc: allowed by default
        let dec = engine.evaluate(&ctx, &AclAction::Read, "doc1", &meta);
        assert!(dec.allowed);

        // Sensitive doc: denied
        let dec = engine.evaluate(&ctx, &AclAction::Read, "sensitive-123", &meta);
        assert!(!dec.allowed);
    }

    #[test]
    fn test_field_level_redaction() {
        let mut engine = AclEngine::allow_by_default();
        engine
            .add_policy(AclPolicy {
                id: "redact-ssn".into(),
                principal: PrincipalMatcher::Any,
                resource: ResourceMatcher::FieldName("ssn".into()),
                effect: Effect::Deny,
                actions: vec![AclAction::ReadMetadata],
                priority: 0,
                description: String::new(),
            })
            .expect("add");

        let ctx = RequestContext::new("user1", vec![]);
        let meta = json!({"name": "Alice", "ssn": "123-45-6789"});
        let dec = engine.evaluate(&ctx, &AclAction::ReadMetadata, "doc1", &meta);
        assert!(dec.redacted_fields.contains(&"ssn".to_string()));
    }

    #[test]
    fn test_row_level_filters() {
        let mut engine = AclEngine::new();
        engine.add_policy(tenant_policy("acme")).expect("add");

        let ctx = RequestContext::new("u1", vec![("tenant_id".into(), "acme".into())]);
        let filters = engine.row_level_filters(&ctx);
        assert_eq!(filters.len(), 1);
        assert_eq!(filters[0], ("tenant_id".to_string(), "acme".to_string()));
    }

    #[test]
    fn test_audit_log() {
        let mut engine = AclEngine::allow_by_default();
        let ctx = RequestContext::new("user1", vec![]);
        let meta = json!({});

        engine.evaluate(&ctx, &AclAction::Read, "doc1", &meta);
        engine.evaluate(&ctx, &AclAction::Write, "doc2", &meta);

        assert_eq!(engine.audit_log().len(), 2);
        assert_eq!(engine.audit_log()[0].vector_id, "doc1");
        assert_eq!(engine.audit_log()[1].action, AclAction::Write);

        engine.clear_audit_log();
        assert!(engine.audit_log().is_empty());
    }

    #[test]
    fn test_duplicate_policy_id() {
        let mut engine = AclEngine::new();
        engine.add_policy(tenant_policy("acme")).expect("first");
        let result = engine.add_policy(tenant_policy("acme"));
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_policy() {
        let mut engine = AclEngine::new();
        engine.add_policy(tenant_policy("acme")).expect("add");
        assert!(engine.remove_policy("tenant-acme"));
        assert!(!engine.remove_policy("tenant-acme")); // already removed
        assert!(engine.list_policies().is_empty());
    }
}
