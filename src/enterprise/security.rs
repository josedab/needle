//! Security - RBAC and Audit Logging
//!
//! Role-based access control and comprehensive audit logging for Needle.
//!
//! # Role-Based Access Control (RBAC)
//!
//! ```rust,ignore
//! use needle::security::{Permission, Resource, Role, User, AccessController};
//!
//! let reader_role = Role::new("reader")
//!     .with_permission(Permission::Read, Resource::AllCollections);
//! let user = User::new("alice").with_role(reader_role);
//! let controller = AccessController::new();
//! assert!(controller.can_access(&user, Permission::Read, &Resource::Collection("docs".into())));
//! ```
//!
//! # Audit Logging
//!
//! ```rust,ignore
//! use needle::security::{AuditLogger, InMemoryAuditLog, AuditEvent, AuditAction, AuditResult};
//!
//! let logger = InMemoryAuditLog::new(1000);
//! logger.log(AuditEvent::new("alice", AuditAction::Search, "docs", AuditResult::Success));
//! let events = logger.execute_query(&logger.query().user("alice")).unwrap();
//! ```
//!
//! # Thread Safety
//!
//! All types use `RwLock` for safe concurrent access.

use crate::error::{NeedleError, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Permissions that can be granted to roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    Read,
    Write,
    Delete,
    Admin,
    Search,
    Export,
}

impl Permission {
    pub fn all() -> Vec<Permission> {
        vec![
            Permission::Read,
            Permission::Write,
            Permission::Delete,
            Permission::Admin,
            Permission::Search,
            Permission::Export,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Permission::Read => "read",
            Permission::Write => "write",
            Permission::Delete => "delete",
            Permission::Admin => "admin",
            Permission::Search => "search",
            Permission::Export => "export",
        }
    }
}

/// Resources that can be protected by permissions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resource {
    Collection(String),
    AllCollections,
    System,
}

impl Resource {
    pub fn matches(&self, other: &Resource) -> bool {
        match (self, other) {
            (Resource::AllCollections, Resource::Collection(_)) => true,
            (Resource::AllCollections, Resource::AllCollections) => true,
            (Resource::Collection(a), Resource::Collection(b)) => a == b,
            (Resource::System, Resource::System) => true,
            _ => false,
        }
    }
}

/// A permission grant associating a permission with a resource.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PermissionGrant {
    pub permission: Permission,
    pub resource: Resource,
}

impl PermissionGrant {
    pub fn new(permission: Permission, resource: Resource) -> Self {
        Self {
            permission,
            resource,
        }
    }
}

/// A role that groups permissions together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub name: String,
    pub description: Option<String>,
    pub permissions: HashSet<PermissionGrant>,
    pub built_in: bool,
}

impl Role {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            permissions: HashSet::new(),
            built_in: false,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_permission(mut self, permission: Permission, resource: Resource) -> Self {
        self.permissions
            .insert(PermissionGrant::new(permission, resource));
        self
    }

    pub fn as_built_in(mut self) -> Self {
        self.built_in = true;
        self
    }

    pub fn has_permission(&self, permission: Permission, resource: &Resource) -> bool {
        self.permissions
            .iter()
            .any(|g| g.permission == permission && g.resource.matches(resource))
    }

    pub fn admin() -> Self {
        let mut role = Self::new("admin")
            .with_description("Full administrative access")
            .as_built_in();
        for perm in Permission::all() {
            role.permissions
                .insert(PermissionGrant::new(perm, Resource::AllCollections));
            role.permissions
                .insert(PermissionGrant::new(perm, Resource::System));
        }
        role
    }

    pub fn reader() -> Self {
        Self::new("reader")
            .with_description("Read-only access")
            .with_permission(Permission::Read, Resource::AllCollections)
            .with_permission(Permission::Search, Resource::AllCollections)
            .as_built_in()
    }

    pub fn writer() -> Self {
        Self::new("writer")
            .with_description("Read and write access")
            .with_permission(Permission::Read, Resource::AllCollections)
            .with_permission(Permission::Write, Resource::AllCollections)
            .with_permission(Permission::Search, Resource::AllCollections)
            .as_built_in()
    }
}

/// A user that can be assigned roles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub name: Option<String>,
    pub email: Option<String>,
    pub roles: Vec<Role>,
    pub active: bool,
    pub created_at: u64,
    pub attributes: HashMap<String, String>,
}

impl User {
    pub fn new(id: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id: id.into(),
            name: None,
            email: None,
            roles: Vec::new(),
            active: true,
            created_at: now,
            attributes: HashMap::new(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    pub fn with_email(mut self, email: impl Into<String>) -> Self {
        self.email = Some(email.into());
        self
    }
    pub fn with_role(mut self, role: Role) -> Self {
        self.roles.push(role);
        self
    }
    pub fn with_attribute(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.attributes.insert(k.into(), v.into());
        self
    }

    pub fn has_permission(&self, permission: Permission, resource: &Resource) -> bool {
        self.active
            && self
                .roles
                .iter()
                .any(|r| r.has_permission(permission, resource))
    }

    pub fn all_permissions(&self) -> HashSet<PermissionGrant> {
        self.roles
            .iter()
            .flat_map(|r| r.permissions.clone())
            .collect()
    }
}

/// Trait for access control implementations.
pub trait AccessControl: Send + Sync {
    fn can_access(&self, user: &User, permission: Permission, resource: &Resource) -> bool;
    fn authorize(&self, user: &User, permission: Permission, resource: &Resource) -> Result<()>;
    fn evaluate(&self, user: &User, permission: Permission, resource: &Resource) -> PolicyDecision;
}

/// The result of a policy evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub allowed: bool,
    pub reason: String,
    pub granting_role: Option<String>,
    // Note: evaluation_time_ns removed to prevent timing attack information leakage.
    // Exposing timing info could allow attackers to infer permission structures.
}

impl PolicyDecision {
    pub fn allow(reason: impl Into<String>, role: impl Into<String>) -> Self {
        Self {
            allowed: true,
            reason: reason.into(),
            granting_role: Some(role.into()),
        }
    }
    pub fn deny(reason: impl Into<String>) -> Self {
        Self {
            allowed: false,
            reason: reason.into(),
            granting_role: None,
        }
    }
}

/// Default access controller implementation.
#[derive(Debug, Clone, Default)]
pub struct AccessController {
    pub enforce: bool,
    pub default_allow: bool,
}

impl AccessController {
    pub fn new() -> Self {
        Self {
            enforce: true,
            default_allow: false,
        }
    }
    pub fn permissive() -> Self {
        Self {
            enforce: false,
            default_allow: true,
        }
    }
    pub fn with_enforcement(mut self, enforce: bool) -> Self {
        self.enforce = enforce;
        self
    }
}

impl AccessControl for AccessController {
    fn can_access(&self, user: &User, permission: Permission, resource: &Resource) -> bool {
        if !self.enforce {
            self.default_allow
        } else {
            user.has_permission(permission, resource)
        }
    }

    fn authorize(&self, user: &User, permission: Permission, resource: &Resource) -> Result<()> {
        if self.can_access(user, permission, resource) {
            Ok(())
        } else {
            Err(NeedleError::InvalidInput(format!(
                "Access denied: user '{}' lacks {} permission on {:?}",
                user.id,
                permission.name(),
                resource
            )))
        }
    }

    fn evaluate(&self, user: &User, permission: Permission, resource: &Resource) -> PolicyDecision {
        // Note: No timing measurement to prevent timing attack information leakage.
        // Attackers could use timing differences to probe permission structures.
        if !self.enforce {
            if self.default_allow {
                PolicyDecision::allow("Enforcement disabled", "none")
            } else {
                PolicyDecision::deny("Enforcement disabled, default deny")
            }
        } else if !user.active {
            PolicyDecision::deny("User is inactive")
        } else if let Some(role) = user
            .roles
            .iter()
            .find(|r| r.has_permission(permission, resource))
        {
            PolicyDecision::allow(
                format!("Permission {} granted", permission.name()),
                &role.name,
            )
        } else {
            PolicyDecision::deny(format!(
                "No role grants {} on {:?}",
                permission.name(),
                resource
            ))
        }
    }
}

/// Actions that can be audited.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditAction {
    Login,
    Logout,
    Read,
    Write,
    Delete,
    Search,
    Export,
    CreateCollection,
    DropCollection,
    CreateUser,
    ModifyUser,
    DeleteUser,
    CreateRole,
    ModifyRole,
    DeleteRole,
    ConfigChange,
    AccessDenied,
}

impl AuditAction {
    pub fn severity(&self) -> u8 {
        match self {
            AuditAction::Read | AuditAction::Search => 1,
            AuditAction::Write | AuditAction::Export => 2,
            AuditAction::Delete | AuditAction::Login | AuditAction::Logout => 3,
            AuditAction::CreateCollection | AuditAction::DropCollection => 4,
            _ => 5,
        }
    }
}

/// The result of an audited action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditResult {
    Success,
    Failure,
    Denied,
}

/// An audit event recording a security-relevant action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub id: u64,
    pub timestamp: u64,
    pub user_id: String,
    pub action: AuditAction,
    pub resource: String,
    pub result: AuditResult,
    pub details: Option<String>,
    pub client_ip: Option<String>,
    pub session_id: Option<String>,
    pub duration_ms: Option<u64>,
}

impl AuditEvent {
    pub fn new(
        user_id: impl Into<String>,
        action: AuditAction,
        resource: impl Into<String>,
        result: AuditResult,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            id: 0,
            timestamp,
            user_id: user_id.into(),
            action,
            resource: resource.into(),
            result,
            details: None,
            client_ip: None,
            session_id: None,
            duration_ms: None,
        }
    }

    pub fn with_details(mut self, d: impl Into<String>) -> Self {
        self.details = Some(d.into());
        self
    }
    pub fn with_client_ip(mut self, ip: impl Into<String>) -> Self {
        self.client_ip = Some(ip.into());
        self
    }
    pub fn with_session_id(mut self, s: impl Into<String>) -> Self {
        self.session_id = Some(s.into());
        self
    }
    pub fn with_duration(mut self, d: Duration) -> Self {
        self.duration_ms = Some(d.as_millis() as u64);
        self
    }
}

/// Trait for audit log implementations.
pub trait AuditLogger: Send + Sync {
    fn log(&self, event: AuditEvent) -> Result<u64>;
    fn query(&self) -> AuditQuery;
    fn execute_query(&self, query: &AuditQuery) -> Result<Vec<AuditEvent>>;
    fn count(&self) -> usize;
    fn clear(&self) -> Result<()>;
    fn rotate(&self) -> Result<()>;
}

/// Query builder for searching audit logs.
#[derive(Debug, Clone, Default)]
pub struct AuditQuery {
    pub user_id: Option<String>,
    pub action: Option<AuditAction>,
    pub result: Option<AuditResult>,
    pub resource_prefix: Option<String>,
    pub from_timestamp: Option<u64>,
    pub to_timestamp: Option<u64>,
    pub min_severity: Option<u8>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub ascending: bool,
}

impl AuditQuery {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn user(mut self, id: impl Into<String>) -> Self {
        self.user_id = Some(id.into());
        self
    }
    pub fn action(mut self, a: AuditAction) -> Self {
        self.action = Some(a);
        self
    }
    pub fn result(mut self, r: AuditResult) -> Self {
        self.result = Some(r);
        self
    }
    pub fn resource(mut self, p: impl Into<String>) -> Self {
        self.resource_prefix = Some(p.into());
        self
    }
    pub fn from(mut self, ts: u64) -> Self {
        self.from_timestamp = Some(ts);
        self
    }
    pub fn to(mut self, ts: u64) -> Self {
        self.to_timestamp = Some(ts);
        self
    }
    pub fn min_severity(mut self, s: u8) -> Self {
        self.min_severity = Some(s);
        self
    }
    pub fn limit(mut self, l: usize) -> Self {
        self.limit = Some(l);
        self
    }
    pub fn offset(mut self, o: usize) -> Self {
        self.offset = Some(o);
        self
    }
    pub fn ascending(mut self) -> Self {
        self.ascending = true;
        self
    }

    pub fn matches(&self, e: &AuditEvent) -> bool {
        if let Some(ref u) = self.user_id {
            if &e.user_id != u {
                return false;
            }
        }
        if let Some(a) = self.action {
            if e.action != a {
                return false;
            }
        }
        if let Some(r) = self.result {
            if e.result != r {
                return false;
            }
        }
        if let Some(ref p) = self.resource_prefix {
            if !e.resource.starts_with(p) {
                return false;
            }
        }
        if let Some(f) = self.from_timestamp {
            if e.timestamp < f {
                return false;
            }
        }
        if let Some(t) = self.to_timestamp {
            if e.timestamp > t {
                return false;
            }
        }
        if let Some(s) = self.min_severity {
            if e.action.severity() < s {
                return false;
            }
        }
        true
    }
}

/// In-memory audit log implementation with configurable capacity.
pub struct InMemoryAuditLog {
    events: RwLock<VecDeque<AuditEvent>>,
    capacity: usize,
    next_id: RwLock<u64>,
}

impl InMemoryAuditLog {
    pub fn new(capacity: usize) -> Self {
        Self {
            events: RwLock::new(VecDeque::with_capacity(capacity)),
            capacity,
            next_id: RwLock::new(1),
        }
    }
    pub fn all_events(&self) -> Vec<AuditEvent> {
        self.events.read().iter().cloned().collect()
    }
}

impl Default for InMemoryAuditLog {
    fn default() -> Self {
        Self::new(10000)
    }
}

impl AuditLogger for InMemoryAuditLog {
    fn log(&self, mut event: AuditEvent) -> Result<u64> {
        let mut next_id = self.next_id.write();
        event.id = *next_id;
        *next_id += 1;
        let id = event.id;
        let mut events = self.events.write();
        if events.len() >= self.capacity {
            events.pop_front();
        }
        events.push_back(event);
        Ok(id)
    }

    fn query(&self) -> AuditQuery {
        AuditQuery::new()
    }

    fn execute_query(&self, query: &AuditQuery) -> Result<Vec<AuditEvent>> {
        let events = self.events.read();
        let mut results: Vec<_> = events
            .iter()
            .filter(|e| query.matches(e))
            .cloned()
            .collect();
        if query.ascending {
            results.sort_by_key(|e| e.timestamp);
        } else {
            results.sort_by_key(|e| std::cmp::Reverse(e.timestamp));
        }
        let results: Vec<_> = results
            .into_iter()
            .skip(query.offset.unwrap_or(0))
            .collect();
        Ok(if let Some(l) = query.limit {
            results.into_iter().take(l).collect()
        } else {
            results
        })
    }

    fn count(&self) -> usize {
        self.events.read().len()
    }
    fn clear(&self) -> Result<()> {
        self.events.write().clear();
        Ok(())
    }
    fn rotate(&self) -> Result<()> {
        let mut events = self.events.write();
        let trim_to = self.capacity / 2;
        while events.len() > trim_to {
            events.pop_front();
        }
        Ok(())
    }
}

/// Configuration for file-based audit logging.
#[derive(Debug, Clone)]
pub struct FileAuditLogConfig {
    pub base_path: PathBuf,
    pub max_file_size: u64,
    pub max_files: usize,
    pub buffered: bool,
}

impl Default for FileAuditLogConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("audit.log"),
            max_file_size: 10 * 1024 * 1024,
            max_files: 5,
            buffered: true,
        }
    }
}

/// File-based audit log implementation with rotation support.
pub struct FileAuditLog {
    config: FileAuditLogConfig,
    current_size: RwLock<u64>,
    next_id: RwLock<u64>,
    write_lock: RwLock<()>,
}

impl FileAuditLog {
    pub fn new(config: FileAuditLogConfig) -> Result<Self> {
        let current_size = if config.base_path.exists() {
            std::fs::metadata(&config.base_path)
                .map(|m| m.len())
                .unwrap_or(0)
        } else {
            0
        };
        let next_id = Self::find_max_id(&config.base_path)? + 1;
        Ok(Self {
            config,
            current_size: RwLock::new(current_size),
            next_id: RwLock::new(next_id),
            write_lock: RwLock::new(()),
        })
    }

    fn find_max_id(path: &Path) -> Result<u64> {
        if !path.exists() {
            return Ok(0);
        }
        let file = File::open(path)?;
        let mut max_id = 0u64;
        for line in BufReader::new(file).lines().map_while(|r| r.ok()) {
            if let Ok(e) = serde_json::from_str::<AuditEvent>(&line) {
                max_id = max_id.max(e.id);
            }
        }
        Ok(max_id)
    }

    fn rotated_path(&self, i: usize) -> PathBuf {
        let ext = self
            .config
            .base_path
            .extension()
            .map(|e| e.to_string_lossy().to_string())
            .unwrap_or_else(|| "log".into());
        let stem = self
            .config
            .base_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "audit".into());
        self.config
            .base_path
            .with_file_name(format!("{}.{}.{}", stem, i, ext))
    }

    fn rotate_internal(&self) -> Result<()> {
        let oldest = self.rotated_path(self.config.max_files - 1);
        if oldest.exists() {
            std::fs::remove_file(&oldest)?;
        }
        for i in (0..self.config.max_files - 1).rev() {
            let current = if i == 0 {
                self.config.base_path.clone()
            } else {
                self.rotated_path(i)
            };
            if current.exists() {
                std::fs::rename(&current, self.rotated_path(i + 1))?;
            }
        }
        *self.current_size.write() = 0;
        Ok(())
    }
}

impl AuditLogger for FileAuditLog {
    fn log(&self, mut event: AuditEvent) -> Result<u64> {
        let _lock = self.write_lock.write();
        let mut next_id = self.next_id.write();
        event.id = *next_id;
        *next_id += 1;
        let id = event.id;
        drop(next_id);
        let line = serde_json::to_string(&event)? + "\n";
        let line_bytes = line.as_bytes();
        let mut current_size = self.current_size.write();
        if *current_size + line_bytes.len() as u64 > self.config.max_file_size {
            drop(current_size);
            self.rotate_internal()?;
            current_size = self.current_size.write();
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.config.base_path)?;
        file.write_all(line_bytes)?;
        if !self.config.buffered {
            file.flush()?;
        }
        *current_size += line_bytes.len() as u64;
        Ok(id)
    }

    fn query(&self) -> AuditQuery {
        AuditQuery::new()
    }

    fn execute_query(&self, query: &AuditQuery) -> Result<Vec<AuditEvent>> {
        let _lock = self.write_lock.read();
        let mut results = Vec::new();
        let mut files = vec![self.config.base_path.clone()];
        for i in 1..self.config.max_files {
            let p = self.rotated_path(i);
            if p.exists() {
                files.push(p);
            }
        }
        for path in files {
            if !path.exists() {
                continue;
            }
            for line in BufReader::new(File::open(&path)?)
                .lines()
                .map_while(|r| r.ok())
            {
                if let Ok(e) = serde_json::from_str::<AuditEvent>(&line) {
                    if query.matches(&e) {
                        results.push(e);
                    }
                }
            }
        }
        if query.ascending {
            results.sort_by_key(|e| e.timestamp);
        } else {
            results.sort_by_key(|e| std::cmp::Reverse(e.timestamp));
        }
        let results: Vec<_> = results
            .into_iter()
            .skip(query.offset.unwrap_or(0))
            .collect();
        Ok(if let Some(l) = query.limit {
            results.into_iter().take(l).collect()
        } else {
            results
        })
    }

    fn count(&self) -> usize {
        let _lock = self.write_lock.read();
        let mut count = 0;
        let mut files = vec![self.config.base_path.clone()];
        for i in 1..self.config.max_files {
            let p = self.rotated_path(i);
            if p.exists() {
                files.push(p);
            }
        }
        for path in files {
            if let Ok(f) = File::open(&path) {
                count += BufReader::new(f).lines().count();
            }
        }
        count
    }

    fn clear(&self) -> Result<()> {
        let _lock = self.write_lock.write();
        if self.config.base_path.exists() {
            std::fs::remove_file(&self.config.base_path)?;
        }
        for i in 1..self.config.max_files {
            let p = self.rotated_path(i);
            if p.exists() {
                std::fs::remove_file(&p)?;
            }
        }
        *self.current_size.write() = 0;
        Ok(())
    }

    fn rotate(&self) -> Result<()> {
        let _lock = self.write_lock.write();
        self.rotate_internal()
    }
}

/// A security context combining access control and audit logging.
pub struct SecurityContext {
    pub access_controller: Arc<dyn AccessControl>,
    pub audit_logger: Arc<dyn AuditLogger>,
}

impl SecurityContext {
    pub fn new(ac: Arc<dyn AccessControl>, al: Arc<dyn AuditLogger>) -> Self {
        Self {
            access_controller: ac,
            audit_logger: al,
        }
    }

    pub fn in_memory() -> Self {
        Self {
            access_controller: Arc::new(AccessController::new()),
            audit_logger: Arc::new(InMemoryAuditLog::new(1000)),
        }
    }

    pub fn permissive() -> Self {
        Self {
            access_controller: Arc::new(AccessController::permissive()),
            audit_logger: Arc::new(InMemoryAuditLog::new(1000)),
        }
    }

    pub fn check_access(
        &self,
        user: &User,
        perm: Permission,
        res: &Resource,
        action: AuditAction,
    ) -> Result<()> {
        let decision = self.access_controller.evaluate(user, perm, res);
        let res_str = match res {
            Resource::Collection(n) => n.clone(),
            Resource::AllCollections => "*".into(),
            Resource::System => "system".into(),
        };
        let result = if decision.allowed {
            AuditResult::Success
        } else {
            AuditResult::Denied
        };
        self.audit_logger.log(
            AuditEvent::new(&user.id, action, res_str, result).with_details(&decision.reason),
        )?;
        if decision.allowed {
            Ok(())
        } else {
            Err(NeedleError::InvalidInput(format!(
                "Access denied: {}",
                decision.reason
            )))
        }
    }

    pub fn log_event(&self, event: AuditEvent) -> Result<u64> {
        self.audit_logger.log(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_permission_all() {
        assert_eq!(Permission::all().len(), 6);
    }

    #[test]
    fn test_resource_matching() {
        let all = Resource::AllCollections;
        let docs = Resource::Collection("docs".into());
        assert!(all.matches(&docs));
        assert!(docs.matches(&docs));
        assert!(!docs.matches(&Resource::Collection("users".into())));
    }

    #[test]
    fn test_role_permissions() {
        let role = Role::new("test").with_permission(Permission::Read, Resource::AllCollections);
        assert!(role.has_permission(Permission::Read, &Resource::Collection("any".into())));
        assert!(!role.has_permission(Permission::Write, &Resource::Collection("any".into())));
    }

    #[test]
    fn test_built_in_roles() {
        let admin = Role::admin();
        assert!(admin.has_permission(Permission::Admin, &Resource::System));
        let reader = Role::reader();
        assert!(reader.has_permission(Permission::Read, &Resource::AllCollections));
        assert!(!reader.has_permission(Permission::Write, &Resource::AllCollections));
    }

    #[test]
    fn test_user_permissions() {
        let user = User::new("alice").with_role(Role::reader());
        assert!(user.has_permission(Permission::Read, &Resource::AllCollections));
        assert!(!user.has_permission(Permission::Admin, &Resource::System));
    }

    #[test]
    fn test_inactive_user() {
        let mut user = User::new("alice").with_role(Role::admin());
        user.active = false;
        assert!(!user.has_permission(Permission::Read, &Resource::AllCollections));
    }

    #[test]
    fn test_access_controller() {
        let ac = AccessController::new();
        let user = User::new("alice").with_role(Role::reader());
        assert!(ac.can_access(&user, Permission::Read, &Resource::AllCollections));
        assert!(!ac.can_access(&user, Permission::Write, &Resource::AllCollections));
    }

    #[test]
    fn test_access_controller_permissive() {
        let ac = AccessController::permissive();
        let user = User::new("alice");
        assert!(ac.can_access(&user, Permission::Admin, &Resource::System));
    }

    #[test]
    fn test_policy_decision() {
        let ac = AccessController::new();
        let user = User::new("alice").with_role(Role::reader());
        let allowed = ac.evaluate(&user, Permission::Read, &Resource::AllCollections);
        assert!(allowed.allowed);
        let denied = ac.evaluate(&user, Permission::Admin, &Resource::System);
        assert!(!denied.allowed);
    }

    #[test]
    fn test_audit_event() {
        let e = AuditEvent::new("alice", AuditAction::Search, "docs", AuditResult::Success)
            .with_details("query")
            .with_client_ip("127.0.0.1");
        assert_eq!(e.user_id, "alice");
        assert_eq!(e.action, AuditAction::Search);
    }

    #[test]
    fn test_audit_action_severity() {
        assert_eq!(AuditAction::Read.severity(), 1);
        assert_eq!(AuditAction::AccessDenied.severity(), 5);
    }

    #[test]
    fn test_audit_query_matching() {
        let e = AuditEvent::new("alice", AuditAction::Search, "docs", AuditResult::Success);
        assert!(AuditQuery::new().user("alice").matches(&e));
        assert!(!AuditQuery::new().user("bob").matches(&e));
        assert!(AuditQuery::new().action(AuditAction::Search).matches(&e));
    }

    #[test]
    fn test_in_memory_audit_log() {
        let log = InMemoryAuditLog::new(100);
        let id1 = log
            .log(AuditEvent::new(
                "alice",
                AuditAction::Read,
                "docs",
                AuditResult::Success,
            ))
            .unwrap();
        let id2 = log
            .log(AuditEvent::new(
                "bob",
                AuditAction::Write,
                "users",
                AuditResult::Success,
            ))
            .unwrap();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(log.count(), 2);
    }

    #[test]
    fn test_in_memory_audit_log_capacity() {
        let log = InMemoryAuditLog::new(5);
        for i in 0..10 {
            log.log(AuditEvent::new(
                format!("user{}", i),
                AuditAction::Read,
                "docs",
                AuditResult::Success,
            ))
            .unwrap();
        }
        assert_eq!(log.count(), 5);
    }

    #[test]
    fn test_in_memory_audit_log_query() {
        let log = InMemoryAuditLog::new(100);
        log.log(AuditEvent::new(
            "alice",
            AuditAction::Read,
            "docs",
            AuditResult::Success,
        ))
        .unwrap();
        log.log(AuditEvent::new(
            "bob",
            AuditAction::Write,
            "docs",
            AuditResult::Success,
        ))
        .unwrap();
        let results = log.execute_query(&AuditQuery::new().user("alice")).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_in_memory_thread_safety() {
        let log = Arc::new(InMemoryAuditLog::new(1000));
        let mut handles = vec![];
        for i in 0..10 {
            let l = Arc::clone(&log);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    l.log(AuditEvent::new(
                        format!("u{}", i),
                        AuditAction::Read,
                        format!("r{}", j),
                        AuditResult::Success,
                    ))
                    .unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(log.count(), 1000);
    }

    #[test]
    fn test_file_audit_log() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = FileAuditLogConfig {
            base_path: dir.path().join("audit.log"),
            max_file_size: 1024 * 1024,
            max_files: 3,
            buffered: false,
        };
        let log = FileAuditLog::new(cfg).unwrap();
        let id1 = log
            .log(AuditEvent::new(
                "alice",
                AuditAction::Read,
                "docs",
                AuditResult::Success,
            ))
            .unwrap();
        assert_eq!(id1, 1);
        assert_eq!(log.count(), 1);
    }

    #[test]
    fn test_file_audit_log_query() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = FileAuditLogConfig {
            base_path: dir.path().join("audit.log"),
            max_file_size: 1024 * 1024,
            max_files: 3,
            buffered: false,
        };
        let log = FileAuditLog::new(cfg).unwrap();
        log.log(AuditEvent::new(
            "alice",
            AuditAction::Read,
            "docs",
            AuditResult::Success,
        ))
        .unwrap();
        log.log(AuditEvent::new(
            "bob",
            AuditAction::Write,
            "docs",
            AuditResult::Success,
        ))
        .unwrap();
        assert_eq!(
            log.execute_query(&AuditQuery::new().user("alice"))
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn test_file_audit_log_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = FileAuditLogConfig {
            base_path: dir.path().join("audit.log"),
            max_file_size: 1024 * 1024,
            max_files: 3,
            buffered: false,
        };
        {
            let log = FileAuditLog::new(cfg.clone()).unwrap();
            log.log(AuditEvent::new(
                "alice",
                AuditAction::Read,
                "docs",
                AuditResult::Success,
            ))
            .unwrap();
        }
        {
            let log = FileAuditLog::new(cfg).unwrap();
            assert_eq!(log.count(), 1);
            assert_eq!(
                log.log(AuditEvent::new(
                    "bob",
                    AuditAction::Read,
                    "docs",
                    AuditResult::Success
                ))
                .unwrap(),
                2
            );
        }
    }

    #[test]
    fn test_security_context() {
        let ctx = SecurityContext::in_memory();
        let user = User::new("alice").with_role(Role::reader());
        assert!(ctx
            .check_access(
                &user,
                Permission::Read,
                &Resource::Collection("docs".into()),
                AuditAction::Read
            )
            .is_ok());
        assert!(ctx
            .check_access(
                &user,
                Permission::Admin,
                &Resource::System,
                AuditAction::ConfigChange
            )
            .is_err());
        assert_eq!(ctx.audit_logger.count(), 2);
    }

    #[test]
    fn test_security_context_permissive() {
        let ctx = SecurityContext::permissive();
        let user = User::new("alice");
        assert!(ctx
            .check_access(
                &user,
                Permission::Admin,
                &Resource::System,
                AuditAction::ConfigChange
            )
            .is_ok());
    }

    #[test]
    fn test_serialization() {
        let role = Role::new("test").with_permission(Permission::Read, Resource::AllCollections);
        let json = serde_json::to_string(&role).unwrap();
        let parsed: Role = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "test");

        let user = User::new("alice").with_role(Role::reader());
        let json = serde_json::to_string(&user).unwrap();
        let parsed: User = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "alice");

        let event = AuditEvent::new("alice", AuditAction::Search, "docs", AuditResult::Success);
        let json = serde_json::to_string(&event).unwrap();
        let parsed: AuditEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.user_id, "alice");
    }
}
