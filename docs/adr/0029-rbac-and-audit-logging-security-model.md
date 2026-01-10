# ADR-0029: RBAC and Audit Logging Security Model

## Status

Accepted

## Context

As Needle moves beyond single-user embedded deployments into multi-user and enterprise environments, security becomes critical. Key requirements emerged:

1. **Multi-tenant access control** — Different users/services need different permissions on different collections
2. **Principle of least privilege** — Grant only necessary permissions, not blanket access
3. **Compliance requirements** — Industries like healthcare (HIPAA) and finance (SOX) require audit trails
4. **Debugging and forensics** — Understanding who did what and when aids incident response
5. **Granular permissions** — Read vs. write vs. admin operations on specific resources

Alternative approaches considered:

| Approach | Pros | Cons |
|----------|------|------|
| Simple API keys | Easy to implement | No granularity, all-or-nothing |
| OAuth/OIDC only | Standard, delegated auth | Doesn't solve authorization |
| ACLs per resource | Fine-grained | Doesn't scale, hard to manage |
| RBAC | Scalable, standard pattern | Requires role design upfront |
| ABAC | Most flexible | Complex policy language needed |

## Decision

Implement **Role-Based Access Control (RBAC)** with **comprehensive audit logging** as an optional security layer.

### Permission Model

```rust
/// Permissions that can be granted to roles
pub enum Permission {
    Read,    // Read vectors and metadata
    Write,   // Insert/update vectors
    Delete,  // Remove vectors
    Admin,   // Manage collection settings
    Search,  // Execute search queries
    Export,  // Export collection data
}

/// Resources that can be protected
pub enum Resource {
    Collection(String),  // Specific collection
    AllCollections,      // Wildcard for all collections
    System,              // System-level operations
}
```

### Role-Based Structure

```
┌─────────────────────────────────────────────────────────┐
│                         User                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │ Role A  │  │ Role B  │  │ Role C  │  (multiple)     │
│  └────┬────┘  └────┬────┘  └────┬────┘                 │
│       │            │            │                       │
│  ┌────▼────────────▼────────────▼────┐                 │
│  │     Permission Grants              │                 │
│  │  (Permission + Resource pairs)     │                 │
│  └───────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

### Audit Logging

Every operation is logged with:

```rust
pub struct AuditEvent {
    pub timestamp: u64,
    pub user: String,
    pub action: AuditAction,      // Search, Insert, Delete, etc.
    pub resource: String,         // Collection name or "system"
    pub result: AuditResult,      // Success or Failure(reason)
    pub details: Option<Value>,   // Additional context
    pub request_id: Option<String>,
    pub ip_address: Option<String>,
}
```

### Storage Options

```rust
// In-memory for development/testing
let logger = InMemoryAuditLog::new(max_events);

// File-based for production (append-only, rotatable)
let logger = FileAuditLog::new("/var/log/needle/audit.log");

// Custom backends via trait
pub trait AuditLogger: Send + Sync {
    fn log(&self, event: AuditEvent);
    fn query(&self) -> AuditQueryBuilder;
    fn execute_query(&self, query: &AuditQuery) -> Result<Vec<AuditEvent>>;
}
```

### Code References

- `src/security.rs:42-61` — Permission enum with all operations
- `src/security.rs:64-82` — Resource enum with wildcard matching
- `src/security.rs:98-130` — Role definition with permission grants
- `src/server.rs` — Integration with HTTP authentication middleware

## Consequences

### Benefits

1. **Enterprise-ready** — Meets compliance requirements for regulated industries
2. **Scalable management** — Roles can be defined once, assigned to many users
3. **Queryable audit trail** — Filter events by user, time, action, or resource
4. **Hierarchical permissions** — `AllCollections` grants access to all without listing each
5. **Separation of concerns** — Auth (who are you) vs. authz (what can you do) are separate
6. **Non-invasive** — Security layer is optional; embedded use cases can skip it

### Tradeoffs

1. **Performance overhead** — Permission checks on every operation (mitigated by caching)
2. **Storage for audit logs** — Grows unbounded without rotation policy
3. **Role explosion risk** — Poor design can lead to too many specialized roles
4. **No attribute-based policies** — Can't express "only during business hours" without ABAC

### What This Enabled

- Multi-tenant SaaS deployments with isolated permissions
- Compliance certifications (SOC 2, HIPAA, GDPR audit requirements)
- Debugging production issues by tracing user actions
- Integration with enterprise identity providers (via role mapping)

### What This Prevented

- Implicit trust models where any authenticated user can do anything
- "Admin-only" deployments that can't delegate limited access
- Audit gaps that fail compliance audits

### Usage Examples

**Define roles:**
```rust
let reader_role = Role::new("reader")
    .with_permission(Permission::Read, Resource::AllCollections)
    .with_permission(Permission::Search, Resource::AllCollections);

let writer_role = Role::new("writer")
    .with_permission(Permission::Read, Resource::Collection("docs".into()))
    .with_permission(Permission::Write, Resource::Collection("docs".into()));

let admin_role = Role::new("admin")
    .with_permissions(Permission::all(), Resource::System);
```

**Check access:**
```rust
let controller = AccessController::new();
controller.add_role(reader_role);
controller.add_role(admin_role);

let user = User::new("alice").with_role("reader");

// Returns true
controller.can_access(&user, Permission::Search, &Resource::Collection("any".into()));

// Returns false (no write permission)
controller.can_access(&user, Permission::Write, &Resource::Collection("docs".into()));
```

**Query audit logs:**
```rust
let events = logger.query()
    .user("alice")
    .action(AuditAction::Delete)
    .since(one_hour_ago)
    .execute()?;
```
