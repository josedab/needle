# ADR-0027: Namespace-Based Multi-Tenancy

## Status

Accepted

## Context

SaaS applications often need to serve multiple tenants (customers, organizations, projects) from a single deployment:

1. **Data isolation** — Tenant A must not see Tenant B's vectors
2. **Resource limits** — Prevent one tenant from consuming all resources
3. **Operational simplicity** — Avoid running N separate databases

### Deployment Options

| Approach | Isolation | Complexity | Resource Efficiency |
|----------|-----------|------------|---------------------|
| Separate databases per tenant | Strong | High (N deployments) | Low (duplication) |
| Separate collections per tenant | Medium | Medium | Medium |
| Namespace prefixing | Logical | Low | High |

For an embedded database, separate databases mean separate files and separate processes—operationally expensive. Collection-per-tenant is viable but lacks quota enforcement.

### Alternatives Considered

1. **Row-level security** — Complex, requires filter on every query
2. **Separate database files** — Operational overhead, no shared resources
3. **Application-level enforcement** — Error-prone, no database-level guarantees

## Decision

Needle implements **namespace-based multi-tenancy** with:
- Collection name prefixing for logical isolation
- Per-namespace resource quotas
- Centralized tenant configuration

### Namespace Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Single Database                           │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Namespace: tenant_a                       ││
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   ││
│  │  │tenant_a/docs  │  │tenant_a/images│  │tenant_a/users │   ││
│  │  └───────────────┘  └───────────────┘  └───────────────┘   ││
│  │  Quota: 1M vectors, 10GB storage, 100 req/s                 ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Namespace: tenant_b                       ││
│  │  ┌───────────────┐  ┌───────────────┐                       ││
│  │  │tenant_b/docs  │  │tenant_b/prods │                       ││
│  │  └───────────────┘  └───────────────┘                       ││
│  │  Quota: 500K vectors, 5GB storage, 50 req/s                 ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Namespace Configuration

```rust
// src/namespace.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Namespace {
    /// Unique namespace identifier (used as collection prefix)
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Resource quotas for this namespace
    pub quotas: TenantQuotas,

    /// Namespace metadata
    pub metadata: HashMap<String, String>,

    /// Creation timestamp
    pub created_at: u64,

    /// Whether namespace is active
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantQuotas {
    /// Maximum number of collections
    pub max_collections: Option<usize>,

    /// Maximum total vectors across all collections
    pub max_vectors: Option<usize>,

    /// Maximum storage in bytes
    pub max_storage_bytes: Option<usize>,

    /// Maximum vector dimensions (prevents abuse with huge vectors)
    pub max_dimensions: Option<usize>,

    /// Rate limit: operations per second
    pub rate_limit_ops: Option<u32>,

    /// Rate limit: bytes per second
    pub rate_limit_bytes: Option<u64>,
}

impl Default for TenantQuotas {
    fn default() -> Self {
        Self {
            max_collections: Some(100),
            max_vectors: Some(1_000_000),
            max_storage_bytes: Some(10 * 1024 * 1024 * 1024), // 10 GB
            max_dimensions: Some(4096),
            rate_limit_ops: Some(1000),
            rate_limit_bytes: Some(100 * 1024 * 1024), // 100 MB/s
        }
    }
}
```

### Namespace Manager

```rust
// src/namespace.rs
pub struct NamespaceManager {
    namespaces: RwLock<HashMap<String, Namespace>>,
    usage: RwLock<HashMap<String, NamespaceUsage>>,
    rate_limiters: RwLock<HashMap<String, RateLimiter>>,
}

#[derive(Debug, Clone, Default)]
pub struct NamespaceUsage {
    pub collection_count: usize,
    pub vector_count: usize,
    pub storage_bytes: usize,
}

impl NamespaceManager {
    /// Create a new namespace
    pub fn create_namespace(&self, id: &str, config: NamespaceConfig) -> Result<Namespace> {
        let mut namespaces = self.namespaces.write();

        if namespaces.contains_key(id) {
            return Err(NeedleError::NamespaceExists(id.to_string()));
        }

        let namespace = Namespace {
            id: id.to_string(),
            name: config.name,
            quotas: config.quotas.unwrap_or_default(),
            metadata: config.metadata,
            created_at: current_timestamp(),
            active: true,
        };

        namespaces.insert(id.to_string(), namespace.clone());
        self.usage.write().insert(id.to_string(), NamespaceUsage::default());

        Ok(namespace)
    }

    /// Get the full collection name with namespace prefix
    pub fn prefixed_name(&self, namespace: &str, collection: &str) -> String {
        format!("{}/{}", namespace, collection)
    }

    /// Extract namespace from a prefixed collection name
    pub fn extract_namespace(&self, prefixed: &str) -> Option<(&str, &str)> {
        prefixed.split_once('/')
    }

    /// Check if an operation is allowed under quotas
    pub fn check_quota(&self, namespace: &str, operation: &QuotaCheck) -> Result<()> {
        let namespaces = self.namespaces.read();
        let namespace = namespaces.get(namespace)
            .ok_or_else(|| NeedleError::NamespaceNotFound(namespace.to_string()))?;

        let usage = self.usage.read();
        let current = usage.get(namespace).cloned().unwrap_or_default();

        match operation {
            QuotaCheck::CreateCollection => {
                if let Some(max) = namespace.quotas.max_collections {
                    if current.collection_count >= max {
                        return Err(NeedleError::QuotaExceeded(
                            format!("Collection limit ({}) reached", max)
                        ));
                    }
                }
            }
            QuotaCheck::InsertVectors(count) => {
                if let Some(max) = namespace.quotas.max_vectors {
                    if current.vector_count + count > max {
                        return Err(NeedleError::QuotaExceeded(
                            format!("Vector limit ({}) would be exceeded", max)
                        ));
                    }
                }
            }
            QuotaCheck::StorageIncrease(bytes) => {
                if let Some(max) = namespace.quotas.max_storage_bytes {
                    if current.storage_bytes + bytes > max {
                        return Err(NeedleError::QuotaExceeded(
                            format!("Storage limit ({} bytes) would be exceeded", max)
                        ));
                    }
                }
            }
            QuotaCheck::VectorDimensions(dims) => {
                if let Some(max) = namespace.quotas.max_dimensions {
                    if *dims > max {
                        return Err(NeedleError::QuotaExceeded(
                            format!("Dimension limit ({}) exceeded", max)
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply rate limiting
    pub fn check_rate_limit(&self, namespace: &str) -> Result<()> {
        let mut limiters = self.rate_limiters.write();
        let limiter = limiters.entry(namespace.to_string())
            .or_insert_with(|| {
                let quotas = self.namespaces.read()
                    .get(namespace)
                    .map(|n| n.quotas.clone())
                    .unwrap_or_default();

                RateLimiter::new(quotas.rate_limit_ops.unwrap_or(1000))
            });

        if limiter.try_acquire() {
            Ok(())
        } else {
            Err(NeedleError::RateLimitExceeded(namespace.to_string()))
        }
    }
}

#[derive(Debug)]
pub enum QuotaCheck {
    CreateCollection,
    InsertVectors(usize),
    StorageIncrease(usize),
    VectorDimensions(usize),
}
```

### Database Integration

```rust
// src/database.rs
impl Database {
    /// Create a collection within a namespace
    pub fn create_collection_namespaced(
        &mut self,
        namespace: &str,
        name: &str,
        dimensions: usize,
    ) -> Result<()> {
        // Check namespace exists and is active
        let ns = self.namespace_manager.get(namespace)?;
        if !ns.active {
            return Err(NeedleError::NamespaceInactive(namespace.to_string()));
        }

        // Check quotas
        self.namespace_manager.check_quota(namespace, &QuotaCheck::CreateCollection)?;
        self.namespace_manager.check_quota(namespace, &QuotaCheck::VectorDimensions(dimensions))?;

        // Create with prefixed name
        let prefixed = self.namespace_manager.prefixed_name(namespace, name);
        self.create_collection(&prefixed, dimensions)?;

        // Update usage
        self.namespace_manager.increment_collection_count(namespace);

        Ok(())
    }

    /// Insert vectors with quota checking
    pub fn insert_namespaced(
        &mut self,
        namespace: &str,
        collection: &str,
        id: &str,
        vector: &[f32],
    ) -> Result<()> {
        // Rate limit check
        self.namespace_manager.check_rate_limit(namespace)?;

        // Quota check
        self.namespace_manager.check_quota(namespace, &QuotaCheck::InsertVectors(1))?;

        let storage_increase = vector.len() * 4 + id.len(); // Rough estimate
        self.namespace_manager.check_quota(
            namespace,
            &QuotaCheck::StorageIncrease(storage_increase)
        )?;

        // Perform insert
        let prefixed = self.namespace_manager.prefixed_name(namespace, collection);
        self.insert(&prefixed, id, vector)?;

        // Update usage
        self.namespace_manager.increment_vectors(namespace, 1);
        self.namespace_manager.increment_storage(namespace, storage_increase);

        Ok(())
    }

    /// List collections for a namespace only
    pub fn list_collections_namespaced(&self, namespace: &str) -> Result<Vec<String>> {
        let prefix = format!("{}/", namespace);
        Ok(self.list_collections()?
            .into_iter()
            .filter(|c| c.starts_with(&prefix))
            .map(|c| c[prefix.len()..].to_string())
            .collect())
    }
}
```

### HTTP API Integration

```rust
// src/server.rs
// Namespace extracted from path or header
async fn insert_handler(
    Path((namespace, collection)): Path<(String, String)>,
    State(db): State<Database>,
    Json(request): Json<InsertRequest>,
) -> Result<Json<InsertResponse>, ApiError> {
    db.insert_namespaced(&namespace, &collection, &request.id, &request.vector)?;
    Ok(Json(InsertResponse { success: true }))
}

// Routes: /namespaces/{namespace}/collections/{collection}/vectors
```

## Consequences

### Benefits

1. **Logical isolation** — Tenants can't access each other's data
2. **Resource fairness** — Quotas prevent noisy neighbor problems
3. **Operational simplicity** — Single database serves all tenants
4. **Flexible limits** — Per-tenant quotas (free tier vs enterprise)

### Tradeoffs

1. **Soft isolation** — Not cryptographic; relies on prefix enforcement
2. **Shared failure domain** — Database crash affects all tenants
3. **Quota tracking overhead** — Every operation updates counters

### What This Enabled

- **SaaS deployments** — Multi-tenant vector search as a service
- **Free tier limits** — Different quotas for different plans
- **Per-tenant billing** — Track usage for billing purposes

### What This Prevented

- **Resource starvation** — No single tenant can consume everything
- **Accidental cross-tenant access** — Prefix enforcement at database level

## References

- Namespace implementation: `src/namespace.rs`
- Quota configuration: `src/namespace.rs:30-60`
- Rate limiting: `src/namespace.rs:150-200`
- HTTP integration: `src/server.rs` (namespaced routes)
