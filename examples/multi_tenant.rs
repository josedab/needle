//! Multi-Tenant Vector Database Example
//!
//! This example demonstrates building a multi-tenant system where each tenant
//! has isolated data but shares the same Needle infrastructure.
//!
//! Run with: cargo run --example multi_tenant

use needle::{Database, Filter, CollectionConfig, DistanceFunction};
use needle::namespace::{Namespace, NamespaceManager, TenantConfig};
use serde_json::json;
use std::collections::HashMap;

/// Tenant information
#[derive(Debug, Clone)]
struct Tenant {
    id: String,
    name: String,
    plan: TenantPlan,
}

/// Tenant subscription plan
#[derive(Debug, Clone, Copy)]
enum TenantPlan {
    Free,
    Pro,
    Enterprise,
}

impl TenantPlan {
    fn max_vectors(&self) -> usize {
        match self {
            TenantPlan::Free => 1_000,
            TenantPlan::Pro => 100_000,
            TenantPlan::Enterprise => usize::MAX,
        }
    }

    fn max_collections(&self) -> usize {
        match self {
            TenantPlan::Free => 3,
            TenantPlan::Pro => 10,
            TenantPlan::Enterprise => usize::MAX,
        }
    }

    fn max_dimensions(&self) -> usize {
        match self {
            TenantPlan::Free => 384,
            TenantPlan::Pro => 1536,
            TenantPlan::Enterprise => 4096,
        }
    }
}

/// Multi-tenant vector database manager
struct MultiTenantVectorDB {
    db: Database,
    tenants: HashMap<String, Tenant>,
    tenant_stats: HashMap<String, TenantStats>,
}

/// Statistics for a tenant
#[derive(Debug, Default, Clone)]
struct TenantStats {
    total_vectors: usize,
    total_collections: usize,
    total_queries: usize,
}

impl MultiTenantVectorDB {
    /// Create a new multi-tenant database
    fn new() -> Self {
        Self {
            db: Database::in_memory(),
            tenants: HashMap::new(),
            tenant_stats: HashMap::new(),
        }
    }

    /// Register a new tenant
    fn register_tenant(&mut self, tenant: Tenant) -> Result<(), String> {
        if self.tenants.contains_key(&tenant.id) {
            return Err(format!("Tenant {} already exists", tenant.id));
        }

        println!("Registered tenant: {} ({:?} plan)", tenant.name, tenant.plan);
        self.tenants.insert(tenant.id.clone(), tenant.clone());
        self.tenant_stats.insert(tenant.id.clone(), TenantStats::default());

        Ok(())
    }

    /// Get tenant by ID
    fn get_tenant(&self, tenant_id: &str) -> Option<&Tenant> {
        self.tenants.get(tenant_id)
    }

    /// Create a collection for a tenant
    fn create_collection(
        &mut self,
        tenant_id: &str,
        name: &str,
        dimensions: usize,
    ) -> needle::Result<()> {
        let tenant = self.tenants.get(tenant_id)
            .ok_or_else(|| needle::NeedleError::NotFound(format!("Tenant {} not found", tenant_id)))?;

        let stats = self.tenant_stats.get(tenant_id)
            .ok_or_else(|| needle::NeedleError::NotFound("Tenant stats not found".to_string()))?;

        // Check plan limits
        if stats.total_collections >= tenant.plan.max_collections() {
            return Err(needle::NeedleError::QuotaExceeded(format!(
                "Collection limit reached for {:?} plan ({} max)",
                tenant.plan, tenant.plan.max_collections()
            )));
        }

        if dimensions > tenant.plan.max_dimensions() {
            return Err(needle::NeedleError::QuotaExceeded(format!(
                "Dimension limit exceeded for {:?} plan ({} max)",
                tenant.plan, tenant.plan.max_dimensions()
            )));
        }

        // Create tenant-prefixed collection name
        let full_name = format!("{}_{}", tenant_id, name);

        let config = CollectionConfig::new(&full_name, dimensions)
            .with_distance(DistanceFunction::Cosine);

        self.db.create_collection_with_config(config)?;

        // Update stats
        if let Some(stats) = self.tenant_stats.get_mut(tenant_id) {
            stats.total_collections += 1;
        }

        println!("Created collection '{}' for tenant {}", name, tenant_id);
        Ok(())
    }

    /// Insert a vector for a tenant
    fn insert(
        &mut self,
        tenant_id: &str,
        collection: &str,
        id: &str,
        vector: &[f32],
        metadata: Option<serde_json::Value>,
    ) -> needle::Result<()> {
        let tenant = self.tenants.get(tenant_id)
            .ok_or_else(|| needle::NeedleError::NotFound(format!("Tenant {} not found", tenant_id)))?;

        let stats = self.tenant_stats.get(tenant_id)
            .ok_or_else(|| needle::NeedleError::NotFound("Tenant stats not found".to_string()))?;

        // Check vector limit
        if stats.total_vectors >= tenant.plan.max_vectors() {
            return Err(needle::NeedleError::QuotaExceeded(format!(
                "Vector limit reached for {:?} plan ({} max)",
                tenant.plan, tenant.plan.max_vectors()
            )));
        }

        let full_name = format!("{}_{}", tenant_id, collection);
        let full_id = format!("{}_{}", tenant_id, id);

        let coll = self.db.collection(&full_name)?;

        // Add tenant_id to metadata for audit
        let mut meta = metadata.unwrap_or_else(|| json!({}));
        if let Some(obj) = meta.as_object_mut() {
            obj.insert("__tenant_id".to_string(), json!(tenant_id));
        }

        coll.insert(&full_id, vector, Some(meta))?;

        // Update stats
        if let Some(stats) = self.tenant_stats.get_mut(tenant_id) {
            stats.total_vectors += 1;
        }

        Ok(())
    }

    /// Search within a tenant's collection
    fn search(
        &mut self,
        tenant_id: &str,
        collection: &str,
        query: &[f32],
        top_k: usize,
        filter: Option<Filter>,
    ) -> needle::Result<Vec<TenantSearchResult>> {
        if !self.tenants.contains_key(tenant_id) {
            return Err(needle::NeedleError::NotFound(format!("Tenant {} not found", tenant_id)));
        }

        let full_name = format!("{}_{}", tenant_id, collection);
        let coll = self.db.collection(&full_name)?;

        let results = match filter {
            Some(f) => coll.search_with_filter(query, top_k, &f)?,
            None => coll.search(query, top_k)?,
        };

        // Update query count
        if let Some(stats) = self.tenant_stats.get_mut(tenant_id) {
            stats.total_queries += 1;
        }

        // Strip tenant prefix from IDs
        let prefix = format!("{}_", tenant_id);
        Ok(results
            .into_iter()
            .map(|r| TenantSearchResult {
                id: r.id.strip_prefix(&prefix).unwrap_or(&r.id).to_string(),
                distance: r.distance,
                metadata: r.metadata,
            })
            .collect())
    }

    /// Get tenant statistics
    fn get_stats(&self, tenant_id: &str) -> Option<&TenantStats> {
        self.tenant_stats.get(tenant_id)
    }

    /// List all collections for a tenant
    fn list_collections(&self, tenant_id: &str) -> Vec<String> {
        let prefix = format!("{}_", tenant_id);
        self.db
            .list_collections()
            .into_iter()
            .filter(|name| name.starts_with(&prefix))
            .map(|name| name.strip_prefix(&prefix).unwrap_or(&name).to_string())
            .collect()
    }

    /// Delete a tenant and all their data
    #[allow(dead_code)] // Example method for tenant cleanup
    fn delete_tenant(&mut self, tenant_id: &str) -> needle::Result<()> {
        if !self.tenants.contains_key(tenant_id) {
            return Err(needle::NeedleError::NotFound(format!("Tenant {} not found", tenant_id)));
        }

        // Delete all collections
        let collections = self.list_collections(tenant_id);
        for collection in collections {
            let full_name = format!("{}_{}", tenant_id, collection);
            self.db.drop_collection(&full_name)?;
        }

        self.tenants.remove(tenant_id);
        self.tenant_stats.remove(tenant_id);

        println!("Deleted tenant {}", tenant_id);
        Ok(())
    }
}

/// Tenant search result
#[derive(Debug)]
struct TenantSearchResult {
    id: String,
    distance: f32,
    metadata: Option<serde_json::Value>,
}

/// Namespace-based multi-tenancy using Needle's built-in Namespace system
struct NamespaceMultiTenant {
    manager: NamespaceManager,
}

impl NamespaceMultiTenant {
    fn new() -> Self {
        Self {
            manager: NamespaceManager::new(),
        }
    }

    /// Create a namespace for a tenant
    fn create_tenant_namespace(&mut self, tenant_id: &str) -> needle::Result<()> {
        let config = TenantConfig {
            max_vectors: Some(100_000),
            max_collections: Some(10),
            ..TenantConfig::default()
        };

        self.manager.create_namespace(tenant_id, config)?;
        println!("Created namespace for tenant: {}", tenant_id);
        Ok(())
    }

    /// Get tenant namespace
    fn get_namespace(&self, tenant_id: &str) -> Option<std::sync::Arc<Namespace>> {
        self.manager.namespace(tenant_id)
    }
}

/// Generate a mock embedding
fn mock_embedding(seed: u64, dim: usize) -> Vec<f32> {
    let mut rng_state = seed;
    let embedding: Vec<f32> = (0..dim)
        .map(|_| {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng_state >> 16) as f32 / 32768.0) - 1.0
        })
        .collect();

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    embedding.into_iter().map(|x| x / norm).collect()
}

fn main() -> needle::Result<()> {
    println!("=== Multi-Tenant Vector Database Example ===\n");

    // Create multi-tenant manager
    let mut mtdb = MultiTenantVectorDB::new();

    // Register tenants with different plans
    mtdb.register_tenant(Tenant {
        id: "acme".to_string(),
        name: "Acme Corporation".to_string(),
        plan: TenantPlan::Enterprise,
    }).unwrap();

    mtdb.register_tenant(Tenant {
        id: "startup".to_string(),
        name: "Cool Startup Inc".to_string(),
        plan: TenantPlan::Pro,
    }).unwrap();

    mtdb.register_tenant(Tenant {
        id: "hobbyist".to_string(),
        name: "Hobbyist User".to_string(),
        plan: TenantPlan::Free,
    }).unwrap();

    println!();

    // Create collections for each tenant
    let embedding_dim = 384;

    mtdb.create_collection("acme", "products", embedding_dim)?;
    mtdb.create_collection("acme", "documents", embedding_dim)?;
    mtdb.create_collection("startup", "embeddings", embedding_dim)?;
    mtdb.create_collection("hobbyist", "vectors", embedding_dim)?;

    println!();

    // Insert data for each tenant
    println!("=== Inserting Data ===");

    // Acme products
    for i in 0..10 {
        let embedding = mock_embedding(1000 + i, embedding_dim);
        mtdb.insert(
            "acme",
            "products",
            &format!("product_{}", i),
            &embedding,
            Some(json!({
                "name": format!("Product {}", i),
                "category": if i % 2 == 0 { "electronics" } else { "clothing" },
                "price": 10.0 + (i as f64 * 5.0),
            })),
        )?;
    }
    println!("Inserted 10 products for Acme");

    // Startup embeddings
    for i in 0..5 {
        let embedding = mock_embedding(2000 + i, embedding_dim);
        mtdb.insert(
            "startup",
            "embeddings",
            &format!("doc_{}", i),
            &embedding,
            Some(json!({ "title": format!("Document {}", i) })),
        )?;
    }
    println!("Inserted 5 embeddings for Startup");

    // Hobbyist vectors
    for i in 0..3 {
        let embedding = mock_embedding(3000 + i, embedding_dim);
        mtdb.insert(
            "hobbyist",
            "vectors",
            &format!("vec_{}", i),
            &embedding,
            None,
        )?;
    }
    println!("Inserted 3 vectors for Hobbyist");

    println!();

    // Search within tenant scope
    println!("=== Tenant-Scoped Search ===");

    let query = mock_embedding(1000, embedding_dim); // Similar to product_0

    println!("\nAcme searching products:");
    let results = mtdb.search("acme", "products", &query, 3, None)?;
    for r in &results {
        println!("  {} (distance: {:.4}) - {:?}",
            r.id, r.distance, r.metadata.as_ref().and_then(|m| m.get("name"))
        );
    }

    println!("\nStartup searching embeddings:");
    let results = mtdb.search("startup", "embeddings", &query, 3, None)?;
    for r in &results {
        println!("  {} (distance: {:.4})", r.id, r.distance);
    }

    // Search with filter
    println!("\nAcme searching electronics only:");
    let filter = Filter::eq("category", "electronics");
    let results = mtdb.search("acme", "products", &query, 5, Some(filter))?;
    for r in &results {
        println!("  {} (distance: {:.4}) - {:?}",
            r.id, r.distance, r.metadata.as_ref().and_then(|m| m.get("category"))
        );
    }

    println!();

    // Show tenant stats
    println!("=== Tenant Statistics ===");

    for tenant_id in &["acme", "startup", "hobbyist"] {
        if let (Some(tenant), Some(stats)) = (mtdb.get_tenant(tenant_id), mtdb.get_stats(tenant_id)) {
            println!("\n{} ({:?} plan):", tenant.name, tenant.plan);
            println!("  Collections: {}/{}", stats.total_collections, tenant.plan.max_collections());
            println!("  Vectors: {}/{}", stats.total_vectors,
                if tenant.plan.max_vectors() == usize::MAX { "unlimited".to_string() }
                else { tenant.plan.max_vectors().to_string() }
            );
            println!("  Queries: {}", stats.total_queries);
        }
    }

    println!();

    // Demonstrate plan limits
    println!("=== Plan Limit Enforcement ===");

    // Try to create too many collections for free tier
    println!("\nTrying to create 4th collection for Free tier user...");
    match mtdb.create_collection("hobbyist", "extra1", embedding_dim) {
        Ok(_) => println!("  Success: Created collection"),
        Err(e) => println!("  Expected error: {}", e),
    }
    match mtdb.create_collection("hobbyist", "extra2", embedding_dim) {
        Ok(_) => println!("  Success: Created collection"),
        Err(e) => println!("  Expected error: {}", e),
    }
    match mtdb.create_collection("hobbyist", "extra3", embedding_dim) {
        Ok(_) => println!("  Success: Created collection"),
        Err(e) => println!("  Error (quota exceeded): {}", e),
    }

    // Try to use dimensions exceeding plan limit
    println!("\nTrying to create collection with 1024 dimensions for Free tier...");
    match mtdb.create_collection("hobbyist", "large_dim", 1024) {
        Ok(_) => println!("  Success: Created collection"),
        Err(e) => println!("  Error (quota exceeded): {}", e),
    }

    println!();

    // List collections per tenant
    println!("=== Collections per Tenant ===");
    for tenant_id in &["acme", "startup", "hobbyist"] {
        let collections = mtdb.list_collections(tenant_id);
        println!("{}: {:?}", tenant_id, collections);
    }

    println!();

    // Demonstrate data isolation
    println!("=== Data Isolation Verification ===");

    println!("Verifying Startup cannot see Acme's products...");
    // Startup tries to search Acme's products
    match mtdb.search("startup", "products", &query, 5, None) {
        Ok(_) => println!("  ERROR: Startup can access Acme's products!"),
        Err(_) => println!("  OK: Startup cannot access Acme's products (collection not found)"),
    }

    // Using Needle's built-in namespace system
    println!("\n=== Using Needle's Built-in Namespace System ===");

    let mut ns_mtdb = NamespaceMultiTenant::new();

    ns_mtdb.create_tenant_namespace("tenant_a")?;
    ns_mtdb.create_tenant_namespace("tenant_b")?;

    if let Some(ns) = ns_mtdb.get_namespace("tenant_a") {
        println!("Tenant A namespace created: {}", ns.id());
        println!("  Max vectors: {:?}", ns.config().max_vectors);
    }

    println!("\nMulti-tenant example complete!");
    Ok(())
}
