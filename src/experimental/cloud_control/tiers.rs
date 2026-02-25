#![allow(clippy::unwrap_used)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Resource tier for tenant plans
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceTier {
    /// Free tier - limited resources
    Free,
    /// Developer tier - for small projects
    Developer,
    /// Professional tier - for production use
    Professional,
    /// Enterprise tier - custom limits
    Enterprise,
}

impl Default for ResourceTier {
    fn default() -> Self {
        ResourceTier::Free
    }
}

impl ResourceTier {
    /// Get tier limits
    pub fn limits(&self) -> TierLimits {
        match self {
            ResourceTier::Free => TierLimits {
                max_collections: 3,
                max_vectors: 10_000,
                max_dimensions: 768,
                max_storage_bytes: 100 * 1024 * 1024, // 100MB
                queries_per_minute: 60,
                requests_per_day: 10_000,
                support_level: SupportLevel::Community,
            },
            ResourceTier::Developer => TierLimits {
                max_collections: 10,
                max_vectors: 100_000,
                max_dimensions: 1536,
                max_storage_bytes: 1024 * 1024 * 1024, // 1GB
                queries_per_minute: 300,
                requests_per_day: 100_000,
                support_level: SupportLevel::Email,
            },
            ResourceTier::Professional => TierLimits {
                max_collections: 50,
                max_vectors: 1_000_000,
                max_dimensions: 4096,
                max_storage_bytes: 10 * 1024 * 1024 * 1024, // 10GB
                queries_per_minute: 1000,
                requests_per_day: 1_000_000,
                support_level: SupportLevel::Priority,
            },
            ResourceTier::Enterprise => TierLimits {
                max_collections: u64::MAX,
                max_vectors: u64::MAX,
                max_dimensions: 8192,
                max_storage_bytes: u64::MAX,
                queries_per_minute: u64::MAX,
                requests_per_day: u64::MAX,
                support_level: SupportLevel::Dedicated,
            },
        }
    }

    /// Get monthly price in cents
    pub fn price_cents(&self) -> u64 {
        match self {
            ResourceTier::Free => 0,
            ResourceTier::Developer => 2900,    // $29/month
            ResourceTier::Professional => 9900, // $99/month
            ResourceTier::Enterprise => 0,      // Custom pricing
        }
    }
}

/// Tier resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierLimits {
    pub max_collections: u64,
    pub max_vectors: u64,
    pub max_dimensions: usize,
    pub max_storage_bytes: u64,
    pub queries_per_minute: u64,
    pub requests_per_day: u64,
    pub support_level: SupportLevel,
}

/// Support level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupportLevel {
    Community,
    Email,
    Priority,
    Dedicated,
}

/// Tenant configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    /// Tenant name
    pub name: String,
    /// Resource tier
    pub tier: ResourceTier,
    /// Organization email
    pub email: String,
    /// Custom limits override (for Enterprise)
    pub custom_limits: Option<TierLimits>,
    /// Enabled features
    pub features: Vec<String>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl Default for TenantConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            tier: ResourceTier::Free,
            email: String::new(),
            custom_limits: None,
            features: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}
