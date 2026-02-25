#![allow(clippy::unwrap_used)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::tiers::TenantConfig;

/// Tenant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tenant {
    /// Unique tenant ID
    pub id: String,
    /// Configuration
    pub config: TenantConfig,
    /// Status
    pub status: TenantStatus,
    /// Created timestamp
    pub created_at: u64,
    /// Updated timestamp
    pub updated_at: u64,
    /// Current usage
    pub usage: TenantUsage,
}

/// Tenant status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TenantStatus {
    /// Active and operational
    Active,
    /// Provisioning in progress
    Provisioning,
    /// Suspended (e.g., payment issue)
    Suspended,
    /// Scheduled for deletion
    PendingDeletion,
    /// Deleted
    Deleted,
}

/// Current tenant usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TenantUsage {
    pub collections_count: u64,
    pub vectors_count: u64,
    pub storage_bytes: u64,
    pub queries_this_minute: u64,
    pub requests_today: u64,
    pub total_queries: u64,
    pub total_inserts: u64,
    pub total_deletes: u64,
}

/// API key for tenant access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// Key ID (public)
    pub id: String,
    /// Key hash (stored, never exposed)
    pub key_hash: String,
    /// Display name
    pub name: String,
    /// Tenant ID this key belongs to
    pub tenant_id: String,
    /// Permissions
    pub permissions: Vec<Permission>,
    /// Created timestamp
    pub created_at: u64,
    /// Expires timestamp (if any)
    pub expires_at: Option<u64>,
    /// Last used timestamp
    pub last_used_at: Option<u64>,
    /// Is active
    pub active: bool,
}

/// API permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Permission {
    /// Read vectors
    Read,
    /// Write vectors
    Write,
    /// Delete vectors
    Delete,
    /// Manage collections
    ManageCollections,
    /// View metrics
    ViewMetrics,
    /// Admin access
    Admin,
}

/// Billing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingInfo {
    /// Tenant ID
    pub tenant_id: String,
    /// Current billing period start
    pub period_start: u64,
    /// Current billing period end
    pub period_end: u64,
    /// Base charge (tier subscription)
    pub base_charge_cents: u64,
    /// Overage charges
    pub overage_charges: Vec<OverageCharge>,
    /// Total amount due
    pub total_cents: u64,
    /// Payment status
    pub status: PaymentStatus,
}

/// Overage charge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverageCharge {
    /// Resource type
    pub resource: String,
    /// Amount over limit
    pub overage_amount: u64,
    /// Unit price in cents
    pub unit_price_cents: u64,
    /// Total charge
    pub charge_cents: u64,
}

/// Payment status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaymentStatus {
    Pending,
    Paid,
    Failed,
    Waived,
}

/// Usage event for metering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEvent {
    /// Event ID
    pub id: String,
    /// Tenant ID
    pub tenant_id: String,
    /// Event type
    pub event_type: UsageEventType,
    /// Quantity
    pub quantity: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Usage event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsageEventType {
    Query,
    Insert,
    Delete,
    StorageIncrease,
    StorageDecrease,
    ApiCall,
}
