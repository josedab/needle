#![allow(clippy::unwrap_used)]

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::control_plane::{current_timestamp, generate_api_key, hash_api_key};
use super::tenant::Permission;

/// An API key entry with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyEntry {
    pub key_id: String,
    pub key_hash: String,
    pub tenant_id: String,
    pub name: String,
    pub permissions: Vec<Permission>,
    pub created_at: u64,
    pub last_used: Option<u64>,
    pub expires_at: Option<u64>,
    pub active: bool,
}

/// Manages API keys independently of the control plane.
pub struct ApiKeyManager {
    keys: RwLock<HashMap<String, ApiKeyEntry>>,
    next_id: AtomicU64,
}

impl ApiKeyManager {
    /// Create a new API key manager.
    pub fn new() -> Self {
        Self {
            keys: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Create a new API key for a tenant. Returns `(raw_key, entry)`.
    pub fn create_key(
        &self,
        tenant_id: &str,
        name: &str,
        permissions: Vec<Permission>,
    ) -> (String, ApiKeyEntry) {
        let key_id = format!("akey_{}", self.next_id.fetch_add(1, Ordering::SeqCst));
        let raw_key = generate_api_key();
        let key_hash = hash_api_key(&raw_key);

        let entry = ApiKeyEntry {
            key_id: key_id.clone(),
            key_hash,
            tenant_id: tenant_id.to_string(),
            name: name.to_string(),
            permissions,
            created_at: current_timestamp(),
            last_used: None,
            expires_at: None,
            active: true,
        };

        self.keys.write().insert(key_id, entry.clone());
        (raw_key, entry)
    }

    /// Validate a key by its hash. Returns the matching entry if found and active.
    pub fn validate_key(&self, key_hash: &str) -> Option<ApiKeyEntry> {
        self.keys
            .read()
            .values()
            .find(|e| e.key_hash == key_hash && e.active)
            .cloned()
    }

    /// Revoke a key by ID. Returns `true` if the key was found and revoked.
    pub fn revoke_key(&self, key_id: &str) -> bool {
        let mut keys = self.keys.write();
        if let Some(entry) = keys.get_mut(key_id) {
            entry.active = false;
            true
        } else {
            false
        }
    }

    /// List all keys belonging to a tenant.
    pub fn list_keys(&self, tenant_id: &str) -> Vec<ApiKeyEntry> {
        self.keys
            .read()
            .values()
            .filter(|e| e.tenant_id == tenant_id)
            .cloned()
            .collect()
    }
}

impl Default for ApiKeyManager {
    fn default() -> Self {
        Self::new()
    }
}
