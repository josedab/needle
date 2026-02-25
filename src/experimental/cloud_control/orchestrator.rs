#![allow(clippy::unwrap_used)]

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{NeedleError, Result};

use super::control_plane::current_timestamp;
use super::regions::Region;

/// Represents a managed Needle database instance for a tenant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedInstance {
    pub instance_id: String,
    pub tenant_id: String,
    pub region: Region,
    pub status: InstanceStatus,
    pub database_path: String,
    pub allocated_memory_bytes: u64,
    pub allocated_storage_bytes: u64,
    pub created_at: u64,
    pub last_heartbeat: u64,
}

/// Status of a managed instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstanceStatus {
    Provisioning,
    Running,
    Stopping,
    Stopped,
    Failed,
    Migrating,
}

/// Configuration for the service orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    pub default_region: Region,
    pub heartbeat_timeout_seconds: u64,
    pub max_instances_per_tenant: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            default_region: Region::UsEast1,
            heartbeat_timeout_seconds: 60,
            max_instances_per_tenant: 5,
        }
    }
}

/// Orchestrates provisioning, scaling, and lifecycle of managed Needle instances.
pub struct ServiceOrchestrator {
    config: OrchestratorConfig,
    instances: RwLock<HashMap<String, ManagedInstance>>,
    instance_counter: AtomicU64,
}

impl ServiceOrchestrator {
    pub fn new(config: OrchestratorConfig) -> Self {
        Self {
            config,
            instances: RwLock::new(HashMap::new()),
            instance_counter: AtomicU64::new(0),
        }
    }

    /// Provision a new managed instance for a tenant in a given region.
    pub fn provision(
        &self,
        tenant_id: &str,
        region: Option<Region>,
        memory_bytes: u64,
        storage_bytes: u64,
    ) -> Result<ManagedInstance> {
        let tenant_instances: Vec<_> = self
            .instances
            .read()
            .values()
            .filter(|i| i.tenant_id == tenant_id && i.status != InstanceStatus::Stopped)
            .cloned()
            .collect();

        if tenant_instances.len() >= self.config.max_instances_per_tenant {
            return Err(NeedleError::InvalidOperation(format!(
                "Tenant {} already has {} instances (max {})",
                tenant_id,
                tenant_instances.len(),
                self.config.max_instances_per_tenant
            )));
        }

        let seq = self.instance_counter.fetch_add(1, Ordering::Relaxed);
        let region = region.unwrap_or_else(|| self.config.default_region.clone());
        let instance_id = format!("inst_{:012x}", seq);
        let database_path = format!("/data/{}/{}.needle", tenant_id, instance_id);

        let instance = ManagedInstance {
            instance_id: instance_id.clone(),
            tenant_id: tenant_id.to_string(),
            region,
            status: InstanceStatus::Provisioning,
            database_path,
            allocated_memory_bytes: memory_bytes,
            allocated_storage_bytes: storage_bytes,
            created_at: current_timestamp(),
            last_heartbeat: current_timestamp(),
        };

        self.instances
            .write()
            .insert(instance_id.clone(), instance.clone());
        Ok(instance)
    }

    /// Mark an instance as running after provisioning completes.
    pub fn mark_running(&self, instance_id: &str) -> Result<()> {
        let mut instances = self.instances.write();
        let inst = instances
            .get_mut(instance_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Instance {} not found", instance_id)))?;
        inst.status = InstanceStatus::Running;
        inst.last_heartbeat = current_timestamp();
        Ok(())
    }

    /// Record a heartbeat from a running instance.
    pub fn heartbeat(&self, instance_id: &str) -> Result<()> {
        let mut instances = self.instances.write();
        let inst = instances
            .get_mut(instance_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Instance {} not found", instance_id)))?;
        inst.last_heartbeat = current_timestamp();
        Ok(())
    }

    /// Stop an instance.
    pub fn stop(&self, instance_id: &str) -> Result<()> {
        let mut instances = self.instances.write();
        let inst = instances
            .get_mut(instance_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Instance {} not found", instance_id)))?;
        inst.status = InstanceStatus::Stopped;
        Ok(())
    }

    /// Detect instances that have missed heartbeats and mark them failed.
    pub fn detect_failures(&self) -> Vec<String> {
        let now = current_timestamp();
        let timeout = self.config.heartbeat_timeout_seconds;
        let mut failed = Vec::new();
        let mut instances = self.instances.write();
        for inst in instances.values_mut() {
            if inst.status == InstanceStatus::Running
                && now.saturating_sub(inst.last_heartbeat) > timeout
            {
                inst.status = InstanceStatus::Failed;
                failed.push(inst.instance_id.clone());
            }
        }
        failed
    }

    /// List all instances for a tenant.
    pub fn list_instances(&self, tenant_id: &str) -> Vec<ManagedInstance> {
        self.instances
            .read()
            .values()
            .filter(|i| i.tenant_id == tenant_id)
            .cloned()
            .collect()
    }

    /// Get a single instance by ID.
    pub fn get_instance(&self, instance_id: &str) -> Option<ManagedInstance> {
        self.instances.read().get(instance_id).cloned()
    }
}
