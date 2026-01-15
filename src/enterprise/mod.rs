//! Enterprise features: security, encryption, multi-tenancy, RBAC, Raft consensus.

pub mod security;
#[cfg(feature = "encryption")]
pub mod encryption;
pub mod raft;
pub mod replicated_database;
pub mod tenant_isolation;
pub mod namespace;
pub mod autoscaling;
