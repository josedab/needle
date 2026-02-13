//! Enterprise features: security, encryption, multi-tenancy, RBAC, Raft consensus.

#![allow(clippy::unwrap_used)] // tech debt: 200 unwrap() calls remaining
pub mod autoscaling;
#[cfg(feature = "encryption")]
pub mod encryption;
pub mod namespace;
pub mod raft;
pub mod replicated_database;
pub mod security;
pub mod tenant_isolation;
/// Per-vector access control lists (ACLs) and row-level security policies.
pub mod vector_acl;
