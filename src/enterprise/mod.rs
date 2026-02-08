//! Enterprise features: security, encryption, multi-tenancy, RBAC, Raft consensus.

#![allow(clippy::unwrap_used)] // tech debt: per-module unwrap cleanup in progress
pub mod autoscaling;
#[cfg(feature = "encryption")]
pub mod encryption;
pub mod namespace;
pub mod raft;
pub mod replicated_database;
pub mod security;
pub mod tenant_isolation;
