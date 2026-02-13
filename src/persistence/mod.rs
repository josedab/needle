//! WAL, transactions, time-travel, backups, cloud storage.
#![allow(clippy::unwrap_used)] // tech debt: 609 unwrap() calls remaining
pub mod backup;
pub mod cloud_storage;
pub mod cow_storage;
pub mod managed_backup;
pub mod migrations;
pub mod schema_evolution;
pub mod shard;
pub mod snapshot_replication;
pub mod sync_protocol;
pub mod tiered;
pub mod time_travel;
pub mod transaction;
pub mod versioning;
pub mod vector_versioning;
pub mod replica_manager;
pub mod wal;
/// Incremental backup with WAL-based point-in-time recovery (PITR).
pub mod incremental_backup;
