//! Schema Versioning & Migrations
//!
//! Provides database schema versioning and migration capabilities for Needle databases.
//! Supports forward and backward migrations with automatic version tracking.
//!
//! # Features
//!
//! - **Version tracking**: Track schema version in database metadata
//! - **Migration definitions**: Define up/down migrations
//! - **Automatic migrations**: Run pending migrations on database open
//! - **Migration history**: Track which migrations have been applied
//! - **Rollback support**: Revert to previous schema versions
//! - **Compatibility checks**: Validate database compatibility before operations
//!
//! # Example
//!
//! ```ignore
//! use needle::migrations::{MigrationManager, Migration, SchemaVersion};
//!
//! let mut manager = MigrationManager::new();
//!
//! // Register migrations
//! manager.register(Migration::new(
//!     "001_initial_schema",
//!     SchemaVersion::new(1, 0, 0),
//!     |db| {
//!         // Migration up logic
//!         Ok(())
//!     },
//!     |db| {
//!         // Migration down logic
//!         Ok(())
//!     },
//! ));
//!
//! // Run pending migrations
//! manager.migrate_up(&mut db)?;
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// Semantic version for schema versioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SchemaVersion {
    /// Major version (breaking changes).
    pub major: u32,
    /// Minor version (backward-compatible features).
    pub minor: u32,
    /// Patch version (backward-compatible fixes).
    pub patch: u32,
}

impl SchemaVersion {
    /// Create a new schema version.
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse from string (e.g., "1.2.3").
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return None;
        }

        Some(Self {
            major: parts[0].parse().ok()?,
            minor: parts[1].parse().ok()?,
            patch: parts[2].parse().ok()?,
        })
    }

    /// Check if this version is compatible with another.
    /// Compatible means same major version.
    pub fn is_compatible(&self, other: &SchemaVersion) -> bool {
        self.major == other.major
    }

    /// Check if this version is newer than another.
    pub fn is_newer_than(&self, other: &SchemaVersion) -> bool {
        self > other
    }

    /// Get the current Needle schema version.
    pub fn current() -> Self {
        Self::new(1, 0, 0)
    }
}

impl Default for SchemaVersion {
    fn default() -> Self {
        Self::current()
    }
}

impl fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Migration status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationStatus {
    /// Migration is pending.
    Pending,
    /// Migration is running.
    Running,
    /// Migration completed successfully.
    Completed,
    /// Migration failed.
    Failed,
    /// Migration was rolled back.
    RolledBack,
}

/// Record of an applied migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationRecord {
    /// Migration ID.
    pub id: String,
    /// Target version.
    pub version: SchemaVersion,
    /// Migration status.
    pub status: MigrationStatus,
    /// Timestamp when applied.
    pub applied_at: u64,
    /// Timestamp when completed (if applicable).
    pub completed_at: Option<u64>,
    /// Error message (if failed).
    pub error: Option<String>,
    /// Duration in milliseconds.
    pub duration_ms: Option<u64>,
}

/// Migration definition using closures.
pub struct Migration {
    /// Migration ID (should be unique and sortable, e.g., "001_initial").
    pub id: String,
    /// Target schema version after this migration.
    pub version: SchemaVersion,
    /// Description of what this migration does.
    pub description: String,
    /// Up migration function.
    up_fn: Box<dyn Fn(&mut MigrationContext) -> Result<()> + Send + Sync>,
    /// Down migration function (for rollback).
    down_fn: Box<dyn Fn(&mut MigrationContext) -> Result<()> + Send + Sync>,
    /// Whether this migration is reversible.
    pub reversible: bool,
}

impl Migration {
    /// Create a new migration.
    pub fn new<F, G>(
        id: impl Into<String>,
        version: SchemaVersion,
        description: impl Into<String>,
        up_fn: F,
        down_fn: G,
    ) -> Self
    where
        F: Fn(&mut MigrationContext) -> Result<()> + Send + Sync + 'static,
        G: Fn(&mut MigrationContext) -> Result<()> + Send + Sync + 'static,
    {
        Self {
            id: id.into(),
            version,
            description: description.into(),
            up_fn: Box::new(up_fn),
            down_fn: Box::new(down_fn),
            reversible: true,
        }
    }

    /// Create a non-reversible migration.
    pub fn one_way<F>(
        id: impl Into<String>,
        version: SchemaVersion,
        description: impl Into<String>,
        up_fn: F,
    ) -> Self
    where
        F: Fn(&mut MigrationContext) -> Result<()> + Send + Sync + 'static,
    {
        Self {
            id: id.into(),
            version,
            description: description.into(),
            up_fn: Box::new(up_fn),
            down_fn: Box::new(|_| {
                Err(NeedleError::InvalidInput(
                    "Migration is not reversible".to_string(),
                ))
            }),
            reversible: false,
        }
    }

    /// Run the up migration.
    pub fn up(&self, ctx: &mut MigrationContext) -> Result<()> {
        (self.up_fn)(ctx)
    }

    /// Run the down migration.
    pub fn down(&self, ctx: &mut MigrationContext) -> Result<()> {
        if !self.reversible {
            return Err(NeedleError::InvalidInput(format!(
                "Migration {} is not reversible",
                self.id
            )));
        }
        (self.down_fn)(ctx)
    }
}

impl fmt::Debug for Migration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Migration")
            .field("id", &self.id)
            .field("version", &self.version)
            .field("description", &self.description)
            .field("reversible", &self.reversible)
            .finish()
    }
}

/// Context passed to migration functions.
#[derive(Debug)]
pub struct MigrationContext {
    /// Current schema version.
    pub current_version: SchemaVersion,
    /// Target schema version.
    pub target_version: SchemaVersion,
    /// Migration metadata.
    pub metadata: HashMap<String, String>,
    /// Operations performed (for logging/auditing).
    operations: Vec<MigrationOperation>,
    /// Whether this is a dry run.
    pub dry_run: bool,
}

impl MigrationContext {
    /// Create a new migration context.
    pub fn new(current_version: SchemaVersion, target_version: SchemaVersion) -> Self {
        Self {
            current_version,
            target_version,
            metadata: HashMap::new(),
            operations: Vec::new(),
            dry_run: false,
        }
    }

    /// Record an operation.
    pub fn record_operation(&mut self, op: MigrationOperation) {
        self.operations.push(op);
    }

    /// Get recorded operations.
    pub fn operations(&self) -> &[MigrationOperation] {
        &self.operations
    }

    /// Add metadata.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Create collection operation.
    pub fn create_collection(&mut self, name: &str, dimensions: usize) {
        self.record_operation(MigrationOperation::CreateCollection {
            name: name.to_string(),
            dimensions,
        });
    }

    /// Drop collection operation.
    pub fn drop_collection(&mut self, name: &str) {
        self.record_operation(MigrationOperation::DropCollection {
            name: name.to_string(),
        });
    }

    /// Rename collection operation.
    pub fn rename_collection(&mut self, old_name: &str, new_name: &str) {
        self.record_operation(MigrationOperation::RenameCollection {
            old_name: old_name.to_string(),
            new_name: new_name.to_string(),
        });
    }

    /// Add index operation.
    pub fn create_index(&mut self, collection: &str, index_type: IndexType) {
        self.record_operation(MigrationOperation::CreateIndex {
            collection: collection.to_string(),
            index_type,
        });
    }

    /// Drop index operation.
    pub fn drop_index(&mut self, collection: &str, index_type: IndexType) {
        self.record_operation(MigrationOperation::DropIndex {
            collection: collection.to_string(),
            index_type,
        });
    }

    /// Update collection config operation.
    pub fn update_config(&mut self, collection: &str, config_key: &str, config_value: &str) {
        self.record_operation(MigrationOperation::UpdateConfig {
            collection: collection.to_string(),
            key: config_key.to_string(),
            value: config_value.to_string(),
        });
    }

    /// Add metadata field operation.
    pub fn add_metadata_field(&mut self, collection: &str, field_name: &str, default_value: &str) {
        self.record_operation(MigrationOperation::AddMetadataField {
            collection: collection.to_string(),
            field_name: field_name.to_string(),
            default_value: default_value.to_string(),
        });
    }

    /// Remove metadata field operation.
    pub fn remove_metadata_field(&mut self, collection: &str, field_name: &str) {
        self.record_operation(MigrationOperation::RemoveMetadataField {
            collection: collection.to_string(),
            field_name: field_name.to_string(),
        });
    }

    /// Execute custom SQL-like command.
    pub fn execute(&mut self, command: &str) {
        self.record_operation(MigrationOperation::Custom {
            command: command.to_string(),
        });
    }
}

/// Index type for migrations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    /// HNSW index.
    Hnsw,
    /// IVF index.
    Ivf,
    /// DiskANN index.
    DiskAnn,
    /// Metadata index on a field.
    Metadata(String),
}

/// Migration operation types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationOperation {
    /// Create a new collection.
    CreateCollection { name: String, dimensions: usize },
    /// Drop a collection.
    DropCollection { name: String },
    /// Rename a collection.
    RenameCollection { old_name: String, new_name: String },
    /// Create an index.
    CreateIndex {
        collection: String,
        index_type: IndexType,
    },
    /// Drop an index.
    DropIndex {
        collection: String,
        index_type: IndexType,
    },
    /// Update collection configuration.
    UpdateConfig {
        collection: String,
        key: String,
        value: String,
    },
    /// Add a metadata field with default value.
    AddMetadataField {
        collection: String,
        field_name: String,
        default_value: String,
    },
    /// Remove a metadata field.
    RemoveMetadataField {
        collection: String,
        field_name: String,
    },
    /// Custom operation.
    Custom { command: String },
}

/// Migration manager for tracking and running migrations.
pub struct MigrationManager {
    /// Registered migrations.
    migrations: Vec<Migration>,
    /// Applied migrations.
    history: Vec<MigrationRecord>,
    /// Current schema version.
    current_version: SchemaVersion,
    /// Whether to allow destructive migrations.
    allow_destructive: bool,
}

impl MigrationManager {
    /// Create a new migration manager.
    pub fn new() -> Self {
        Self {
            migrations: Vec::new(),
            history: Vec::new(),
            current_version: SchemaVersion::new(0, 0, 0),
            allow_destructive: false,
        }
    }

    /// Create with initial version.
    pub fn with_version(version: SchemaVersion) -> Self {
        Self {
            migrations: Vec::new(),
            history: Vec::new(),
            current_version: version,
            allow_destructive: false,
        }
    }

    /// Allow destructive migrations.
    pub fn allow_destructive(mut self) -> Self {
        self.allow_destructive = true;
        self
    }

    /// Register a migration.
    pub fn register(&mut self, migration: Migration) {
        // Insert in sorted order by ID
        let pos = self
            .migrations
            .binary_search_by(|m| m.id.cmp(&migration.id))
            .unwrap_or_else(|p| p);
        self.migrations.insert(pos, migration);
    }

    /// Get current schema version.
    pub fn current_version(&self) -> SchemaVersion {
        self.current_version
    }

    /// Set current version (from database metadata).
    pub fn set_current_version(&mut self, version: SchemaVersion) {
        self.current_version = version;
    }

    /// Load migration history.
    pub fn load_history(&mut self, history: Vec<MigrationRecord>) {
        self.history = history;
    }

    /// Get migration history.
    pub fn history(&self) -> &[MigrationRecord] {
        &self.history
    }

    /// Get pending migrations.
    pub fn pending(&self) -> Vec<&Migration> {
        let applied: std::collections::HashSet<_> = self
            .history
            .iter()
            .filter(|r| r.status == MigrationStatus::Completed)
            .map(|r| &r.id)
            .collect();

        self.migrations
            .iter()
            .filter(|m| !applied.contains(&m.id))
            .collect()
    }

    /// Get applied migrations.
    pub fn applied(&self) -> Vec<&MigrationRecord> {
        self.history
            .iter()
            .filter(|r| r.status == MigrationStatus::Completed)
            .collect()
    }

    /// Check if there are pending migrations.
    pub fn has_pending(&self) -> bool {
        !self.pending().is_empty()
    }

    /// Run all pending migrations.
    pub fn migrate_up(&mut self) -> Result<MigrationResult> {
        // Collect pending migration indices to avoid borrow issues
        let applied: std::collections::HashSet<_> = self
            .history
            .iter()
            .filter(|r| r.status == MigrationStatus::Completed)
            .map(|r| r.id.clone())
            .collect();

        let pending_indices: Vec<usize> = self
            .migrations
            .iter()
            .enumerate()
            .filter(|(_, m)| !applied.contains(&m.id))
            .map(|(i, _)| i)
            .collect();

        if pending_indices.is_empty() {
            return Ok(MigrationResult {
                migrations_run: 0,
                final_version: self.current_version,
                operations: vec![],
                errors: vec![],
            });
        }

        let mut result = MigrationResult {
            migrations_run: 0,
            final_version: self.current_version,
            operations: vec![],
            errors: vec![],
        };

        for idx in pending_indices {
            let migration_id = self.migrations[idx].id.clone();
            let target_version = self.migrations[idx].version;

            // Create migration record
            let record = MigrationRecord {
                id: migration_id.clone(),
                version: target_version,
                status: MigrationStatus::Running,
                applied_at: Self::now(),
                completed_at: None,
                error: None,
                duration_ms: None,
            };
            self.history.push(record);

            // Create context
            let mut ctx = MigrationContext::new(self.current_version, target_version);

            // Run migration
            let start = std::time::Instant::now();
            match self.migrations[idx].up(&mut ctx) {
                Ok(()) => {
                    let duration = start.elapsed().as_millis() as u64;
                    let record_idx = self.history.len() - 1;
                    self.history[record_idx].status = MigrationStatus::Completed;
                    self.history[record_idx].completed_at = Some(Self::now());
                    self.history[record_idx].duration_ms = Some(duration);

                    self.current_version = target_version;
                    result.migrations_run += 1;
                    result.final_version = target_version;
                    result.operations.extend(ctx.operations().iter().cloned());
                }
                Err(e) => {
                    let record_idx = self.history.len() - 1;
                    self.history[record_idx].status = MigrationStatus::Failed;
                    self.history[record_idx].error = Some(e.to_string());

                    result.errors.push(MigrationError {
                        migration_id,
                        error: e.to_string(),
                    });

                    // Stop on first error
                    break;
                }
            }
        }

        Ok(result)
    }

    /// Migrate to a specific version.
    pub fn migrate_to(&mut self, target_version: SchemaVersion) -> Result<MigrationResult> {
        if target_version > self.current_version {
            // Migrate up
            self.migrate_up_to(target_version)
        } else if target_version < self.current_version {
            // Migrate down
            self.migrate_down_to(target_version)
        } else {
            // Already at target version
            Ok(MigrationResult {
                migrations_run: 0,
                final_version: self.current_version,
                operations: vec![],
                errors: vec![],
            })
        }
    }

    /// Migrate up to a specific version.
    fn migrate_up_to(&mut self, target_version: SchemaVersion) -> Result<MigrationResult> {
        // Collect applied IDs
        let applied: std::collections::HashSet<_> = self
            .history
            .iter()
            .filter(|r| r.status == MigrationStatus::Completed)
            .map(|r| r.id.clone())
            .collect();

        // Find pending migrations up to target version
        let pending_indices: Vec<usize> = self
            .migrations
            .iter()
            .enumerate()
            .filter(|(_, m)| !applied.contains(&m.id) && m.version <= target_version)
            .map(|(i, _)| i)
            .collect();

        let mut result = MigrationResult {
            migrations_run: 0,
            final_version: self.current_version,
            operations: vec![],
            errors: vec![],
        };

        for idx in pending_indices {
            let migration_id = self.migrations[idx].id.clone();
            let migration_version = self.migrations[idx].version;

            let record = MigrationRecord {
                id: migration_id.clone(),
                version: migration_version,
                status: MigrationStatus::Running,
                applied_at: Self::now(),
                completed_at: None,
                error: None,
                duration_ms: None,
            };
            self.history.push(record);

            let mut ctx = MigrationContext::new(self.current_version, migration_version);

            let start = std::time::Instant::now();
            match self.migrations[idx].up(&mut ctx) {
                Ok(()) => {
                    let duration = start.elapsed().as_millis() as u64;
                    let record_idx = self.history.len() - 1;
                    self.history[record_idx].status = MigrationStatus::Completed;
                    self.history[record_idx].completed_at = Some(Self::now());
                    self.history[record_idx].duration_ms = Some(duration);

                    self.current_version = migration_version;
                    result.migrations_run += 1;
                    result.final_version = migration_version;
                    result.operations.extend(ctx.operations().iter().cloned());
                }
                Err(e) => {
                    let record_idx = self.history.len() - 1;
                    self.history[record_idx].status = MigrationStatus::Failed;
                    self.history[record_idx].error = Some(e.to_string());

                    result.errors.push(MigrationError {
                        migration_id,
                        error: e.to_string(),
                    });
                    break;
                }
            }
        }

        Ok(result)
    }

    /// Migrate down to a specific version.
    fn migrate_down_to(&mut self, target_version: SchemaVersion) -> Result<MigrationResult> {
        // Collect rollback info: (migration_id, history_index, migration_index)
        let rollback_info: Vec<(String, usize, Option<usize>)> = self
            .history
            .iter()
            .enumerate()
            .filter(|(_, r)| r.status == MigrationStatus::Completed && r.version > target_version)
            .map(|(hist_idx, r)| {
                let mig_idx = self.migrations.iter().position(|m| m.id == r.id);
                (r.id.clone(), hist_idx, mig_idx)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        let mut result = MigrationResult {
            migrations_run: 0,
            final_version: self.current_version,
            operations: vec![],
            errors: vec![],
        };

        for (migration_id, history_idx, mig_idx_opt) in rollback_info {
            if let Some(mig_idx) = mig_idx_opt {
                if !self.migrations[mig_idx].reversible {
                    result.errors.push(MigrationError {
                        migration_id: self.migrations[mig_idx].id.clone(),
                        error: "Migration is not reversible".to_string(),
                    });
                    break;
                }

                let prev_version = self.migrations[mig_idx]
                    .version
                    .major
                    .checked_sub(1)
                    .map(|m| SchemaVersion::new(m, 0, 0))
                    .unwrap_or(SchemaVersion::new(0, 0, 0));

                let mut ctx = MigrationContext::new(self.current_version, prev_version);

                match self.migrations[mig_idx].down(&mut ctx) {
                    Ok(()) => {
                        // Update history
                        self.history[history_idx].status = MigrationStatus::RolledBack;

                        self.current_version = prev_version;
                        result.migrations_run += 1;
                        result.final_version = prev_version;
                        result.operations.extend(ctx.operations().iter().cloned());
                    }
                    Err(e) => {
                        result.errors.push(MigrationError {
                            migration_id,
                            error: e.to_string(),
                        });
                        break;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Rollback the last migration.
    pub fn rollback(&mut self) -> Result<MigrationResult> {
        // Find the last applied migration
        let last_applied_idx = self
            .history
            .iter()
            .enumerate()
            .rev()
            .find(|(_, r)| r.status == MigrationStatus::Completed)
            .map(|(i, _)| i);

        let last_applied_idx = match last_applied_idx {
            Some(idx) => idx,
            None => {
                return Ok(MigrationResult {
                    migrations_run: 0,
                    final_version: self.current_version,
                    operations: vec![],
                    errors: vec![],
                })
            }
        };

        let migration_id = self.history[last_applied_idx].id.clone();

        // Find the migration
        let mig_idx = self.migrations.iter().position(|m| m.id == migration_id);

        let mig_idx = match mig_idx {
            Some(idx) => idx,
            None => {
                return Ok(MigrationResult {
                    migrations_run: 0,
                    final_version: self.current_version,
                    operations: vec![],
                    errors: vec![],
                })
            }
        };

        // Check if reversible
        if !self.migrations[mig_idx].reversible {
            return Ok(MigrationResult {
                migrations_run: 0,
                final_version: self.current_version,
                operations: vec![],
                errors: vec![MigrationError {
                    migration_id,
                    error: "Migration is not reversible".to_string(),
                }],
            });
        }

        // Find the previous version
        let prev_version = self
            .history
            .iter()
            .filter(|r| r.status == MigrationStatus::Completed && r.id != migration_id)
            .map(|r| r.version)
            .max()
            .unwrap_or(SchemaVersion::new(0, 0, 0));

        let mut ctx = MigrationContext::new(self.current_version, prev_version);

        match self.migrations[mig_idx].down(&mut ctx) {
            Ok(()) => {
                self.history[last_applied_idx].status = MigrationStatus::RolledBack;
                self.current_version = prev_version;

                Ok(MigrationResult {
                    migrations_run: 1,
                    final_version: prev_version,
                    operations: ctx.operations().to_vec(),
                    errors: vec![],
                })
            }
            Err(e) => Ok(MigrationResult {
                migrations_run: 0,
                final_version: self.current_version,
                operations: vec![],
                errors: vec![MigrationError {
                    migration_id,
                    error: e.to_string(),
                }],
            }),
        }
    }

    /// Preview migrations without applying.
    pub fn preview(&self) -> Vec<MigrationPreview> {
        self.pending()
            .into_iter()
            .map(|m| {
                let mut ctx = MigrationContext::new(self.current_version, m.version);
                ctx.dry_run = true;
                let _ = m.up(&mut ctx);

                MigrationPreview {
                    id: m.id.clone(),
                    version: m.version,
                    description: m.description.clone(),
                    operations: ctx.operations().to_vec(),
                    reversible: m.reversible,
                }
            })
            .collect()
    }

    /// Validate that migrations can be applied.
    pub fn validate(&self) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        // Check for duplicate IDs
        let mut seen_ids = std::collections::HashSet::new();
        for m in &self.migrations {
            if !seen_ids.insert(&m.id) {
                issues.push(ValidationIssue {
                    migration_id: Some(m.id.clone()),
                    severity: IssueSeverity::Error,
                    message: format!("Duplicate migration ID: {}", m.id),
                });
            }
        }

        // Check version ordering
        let mut prev_version = SchemaVersion::new(0, 0, 0);
        for m in &self.migrations {
            if m.version <= prev_version {
                issues.push(ValidationIssue {
                    migration_id: Some(m.id.clone()),
                    severity: IssueSeverity::Warning,
                    message: format!(
                        "Migration {} has version {} which is not greater than previous version {}",
                        m.id, m.version, prev_version
                    ),
                });
            }
            prev_version = m.version;
        }

        // Check for non-reversible migrations
        for m in &self.migrations {
            if !m.reversible {
                issues.push(ValidationIssue {
                    migration_id: Some(m.id.clone()),
                    severity: IssueSeverity::Warning,
                    message: format!("Migration {} is not reversible", m.id),
                });
            }
        }

        issues
    }

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

impl Default for MigrationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of running migrations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationResult {
    /// Number of migrations run.
    pub migrations_run: usize,
    /// Final schema version.
    pub final_version: SchemaVersion,
    /// Operations performed.
    pub operations: Vec<MigrationOperation>,
    /// Errors encountered.
    pub errors: Vec<MigrationError>,
}

impl MigrationResult {
    /// Check if migration was successful.
    pub fn is_success(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Migration error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationError {
    /// Migration ID that failed.
    pub migration_id: String,
    /// Error message.
    pub error: String,
}

/// Preview of a migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPreview {
    /// Migration ID.
    pub id: String,
    /// Target version.
    pub version: SchemaVersion,
    /// Description.
    pub description: String,
    /// Operations that will be performed.
    pub operations: Vec<MigrationOperation>,
    /// Whether migration is reversible.
    pub reversible: bool,
}

/// Validation issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Migration ID (if applicable).
    pub migration_id: Option<String>,
    /// Severity level.
    pub severity: IssueSeverity,
    /// Issue message.
    pub message: String,
}

/// Issue severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Warning (migration can proceed).
    Warning,
    /// Error (migration should not proceed).
    Error,
}

/// Schema compatibility result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityResult {
    /// Whether schemas are compatible.
    pub compatible: bool,
    /// Database version.
    pub database_version: SchemaVersion,
    /// Application version.
    pub application_version: SchemaVersion,
    /// Compatibility issues.
    pub issues: Vec<String>,
    /// Whether upgrade is needed.
    pub upgrade_needed: bool,
    /// Whether downgrade is needed.
    pub downgrade_needed: bool,
}

impl CompatibilityResult {
    /// Check compatibility between database and application versions.
    pub fn check(database_version: SchemaVersion, application_version: SchemaVersion) -> Self {
        let compatible = database_version.is_compatible(&application_version);
        let upgrade_needed = database_version < application_version;
        let downgrade_needed = database_version > application_version;

        let mut issues = Vec::new();

        if database_version.major != application_version.major {
            issues.push(format!(
                "Major version mismatch: database={}, application={}",
                database_version.major, application_version.major
            ));
        }

        if downgrade_needed && !compatible {
            issues.push(
                "Database version is newer than application - downgrade may lose data".to_string(),
            );
        }

        Self {
            compatible,
            database_version,
            application_version,
            issues,
            upgrade_needed,
            downgrade_needed,
        }
    }
}

/// Built-in migrations for Needle.
pub fn built_in_migrations() -> Vec<Migration> {
    vec![
        // Initial schema
        Migration::new(
            "001_initial_schema",
            SchemaVersion::new(1, 0, 0),
            "Initial Needle database schema",
            |_ctx| {
                // Initial schema - nothing to migrate
                Ok(())
            },
            |_ctx| {
                // Cannot rollback initial schema
                Err(NeedleError::InvalidInput(
                    "Cannot rollback initial schema".to_string(),
                ))
            },
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_version() {
        let v1 = SchemaVersion::new(1, 0, 0);
        let v2 = SchemaVersion::new(1, 1, 0);
        let v3 = SchemaVersion::new(2, 0, 0);

        assert!(v2.is_newer_than(&v1));
        assert!(v3.is_newer_than(&v2));
        assert!(v1.is_compatible(&v2));
        assert!(!v1.is_compatible(&v3));
    }

    #[test]
    fn test_schema_version_parse() {
        let v = SchemaVersion::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert_eq!(v.to_string(), "1.2.3");

        assert!(SchemaVersion::parse("invalid").is_none());
        assert!(SchemaVersion::parse("1.2").is_none());
    }

    #[test]
    fn test_migration_manager() {
        let mut manager = MigrationManager::new();

        manager.register(Migration::new(
            "001_initial",
            SchemaVersion::new(1, 0, 0),
            "Initial schema",
            |ctx| {
                ctx.create_collection("default", 128);
                Ok(())
            },
            |ctx| {
                ctx.drop_collection("default");
                Ok(())
            },
        ));

        manager.register(Migration::new(
            "002_add_index",
            SchemaVersion::new(1, 1, 0),
            "Add HNSW index",
            |ctx| {
                ctx.create_index("default", IndexType::Hnsw);
                Ok(())
            },
            |ctx| {
                ctx.drop_index("default", IndexType::Hnsw);
                Ok(())
            },
        ));

        assert_eq!(manager.pending().len(), 2);

        let result = manager.migrate_up().unwrap();
        assert!(result.is_success());
        assert_eq!(result.migrations_run, 2);
        assert_eq!(result.final_version, SchemaVersion::new(1, 1, 0));
        assert_eq!(manager.pending().len(), 0);
    }

    #[test]
    fn test_migration_rollback() {
        let mut manager = MigrationManager::new();

        manager.register(Migration::new(
            "001_initial",
            SchemaVersion::new(1, 0, 0),
            "Initial schema",
            |_| Ok(()),
            |_| Ok(()),
        ));

        manager.register(Migration::new(
            "002_feature",
            SchemaVersion::new(1, 1, 0),
            "Add feature",
            |_| Ok(()),
            |_| Ok(()),
        ));

        // Migrate up
        manager.migrate_up().unwrap();
        assert_eq!(manager.current_version(), SchemaVersion::new(1, 1, 0));

        // Rollback
        let result = manager.rollback().unwrap();
        assert!(result.is_success());
        assert_eq!(result.migrations_run, 1);
    }

    #[test]
    fn test_one_way_migration() {
        let mut manager = MigrationManager::new();

        manager.register(Migration::one_way(
            "001_irreversible",
            SchemaVersion::new(1, 0, 0),
            "Irreversible change",
            |_| Ok(()),
        ));

        // Migrate up works
        let result = manager.migrate_up().unwrap();
        assert!(result.is_success());

        // Rollback fails
        let result = manager.rollback().unwrap();
        assert!(!result.is_success());
    }

    #[test]
    fn test_migration_preview() {
        let mut manager = MigrationManager::new();

        manager.register(Migration::new(
            "001_test",
            SchemaVersion::new(1, 0, 0),
            "Test migration",
            |ctx| {
                ctx.create_collection("test", 64);
                ctx.create_index("test", IndexType::Hnsw);
                Ok(())
            },
            |ctx| {
                ctx.drop_index("test", IndexType::Hnsw);
                ctx.drop_collection("test");
                Ok(())
            },
        ));

        let previews = manager.preview();
        assert_eq!(previews.len(), 1);
        assert_eq!(previews[0].operations.len(), 2);
    }

    #[test]
    fn test_migration_validation() {
        let mut manager = MigrationManager::new();

        // Register with duplicate ID
        manager.register(Migration::new(
            "001_test",
            SchemaVersion::new(1, 0, 0),
            "Test 1",
            |_| Ok(()),
            |_| Ok(()),
        ));

        manager.register(Migration::one_way(
            "002_test",
            SchemaVersion::new(1, 1, 0),
            "Test 2 (non-reversible)",
            |_| Ok(()),
        ));

        let issues = manager.validate();
        // Should have warning about non-reversible migration
        assert!(issues.iter().any(|i| i.message.contains("not reversible")));
    }

    #[test]
    fn test_compatibility_check() {
        // Compatible versions
        let result =
            CompatibilityResult::check(SchemaVersion::new(1, 0, 0), SchemaVersion::new(1, 2, 0));
        assert!(result.compatible);
        assert!(result.upgrade_needed);
        assert!(!result.downgrade_needed);

        // Incompatible versions
        let result =
            CompatibilityResult::check(SchemaVersion::new(1, 0, 0), SchemaVersion::new(2, 0, 0));
        assert!(!result.compatible);
        assert!(result.upgrade_needed);
    }

    #[test]
    fn test_migration_context() {
        let mut ctx =
            MigrationContext::new(SchemaVersion::new(0, 0, 0), SchemaVersion::new(1, 0, 0));

        ctx.create_collection("test", 128);
        ctx.create_index("test", IndexType::Hnsw);
        ctx.add_metadata_field("test", "category", "default");
        ctx.set_metadata("migrated_at", "2024-01-01");

        assert_eq!(ctx.operations().len(), 3);
        assert_eq!(
            ctx.metadata.get("migrated_at"),
            Some(&"2024-01-01".to_string())
        );
    }

    #[test]
    fn test_built_in_migrations() {
        let migrations = built_in_migrations();
        assert!(!migrations.is_empty());
        assert_eq!(migrations[0].id, "001_initial_schema");
    }
}
