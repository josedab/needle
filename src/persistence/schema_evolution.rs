#![allow(dead_code)]

//! Zero-Downtime Schema Evolution
//!
//! Supports adding/removing metadata fields, changing distance functions, and
//! re-dimensioning vectors without downtime. Implements lazy migration with
//! dual-read during transition.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │              SchemaRegistry                       │
//! │  version_chain: [v1] → [v2] → [v3]              │
//! ├──────────────────────────────────────────────────┤
//! │              MigrationEngine                      │
//! │  ┌─────────┐   ┌──────────┐   ┌──────────────┐  │
//! │  │Old Index│──►│ Dual-Read│──►│Migrated Index│  │
//! │  └─────────┘   └──────────┘   └──────────────┘  │
//! │     background batch re-indexing                  │
//! └──────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::persistence::schema_evolution::{
//!     SchemaRegistry, SchemaChange, MigrationEngine, MigrationConfig,
//! };
//!
//! let mut registry = SchemaRegistry::new("my_collection", 1, 384);
//!
//! // Add a metadata field
//! registry.apply_change(SchemaChange::AddMetadataField {
//!     name: "category".into(),
//!     default_value: Some(serde_json::Value::String("uncategorized".into())),
//! })?;
//!
//! // Change distance function
//! registry.apply_change(SchemaChange::ChangeDistance {
//!     from: DistanceFunction::Cosine,
//!     to: DistanceFunction::Euclidean,
//! })?;
//!
//! // Re-dimension vectors (e.g., PCA projection)
//! registry.apply_change(SchemaChange::ChangeDimension {
//!     from: 384,
//!     to: 256,
//!     strategy: DimensionStrategy::ZeroPad,
//! })?;
//! ```

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Schema Version
// ============================================================================

/// A schema version describing the collection's structure at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSchema {
    /// Schema version number (monotonically increasing).
    pub version: u64,
    /// Collection name.
    pub collection_name: String,
    /// Vector dimensionality.
    pub dimensions: usize,
    /// Distance function.
    pub distance_function: DistanceFunction,
    /// Known metadata fields with optional default values.
    pub metadata_fields: HashMap<String, MetadataFieldDef>,
    /// Timestamp when this schema version was created.
    pub created_at: u64,
    /// Optional description of the change.
    pub description: Option<String>,
    /// Parent version (None for initial schema).
    pub parent_version: Option<u64>,
}

/// Metadata field definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataFieldDef {
    /// Field name.
    pub name: String,
    /// Whether the field is required.
    pub required: bool,
    /// Default value for new/migrated records.
    pub default_value: Option<serde_json::Value>,
    /// Added in schema version.
    pub added_in_version: u64,
    /// Removed in schema version (None if still active).
    pub removed_in_version: Option<u64>,
}

// ============================================================================
// Schema Changes
// ============================================================================

/// A schema change that can be applied to a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaChange {
    /// Add a new metadata field.
    AddMetadataField {
        /// Field name.
        name: String,
        /// Default value for existing records.
        default_value: Option<serde_json::Value>,
    },
    /// Remove a metadata field.
    RemoveMetadataField {
        /// Field name to remove.
        name: String,
    },
    /// Change the distance function.
    ChangeDistance {
        /// Current distance function.
        from: DistanceFunction,
        /// New distance function.
        to: DistanceFunction,
    },
    /// Change vector dimensionality.
    ChangeDimension {
        /// Current dimensions.
        from: usize,
        /// New dimensions.
        to: usize,
        /// Strategy for dimension change.
        strategy: DimensionStrategy,
    },
}

/// Strategy for changing vector dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DimensionStrategy {
    /// Pad with zeros (when increasing dimensions).
    ZeroPad,
    /// Truncate (when decreasing dimensions).
    Truncate,
    /// PCA projection (requires pre-computed projection matrix).
    PcaProject {
        /// Projection matrix (flattened, row-major).
        matrix: Vec<f32>,
    },
    /// Random projection.
    RandomProject {
        /// Random seed for reproducibility.
        seed: u64,
    },
}

// ============================================================================
// Schema Registry
// ============================================================================

/// Registry that tracks all schema versions for a collection.
#[derive(Debug)]
pub struct SchemaRegistry {
    /// All schema versions in order.
    versions: Vec<CollectionSchema>,
    /// Current (active) schema version.
    current_version: u64,
}

impl SchemaRegistry {
    /// Create a new schema registry with the initial schema.
    pub fn new(collection_name: &str, initial_version: u64, dimensions: usize) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let initial = CollectionSchema {
            version: initial_version,
            collection_name: collection_name.to_string(),
            dimensions,
            distance_function: DistanceFunction::Cosine,
            metadata_fields: HashMap::new(),
            created_at: now,
            description: Some("Initial schema".into()),
            parent_version: None,
        };

        Self {
            versions: vec![initial],
            current_version: initial_version,
        }
    }

    /// Apply a schema change, creating a new version.
    pub fn apply_change(&mut self, change: SchemaChange) -> Result<u64> {
        let current = self
            .current_schema()
            .ok_or_else(|| NeedleError::InvalidState("No current schema".into()))?
            .clone();

        let new_version = self.current_version + 1;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut new_schema = CollectionSchema {
            version: new_version,
            collection_name: current.collection_name.clone(),
            dimensions: current.dimensions,
            distance_function: current.distance_function,
            metadata_fields: current.metadata_fields.clone(),
            created_at: now,
            description: None,
            parent_version: Some(current.version),
        };

        match &change {
            SchemaChange::AddMetadataField {
                name,
                default_value,
            } => {
                if new_schema.metadata_fields.contains_key(name) {
                    return Err(NeedleError::InvalidOperation(format!(
                        "Metadata field '{name}' already exists"
                    )));
                }
                new_schema.metadata_fields.insert(
                    name.clone(),
                    MetadataFieldDef {
                        name: name.clone(),
                        required: false,
                        default_value: default_value.clone(),
                        added_in_version: new_version,
                        removed_in_version: None,
                    },
                );
                new_schema.description =
                    Some(format!("Added metadata field '{name}'"));
            }
            SchemaChange::RemoveMetadataField { name } => {
                let field = new_schema.metadata_fields.get_mut(name).ok_or_else(|| {
                    NeedleError::NotFound(format!("Metadata field '{name}'"))
                })?;
                if field.removed_in_version.is_some() {
                    return Err(NeedleError::InvalidOperation(format!(
                        "Metadata field '{name}' already removed"
                    )));
                }
                field.removed_in_version = Some(new_version);
                new_schema.description =
                    Some(format!("Removed metadata field '{name}'"));
            }
            SchemaChange::ChangeDistance { from, to } => {
                if current.distance_function != *from {
                    return Err(NeedleError::InvalidOperation(format!(
                        "Current distance function is {:?}, not {:?}",
                        current.distance_function, from
                    )));
                }
                new_schema.distance_function = *to;
                new_schema.description =
                    Some(format!("Changed distance from {from:?} to {to:?}"));
            }
            SchemaChange::ChangeDimension { from, to, .. } => {
                if current.dimensions != *from {
                    return Err(NeedleError::InvalidOperation(format!(
                        "Current dimensions is {}, not {from}",
                        current.dimensions
                    )));
                }
                if *to == 0 {
                    return Err(NeedleError::InvalidConfig(
                        "Dimensions must be > 0".into(),
                    ));
                }
                new_schema.dimensions = *to;
                new_schema.description =
                    Some(format!("Changed dimensions from {from} to {to}"));
            }
        }

        self.versions.push(new_schema);
        self.current_version = new_version;
        Ok(new_version)
    }

    /// Get the current schema.
    pub fn current_schema(&self) -> Option<&CollectionSchema> {
        self.versions
            .iter()
            .find(|s| s.version == self.current_version)
    }

    /// Get a schema by version number.
    pub fn schema_at(&self, version: u64) -> Option<&CollectionSchema> {
        self.versions.iter().find(|s| s.version == version)
    }

    /// Get all schema versions.
    pub fn versions(&self) -> &[CollectionSchema] {
        &self.versions
    }

    /// Get the version history (version numbers only).
    pub fn version_history(&self) -> Vec<u64> {
        self.versions.iter().map(|s| s.version).collect()
    }

    /// Get active metadata fields (not removed) at current version.
    pub fn active_metadata_fields(&self) -> Vec<&MetadataFieldDef> {
        self.current_schema()
            .map(|s| {
                s.metadata_fields
                    .values()
                    .filter(|f| f.removed_in_version.is_none())
                    .collect()
            })
            .unwrap_or_default()
    }
}

// ============================================================================
// Migration Engine
// ============================================================================

/// Configuration for the migration engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationConfig {
    /// Batch size for background re-indexing.
    pub batch_size: usize,
    /// Maximum concurrent migration workers.
    pub max_workers: usize,
    /// Whether to enable dual-read during migration.
    pub dual_read: bool,
    /// Maximum migration time before timeout (seconds).
    pub timeout_seconds: u64,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            max_workers: 4,
            dual_read: true,
            timeout_seconds: 3600,
        }
    }
}

/// Status of an ongoing migration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MigrationPhase {
    /// No migration in progress.
    Idle,
    /// Preparing migration (computing projection matrices, etc.).
    Preparing,
    /// Actively migrating vectors (dual-read mode).
    Migrating {
        /// Total vectors to migrate.
        total: usize,
        /// Vectors migrated so far.
        migrated: usize,
        /// Start timestamp.
        started_at: u64,
    },
    /// Migration complete, ready to finalize.
    Finalizing,
    /// Migration complete and finalized.
    Complete {
        /// Total vectors migrated.
        total_migrated: usize,
        /// Duration in seconds.
        duration_seconds: u64,
    },
    /// Migration failed.
    Failed {
        /// Error message.
        reason: String,
    },
}

/// Engine that coordinates schema migrations with zero downtime.
#[derive(Debug)]
pub struct MigrationEngine {
    config: MigrationConfig,
    phase: MigrationPhase,
    /// Source schema version.
    source_version: Option<u64>,
    /// Target schema version.
    target_version: Option<u64>,
    /// The schema change being applied.
    active_change: Option<SchemaChange>,
}

impl MigrationEngine {
    /// Create a new migration engine.
    pub fn new(config: MigrationConfig) -> Self {
        Self {
            config,
            phase: MigrationPhase::Idle,
            source_version: None,
            target_version: None,
            active_change: None,
        }
    }

    /// Start a migration for a schema change.
    pub fn start_migration(
        &mut self,
        source_version: u64,
        target_version: u64,
        change: SchemaChange,
    ) -> Result<()> {
        if self.phase != MigrationPhase::Idle {
            return Err(NeedleError::OperationInProgress(
                "Migration already in progress".into(),
            ));
        }

        self.source_version = Some(source_version);
        self.target_version = Some(target_version);
        self.active_change = Some(change);
        self.phase = MigrationPhase::Preparing;
        Ok(())
    }

    /// Begin the migration phase (after preparation).
    pub fn begin_migrate(&mut self, total_vectors: usize) -> Result<()> {
        if !matches!(self.phase, MigrationPhase::Preparing) {
            return Err(NeedleError::InvalidState(
                "Must be in Preparing phase to begin migration".into(),
            ));
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.phase = MigrationPhase::Migrating {
            total: total_vectors,
            migrated: 0,
            started_at: now,
        };
        Ok(())
    }

    /// Migrate a single vector. Returns the adapted vector.
    pub fn migrate_vector(
        &self,
        vector: &[f32],
        _metadata: Option<&serde_json::Value>,
    ) -> Result<Vec<f32>> {
        let change = self
            .active_change
            .as_ref()
            .ok_or_else(|| NeedleError::InvalidState("No active migration".into()))?;

        match change {
            SchemaChange::ChangeDimension { to, strategy, .. } => {
                adapt_dimensions(vector, *to, strategy)
            }
            SchemaChange::ChangeDistance { .. } => {
                // Distance change doesn't modify vectors, only the index
                Ok(vector.to_vec())
            }
            SchemaChange::AddMetadataField { .. }
            | SchemaChange::RemoveMetadataField { .. } => {
                // Metadata changes don't modify vectors
                Ok(vector.to_vec())
            }
        }
    }

    /// Record that a batch of vectors has been migrated.
    pub fn record_progress(&mut self, count: usize) {
        if let MigrationPhase::Migrating {
            ref mut migrated, ..
        } = self.phase
        {
            *migrated += count;
        }
    }

    /// Finalize the migration.
    pub fn finalize(&mut self) -> Result<MigrationPhase> {
        let (total, started_at) = match self.phase {
            MigrationPhase::Migrating {
                total,
                migrated,
                started_at,
            } => {
                if migrated < total {
                    return Err(NeedleError::InvalidState(format!(
                        "Migration incomplete: {migrated}/{total} vectors migrated"
                    )));
                }
                (total, started_at)
            }
            _ => {
                return Err(NeedleError::InvalidState(
                    "Not in Migrating phase".into(),
                ));
            }
        };

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let result = MigrationPhase::Complete {
            total_migrated: total,
            duration_seconds: now.saturating_sub(started_at),
        };
        self.phase = result.clone();
        self.active_change = None;
        Ok(result)
    }

    /// Cancel an in-progress migration.
    pub fn cancel(&mut self) {
        self.phase = MigrationPhase::Idle;
        self.active_change = None;
        self.source_version = None;
        self.target_version = None;
    }

    /// Get current migration phase.
    pub fn phase(&self) -> &MigrationPhase {
        &self.phase
    }

    /// Get migration progress as a percentage (0.0 - 100.0).
    pub fn progress_pct(&self) -> f64 {
        match &self.phase {
            MigrationPhase::Migrating {
                total, migrated, ..
            } => {
                if *total > 0 {
                    *migrated as f64 / *total as f64 * 100.0
                } else {
                    100.0
                }
            }
            MigrationPhase::Complete { .. } => 100.0,
            _ => 0.0,
        }
    }
}

// ============================================================================
// Dimension Adaptation
// ============================================================================

/// Adapt a vector to new dimensions using the given strategy.
pub fn adapt_dimensions(
    vector: &[f32],
    target_dim: usize,
    strategy: &DimensionStrategy,
) -> Result<Vec<f32>> {
    if target_dim == 0 {
        return Err(NeedleError::InvalidConfig(
            "Target dimensions must be > 0".into(),
        ));
    }

    match strategy {
        DimensionStrategy::ZeroPad => {
            let mut result = vector.to_vec();
            result.resize(target_dim, 0.0);
            Ok(result)
        }
        DimensionStrategy::Truncate => {
            if target_dim > vector.len() {
                return Err(NeedleError::InvalidOperation(format!(
                    "Cannot truncate {} dims to {} dims",
                    vector.len(),
                    target_dim
                )));
            }
            Ok(vector[..target_dim].to_vec())
        }
        DimensionStrategy::PcaProject { matrix } => {
            let source_dim = vector.len();
            let expected_size = source_dim * target_dim;
            if matrix.len() != expected_size {
                return Err(NeedleError::InvalidConfig(format!(
                    "PCA matrix size mismatch: expected {expected_size}, got {}",
                    matrix.len()
                )));
            }
            // Matrix multiply: result[i] = sum(vector[j] * matrix[i * source_dim + j])
            let mut result = vec![0.0f32; target_dim];
            for (i, val) in result.iter_mut().enumerate() {
                for (j, &v) in vector.iter().enumerate() {
                    *val += v * matrix[i * source_dim + j];
                }
            }
            Ok(result)
        }
        DimensionStrategy::RandomProject { seed } => {
            // Simple random projection using a seeded PRNG
            let source_dim = vector.len();
            let mut result = vec![0.0f32; target_dim];
            let scale = 1.0 / (source_dim as f32).sqrt();

            for (i, val) in result.iter_mut().enumerate() {
                let mut state = seed.wrapping_mul(6364136223846793005)
                    .wrapping_add(i as u64 * 1442695040888963407);
                for (j, &v) in vector.iter().enumerate() {
                    state = state.wrapping_mul(6364136223846793005)
                        .wrapping_add(j as u64);
                    // Generate +1 or -1 based on high bit
                    let sign = if (state >> 63) == 0 { 1.0 } else { -1.0 };
                    *val += v * sign * scale;
                }
            }
            Ok(result)
        }
    }
}

/// Adapt metadata from one schema version to another during lazy migration.
///
/// Applies field additions (with defaults) and removals based on the schema
/// registry's active field definitions.
pub fn adapt_metadata(
    metadata: &mut serde_json::Value,
    registry: &SchemaRegistry,
) {
    let Some(schema) = registry.current_schema() else { return };
    let obj = match metadata.as_object_mut() {
        Some(o) => o,
        None => return,
    };

    // Add missing fields with defaults
    for field in schema.metadata_fields.values() {
        if field.removed_in_version.is_some() {
            continue; // Skip removed fields
        }
        if !obj.contains_key(&field.name) {
            if let Some(ref default) = field.default_value {
                obj.insert(field.name.clone(), default.clone());
            }
        }
    }

    // Remove fields that were dropped
    for field in schema.metadata_fields.values() {
        if field.removed_in_version.is_some() {
            obj.remove(&field.name);
        }
    }
}

/// Perform a dual-read: try to read from migrated data first, fall back to old.
/// Applies metadata adaptation on-the-fly.
pub fn dual_read_metadata(
    old_metadata: Option<serde_json::Value>,
    migrated_metadata: Option<serde_json::Value>,
    registry: &SchemaRegistry,
) -> Option<serde_json::Value> {
    // Prefer migrated data if available
    if let Some(m) = migrated_metadata {
        return Some(m);
    }
    // Fall back to old data, adapting it lazily
    if let Some(mut m) = old_metadata {
        adapt_metadata(&mut m, registry);
        return Some(m);
    }
    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_registry_creation() {
        let registry = SchemaRegistry::new("test_col", 1, 384);
        assert_eq!(registry.current_version, 1);
        let schema = registry.current_schema().expect("should have schema");
        assert_eq!(schema.dimensions, 384);
        assert_eq!(schema.collection_name, "test_col");
    }

    #[test]
    fn test_add_metadata_field() {
        let mut registry = SchemaRegistry::new("test", 1, 128);
        let v = registry
            .apply_change(SchemaChange::AddMetadataField {
                name: "category".into(),
                default_value: Some(serde_json::Value::String("unknown".into())),
            })
            .expect("apply");
        assert_eq!(v, 2);

        let fields = registry.active_metadata_fields();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "category");
    }

    #[test]
    fn test_remove_metadata_field() {
        let mut registry = SchemaRegistry::new("test", 1, 128);
        registry
            .apply_change(SchemaChange::AddMetadataField {
                name: "tag".into(),
                default_value: None,
            })
            .expect("add");
        registry
            .apply_change(SchemaChange::RemoveMetadataField {
                name: "tag".into(),
            })
            .expect("remove");

        let fields = registry.active_metadata_fields();
        assert_eq!(fields.len(), 0);
    }

    #[test]
    fn test_change_distance() {
        let mut registry = SchemaRegistry::new("test", 1, 128);
        registry
            .apply_change(SchemaChange::ChangeDistance {
                from: DistanceFunction::Cosine,
                to: DistanceFunction::Euclidean,
            })
            .expect("change distance");

        let schema = registry.current_schema().expect("schema");
        assert_eq!(schema.distance_function, DistanceFunction::Euclidean);
    }

    #[test]
    fn test_change_distance_wrong_current() {
        let mut registry = SchemaRegistry::new("test", 1, 128);
        let result = registry.apply_change(SchemaChange::ChangeDistance {
            from: DistanceFunction::Euclidean, // Wrong: current is Cosine
            to: DistanceFunction::DotProduct,
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_change_dimensions() {
        let mut registry = SchemaRegistry::new("test", 1, 384);
        registry
            .apply_change(SchemaChange::ChangeDimension {
                from: 384,
                to: 256,
                strategy: DimensionStrategy::Truncate,
            })
            .expect("change dimensions");

        let schema = registry.current_schema().expect("schema");
        assert_eq!(schema.dimensions, 256);
    }

    #[test]
    fn test_version_history() {
        let mut registry = SchemaRegistry::new("test", 1, 128);
        registry
            .apply_change(SchemaChange::AddMetadataField {
                name: "a".into(),
                default_value: None,
            })
            .expect("v2");
        registry
            .apply_change(SchemaChange::AddMetadataField {
                name: "b".into(),
                default_value: None,
            })
            .expect("v3");

        assert_eq!(registry.version_history(), vec![1, 2, 3]);
    }

    #[test]
    fn test_adapt_zero_pad() {
        let vec = vec![1.0, 2.0, 3.0];
        let result = adapt_dimensions(&vec, 5, &DimensionStrategy::ZeroPad).expect("pad");
        assert_eq!(result, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_adapt_truncate() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result =
            adapt_dimensions(&vec, 3, &DimensionStrategy::Truncate).expect("truncate");
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_adapt_pca_project() {
        // 3D → 2D with identity-like projection
        let vec = vec![1.0, 2.0, 3.0];
        let matrix = vec![
            1.0, 0.0, 0.0, // row 0: extract dim 0
            0.0, 1.0, 0.0, // row 1: extract dim 1
        ];
        let result = adapt_dimensions(
            &vec,
            2,
            &DimensionStrategy::PcaProject { matrix },
        )
        .expect("pca");
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_adapt_random_project() {
        let vec = vec![1.0, 0.0, 0.0, 0.0];
        let result = adapt_dimensions(
            &vec,
            2,
            &DimensionStrategy::RandomProject { seed: 42 },
        )
        .expect("random");
        assert_eq!(result.len(), 2);
        // Random projection preserves approximate norms
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_migration_lifecycle() {
        let mut engine = MigrationEngine::new(MigrationConfig::default());
        assert_eq!(*engine.phase(), MigrationPhase::Idle);

        engine
            .start_migration(
                1,
                2,
                SchemaChange::ChangeDimension {
                    from: 384,
                    to: 256,
                    strategy: DimensionStrategy::Truncate,
                },
            )
            .expect("start");
        assert!(matches!(*engine.phase(), MigrationPhase::Preparing));

        engine.begin_migrate(100).expect("begin");
        assert!(matches!(
            *engine.phase(),
            MigrationPhase::Migrating { total: 100, .. }
        ));

        // Migrate vectors
        let vec = vec![1.0; 384];
        let migrated = engine.migrate_vector(&vec, None).expect("migrate");
        assert_eq!(migrated.len(), 256);

        engine.record_progress(100);
        assert!((engine.progress_pct() - 100.0).abs() < 0.01);

        let result = engine.finalize().expect("finalize");
        assert!(matches!(result, MigrationPhase::Complete { .. }));
    }

    #[test]
    fn test_migration_cannot_finalize_early() {
        let mut engine = MigrationEngine::new(MigrationConfig::default());
        engine
            .start_migration(
                1,
                2,
                SchemaChange::AddMetadataField {
                    name: "x".into(),
                    default_value: None,
                },
            )
            .expect("start");
        engine.begin_migrate(100).expect("begin");
        engine.record_progress(50); // Only 50 of 100

        let result = engine.finalize();
        assert!(result.is_err());
    }

    #[test]
    fn test_migration_cancel() {
        let mut engine = MigrationEngine::new(MigrationConfig::default());
        engine
            .start_migration(
                1,
                2,
                SchemaChange::AddMetadataField {
                    name: "x".into(),
                    default_value: None,
                },
            )
            .expect("start");
        engine.cancel();
        assert_eq!(*engine.phase(), MigrationPhase::Idle);
    }

    #[test]
    fn test_adapt_metadata_adds_defaults() {
        let mut registry = SchemaRegistry::new("test", 1, 128);
        registry
            .apply_change(SchemaChange::AddMetadataField {
                name: "category".into(),
                default_value: Some(serde_json::json!("uncategorized")),
            })
            .expect("add");

        let mut metadata = serde_json::json!({"title": "hello"});
        adapt_metadata(&mut metadata, &registry);

        assert_eq!(metadata["category"], "uncategorized");
        assert_eq!(metadata["title"], "hello");
    }

    #[test]
    fn test_adapt_metadata_removes_dropped_fields() {
        let mut registry = SchemaRegistry::new("test", 1, 128);
        registry
            .apply_change(SchemaChange::AddMetadataField {
                name: "temp".into(),
                default_value: None,
            })
            .expect("add");
        registry
            .apply_change(SchemaChange::RemoveMetadataField {
                name: "temp".into(),
            })
            .expect("remove");

        let mut metadata = serde_json::json!({"title": "hello", "temp": "value"});
        adapt_metadata(&mut metadata, &registry);

        assert!(metadata.get("temp").is_none());
        assert_eq!(metadata["title"], "hello");
    }

    #[test]
    fn test_dual_read_prefers_migrated() {
        let registry = SchemaRegistry::new("test", 1, 128);
        let old = Some(serde_json::json!({"old": true}));
        let migrated = Some(serde_json::json!({"new": true}));

        let result = dual_read_metadata(old, migrated, &registry);
        assert_eq!(result.expect("result")["new"], true);
    }

    #[test]
    fn test_dual_read_falls_back_to_old() {
        let mut registry = SchemaRegistry::new("test", 1, 128);
        registry
            .apply_change(SchemaChange::AddMetadataField {
                name: "added".into(),
                default_value: Some(serde_json::json!("default")),
            })
            .expect("add");

        let old = Some(serde_json::json!({"existing": "value"}));
        let result = dual_read_metadata(old, None, &registry);
        let r = result.expect("result");
        assert_eq!(r["existing"], "value");
        assert_eq!(r["added"], "default");
    }
}
