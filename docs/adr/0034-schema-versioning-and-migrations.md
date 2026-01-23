# ADR-0034: Schema Versioning and Migrations

## Status

Accepted

## Context

As Needle evolves, the database schema changes:

1. **New features** — Adding fields to metadata, new index types
2. **Format changes** — Optimizing storage layout for performance
3. **Bug fixes** — Correcting data inconsistencies from previous versions
4. **Breaking changes** — Major version upgrades that aren't backward compatible

Without schema versioning:
- Opening old databases with new code may crash or corrupt data
- Users can't safely upgrade without manual intervention
- Rollback after failed upgrade is impossible
- No way to know if a database is compatible before loading

## Decision

Implement **semantic versioning for schemas** with **forward and backward migration support**.

### Schema Version

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SchemaVersion {
    pub major: u32,  // Breaking changes (incompatible)
    pub minor: u32,  // New features (backward compatible)
    pub patch: u32,  // Bug fixes (backward compatible)
}

impl SchemaVersion {
    /// Check if this version can read data from another version
    pub fn is_compatible(&self, other: &SchemaVersion) -> bool {
        self.major == other.major  // Same major = compatible
    }

    /// Current Needle schema version
    pub fn current() -> Self {
        Self::new(1, 0, 0)
    }
}
```

### Compatibility Rules

| Data Version | Code Version | Compatible? | Action |
|--------------|--------------|-------------|--------|
| 1.0.0 | 1.0.0 | Yes | Direct load |
| 1.0.0 | 1.2.0 | Yes | Auto-migrate to 1.2.0 |
| 1.2.0 | 1.0.0 | Yes | Load (newer features ignored) |
| 1.0.0 | 2.0.0 | No | Explicit migration required |
| 2.0.0 | 1.0.0 | No | Error: cannot downgrade major |

### Migration Definition

```rust
pub struct Migration {
    pub name: String,
    pub from_version: SchemaVersion,
    pub to_version: SchemaVersion,
    pub up: Box<dyn Fn(&mut Database) -> Result<()>>,
    pub down: Box<dyn Fn(&mut Database) -> Result<()>>,
}

impl Migration {
    pub fn new<F, G>(
        name: &str,
        from: SchemaVersion,
        to: SchemaVersion,
        up: F,
        down: G,
    ) -> Self
    where
        F: Fn(&mut Database) -> Result<()> + 'static,
        G: Fn(&mut Database) -> Result<()> + 'static,
    {
        Self {
            name: name.to_string(),
            from_version: from,
            to_version: to,
            up: Box::new(up),
            down: Box::new(down),
        }
    }
}
```

### Migration Manager

```rust
pub struct MigrationManager {
    migrations: Vec<Migration>,
    history: Vec<MigrationRecord>,
}

impl MigrationManager {
    /// Register a migration
    pub fn register(&mut self, migration: Migration) {
        self.migrations.push(migration);
        self.migrations.sort_by(|a, b| a.to_version.cmp(&b.to_version));
    }

    /// Run all pending migrations
    pub fn migrate_up(&self, db: &mut Database) -> Result<MigrateResult> {
        let current = db.schema_version();
        let target = SchemaVersion::current();

        if !current.is_compatible(&target) {
            return Err(NeedleError::IncompatibleVersion {
                current,
                required: target,
            });
        }

        let pending: Vec<&Migration> = self.migrations.iter()
            .filter(|m| m.from_version >= current && m.to_version <= target)
            .collect();

        for migration in pending {
            // Create checkpoint before migration
            let checkpoint = db.checkpoint()?;

            match (migration.up)(db) {
                Ok(()) => {
                    self.record_migration(migration, MigrationDirection::Up)?;
                    db.set_schema_version(migration.to_version)?;
                }
                Err(e) => {
                    // Rollback to checkpoint
                    db.restore_checkpoint(checkpoint)?;
                    return Err(e);
                }
            }
        }

        Ok(MigrateResult {
            from_version: current,
            to_version: target,
            migrations_applied: pending.len(),
        })
    }

    /// Rollback to a previous version
    pub fn migrate_down(&self, db: &mut Database, target: SchemaVersion) -> Result<MigrateResult> {
        let current = db.schema_version();

        let rollback: Vec<&Migration> = self.migrations.iter()
            .filter(|m| m.to_version <= current && m.from_version >= target)
            .rev()  // Apply in reverse order
            .collect();

        for migration in rollback {
            (migration.down)(db)?;
            self.record_migration(migration, MigrationDirection::Down)?;
            db.set_schema_version(migration.from_version)?;
        }

        Ok(MigrateResult {
            from_version: current,
            to_version: target,
            migrations_applied: rollback.len(),
        })
    }
}
```

### Migration History

```rust
pub struct MigrationRecord {
    pub migration_name: String,
    pub direction: MigrationDirection,
    pub applied_at: u64,
    pub from_version: SchemaVersion,
    pub to_version: SchemaVersion,
}

pub enum MigrationDirection {
    Up,
    Down,
}
```

### Code References

- `src/migrations.rs:46-92` — SchemaVersion with compatibility checking
- `src/migrations.rs` — MigrationManager and Migration definitions
- `src/storage.rs` — Version field in file header

## Consequences

### Benefits

1. **Safe upgrades** — Automatic migration on database open
2. **Rollback support** — Revert to previous version if upgrade fails
3. **Compatibility detection** — Clear error before attempting incompatible load
4. **Audit trail** — History of all migrations applied
5. **Incremental updates** — Skip already-applied migrations

### Tradeoffs

1. **Migration code maintenance** — Must write up/down for each change
2. **Testing burden** — Migrations need testing with real data
3. **Storage overhead** — Migration history stored in database
4. **Checkpoint cost** — Pre-migration checkpoint uses disk space

### What This Enabled

- Zero-downtime upgrades (migrate, verify, cutover)
- Blue-green deployments (old and new versions coexist)
- Safe experimentation (rollback if new version has issues)
- Clear upgrade path documentation

### What This Prevented

- Silent data corruption from version mismatches
- Unrecoverable upgrade failures
- Ambiguity about database compatibility
- Need for manual schema updates

### Example Migrations

```rust
// Migration: Add created_at timestamp to vectors
Migration::new(
    "001_add_created_at",
    SchemaVersion::new(1, 0, 0),
    SchemaVersion::new(1, 1, 0),
    |db| {
        // Up: Add created_at with default value
        for collection in db.collections() {
            for vector in collection.vectors_mut() {
                if !vector.metadata.contains_key("created_at") {
                    vector.metadata.insert("created_at", now());
                }
            }
        }
        Ok(())
    },
    |db| {
        // Down: Remove created_at
        for collection in db.collections() {
            for vector in collection.vectors_mut() {
                vector.metadata.remove("created_at");
            }
        }
        Ok(())
    },
)
```

### Version Checking Flow

```
Database::open(path)
     │
     ▼
Read header version
     │
     ▼
┌────────────────────────┐
│ Compare with current   │
└───────────┬────────────┘
            │
     ┌──────┴──────┐
     │             │
 Same major    Different major
     │             │
     ▼             ▼
Auto-migrate   Return error
  if needed    (explicit migration
     │          required)
     ▼
 Load database
```
