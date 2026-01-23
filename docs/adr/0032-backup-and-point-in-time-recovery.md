# ADR-0032: Backup and Point-in-Time Recovery

## Status

Accepted

## Context

Production deployments require robust backup and recovery capabilities:

1. **Disaster recovery** — Hardware failures, data center outages, ransomware
2. **Human error recovery** — Accidental deletions or corruptions
3. **Compliance requirements** — Data retention policies (7 years for financial data)
4. **Testing/staging** — Clone production data for development
5. **Migration** — Move databases between environments

Backup strategy requirements:

| Requirement | Priority |
|-------------|----------|
| Full database backup | Critical |
| Point-in-time recovery | High |
| Incremental backups | High |
| Backup verification | High |
| Encryption at rest | Medium |
| Compression | Medium |

## Decision

Implement a **BackupManager** with support for full backups, incremental backups, point-in-time recovery, and verification — with security-hardened path handling.

### Security-First Design

All backup operations validate inputs to prevent path traversal attacks:

```rust
/// Validate backup ID (alphanumeric, underscore, hyphen only)
fn validate_backup_id(id: &str) -> Result<()> {
    if id.is_empty() || id.len() > 256 {
        return Err(NeedleError::InvalidInput("Invalid backup ID length"));
    }

    for c in id.chars() {
        if !c.is_ascii_alphanumeric() && c != '_' && c != '-' {
            return Err(NeedleError::InvalidInput(
                format!("Invalid character '{}' in backup ID", c)
            ));
        }
    }

    // Reject path traversal patterns
    if id.contains("..") || id.starts_with('.') {
        return Err(NeedleError::InvalidInput("Path traversal detected"));
    }

    Ok(())
}

/// Verify path is contained within base directory
fn ensure_path_contained(base: &Path, path: &Path) -> Result<PathBuf> {
    let canonical_base = base.canonicalize()?;
    let canonical_path = path.canonicalize()?;

    if !canonical_path.starts_with(&canonical_base) {
        return Err(NeedleError::InvalidInput("Path escapes backup directory"));
    }

    Ok(canonical_path)
}
```

### Backup Types

```rust
pub enum BackupType {
    Full,        // Complete database snapshot
    Incremental, // Changes since last backup
    Differential, // Changes since last full backup
}

pub struct BackupConfig {
    pub backup_type: BackupType,
    pub compression: CompressionLevel,  // None, Fast, Best
    pub encryption_key: Option<Vec<u8>>,
    pub retention_days: u32,
    pub verify_after_backup: bool,
}
```

### Backup Metadata

```rust
pub struct BackupInfo {
    pub id: String,
    pub created_at: u64,
    pub backup_type: BackupType,
    pub size_bytes: u64,
    pub checksum: String,
    pub parent_backup: Option<String>,  // For incremental
    pub collections: Vec<String>,
    pub vector_count: u64,
}
```

### BackupManager API

```rust
impl BackupManager {
    /// Create a new backup
    pub fn create_backup(&self, db: &Database) -> Result<BackupInfo> {
        let backup_id = self.generate_backup_id();
        validate_backup_id(&backup_id)?;

        let backup_path = ensure_path_contained(
            &self.backup_dir,
            &self.backup_dir.join(&backup_id)
        )?;

        // Snapshot database state
        let snapshot = db.snapshot()?;

        // Write to backup location
        self.write_backup(&backup_path, &snapshot)?;

        // Verify if configured
        if self.config.verify_after_backup {
            self.verify_backup(&backup_id)?;
        }

        Ok(BackupInfo { id: backup_id, ... })
    }

    /// Restore from backup
    pub fn restore_backup(&self, backup_id: &str) -> Result<Database> {
        validate_backup_id(backup_id)?;

        let backup_path = ensure_path_contained(
            &self.backup_dir,
            &self.backup_dir.join(backup_id)
        )?;

        let snapshot = self.read_backup(&backup_path)?;
        Database::from_snapshot(snapshot)
    }

    /// Point-in-time recovery (requires WAL)
    pub fn restore_to_point_in_time(
        &self,
        backup_id: &str,
        target_time: SystemTime,
    ) -> Result<Database> {
        let base_db = self.restore_backup(backup_id)?;
        let wal = self.get_wal_since(backup_id)?;
        wal.replay_until(&base_db, target_time)
    }

    /// Verify backup integrity
    pub fn verify_backup(&self, backup_id: &str) -> Result<VerifyResult> {
        validate_backup_id(backup_id)?;

        let info = self.get_backup_info(backup_id)?;
        let actual_checksum = self.compute_checksum(backup_id)?;

        Ok(VerifyResult {
            valid: actual_checksum == info.checksum,
            expected_checksum: info.checksum,
            actual_checksum,
        })
    }
}
```

### Code References

- `src/backup.rs:31-65` — Path traversal prevention (validate_backup_id)
- `src/backup.rs:67-96` — Path containment verification
- `src/backup.rs:98-120` — BackupConfig structure
- `src/backup.rs` — BackupManager implementation

## Consequences

### Benefits

1. **Disaster recovery** — Restore from any backup in the retention window
2. **Point-in-time recovery** — Recover to specific moment (with WAL integration)
3. **Security hardened** — Path traversal attacks prevented at input validation
4. **Verifiable backups** — Checksums ensure backup integrity before restore
5. **Flexible retention** — Configure retention per backup type

### Tradeoffs

1. **Storage requirements** — Full backups consume space equal to database size
2. **Backup time** — Full backups require reading entire database
3. **WAL dependency** — Point-in-time recovery requires WAL to be enabled
4. **Incremental complexity** — Restoring incrementals requires base + all deltas

### What This Enabled

- Automated backup schedules via cron/systemd
- Multi-region backup replication (copy files to remote storage)
- Database cloning for testing
- Regulatory compliance for data retention

### What This Prevented

- Unrecoverable data loss from hardware failure
- Security vulnerabilities from path traversal attacks
- Silent backup corruption (detected by verification)
- Incomplete restores from corrupt incrementals

### Backup Strategy Recommendations

| Environment | Strategy |
|-------------|----------|
| Development | Manual full backups as needed |
| Staging | Daily full backups, 7-day retention |
| Production | Daily full + hourly incremental, 30-day retention |
| Regulated | Full + incremental + PITR, multi-region, 7-year retention |

### Backup Chain Example

```
Day 1: Full backup (base)
         │
Day 2: Incremental ─┐
Day 3: Incremental ─┤
Day 4: Incremental ─┤
Day 5: Incremental ─┤
Day 6: Incremental ─┤
Day 7: Incremental ─┤
         │
Day 8: Full backup (new base)
         │
Day 9: Incremental ─┐
  ...
```

Restoring Day 6 requires: Day 1 full + Days 2-6 incrementals applied in order.
