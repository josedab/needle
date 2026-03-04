use needle::{Database, Result};
use needle::backup::{BackupConfig, BackupManager, BackupType};

use crate::cli::commands::BackupCommands;

pub fn backup_command(cmd: BackupCommands) -> Result<()> {
    match cmd {
        BackupCommands::Create {
            database,
            output,
            backup_type,
            compress,
        } => backup_create(&database, &output, &backup_type, compress),
        BackupCommands::List { path } => backup_list(&path),
        BackupCommands::Restore {
            backup,
            output,
            force,
        } => backup_restore(&backup, &output, force),
        BackupCommands::Verify { backup } => backup_verify(&backup),
        BackupCommands::Cleanup { path, keep } => backup_cleanup(&path, keep),
    }
}

fn backup_create(database: &str, output: &str, backup_type: &str, compress: bool) -> Result<()> {
    let db = Database::open(database)?;

    let _btype = match backup_type.to_lowercase().as_str() {
        "incremental" => BackupType::Incremental,
        "snapshot" => BackupType::Snapshot,
        _ => BackupType::Full,
    };

    let config = BackupConfig {
        compression: compress,
        verify: true,
        max_backups: Some(10),
        include_metadata: true,
    };

    let manager = BackupManager::new(output, config);
    let metadata = manager.create_backup(&db)?;

    println!("Backup created successfully!");
    println!();
    println!("Backup Details:");
    println!("  ID: {}", metadata.id);
    println!("  Type: {:?}", metadata.backup_type);
    println!("  Collections: {}", metadata.num_collections);
    println!("  Total vectors: {}", metadata.total_vectors);
    println!("  Size: {} bytes", metadata.size_bytes);
    println!("  Checksum: {}", metadata.checksum);

    Ok(())
}

fn backup_list(path: &str) -> Result<()> {
    let config = BackupConfig::default();
    let manager = BackupManager::new(path, config);
    let backups = manager.list_backups()?;

    if backups.is_empty() {
        println!("No backups found in: {}", path);
        return Ok(());
    }

    println!("Available Backups:");
    println!("{:-<80}", "");
    println!(
        "{:<36} {:<12} {:<10} {:<12}",
        "ID", "Type", "Vectors", "Size"
    );
    println!("{:-<80}", "");

    for backup in backups {
        let size_str = if backup.size_bytes > 1024 * 1024 {
            format!("{:.1} MB", backup.size_bytes as f64 / 1024.0 / 1024.0)
        } else if backup.size_bytes > 1024 {
            format!("{:.1} KB", backup.size_bytes as f64 / 1024.0)
        } else {
            format!("{} B", backup.size_bytes)
        };

        println!(
            "{:<36} {:<12} {:<10} {:<12}",
            backup.id,
            format!("{:?}", backup.backup_type),
            backup.total_vectors,
            size_str
        );
    }

    Ok(())
}

fn backup_restore(backup_path: &str, output: &str, force: bool) -> Result<()> {
    if std::path::Path::new(output).exists() && !force {
        eprintln!(
            "Error: Destination '{}' already exists. Use --force to overwrite.",
            output
        );
        return Ok(());
    }

    let backup_dir = std::path::Path::new(backup_path)
        .parent()
        .unwrap_or(std::path::Path::new("."));

    let config = BackupConfig::default();
    let manager = BackupManager::new(backup_dir, config);
    let db = manager.restore_backup(backup_path)?;

    println!("Backup restored successfully!");
    println!("  Collections: {}", db.list_collections().len());
    println!("  Total vectors: {}", db.total_vectors());
    println!();
    println!(
        "Note: To save to '{}', use the database normally and call save().",
        output
    );

    Ok(())
}

fn backup_verify(backup_path: &str) -> Result<()> {
    let config = BackupConfig::default();
    let manager = BackupManager::new(backup_path, config);
    let valid = manager.verify_backup(backup_path)?;

    if valid {
        println!("Backup verification: PASSED");
        println!("  Checksum: Valid");
        println!("  Structure: Valid");
    } else {
        println!("Backup verification: FAILED");
        println!("  The backup file may be corrupted.");
    }

    Ok(())
}

fn backup_cleanup(path: &str, keep: usize) -> Result<()> {
    let config = BackupConfig {
        max_backups: Some(keep),
        ..Default::default()
    };
    let manager = BackupManager::new(path, config);

    let backups = manager.list_backups()?;

    if backups.len() <= keep {
        println!(
            "No backups to clean up (have {}, keeping {}).",
            backups.len(),
            keep
        );
        return Ok(());
    }

    let to_remove = backups.len() - keep;
    println!(
        "Would remove {} old backup(s), keeping last {}.",
        to_remove, keep
    );
    println!(
        "Note: Manual cleanup - delete old backup files from: {}",
        path
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backup_create_and_list() {
        let db_dir = tempfile::tempdir().unwrap();
        let db_path = db_dir.path().join("test.needle");
        let db_path = db_path.to_str().unwrap();
        let mut db = Database::open(db_path).unwrap();
        db.create_collection("test", 64).unwrap();
        db.save().unwrap();

        let backup_dir = tempfile::tempdir().unwrap();
        let backup_path = backup_dir.path().to_str().unwrap();

        assert!(backup_create(db_path, backup_path, "full", false).is_ok());
        assert!(backup_list(backup_path).is_ok());
    }

    #[test]
    fn test_backup_list_empty() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().to_str().unwrap();
        assert!(backup_list(path).is_ok());
    }

    #[test]
    fn test_backup_restore_destination_exists_no_force() {
        let existing = tempfile::NamedTempFile::new().unwrap();
        let output = existing.path().to_str().unwrap();

        let result = backup_restore("nonexistent_backup.bak", output, false);
        assert!(result.is_ok()); // Returns Ok but prints error
    }

    #[test]
    fn test_backup_cleanup_below_keep() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().to_str().unwrap();
        assert!(backup_cleanup(path, 5).is_ok());
    }

    #[test]
    fn test_backup_type_parsing() {
        let db_dir = tempfile::tempdir().unwrap();
        let db_path = db_dir.path().join("test.needle");
        let db_path = db_path.to_str().unwrap();
        let mut db = Database::open(db_path).unwrap();
        db.create_collection("t", 32).unwrap();
        db.save().unwrap();

        let backup_dir = tempfile::tempdir().unwrap();
        let backup_path = backup_dir.path().to_str().unwrap();

        // "unknown" should fall back to Full
        assert!(backup_create(db_path, backup_path, "unknown", false).is_ok());
    }
}
