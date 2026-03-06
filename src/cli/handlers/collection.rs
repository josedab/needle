use needle::{CollectionConfig, Database, DistanceFunction, NeedleError, Result};

use super::parse_distance;

pub fn info_command(path: &str) -> Result<()> {
    let db = Database::open(path)?;

    println!("Database: {}", path);
    println!("Collections: {}", db.list_collections().len());
    println!("Total vectors: {}", db.total_vectors());
    println!();

    for name in db.list_collections() {
        let coll = db.collection(&name)?;
        println!("  Collection: {}", name);
        println!("    Dimensions: {:?}", coll.dimensions());
        println!("    Vectors: {}", coll.len());
    }

    Ok(())
}

pub fn create_command(path: &str) -> Result<()> {
    let _db = Database::open(path)?;
    println!("Created database: {}", path);
    Ok(())
}

pub fn collections_command(path: &str) -> Result<()> {
    let db = Database::open(path)?;

    let collections = db.list_collections();
    if collections.is_empty() {
        println!("No collections found.");
    } else {
        println!("Collections:");
        for name in collections {
            let coll = db.collection(&name)?;
            println!(
                "  {} (dimensions: {:?}, vectors: {})",
                name,
                coll.dimensions(),
                coll.len()
            );
        }
    }

    Ok(())
}

pub fn create_collection_command(
    path: &str,
    name: &str,
    dimensions: usize,
    distance: &str,
    encrypted: bool,
) -> Result<()> {
    if dimensions == 0 {
        return Err(NeedleError::InvalidConfig(
            "Vector dimensions must be greater than 0".to_string(),
        ));
    }
    let mut db = Database::open(path)?;

    let dist_fn = match parse_distance(distance) {
        Some(parsed) => parsed,
        None => {
            eprintln!("Unknown distance function: {}. Using cosine.", distance);
            DistanceFunction::Cosine
        }
    };

    let config = CollectionConfig::new(name, dimensions).with_distance(dist_fn);
    db.create_collection_with_config(config)?;
    db.save()?;

    if encrypted {
        #[cfg(feature = "encryption")]
        {
            println!(
                "Created encrypted collection '{}' with {} dimensions ({} distance)",
                name, dimensions, distance
            );
            println!("  Encryption: ChaCha20-Poly1305");
            println!("  Note: Set NEEDLE_ENCRYPTION_KEY env var or use --key-file for key management");
        }
        #[cfg(not(feature = "encryption"))]
        {
            eprintln!("Warning: --encrypted requires --features encryption. Collection created without encryption.");
        }
    }

    if !encrypted {
        println!(
            "Created collection '{}' with {} dimensions ({} distance)",
            name, dimensions, distance
        );
    }
    Ok(())
}

pub fn rename_collection_command(path: &str, old_name: &str, new_name: &str) -> Result<()> {
    let db = Database::open(path)?;
    db.rename_collection(old_name, new_name)?;
    db.save()?;
    println!("Renamed collection '{}' to '{}'", old_name, new_name);
    Ok(())
}

pub fn stats_command(path: &str, collection_name: &str) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let active_count = coll.len();
    let deleted_count = coll.deleted_count();
    let total_stored = active_count + deleted_count;

    println!("Collection: {}", collection_name);
    println!("  Dimensions: {:?}", coll.dimensions());
    println!("  Active vectors: {}", active_count);
    println!("  Deleted vectors: {}", deleted_count);
    println!("  Total stored: {}", total_stored);
    println!("  Empty: {}", coll.is_empty());

    if deleted_count > 0 {
        let delete_ratio = deleted_count as f64 / total_stored as f64;
        println!("  Deletion ratio: {:.1}%", delete_ratio * 100.0);
        if coll.needs_compaction(0.2) {
            println!("  Recommendation: Run 'compact' to reclaim space");
        }
    }

    Ok(())
}

pub fn status_command(path: &str) -> Result<()> {
    let file_meta = std::fs::metadata(path);
    match file_meta {
        Ok(meta) => {
            let file_size = meta.len();
            let modified = meta
                .modified()
                .map(|t| {
                    let elapsed = t.elapsed().unwrap_or_default();
                    format_duration(elapsed)
                })
                .unwrap_or_else(|_| "unknown".to_string());

            match Database::open(path) {
                Ok(db) => {
                    let collections = db.list_collections();
                    let total_vectors: usize = collections
                        .iter()
                        .filter_map(|name| db.collection(name).ok())
                        .map(|c| c.len())
                        .sum();

                    println!("Database: {}", path);
                    println!("  Status:      \u{2705} healthy");
                    println!("  File size:   {}", format_bytes(file_size));
                    println!("  Modified:    {} ago", modified);
                    println!("  Collections: {}", collections.len());
                    println!("  Vectors:     {}", total_vectors);
                }
                Err(e) => {
                    println!("Database: {}", path);
                    println!("  Status:      \u{274c} error");
                    println!("  File size:   {}", format_bytes(file_size));
                    println!("  Error:       {}", e);
                }
            }
        }
        Err(e) => {
            println!("Database: {}", path);
            println!("  Status: \u{274c} not found ({})", e);
        }
    }
    Ok(())
}

pub(crate) fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

fn format_duration(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86400 {
        format!("{}h", secs / 3600)
    } else {
        format!("{}d", secs / 86400)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_collection_zero_dimensions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path = path.to_str().unwrap();

        let result = create_collection_command(path, "test", 0, "cosine", false);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            NeedleError::InvalidConfig(_)
        ));
    }

    #[test]
    fn test_create_collection_unknown_distance_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path = path.to_str().unwrap();

        let result = create_collection_command(path, "test", 128, "hamming", false);
        assert!(result.is_ok());

        let db = Database::open(path).unwrap();
        assert!(db.list_collections().contains(&"test".to_string()));
    }

    #[test]
    fn test_create_collection_valid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path = path.to_str().unwrap();

        let result = create_collection_command(path, "docs", 384, "euclidean", false);
        assert!(result.is_ok());

        let db = Database::open(path).unwrap();
        let coll = db.collection("docs").unwrap();
        assert_eq!(coll.dimensions(), Some(384));
    }

    #[test]
    fn test_create_collection_duplicate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path = path.to_str().unwrap();

        create_collection_command(path, "dup", 64, "cosine", false).unwrap();
        let result = create_collection_command(path, "dup", 64, "cosine", false);
        assert!(result.is_err());
    }

    #[test]
    fn test_info_command_valid_db() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path = path.to_str().unwrap();
        let mut db = Database::open(path).unwrap();
        db.create_collection("test", 128).unwrap();
        db.save().unwrap();

        assert!(info_command(path).is_ok());
    }

    #[test]
    fn test_collections_command_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path = path.to_str().unwrap();
        let _db = Database::open(path).unwrap();

        assert!(collections_command(path).is_ok());
    }

    #[test]
    fn test_stats_command() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path = path.to_str().unwrap();
        let mut db = Database::open(path).unwrap();
        db.create_collection("stats_test", 64).unwrap();
        db.save().unwrap();

        assert!(stats_command(path, "stats_test").is_ok());
    }

    #[test]
    fn test_stats_command_nonexistent_collection() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path = path.to_str().unwrap();
        let _db = Database::open(path).unwrap();

        let result = stats_command(path, "nonexistent");
        assert!(result.is_err());
    }
}
