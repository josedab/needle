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
