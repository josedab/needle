use needle::{Database, Result};
use serde_json::json;
use std::io::{self, BufRead};

use super::{parse_distance, parse_query_vector};

/// Maximum import file size (1 GB).
const MAX_IMPORT_FILE_SIZE: u64 = 1024 * 1024 * 1024;

pub fn insert_command(path: &str, collection_name: &str) -> Result<()> {
    let mut db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    println!("Reading vectors from stdin (JSON format: {{\"id\": \"...\", \"vector\": [...], \"metadata\": {{...}}}})");
    println!("Press Ctrl+D when done.");

    let stdin = io::stdin();
    let mut count = 0;

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Invalid JSON: {}", e);
                continue;
            }
        };

        let id = value["id"].as_str().unwrap_or("").to_string();
        let vector: Vec<f32> = match value["vector"].as_array() {
            Some(arr) => arr
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect(),
            None => {
                eprintln!("Missing 'vector' field");
                continue;
            }
        };

        let metadata = value.get("metadata").cloned();

        match coll.insert(&id, &vector, metadata) {
            Ok(_) => count += 1,
            Err(e) => eprintln!("Error inserting '{}': {}", id, e),
        }
    }

    db.save()?;
    println!("Inserted {} vectors.", count);
    Ok(())
}

pub fn search_command(
    path: &str,
    collection_name: &str,
    query_str: &str,
    k: usize,
    explain: bool,
    distance_override: Option<&str>,
    max_age: Option<u64>,
    truncate_dims: Option<usize>,
) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let query = parse_query_vector(query_str)?;

    let distance_fn = distance_override.and_then(|d| match parse_distance(d) {
        Some(parsed) => Some(parsed),
        None => {
            eprintln!(
                "Warning: Unknown distance function '{}', using collection default",
                d
            );
            None
        }
    });

    if explain {
        if distance_override.is_some() {
            eprintln!("Warning: --explain and --distance cannot be combined; ignoring --distance");
        }
        let (results, explain_data) = coll.search_explain(&query, k)?;

        println!("Search results (k={}):", k);
        for result in &results {
            let meta = result
                .metadata
                .as_ref()
                .map_or_else(|| "null".to_string(), |m| m.to_string());
            println!(
                "  ID: {}, Distance: {:.6}, Metadata: {}",
                result.id, result.distance, meta
            );
        }

        println!();
        println!("Query Profiling:");
        println!("  Total time: {}μs", explain_data.total_time_us);
        println!("  Index traversal: {}μs", explain_data.index_time_us);
        println!("  Filter evaluation: {}μs", explain_data.filter_time_us);
        println!("  Result enrichment: {}μs", explain_data.enrich_time_us);
        println!();
        println!("HNSW Statistics:");
        println!("  Visited nodes: {}", explain_data.hnsw_stats.visited_nodes);
        println!(
            "  Layers traversed: {}",
            explain_data.hnsw_stats.layers_traversed
        );
        println!(
            "  Distance computations: {}",
            explain_data.hnsw_stats.distance_computations
        );
        println!(
            "  Traversal time: {}μs",
            explain_data.hnsw_stats.traversal_time_us
        );
        println!();
        println!("Query Parameters:");
        println!("  Dimensions: {}", explain_data.dimensions);
        println!("  Collection size: {}", explain_data.collection_size);
        println!("  Requested k: {}", explain_data.requested_k);
        println!("  Effective k: {}", explain_data.effective_k);
        println!("  ef_search: {}", explain_data.ef_search);
        println!("  Distance function: {}", explain_data.distance_function);
    } else {
        // Build a SearchParams query with all configured options
        let mut search = coll.query(&query).limit(k);

        if let Some(dist) = distance_fn {
            search = search.distance(dist);
        }
        if let Some(dims) = truncate_dims {
            search = search.with_dimensions(dims);
        }
        if let Some(age) = max_age {
            search = search.with_time_decay(
                needle::collection::pipeline::TimeDecay::Step { window_seconds: age }
            );
        }

        let results = search.execute()?;

        println!("Search results (k={}):", k);
        if distance_override.is_some() {
            println!("  (using distance override: {:?})", distance_fn);
        }
        if truncate_dims.is_some() {
            println!("  (using Matryoshka truncation to {} dims)", truncate_dims.expect("checked"));
        }
        if max_age.is_some() {
            println!("  (using max_age filter: {}s)", max_age.expect("checked"));
        }
        for result in results {
            let meta = result
                .metadata
                .as_ref()
                .map_or_else(|| "null".to_string(), |m| m.to_string());
            println!(
                "  ID: {}, Distance: {:.6}, Metadata: {}",
                result.id, result.distance, meta
            );
        }
    }

    Ok(())
}

pub fn delete_command(path: &str, collection_name: &str, id: &str) -> Result<()> {
    let mut db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let deleted = coll.delete(id)?;
    db.save()?;

    if deleted {
        println!("Deleted vector '{}'", id);
    } else {
        println!("Vector '{}' not found", id);
    }

    Ok(())
}

pub fn get_command(path: &str, collection_name: &str, id: &str) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    match coll.get(id) {
        Some((vector, metadata)) => {
            let output = json!({
                "id": id,
                "vector": vector,
                "metadata": metadata
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        None => {
            println!("Vector '{}' not found", id);
        }
    }

    Ok(())
}

pub fn compact_command(path: &str) -> Result<()> {
    let mut db = Database::open(path)?;

    let mut total_deleted = 0;
    for name in db.list_collections() {
        let coll = db.collection(&name)?;
        let deleted = coll.compact()?;
        if deleted > 0 {
            println!("  {}: removed {} deleted vectors", name, deleted);
            total_deleted += deleted;
        }
    }

    db.save()?;

    if total_deleted > 0 {
        println!(
            "Compaction complete: removed {} total deleted vectors",
            total_deleted
        );
    } else {
        println!("No deleted vectors to compact");
    }

    Ok(())
}

pub fn export_command(path: &str, collection_name: &str) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let vectors = coll.export_all()?;

    let output = json!({
        "collection": collection_name,
        "dimensions": coll.dimensions(),
        "count": vectors.len(),
        "vectors": vectors.iter().map(|(id, vec, meta)| {
            json!({
                "id": id,
                "vector": vec,
                "metadata": meta
            })
        }).collect::<Vec<_>>()
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}

pub fn import_command(path: &str, collection_name: &str, file_path: &str) -> Result<()> {
    let mut db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let content: String = if file_path == "-" {
        let stdin = io::stdin();
        let mut buffer = String::new();
        for line in stdin.lock().lines() {
            buffer.push_str(&line?);
            buffer.push('\n');
        }
        buffer
    } else {
        let metadata = std::fs::metadata(file_path)?;
        if metadata.len() > MAX_IMPORT_FILE_SIZE {
            return Err(needle::NeedleError::InvalidInput(format!(
                "Import file too large ({} bytes). Maximum allowed size is {} bytes",
                metadata.len(),
                MAX_IMPORT_FILE_SIZE
            )));
        }
        std::fs::read_to_string(file_path)?
    };

    let data: serde_json::Value = serde_json::from_str(&content)?;

    let vectors = data["vectors"]
        .as_array()
        .ok_or_else(|| needle::NeedleError::InvalidConfig("Missing 'vectors' array".to_string()))?;

    let mut count = 0;
    let mut errors = 0;

    for entry in vectors {
        let id = match entry["id"].as_str() {
            Some(id) => id.to_string(),
            None => {
                eprintln!("Skipping entry without 'id'");
                errors += 1;
                continue;
            }
        };

        let vector: Vec<f32> = match entry["vector"].as_array() {
            Some(arr) => arr
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect(),
            None => {
                eprintln!("Skipping '{}': missing 'vector'", id);
                errors += 1;
                continue;
            }
        };

        let metadata = entry.get("metadata").cloned();

        match coll.insert(&id, &vector, metadata) {
            Ok(_) => count += 1,
            Err(e) => {
                eprintln!("Error inserting '{}': {}", id, e);
                errors += 1;
            }
        }
    }

    db.save()?;

    println!("Imported {} vectors", count);
    if errors > 0 {
        println!("Skipped {} entries due to errors", errors);
    }

    Ok(())
}

pub fn count_command(path: &str, collection_name: &str) -> Result<()> {
    let db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    println!("{}", coll.len());

    Ok(())
}

pub fn clear_command(path: &str, collection_name: &str, force: bool) -> Result<()> {
    if !force {
        eprint!(
            "Are you sure you want to delete all vectors from '{}'? [y/N] ",
            collection_name
        );
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    let mut db = Database::open(path)?;
    let coll = db.collection(collection_name)?;

    let ids = coll.ids()?;
    let count = ids.len();

    for id in ids {
        coll.delete(&id)?;
    }

    coll.compact()?;
    db.save()?;

    println!("Deleted {} vectors from '{}'", count, collection_name);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use needle::Database;

    fn setup_db_with_vectors() -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle").to_str().unwrap().to_string();
        let mut db = Database::open(&path).unwrap();
        db.create_collection("test", 4).unwrap();
        let coll = db.collection("test").unwrap();
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], Some(serde_json::json!({"tag": "a"}))).unwrap();
        db.save().unwrap();
        (dir, path)
    }

    #[test]
    fn test_get_command_found() {
        let (_dir, path) = setup_db_with_vectors();
        assert!(get_command(&path, "test", "v1").is_ok());
    }

    #[test]
    fn test_get_command_not_found() {
        let (_dir, path) = setup_db_with_vectors();
        assert!(get_command(&path, "test", "nonexistent").is_ok());
    }

    #[test]
    fn test_get_command_nonexistent_collection() {
        let (_dir, path) = setup_db_with_vectors();
        let result = get_command(&path, "no_coll", "v1");
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_command() {
        let (_dir, path) = setup_db_with_vectors();
        assert!(delete_command(&path, "test", "v1").is_ok());
    }

    #[test]
    fn test_delete_command_not_found() {
        let (_dir, path) = setup_db_with_vectors();
        assert!(delete_command(&path, "test", "nonexistent").is_ok());
    }

    #[test]
    fn test_count_command() {
        let (_dir, path) = setup_db_with_vectors();
        assert!(count_command(&path, "test").is_ok());
    }

    #[test]
    fn test_compact_command() {
        let (_dir, path) = setup_db_with_vectors();
        assert!(compact_command(&path).is_ok());
    }

    #[test]
    fn test_export_command() {
        let (_dir, path) = setup_db_with_vectors();
        assert!(export_command(&path, "test").is_ok());
    }

    #[test]
    fn test_clear_command_with_force() {
        let (_dir, path) = setup_db_with_vectors();
        assert!(clear_command(&path, "test", true).is_ok());

        let db = Database::open(&path).unwrap();
        let coll = db.collection("test").unwrap();
        assert_eq!(coll.len(), 0);
    }

    #[test]
    fn test_import_command_valid_file() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.needle");
        let db_path = db_path.to_str().unwrap();
        let mut db = Database::open(db_path).unwrap();
        db.create_collection("import_test", 3).unwrap();
        db.save().unwrap();

        let import_file = tempfile::NamedTempFile::new().unwrap();
        let import_data = serde_json::json!({
            "vectors": [
                {"id": "d1", "vector": [1.0, 0.0, 0.0]},
                {"id": "d2", "vector": [0.0, 1.0, 0.0], "metadata": {"key": "val"}}
            ]
        });
        std::fs::write(import_file.path(), import_data.to_string()).unwrap();

        let result = import_command(db_path, "import_test", import_file.path().to_str().unwrap());
        assert!(result.is_ok());

        let db = Database::open(db_path).unwrap();
        let coll = db.collection("import_test").unwrap();
        assert_eq!(coll.len(), 2);
    }

    #[test]
    fn test_import_command_missing_vectors_array() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.needle");
        let db_path = db_path.to_str().unwrap();
        let mut db = Database::open(db_path).unwrap();
        db.create_collection("import_err", 3).unwrap();
        db.save().unwrap();

        let import_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(import_file.path(), r#"{"data": []}"#).unwrap();

        let result = import_command(db_path, "import_err", import_file.path().to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_import_command_entry_without_id() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.needle");
        let db_path = db_path.to_str().unwrap();
        let mut db = Database::open(db_path).unwrap();
        db.create_collection("import_skip", 2).unwrap();
        db.save().unwrap();

        let import_file = tempfile::NamedTempFile::new().unwrap();
        let import_data = serde_json::json!({
            "vectors": [
                {"vector": [1.0, 0.0]},
                {"id": "good", "vector": [0.0, 1.0]}
            ]
        });
        std::fs::write(import_file.path(), import_data.to_string()).unwrap();

        let result = import_command(db_path, "import_skip", import_file.path().to_str().unwrap());
        assert!(result.is_ok());

        let db = Database::open(db_path).unwrap();
        let coll = db.collection("import_skip").unwrap();
        assert_eq!(coll.len(), 1);
    }

    #[test]
    fn test_search_command_basic() {
        let (_dir, path) = setup_db_with_vectors();
        assert!(search_command(
            &path, "test", "1.0,0.0,0.0,0.0", 2, false, None, None, None
        ).is_ok());
    }

    #[test]
    fn test_search_command_with_distance_override() {
        let (_dir, path) = setup_db_with_vectors();
        assert!(search_command(
            &path, "test", "1.0,0.0,0.0,0.0", 2, false, Some("euclidean"), None, None
        ).is_ok());
    }

    #[test]
    fn test_search_command_invalid_distance() {
        let (_dir, path) = setup_db_with_vectors();
        // Invalid distance should print warning and continue with default
        assert!(search_command(
            &path, "test", "1.0,0.0,0.0,0.0", 2, false, Some("invalid_dist"), None, None
        ).is_ok());
    }
}
