#![allow(clippy::unwrap_used)]
//! Columnar Export/Import for Vector Collections
//!
//! Provides export/import in a columnar JSON format that maps directly to
//! Apache Arrow / Parquet schemas. This enables interoperability with
//! data engineering tools (Pandas, Polars, DuckDB) without requiring
//! Arrow/Parquet dependencies at compile time.
//!
//! # Schema Convention
//!
//! The Needle columnar format uses this Arrow-compatible schema:
//!
//! | Column | Arrow Type | Description |
//! |--------|-----------|-------------|
//! | `id` | `Utf8` | Vector identifier |
//! | `vector` | `FixedSizeList<Float32>` | Embedding values |
//! | `metadata` | `Utf8` (JSON) | Serialized metadata |
//!
//! # Export Format (JSON Lines)
//!
//! ```json
//! {"id":"doc1","vector":[0.1,0.2,0.3],"metadata":{"title":"Hello"}}
//! {"id":"doc2","vector":[0.4,0.5,0.6],"metadata":{"title":"World"}}
//! ```
//!
//! This format is directly loadable by:
//! - `pandas.read_json("export.jsonl", lines=True)`
//! - `polars.read_ndjson("export.jsonl")`
//! - `DuckDB: SELECT * FROM read_json_auto('export.jsonl')`
//!
//! # Example
//!
//! ```rust
//! use needle::Database;
//! use needle::columnar::{export_jsonl, import_jsonl};
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 3).unwrap();
//! let coll = db.collection("docs").unwrap();
//! coll.insert("d1", &[1.0, 2.0, 3.0], None).unwrap();
//!
//! // Export
//! let data = export_jsonl(&db, "docs").unwrap();
//! assert!(String::from_utf8_lossy(&data).contains("d1"));
//!
//! // Import into a new collection
//! let db2 = Database::in_memory();
//! db2.create_collection("imported", 3).unwrap();
//! import_jsonl(&db2, "imported", &data).unwrap();
//! ```

use crate::database::Database;
use crate::error::{NeedleError, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};

/// Export a collection to JSON Lines (newline-delimited JSON).
///
/// Each line contains: `{"id": "...", "vector": [...], "metadata": {...}}`
pub fn export_jsonl(db: &Database, collection: &str) -> Result<Vec<u8>> {
    let coll = db.collection(collection)?;
    let entries = coll.export_all()?;

    let mut buf = Vec::new();
    for (id, vector, metadata) in &entries {
        let line = json!({
            "id": id,
            "vector": vector,
            "metadata": metadata,
        });
        serde_json::to_writer(&mut buf, &line)
            .map_err(|e| NeedleError::InvalidInput(format!("Serialization failed: {e}")))?;
        buf.push(b'\n');
    }

    Ok(buf)
}

/// Import vectors from JSON Lines format into a collection.
///
/// Each line must contain `{"id": "...", "vector": [...]}` with optional `metadata`.
pub fn import_jsonl(db: &Database, collection: &str, data: &[u8]) -> Result<usize> {
    let coll = db.collection(collection)?;
    let reader = BufReader::new(data);
    let mut count = 0;

    for line in reader.lines() {
        let line = line.map_err(|e| NeedleError::InvalidInput(format!("Read error: {e}")))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let entry: Value = serde_json::from_str(trimmed)
            .map_err(|e| NeedleError::InvalidInput(format!("Invalid JSON on line {}: {e}", count + 1)))?;

        let id = entry.get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput(format!("Missing 'id' on line {}", count + 1)))?;

        let vector: Vec<f32> = entry.get("vector")
            .and_then(|v| v.as_array())
            .ok_or_else(|| NeedleError::InvalidInput(format!("Missing 'vector' on line {}", count + 1)))?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        let metadata = entry.get("metadata").cloned();

        coll.insert(id, &vector, metadata)?;
        count += 1;
    }

    Ok(count)
}

/// Export to CSV format (id, vector as semicolon-separated floats, metadata JSON).
pub fn export_csv(db: &Database, collection: &str) -> Result<Vec<u8>> {
    let coll = db.collection(collection)?;
    let entries = coll.export_all()?;

    let mut buf = Vec::new();
    writeln!(buf, "id,vector,metadata")
        .map_err(|e| NeedleError::InvalidInput(e.to_string()))?;

    for (id, vector, metadata) in &entries {
        let vec_str: Vec<String> = vector.iter().map(|v| format!("{v}")).collect();
        let meta_str = metadata
            .as_ref()
            .map(|m| serde_json::to_string(m).unwrap_or_default())
            .unwrap_or_default();
        writeln!(buf, "\"{id}\",\"{}\",\"{}\"",
            vec_str.join(";"),
            meta_str.replace('"', "\"\""))
            .map_err(|e| NeedleError::InvalidInput(e.to_string()))?;
    }

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_import_jsonl_roundtrip() {
        let db = Database::in_memory();
        db.create_collection("test", 3).unwrap();
        let coll = db.collection("test").unwrap();
        coll.insert("d1", &[1.0, 2.0, 3.0], Some(json!({"tag": "a"}))).unwrap();
        coll.insert("d2", &[4.0, 5.0, 6.0], None).unwrap();

        let exported = export_jsonl(&db, "test").unwrap();
        let text = String::from_utf8_lossy(&exported);
        assert!(text.contains("d1"));
        assert!(text.contains("d2"));

        // Import into fresh collection
        let db2 = Database::in_memory();
        db2.create_collection("imported", 3).unwrap();
        let count = import_jsonl(&db2, "imported", &exported).unwrap();
        assert_eq!(count, 2);

        let coll2 = db2.collection("imported").unwrap();
        assert_eq!(coll2.len(), 2);
    }

    #[test]
    fn test_export_csv() {
        let db = Database::in_memory();
        db.create_collection("test", 2).unwrap();
        let coll = db.collection("test").unwrap();
        coll.insert("v1", &[1.0, 2.0], Some(json!({"x": 1}))).unwrap();

        let csv = export_csv(&db, "test").unwrap();
        let text = String::from_utf8_lossy(&csv);
        assert!(text.starts_with("id,vector,metadata"));
        assert!(text.contains("v1"));
    }

    #[test]
    fn test_import_empty() {
        let db = Database::in_memory();
        db.create_collection("test", 3).unwrap();
        let count = import_jsonl(&db, "test", b"").unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_import_invalid_json() {
        let db = Database::in_memory();
        db.create_collection("test", 3).unwrap();
        let result = import_jsonl(&db, "test", b"not json\n");
        assert!(result.is_err());
    }
}
