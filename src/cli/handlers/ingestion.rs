use needle::{Database, Result};

use crate::cli::commands::IngestionCommands;

pub fn ingestion_command(cmd: IngestionCommands) -> Result<()> {
    match cmd {
        IngestionCommands::Status { database } => ingestion_status(&database),
        IngestionCommands::Ingest {
            database,
            collection,
            input,
            chunk_size,
            chunk_overlap,
            dimensions,
        } => ingest_documents(&database, &collection, &input, chunk_size, chunk_overlap, dimensions),
    }
}

fn ingestion_status(path: &str) -> Result<()> {
    let db = Database::open(path)?;
    let collections = db.list_collections();

    println!("═══ Ingestion Status ═══");
    println!("  Database: {path}");
    println!("  Collections: {}", collections.len());
    for name in &collections {
        let coll = db.collection(name)?;
        println!("    {name}: {} vectors", coll.len());
    }
    println!();
    println!("Pipeline counters (current process):");
    println!(
        "  dedup_checked:  {}",
        needle::collection::dedup::DEDUP_CHECKED_TOTAL
            .load(std::sync::atomic::Ordering::Relaxed)
    );
    println!(
        "  dedup_rejected: {}",
        needle::collection::dedup::DEDUP_REJECTED_TOTAL
            .load(std::sync::atomic::Ordering::Relaxed)
    );
    println!();
    println!("For streaming pipeline status, use the HTTP API at /ingestion/status.");
    Ok(())
}

fn ingest_documents(
    db_path: &str,
    collection_name: &str,
    input_path: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    dimensions: usize,
) -> Result<()> {
    use needle::ml::rag::chunking::{DocumentLoader, RecursiveTextSplitter};
    use std::path::Path;

    let mut db = Database::open(db_path)?;
    if !db.has_collection(collection_name) {
        db.create_collection(collection_name, dimensions)?;
        println!("Created collection '{collection_name}' with {dimensions} dimensions");
    }
    let coll = db.collection(collection_name)?;

    let path = Path::new(input_path);
    let splitter = RecursiveTextSplitter::new(chunk_size, chunk_overlap);

    let files: Vec<std::path::PathBuf> = if path.is_dir() {
        std::fs::read_dir(path)
            .map_err(|e| needle::NeedleError::InvalidInput(format!("Failed to read directory: {e}")))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| {
                matches!(
                    p.extension().and_then(|e| e.to_str()),
                    Some("txt" | "md" | "json" | "html" | "htm")
                )
            })
            .collect()
    } else {
        vec![path.to_path_buf()]
    };

    let mut total_chunks = 0;
    let mut total_files = 0;

    for file_path in &files {
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| needle::NeedleError::InvalidInput(format!("Failed to read {}: {e}", file_path.display())))?;

        let file_name = file_path.file_name().unwrap_or_default().to_string_lossy().to_string();

        let doc = match file_path.extension().and_then(|e| e.to_str()) {
            Some("md") => DocumentLoader::load_markdown(&file_name, &content),
            _ => DocumentLoader::load_plaintext(&file_name, &content),
        };

        let chunks = splitter.split(&doc.text);
        for (i, (chunk_text, _start, _end)) in chunks.iter().enumerate() {
            let chunk_id = format!("{}_{}", file_name, i);
            // Generate deterministic placeholder embedding from content hash.
            // In production, replace with a real embedding model (e.g., --embed-model flag).
            let mut rng_state = 0u64;
            for b in chunk_text.bytes() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(b as u64);
            }
            let embedding: Vec<f32> = (0..dimensions)
                .map(|d| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(d as u64);
                    ((rng_state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
                })
                .collect();

            let metadata = serde_json::json!({
                "source": file_name,
                "chunk_index": i,
                "text": &chunk_text[..chunk_text.len().min(500)],
            });

            if let Err(e) = coll.insert(&chunk_id, &embedding, Some(metadata)) {
                eprintln!("Warning: failed to insert chunk {chunk_id}: {e}");
                continue;
            }
            total_chunks += 1;
        }
        total_files += 1;
        println!("  Ingested: {} ({} chunks)", file_name, chunks.len());
    }

    db.save()?;
    println!();
    println!("Ingestion complete: {total_files} files, {total_chunks} chunks indexed");
    println!("Note: Using placeholder embeddings. For production use, pair with an embedding model.");
    Ok(())
}
