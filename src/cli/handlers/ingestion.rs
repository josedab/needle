use needle::{Database, Result};

use crate::cli::commands::IngestionCommands;

pub fn ingestion_command(cmd: IngestionCommands) -> Result<()> {
    match cmd {
        IngestionCommands::Status { database } => ingestion_status(&database),
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
