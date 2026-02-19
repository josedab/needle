//! Cache management CLI handlers.

use needle::{Database, Result};

use crate::cli::commands::CacheCommands;

pub fn cache_command(cmd: CacheCommands) -> Result<()> {
    match cmd {
        CacheCommands::Stats { database } => cache_stats_command(&database),
        CacheCommands::Clear { database, force } => cache_clear_command(&database, force),
    }
}

fn cache_stats_command(path: &str) -> Result<()> {
    let db = Database::open(path)?;
    let collections = db.list_collections();

    println!("═══ Cache Statistics ═══");
    println!("  Database: {path}");
    for name in &collections {
        println!("  Collection '{name}':");
        println!("    (Cache stats available via SDK/HTTP API at runtime)");
    }
    println!();
    println!("Note: Query caches are in-memory and per-process.");
    println!("Use the HTTP API /collections/:name/stats for live metrics.");
    Ok(())
}

fn cache_clear_command(path: &str, force: bool) -> Result<()> {
    if !force {
        println!("This will clear all cached query results for: {path}");
        println!("Use --force to skip this confirmation.");
        return Ok(());
    }
    // Caches are in-memory per-process, so clearing via CLI just acknowledges
    println!("In-memory caches will be reset when the database is next opened.");
    println!("For a running server, POST to /collections/:name/cache/clear.");
    Ok(())
}
