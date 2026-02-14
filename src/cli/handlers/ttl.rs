use needle::{Database, Result};

use crate::cli::commands::TtlCommands;

pub fn ttl_command(cmd: TtlCommands) -> Result<()> {
    match cmd {
        TtlCommands::Sweep {
            database,
            collection,
        } => ttl_sweep(&database, &collection),
        TtlCommands::Stats {
            database,
            collection,
        } => ttl_stats(&database, &collection),
    }
}

fn ttl_sweep(path: &str, collection_name: &str) -> Result<()> {
    let mut db = Database::open(path)?;
    let collection = db.collection(collection_name)?;
    let expired = collection.expire_vectors()?;
    db.save()?;

    if expired > 0 {
        println!("Expired {} vectors from '{}'", expired, collection_name);
    } else {
        println!("No expired vectors found in '{}'", collection_name);
    }
    Ok(())
}

fn ttl_stats(path: &str, collection_name: &str) -> Result<()> {
    let db = Database::open(path)?;
    let collection = db.collection(collection_name)?;
    let (total, expired, earliest, latest) = collection.ttl_stats();

    println!("TTL Statistics for '{}':", collection_name);
    println!("{:-<50}", "");
    println!("Vectors with TTL:        {}", total);
    println!("Currently expired:       {}", expired);

    if let Some(earliest_ts) = earliest {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if earliest_ts > now {
            println!("Next expiration in:      {} seconds", earliest_ts - now);
        } else {
            println!("Oldest expired:          {} seconds ago", now - earliest_ts);
        }
    }

    if let Some(latest_ts) = latest {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if latest_ts > now {
            println!("Latest expiration in:    {} seconds", latest_ts - now);
        }
    }

    if collection.needs_expiration_sweep(0.1) {
        println!("\nRecommendation: Run 'needle ttl sweep' to clean up expired vectors.");
    }

    Ok(())
}
