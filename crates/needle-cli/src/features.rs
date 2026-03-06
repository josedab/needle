use needle::backup::{BackupConfig, BackupManager, BackupType};
use needle::{Database, Result};

#[cfg(feature = "observability")]
use needle::NeedleError;
#[cfg(feature = "observability")]
use serde_json::json;

#[cfg(feature = "observability")]
use needle::drift::{DriftConfig, DriftDetector};

use crate::commands::{AliasCommands, BackupCommands, FederateCommands, TtlCommands};
#[cfg(feature = "observability")]
use crate::commands::DriftCommands;

// ============================================================================
// Backup Commands
// ============================================================================

pub(crate) fn backup_command(cmd: BackupCommands) -> Result<()> {
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

    // Get backup directory from backup_path
    let backup_dir = std::path::Path::new(backup_path)
        .parent()
        .unwrap_or(std::path::Path::new("."));

    let config = BackupConfig::default();
    let manager = BackupManager::new(backup_dir, config);
    let db = manager.restore_backup(backup_path)?;

    // The restored database is in-memory, we need to export and re-import
    // For now, just show a message about the restored data
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

    // List backups and manually clean up old ones
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

// ============================================================================
// Drift Detection Commands
// ============================================================================

#[cfg(feature = "observability")]
pub(crate) fn drift_command(cmd: DriftCommands) -> Result<()> {
    match cmd {
        DriftCommands::Baseline {
            database,
            collection,
            output,
            sample_size,
        } => drift_baseline(&database, &collection, &output, sample_size),
        DriftCommands::Detect {
            database,
            collection,
            baseline,
            threshold,
        } => drift_detect(&database, &collection, &baseline, threshold),
        DriftCommands::Report {
            database,
            collection,
            baseline,
            format,
        } => drift_report(&database, &collection, &baseline, &format),
    }
}

#[cfg(feature = "observability")]
fn drift_baseline(
    database: &str,
    collection_name: &str,
    output: &str,
    sample_size: usize,
) -> Result<()> {
    let db = Database::open(database)?;
    let coll = db.collection(collection_name)?;

    // Get dimensions from collection
    let dimensions = coll.dimensions().unwrap_or(0);
    if dimensions == 0 {
        return Err(NeedleError::InvalidInput(
            "Cannot determine vector dimensions".to_string(),
        ));
    }

    // Export vectors for baseline
    let vectors = coll.export_all()?;
    let sample: Vec<Vec<f32>> = if sample_size > 0 && sample_size < vectors.len() {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut sampled: Vec<_> = vectors.iter().map(|(_, v, _)| v.clone()).collect();
        sampled.shuffle(&mut rng);
        sampled.truncate(sample_size);
        sampled
    } else {
        vectors.iter().map(|(_, v, _)| v.clone()).collect()
    };

    // Create detector and add baseline
    let config = DriftConfig::default();
    let mut detector = DriftDetector::new(dimensions, config);
    detector.add_baseline(&sample)?;

    // Compute baseline statistics for saving
    let centroid: Vec<f32> = (0..dimensions)
        .map(|d| sample.iter().map(|v| v[d]).sum::<f32>() / sample.len() as f32)
        .collect();

    let variance: Vec<f32> = (0..dimensions)
        .map(|d| {
            let mean = centroid[d];
            sample.iter().map(|v| (v[d] - mean).powi(2)).sum::<f32>() / sample.len() as f32
        })
        .collect();

    // Save baseline to file
    let baseline_json = serde_json::to_string_pretty(&json!({
        "collection": collection_name,
        "sample_size": sample.len(),
        "dimensions": dimensions,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "centroid": centroid,
        "variance": variance
    }))?;

    std::fs::write(output, baseline_json)?;

    println!("Baseline created successfully!");
    println!();
    println!("Details:");
    println!("  Collection: {}", collection_name);
    println!("  Sample size: {}", sample.len());
    println!("  Dimensions: {}", dimensions);
    println!("  Output: {}", output);

    Ok(())
}

#[cfg(feature = "observability")]
fn drift_detect(
    database: &str,
    collection_name: &str,
    baseline_path: &str,
    threshold: f64,
) -> Result<()> {
    let db = Database::open(database)?;
    let coll = db.collection(collection_name)?;

    // Load baseline
    let baseline_content = std::fs::read_to_string(baseline_path)?;
    let baseline_data: serde_json::Value = serde_json::from_str(&baseline_content)?;

    let dimensions = baseline_data["dimensions"].as_u64().ok_or_else(|| {
        NeedleError::InvalidInput("Invalid baseline: missing dimensions".to_string())
    })? as usize;

    let baseline_centroid: Vec<f32> = baseline_data["centroid"]
        .as_array()
        .ok_or_else(|| {
            NeedleError::InvalidInput("Invalid baseline: missing centroid".to_string())
        })?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();

    let config = DriftConfig {
        centroid_threshold: threshold as f32,
        variance_threshold: threshold as f32,
        ..Default::default()
    };
    let mut detector = DriftDetector::new(dimensions, config);

    // Reconstruct baseline from saved stats
    // We create synthetic baseline vectors around the saved centroid
    let baseline_variance: Vec<f32> = baseline_data["variance"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect()
        })
        .unwrap_or_else(|| vec![0.1; dimensions]);

    // Create baseline vectors around centroid
    let mut baseline_vectors = Vec::new();
    for _ in 0..100 {
        let vec: Vec<f32> = baseline_centroid
            .iter()
            .zip(baseline_variance.iter())
            .map(|(&c, &v)| c + (rand::random::<f32>() - 0.5) * v.sqrt() * 2.0)
            .collect();
        baseline_vectors.push(vec);
    }
    detector.add_baseline(&baseline_vectors)?;

    // Get current vectors and check for drift
    let vectors = coll.export_all()?;
    let mut drift_detected = false;
    let mut total_drift_score = 0.0f32;
    let mut samples_checked = 0;

    for (_, vec, _) in vectors.iter().take(1000) {
        let report = detector.check(vec)?;
        if report.is_drifting {
            drift_detected = true;
        }
        total_drift_score += report.drift_score;
        samples_checked += 1;
    }

    let avg_drift_score = if samples_checked > 0 {
        total_drift_score / samples_checked as f32
    } else {
        0.0
    };

    println!("Drift Detection Results");
    println!("=======================");
    println!();
    println!("Threshold: {:.2}", threshold);
    println!(
        "Drift detected: {}",
        if drift_detected { "YES" } else { "NO" }
    );
    println!();
    println!("Metrics:");
    println!("  Samples checked: {}", samples_checked);
    println!("  Average drift score: {:.4}", avg_drift_score);

    if drift_detected {
        println!();
        println!("Warning: Significant drift detected!");
        println!("  Consider retraining models or investigating data quality.");
    }

    Ok(())
}

#[cfg(feature = "observability")]
fn drift_report(
    database: &str,
    collection_name: &str,
    baseline_path: &str,
    format: &str,
) -> Result<()> {
    let db = Database::open(database)?;
    let coll = db.collection(collection_name)?;

    // Load baseline
    let baseline_content = std::fs::read_to_string(baseline_path)?;
    let baseline_data: serde_json::Value = serde_json::from_str(&baseline_content)?;

    let dimensions = baseline_data["dimensions"].as_u64().ok_or_else(|| {
        NeedleError::InvalidInput("Invalid baseline: missing dimensions".to_string())
    })? as usize;

    let baseline_centroid: Vec<f32> = baseline_data["centroid"]
        .as_array()
        .ok_or_else(|| {
            NeedleError::InvalidInput("Invalid baseline: missing centroid".to_string())
        })?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();

    let baseline_variance: Vec<f32> = baseline_data["variance"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect()
        })
        .unwrap_or_else(|| vec![0.1; dimensions]);

    // Get current vectors
    let vectors = coll.export_all()?;
    let current_vecs: Vec<Vec<f32>> = vectors.iter().map(|(_, v, _)| v.clone()).collect();

    // Compute current statistics
    let current_centroid: Vec<f32> = (0..dimensions)
        .map(|d| current_vecs.iter().map(|v| v[d]).sum::<f32>() / current_vecs.len().max(1) as f32)
        .collect();

    let current_variance: Vec<f32> = (0..dimensions)
        .map(|d| {
            let mean = current_centroid[d];
            current_vecs
                .iter()
                .map(|v| (v[d] - mean).powi(2))
                .sum::<f32>()
                / current_vecs.len().max(1) as f32
        })
        .collect();

    // Compute drift metrics
    let centroid_shift: f32 = baseline_centroid
        .iter()
        .zip(current_centroid.iter())
        .map(|(b, c)| (b - c).powi(2))
        .sum::<f32>()
        .sqrt();

    let variance_change: f32 = baseline_variance
        .iter()
        .zip(current_variance.iter())
        .map(|(b, c)| ((c / b.max(0.0001)) - 1.0).abs())
        .sum::<f32>()
        / dimensions as f32;

    let drift_score = (centroid_shift * 0.6 + variance_change * 0.4).min(1.0);
    let has_drift = drift_score > 0.1;

    // Find top drifting dimensions
    let mut dimension_drifts: Vec<(usize, f32)> = (0..dimensions)
        .map(|d| {
            let shift = (baseline_centroid[d] - current_centroid[d]).abs();
            let var_change = ((current_variance[d] / baseline_variance[d].max(0.0001)) - 1.0).abs();
            (d, shift + var_change)
        })
        .collect();
    dimension_drifts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let recommendations: Vec<String> = if has_drift {
        vec![
            "Review recent data ingestion for quality issues".to_string(),
            "Consider retraining embedding models".to_string(),
            "Investigate top drifting dimensions".to_string(),
        ]
    } else {
        vec!["Data distribution is stable".to_string()]
    };

    if format == "json" {
        let json_report = json!({
            "collection": collection_name,
            "baseline_file": baseline_path,
            "current_count": current_vecs.len(),
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "drift_detected": has_drift,
            "drift_score": drift_score,
            "metrics": {
                "centroid_shift": centroid_shift,
                "variance_change": variance_change
            },
            "dimension_drifts": dimension_drifts.iter().take(10).collect::<Vec<_>>(),
            "recommendations": recommendations
        });
        println!("{}", serde_json::to_string_pretty(&json_report)?);
    } else {
        println!("Drift Analysis Report");
        println!("=====================");
        println!();
        println!("Collection: {}", collection_name);
        println!("Baseline: {}", baseline_path);
        println!("Current vectors: {}", current_vecs.len());
        println!(
            "Analysis time: {}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        );
        println!();
        println!("Overall Assessment:");
        println!("  Drift detected: {}", if has_drift { "YES" } else { "NO" });
        println!("  Drift score: {:.4}", drift_score);
        println!();
        println!("Detailed Metrics:");
        println!("  Centroid shift: {:.4}", centroid_shift);
        println!("  Variance change: {:.4}", variance_change);

        if !dimension_drifts.is_empty() {
            println!();
            println!("Top Drifting Dimensions:");
            for (i, (dim, drift)) in dimension_drifts.iter().take(5).enumerate() {
                println!("  {}. Dimension {}: {:.4}", i + 1, dim, drift);
            }
        }

        println!();
        println!("Recommendations:");
        for rec in &recommendations {
            println!("  - {}", rec);
        }
    }

    Ok(())
}

// ============================================================================
// Federated Search Commands
// ============================================================================

pub(crate) fn federate_command(cmd: FederateCommands) -> Result<()> {
    match cmd {
        FederateCommands::Search {
            query,
            collection,
            k,
            instances,
            routing,
            merge,
        } => federate_search(&query, &collection, k, &instances, &routing, &merge),
        FederateCommands::Health { instances } => federate_health(&instances),
        FederateCommands::Stats { instances } => federate_stats(&instances),
    }
}

fn federate_search(
    query_str: &str,
    collection: &str,
    k: usize,
    instances_str: &str,
    routing: &str,
    merge: &str,
) -> Result<()> {
    use needle::federated::{
        Federation, FederationConfig, InstanceConfig, MergeStrategy, RoutingStrategy,
    };

    // Parse query vector
    let query: Vec<f32> = query_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if query.is_empty() {
        eprintln!("Invalid query vector. Use comma-separated floats.");
        return Ok(());
    }

    // Parse instances
    let instance_urls: Vec<&str> = instances_str.split(',').map(|s| s.trim()).collect();

    if instance_urls.is_empty() {
        eprintln!("No instances specified.");
        return Ok(());
    }

    // Parse routing strategy
    let routing_strategy = match routing.to_lowercase().as_str() {
        "latency-aware" | "latency" => RoutingStrategy::LatencyAware,
        "round-robin" | "roundrobin" => RoutingStrategy::RoundRobin,
        "geographic" | "geo" => RoutingStrategy::GeographicProximity,
        _ => RoutingStrategy::Broadcast,
    };

    // Parse merge strategy
    let merge_strategy = match merge.to_lowercase().as_str() {
        "rrf" | "reciprocal" => MergeStrategy::ReciprocalRankFusion,
        "consensus" => MergeStrategy::Consensus,
        "first" => MergeStrategy::FirstResponse,
        _ => MergeStrategy::DistanceBased,
    };

    // Create federation
    let config = FederationConfig::default()
        .with_routing(routing_strategy)
        .with_merge(merge_strategy);

    let federation = Federation::new(config);

    // Register instances
    for (i, url) in instance_urls.iter().enumerate() {
        let instance_config = InstanceConfig::new(format!("instance-{}", i), *url);
        federation.register_instance(instance_config);
    }

    println!("Federated Search");
    println!("================");
    println!();
    println!("Query: {} dimensions", query.len());
    println!("Collection: {}", collection);
    println!("K: {}", k);
    println!("Instances: {}", instance_urls.len());
    println!("Routing: {:?}", routing_strategy);
    println!("Merge: {:?}", merge_strategy);
    println!();

    // Note: Actual federated search requires async runtime and HTTP client
    // This demonstrates the CLI interface
    println!("Note: Federated search requires the 'server' feature and running instances.");
    println!("      Use 'needle serve' to start instances, then use this command to query them.");
    println!();
    println!("Configured instances:");
    for url in &instance_urls {
        println!("  - {}", url);
    }

    Ok(())
}

fn federate_health(instances_str: &str) -> Result<()> {
    use needle::federated::{Federation, FederationConfig, InstanceConfig};

    let instance_urls: Vec<&str> = instances_str.split(',').map(|s| s.trim()).collect();

    let config = FederationConfig::default();
    let federation = Federation::new(config);

    for (i, url) in instance_urls.iter().enumerate() {
        let instance_config = InstanceConfig::new(format!("instance-{}", i), *url);
        federation.register_instance(instance_config);
    }

    let health = federation.health();

    println!("Federation Health Status");
    println!("========================");
    println!();
    println!("Overall: {:?}", health.status);
    println!(
        "Healthy instances: {}/{}",
        health.healthy_instances, health.total_instances
    );
    println!("Degraded instances: {}", health.degraded_instances);
    println!("Unhealthy instances: {}", health.unhealthy_instances);
    println!("Average latency: {:.2} ms", health.avg_latency_ms);

    Ok(())
}

fn federate_stats(instances_str: &str) -> Result<()> {
    use needle::federated::{Federation, FederationConfig, InstanceConfig};

    let instance_urls: Vec<&str> = instances_str.split(',').map(|s| s.trim()).collect();

    let config = FederationConfig::default();
    let federation = Federation::new(config);

    for (i, url) in instance_urls.iter().enumerate() {
        let instance_config = InstanceConfig::new(format!("instance-{}", i), *url);
        federation.register_instance(instance_config);
    }

    let stats = federation.stats();

    println!("Federation Statistics");
    println!("=====================");
    println!();
    println!("Total queries: {}", stats.total_queries);
    println!("Failed queries: {}", stats.failed_queries);
    println!("Partial results: {}", stats.partial_results);
    println!("Timeouts: {}", stats.timeouts);

    Ok(())
}

// ============================================================================
// Alias Commands
// ============================================================================

pub(crate) fn alias_command(cmd: AliasCommands) -> Result<()> {
    match cmd {
        AliasCommands::Create {
            database,
            alias,
            collection,
        } => alias_create(&database, &alias, &collection),
        AliasCommands::Delete { database, alias } => alias_delete(&database, &alias),
        AliasCommands::List { database } => alias_list(&database),
        AliasCommands::Resolve { database, alias } => alias_resolve(&database, &alias),
        AliasCommands::Update {
            database,
            alias,
            collection,
        } => alias_update(&database, &alias, &collection),
    }
}

fn alias_create(path: &str, alias: &str, collection: &str) -> Result<()> {
    let db = Database::open(path)?;
    db.create_alias(alias, collection)?;
    db.save()?;

    println!("Created alias '{}' -> '{}'", alias, collection);
    Ok(())
}

fn alias_delete(path: &str, alias: &str) -> Result<()> {
    let db = Database::open(path)?;
    let deleted = db.delete_alias(alias)?;
    db.save()?;

    if deleted {
        println!("Deleted alias '{}'", alias);
    } else {
        println!("Alias '{}' not found", alias);
    }
    Ok(())
}

fn alias_list(path: &str) -> Result<()> {
    let db = Database::open(path)?;
    let aliases = db.list_aliases();

    if aliases.is_empty() {
        println!("No aliases defined.");
    } else {
        println!("Aliases:");
        println!("{:-<50}", "");
        println!("{:<25} {:<25}", "Alias", "Collection");
        println!("{:-<50}", "");
        for (alias, collection) in aliases {
            println!("{:<25} {:<25}", alias, collection);
        }
    }

    Ok(())
}

fn alias_resolve(path: &str, alias: &str) -> Result<()> {
    let db = Database::open(path)?;

    match db.get_canonical_name(alias) {
        Some(collection) => {
            println!("{}", collection);
        }
        None => {
            println!("Alias '{}' not found", alias);
        }
    }

    Ok(())
}

fn alias_update(path: &str, alias: &str, collection: &str) -> Result<()> {
    let db = Database::open(path)?;
    db.update_alias(alias, collection)?;
    db.save()?;

    println!("Updated alias '{}' -> '{}'", alias, collection);
    Ok(())
}

// ============================================================================
// TTL Commands
// ============================================================================

pub(crate) fn ttl_command(cmd: TtlCommands) -> Result<()> {
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
    let db = Database::open(path)?;
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
