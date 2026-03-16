//! CLI module — Command definitions, dispatch, and handler implementations.

pub mod commands;
pub mod handlers;

use clap::Parser;
use needle::{NeedleError, Result};

use commands::Commands;
use handlers::*;

#[derive(Parser)]
#[command(name = "needle")]
#[command(author, version, about = "Embedded Vector Database - SQLite for Vectors", long_about = None)]
pub struct Cli {
    /// Enable debug logging
    #[arg(short, long, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

pub fn print_error(err: &NeedleError) {
    use needle::error::Recoverable;

    eprintln!("Error: {}", err);

    let code = err.error_code();
    eprintln!("  Code: {} ({})", code.code(), code.category());

    let hints = err.recovery_hints();
    if !hints.is_empty() {
        eprintln!();
        for hint in &hints {
            eprintln!("  Hint: {}", hint);
        }
    }

    if std::env::var("RUST_BACKTRACE").is_err() {
        eprintln!();
        eprintln!("  Set RUST_BACKTRACE=1 for a detailed backtrace.");
    }
}

#[allow(clippy::too_many_lines)]
pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Info { database } => info_command(&database),
        Commands::Create { database } => create_command(&database),
        Commands::Collections { database } => collections_command(&database),
        Commands::CreateCollection {
            database,
            name,
            dimensions,
            distance,
            encrypted,
        } => create_collection_command(&database, &name, dimensions, &distance, encrypted),
        Commands::RenameCollection {
            database,
            old_name,
            new_name,
        } => rename_collection_command(&database, &old_name, &new_name),
        Commands::Stats {
            database,
            collection,
        } => stats_command(&database, &collection),
        Commands::Insert {
            database,
            collection,
        } => insert_command(&database, &collection),
        Commands::Search {
            database,
            collection,
            query,
            k,
            explain,
            distance,
            filter,
            max_age,
            truncate_dims,
        } => search_command(
            &database,
            &collection,
            &query,
            k,
            explain,
            distance.as_deref(),
            filter.as_deref(),
            max_age,
            truncate_dims,
        ),
        Commands::Delete {
            database,
            collection,
            id,
        } => delete_command(&database, &collection, &id),
        Commands::Get {
            database,
            collection,
            id,
        } => get_command(&database, &collection, &id),
        Commands::Compact { database } => compact_command(&database),
        Commands::Export {
            database,
            collection,
        } => export_command(&database, &collection),
        Commands::Import {
            database,
            collection,
            file,
            format,
        } => import_command(&database, &collection, &file, &format),
        Commands::Count {
            database,
            collection,
        } => count_command(&database, &collection),
        Commands::Clear {
            database,
            collection,
            force,
        } => clear_command(&database, &collection, force),
        #[cfg(feature = "server")]
        Commands::Serve { address, database } => serve_command(&address, database),
        Commands::Tune {
            vectors,
            dimensions,
            profile,
            memory_mb,
        } => tune_command(vectors, dimensions, &profile, memory_mb),
        Commands::OptimizeDimensions {
            database,
            collection,
            targets,
            sample_size,
        } => optimize_dimensions_command(&database, &collection, targets.as_deref(), sample_size),
        Commands::Demo { count, dimensions } => demo_command(count, dimensions),
        Commands::Query {
            database,
            collection,
            query,
            k,
            analyze,
        } => query_command(&database, &collection, &query, k, analyze),
        Commands::Backup(cmd) => backup_command(cmd),
        #[cfg(feature = "observability")]
        Commands::Drift(cmd) => drift_command(cmd),
        #[cfg(not(feature = "observability"))]
        Commands::Drift(_) => {
            eprintln!("Drift detection requires the 'observability' feature. Rebuild with --features observability");
            std::process::exit(1);
        }
        Commands::Federate(cmd) => federate_command(cmd),
        Commands::Alias(cmd) => alias_command(cmd),
        Commands::Ttl(cmd) => ttl_command(cmd),
        Commands::Dev(cmd) => dev_command(cmd),
        Commands::Mcp { database, read_only } => mcp_command(&database, read_only),
        Commands::Init { directory, database, dimensions } => init_command(&directory, &database, dimensions),
        Commands::Doctor { database } => doctor_command(&database),
        Commands::Snapshot(cmd) => snapshot_command(cmd),
        Commands::Branch(cmd) => branch_command(cmd),
        Commands::Memory(cmd) => memory_command(cmd),
        Commands::Diff { database, source, target, limit, threshold } =>
            diff_command(&database, &source, &target, limit, threshold),
        Commands::Merge { database, source, target, base, strategy, dry_run } =>
            merge_command(&database, &source, &target, base.as_deref(), &strategy, dry_run),
        Commands::Estimate { database, collection, k, with_filter } =>
            estimate_command(&database, &collection, k, with_filter),
        Commands::RecommendIndex { vectors, dimensions, memory_mb, profile } =>
            recommend_index_command(vectors, dimensions, memory_mb, &profile),
        Commands::Sql { database, query, format, vector } =>
            sql_command(&database, &query, &format, vector.as_deref()),
        Commands::Provenance { database, collection, id } =>
            provenance_command(&database, &collection, &id),
        Commands::Evaluate { database, collection, ground_truth, k } =>
            evaluate_command(&database, &collection, &ground_truth, k),
        Commands::ExportBundle { database, collection, output } =>
            export_bundle_command(&database, &collection, &output),
        Commands::ImportBundle { database, bundle, name } =>
            import_bundle_command(&database, &bundle, name.as_deref()),
        Commands::AdviseCompression { database, collection, test_queries, k, targets, apply } =>
            advise_compression_command(&database, &collection, test_queries, k, &targets, apply),
        Commands::Migrate { database, source, url, collection, dry_run, batch_size, resume, rollback } =>
            migrate_command(&database, &source, &url, &collection, dry_run, batch_size, resume.as_deref(), rollback),
        Commands::Function(cmd) => function_command(cmd),
        Commands::Views(cmd) => views_command(cmd),
        Commands::ExplainSearch { database, collection, query, k, format } =>
            explain_search_command(&database, &collection, &query, k, &format),
        Commands::Advise { database, collection, sample_queries } =>
            advise_command(&database, &collection, sample_queries),
        Commands::Watch { database, interval } =>
            watch_file_command(&database, interval),
        Commands::WatchEvents { database, collection, from_sequence, batch_size, consumer_id } =>
            watch_events_command(&database, &collection, from_sequence, batch_size, &consumer_id),
        Commands::Sync { database, replica_id, status } =>
            sync_command(&database, &replica_id, status),
        Commands::Dedup { database, collection, threshold, strategy, dry_run } =>
            dedup_command(&database, &collection, threshold, &strategy, dry_run),
        Commands::Health { database, collection, format } =>
            health_command(&database, &collection, &format),
        Commands::Playground {
            database,
            #[cfg(feature = "experimental")]
            tutorial,
            #[cfg(feature = "experimental")]
            execute,
        } => {
            #[cfg(feature = "experimental")]
            {
                playground_command(database.as_deref(), tutorial.as_deref(), execute.as_deref())
            }
            #[cfg(not(feature = "experimental"))]
            {
                playground_command(database.as_deref())
            }
        }
        Commands::Bench { vectors, dimensions, queries, k_values, format, output, compare, ann_dataset } => {
            if let Some(dataset_name) = ann_dataset {
                ann_bench_command(&dataset_name, &format, output.as_deref())
            } else {
                bench_command(vectors, dimensions, queries, &k_values, &format, output.as_deref(), compare.as_deref())
            }
        }
        Commands::Ingestion(cmd) => ingestion_command(cmd),
        Commands::Cache(cmd) => cache_command(cmd),
        Commands::Models(cmd) => models_command(cmd),
        Commands::Plugin(cmd) => plugin_command(cmd),
        Commands::Partition { database, collection, analyze, target_size } =>
            partition_command(&database, &collection, analyze, target_size),
        Commands::Status { database } => status_command(&database),
    }
}
