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
}

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
        } => search_command(
            &database,
            &collection,
            &query,
            k,
            explain,
            distance.as_deref(),
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
        } => import_command(&database, &collection, &file),
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
        Commands::Demo { count, dimensions } => demo_command(count, dimensions),
        Commands::Query {
            database,
            collection,
            query,
            k,
            analyze,
        } => query_command(&database, &collection, &query, k, analyze),
        Commands::Backup(cmd) => backup_command(cmd),
        Commands::Drift(cmd) => drift_command(cmd),
        Commands::Federate(cmd) => federate_command(cmd),
        Commands::Alias(cmd) => alias_command(cmd),
        Commands::Ttl(cmd) => ttl_command(cmd),
        Commands::Dev(cmd) => dev_command(cmd),
        Commands::Mcp { database, read_only } => mcp_command(&database, read_only),
        Commands::Init { directory, database, dimensions } => init_command(&directory, &database, dimensions),
        Commands::Doctor => doctor_command(),
        Commands::Snapshot(cmd) => snapshot_command(cmd),
        Commands::Memory(cmd) => memory_command(cmd),
        Commands::Diff { database, source, target, limit } =>
            diff_command(&database, &source, &target, limit),
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
    }
}
