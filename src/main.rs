//! Needle CLI — Command line interface for the Needle vector database.
//!
//! Provides commands for database management, vector operations, search,
//! server mode, backup/restore, and administrative tools.
//!
//! # Usage
//!
//! ```bash
//! needle create mydb.needle
//! needle create-collection mydb.needle -n docs -d 384
//! needle info mydb.needle
//! needle serve -a 127.0.0.1:8080  # requires --features server
//! ```

mod cli;

use clap::Parser;

fn main() {
    let parsed = cli::Cli::parse();

    if parsed.verbose {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::builder()
                    .with_default_directive(tracing::level_filters::LevelFilter::DEBUG.into())
                    .from_env_lossy(),
            )
            .init();
    }

    if let Err(err) = cli::run(parsed) {
        cli::print_error(&err);
        std::process::exit(1);
    }
}
