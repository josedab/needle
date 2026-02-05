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
        std::env::set_var("RUST_LOG", "debug");
    }

    if let Err(err) = cli::run(parsed) {
        cli::print_error(&err);
        std::process::exit(1);
    }
}
