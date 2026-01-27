//! Server Example — HTTP REST API
//!
//! Demonstrates starting the Needle HTTP server with custom configuration.
//!
//! Run with: cargo run --example server_usage --features server

#[cfg(feature = "server")]
use needle::server::ServerConfig;

#[cfg(feature = "server")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Needle HTTP Server Example ===\n");

    // Create a server config with default settings (in-memory database)
    let config = ServerConfig {
        db_path: None, // in-memory; use Some("data.needle".into()) for persistence
        ..ServerConfig::default()
    };

    println!("Starting server on http://{}", config.addr);
    println!("Endpoints:");
    println!("  GET  /health                        - Health check");
    println!("  GET  /collections                   - List collections");
    println!("  POST /collections                   - Create collection");
    println!("  POST /collections/:name/vectors     - Insert vectors");
    println!("  POST /collections/:name/search      - Search vectors");
    println!();

    // Start the server (blocks until shutdown)
    needle::server::serve(config).await?;

    Ok(())
}

#[cfg(not(feature = "server"))]
fn main() {
    eprintln!("This example requires the 'server' feature.");
    eprintln!("Run with: cargo run --example server_usage --features server");
}
