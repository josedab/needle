//! CLI Integration Tests
//!
//! Tests for the Needle CLI commands using subprocess execution.

use std::process::Command;
use tempfile::tempdir;

/// Get the path to the needle binary
fn needle_bin() -> String {
    // Build the binary first
    let status = Command::new("cargo")
        .args(["build", "--bin", "needle"])
        .status()
        .expect("Failed to build needle binary");
    assert!(status.success(), "Failed to build needle binary");

    // Return the path to the debug binary
    "target/debug/needle".to_string()
}

#[test]
fn test_cli_create_database() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    let output = Command::new(needle_bin())
        .args(["create", db_path.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "Create command failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(db_path.exists(), "Database file was not created");
}

#[test]
fn test_cli_info_command() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    // Create database first
    Command::new(needle_bin())
        .args(["create", db_path.to_str().unwrap()])
        .output()
        .expect("Failed to create database");

    // Test info command
    let output = Command::new(needle_bin())
        .args(["info", db_path.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "Info command failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Collections:"),
        "Info output should contain 'Collections:'"
    );
}

#[test]
fn test_cli_create_collection() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    // Create database first
    Command::new(needle_bin())
        .args(["create", db_path.to_str().unwrap()])
        .output()
        .expect("Failed to create database");

    // Create collection
    let output = Command::new(needle_bin())
        .args([
            "create-collection",
            db_path.to_str().unwrap(),
            "-n",
            "test_collection",
            "-d",
            "128",
            "--distance",
            "cosine",
        ])
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "Create collection failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify with collections command
    let output = Command::new(needle_bin())
        .args(["collections", db_path.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("test_collection"),
        "Collection should be listed"
    );
}

#[test]
fn test_cli_count_empty_collection() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    // Create database and collection
    Command::new(needle_bin())
        .args(["create", db_path.to_str().unwrap()])
        .output()
        .expect("Failed to create database");

    Command::new(needle_bin())
        .args([
            "create-collection",
            db_path.to_str().unwrap(),
            "-n",
            "test",
            "-d",
            "64",
        ])
        .output()
        .expect("Failed to create collection");

    // Test count
    let output = Command::new(needle_bin())
        .args(["count", db_path.to_str().unwrap(), "-c", "test"])
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "Count command failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("0"),
        "Empty collection should have 0 vectors"
    );
}

#[test]
fn test_cli_insert_and_search() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    // Create database and collection
    Command::new(needle_bin())
        .args(["create", db_path.to_str().unwrap()])
        .output()
        .expect("Failed to create database");

    Command::new(needle_bin())
        .args([
            "create-collection",
            db_path.to_str().unwrap(),
            "-n",
            "vectors",
            "-d",
            "4",
        ])
        .output()
        .expect("Failed to create collection");

    // Insert a vector via stdin
    let mut insert = Command::new(needle_bin())
        .args(["insert", db_path.to_str().unwrap(), "-c", "vectors"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to start insert command");

    use std::io::Write;
    {
        let stdin = insert.stdin.as_mut().expect("Failed to open stdin");
        writeln!(stdin, r#"{{"id": "vec1", "vector": [1.0, 0.0, 0.0, 0.0]}}"#).unwrap();
        writeln!(stdin, r#"{{"id": "vec2", "vector": [0.0, 1.0, 0.0, 0.0]}}"#).unwrap();
    } // stdin ref dropped here, but we also take ownership to close it
    drop(insert.stdin.take());

    let output = insert
        .wait_with_output()
        .expect("Failed to wait for insert");
    assert!(
        output.status.success(),
        "Insert failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Search
    let output = Command::new(needle_bin())
        .args([
            "search",
            db_path.to_str().unwrap(),
            "-c",
            "vectors",
            "-q",
            "1.0,0.0,0.0,0.0",
            "-k",
            "2",
        ])
        .output()
        .expect("Failed to execute search");

    assert!(
        output.status.success(),
        "Search failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("vec1"),
        "Search results should contain vec1"
    );
}

#[test]
fn test_cli_get_vector() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    // Setup database with a vector
    Command::new(needle_bin())
        .args(["create", db_path.to_str().unwrap()])
        .output()
        .unwrap();

    Command::new(needle_bin())
        .args([
            "create-collection",
            db_path.to_str().unwrap(),
            "-n",
            "test",
            "-d",
            "4",
        ])
        .output()
        .unwrap();

    let mut insert = Command::new(needle_bin())
        .args(["insert", db_path.to_str().unwrap(), "-c", "test"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    use std::io::Write;
    {
        let stdin = insert.stdin.as_mut().unwrap();
        writeln!(stdin, r#"{{"id": "test_vec", "vector": [1.0, 2.0, 3.0, 4.0], "metadata": {{"key": "value"}}}}"#).unwrap();
    }
    drop(insert.stdin.take());
    insert.wait().unwrap();

    // Get the vector
    let output = Command::new(needle_bin())
        .args([
            "get",
            db_path.to_str().unwrap(),
            "-c",
            "test",
            "-i",
            "test_vec",
        ])
        .output()
        .expect("Failed to execute get");

    assert!(
        output.status.success(),
        "Get failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("test_vec"),
        "Output should contain vector ID"
    );
}

#[test]
fn test_cli_delete_vector() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    // Setup
    Command::new(needle_bin())
        .args(["create", db_path.to_str().unwrap()])
        .output()
        .unwrap();

    Command::new(needle_bin())
        .args([
            "create-collection",
            db_path.to_str().unwrap(),
            "-n",
            "test",
            "-d",
            "4",
        ])
        .output()
        .unwrap();

    let mut insert = Command::new(needle_bin())
        .args(["insert", db_path.to_str().unwrap(), "-c", "test"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    use std::io::Write;
    {
        let stdin = insert.stdin.as_mut().unwrap();
        writeln!(
            stdin,
            r#"{{"id": "to_delete", "vector": [1.0, 2.0, 3.0, 4.0]}}"#
        )
        .unwrap();
    }
    drop(insert.stdin.take());
    insert.wait().unwrap();

    // Verify count is 1
    let output = Command::new(needle_bin())
        .args(["count", db_path.to_str().unwrap(), "-c", "test"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("1"), "Should have 1 vector");

    // Delete the vector
    let output = Command::new(needle_bin())
        .args([
            "delete",
            db_path.to_str().unwrap(),
            "-c",
            "test",
            "-i",
            "to_delete",
        ])
        .output()
        .expect("Failed to execute delete");

    assert!(
        output.status.success(),
        "Delete failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify count is 0
    let output = Command::new(needle_bin())
        .args(["count", db_path.to_str().unwrap(), "-c", "test"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("0"), "Should have 0 vectors after delete");
}

#[test]
fn test_cli_stats_command() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    // Setup
    Command::new(needle_bin())
        .args(["create", db_path.to_str().unwrap()])
        .output()
        .unwrap();

    Command::new(needle_bin())
        .args([
            "create-collection",
            db_path.to_str().unwrap(),
            "-n",
            "stats_test",
            "-d",
            "128",
        ])
        .output()
        .unwrap();

    // Get stats
    let output = Command::new(needle_bin())
        .args(["stats", db_path.to_str().unwrap(), "-c", "stats_test"])
        .output()
        .expect("Failed to execute stats");

    assert!(
        output.status.success(),
        "Stats failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("128") || stdout.contains("Dimensions"),
        "Stats should show dimensions"
    );
}

#[test]
fn test_cli_tune_command() {
    let output = Command::new(needle_bin())
        .args(["tune", "-v", "100000", "-d", "384", "-p", "balanced"])
        .output()
        .expect("Failed to execute tune");

    assert!(
        output.status.success(),
        "Tune failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("M:") || stdout.contains("ef_construction"),
        "Tune output should contain HNSW parameters"
    );
}

#[test]
fn test_cli_export_import() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");
    let export_file = dir.path().join("export.json");

    // Setup with some vectors
    Command::new(needle_bin())
        .args(["create", db_path.to_str().unwrap()])
        .output()
        .unwrap();

    Command::new(needle_bin())
        .args([
            "create-collection",
            db_path.to_str().unwrap(),
            "-n",
            "export_test",
            "-d",
            "4",
        ])
        .output()
        .unwrap();

    let mut insert = Command::new(needle_bin())
        .args(["insert", db_path.to_str().unwrap(), "-c", "export_test"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    use std::io::Write;
    {
        let stdin = insert.stdin.as_mut().unwrap();
        writeln!(stdin, r#"{{"id": "exp1", "vector": [1.0, 0.0, 0.0, 0.0]}}"#).unwrap();
        writeln!(stdin, r#"{{"id": "exp2", "vector": [0.0, 1.0, 0.0, 0.0]}}"#).unwrap();
    }
    drop(insert.stdin.take());
    insert.wait().unwrap();

    // Export
    let output = Command::new(needle_bin())
        .args(["export", db_path.to_str().unwrap(), "-c", "export_test"])
        .output()
        .expect("Failed to execute export");

    assert!(
        output.status.success(),
        "Export failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Save export to file
    std::fs::write(&export_file, &output.stdout).unwrap();

    // Create new database and import
    let db_path2 = dir.path().join("test2.needle");
    Command::new(needle_bin())
        .args(["create", db_path2.to_str().unwrap()])
        .output()
        .unwrap();

    Command::new(needle_bin())
        .args([
            "create-collection",
            db_path2.to_str().unwrap(),
            "-n",
            "export_test",
            "-d",
            "4",
        ])
        .output()
        .unwrap();

    // Import
    let output = Command::new(needle_bin())
        .args([
            "import",
            db_path2.to_str().unwrap(),
            "-c",
            "export_test",
            "-f",
            export_file.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute import");

    assert!(
        output.status.success(),
        "Import failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify count
    let output = Command::new(needle_bin())
        .args(["count", db_path2.to_str().unwrap(), "-c", "export_test"])
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("2"),
        "Imported collection should have 2 vectors"
    );
}

#[test]
fn test_cli_compact_command() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    // Setup
    Command::new(needle_bin())
        .args(["create", db_path.to_str().unwrap()])
        .output()
        .unwrap();

    // Compact
    let output = Command::new(needle_bin())
        .args(["compact", db_path.to_str().unwrap()])
        .output()
        .expect("Failed to execute compact");

    assert!(
        output.status.success(),
        "Compact failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_cli_nonexistent_database_error() {
    let output = Command::new(needle_bin())
        .args(["info", "/nonexistent/path/to/database.needle"])
        .output()
        .expect("Failed to execute command");

    // Should fail gracefully
    assert!(
        !output.status.success() || !String::from_utf8_lossy(&output.stderr).is_empty(),
        "Should error on nonexistent database"
    );
}

#[test]
fn test_cli_nonexistent_collection_error() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    Command::new(needle_bin())
        .args(["create", db_path.to_str().unwrap()])
        .output()
        .unwrap();

    let output = Command::new(needle_bin())
        .args(["count", db_path.to_str().unwrap(), "-c", "nonexistent"])
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "Should error on nonexistent collection"
    );
}

#[test]
fn test_cli_distance_functions() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    Command::new(needle_bin())
        .args(["create", db_path.to_str().unwrap()])
        .output()
        .unwrap();

    // Test all distance functions
    for distance in &["cosine", "euclidean", "dot", "manhattan"] {
        let output = Command::new(needle_bin())
            .args([
                "create-collection",
                db_path.to_str().unwrap(),
                "-n",
                &format!("col_{}", distance),
                "-d",
                "64",
                "--distance",
                distance,
            ])
            .output()
            .expect("Failed to create collection");

        assert!(
            output.status.success(),
            "Failed to create collection with {} distance: {:?}",
            distance,
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
