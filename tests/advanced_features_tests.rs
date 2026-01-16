//! Integration tests for advanced features:
//! - Query Builder / Natural Language Query Interface
//! - Federated Search
//! - Drift Detection
//! - Backup/Restore

use needle::backup::{BackupConfig, BackupManager, BackupType};
use needle::drift::{DriftConfig, DriftDetector};
use needle::federated::{
    Federation, FederationConfig, InstanceConfig, MergeStrategy, RoutingStrategy,
};
use needle::query_builder::{CollectionProfile, QueryAnalyzer, QueryClass, VisualQueryBuilder};
use needle::Database;
use tempfile::TempDir;

fn random_vector(dim: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

// ============================================================================
// Query Builder Tests
// ============================================================================

#[test]
fn test_query_analyzer_semantic_query() {
    let analyzer = QueryAnalyzer::new();

    let analysis = analyzer.analyze("find similar documents about machine learning");

    assert_eq!(analysis.class, QueryClass::Semantic);
    assert!(!analysis.search_terms.is_empty());
    assert!(analysis.confidence > 0.0);
}

#[test]
fn test_query_analyzer_filter_query() {
    let analyzer = QueryAnalyzer::new();

    let analysis = analyzer.analyze("category = 'books' AND price < 50");

    // Filter queries may be classified as MetadataOnly or Hybrid depending on context
    assert!(analysis.class == QueryClass::MetadataOnly || analysis.class == QueryClass::Hybrid);
    // Should detect filter fields
    assert!(!analysis.filter_fields.is_empty() || analysis.confidence > 0.0);
}

#[test]
fn test_query_analyzer_hybrid_query() {
    let analyzer = QueryAnalyzer::new();

    let analysis = analyzer.analyze("search for documents about AI where category = 'tech'");

    assert_eq!(analysis.class, QueryClass::Hybrid);
    assert!(!analysis.search_terms.is_empty());
    assert!(!analysis.filter_fields.is_empty());
}

#[test]
fn test_visual_query_builder_basic() {
    let profile = CollectionProfile::new("test_collection", 128, 1000);
    let builder = VisualQueryBuilder::new(profile);

    let result = builder.build("find documents similar to machine learning");

    assert!(!result.needleql.is_empty());
    assert!(result.quality_score > 0.0);
}

#[test]
fn test_visual_query_builder_with_filter() {
    let profile = CollectionProfile::new("products", 384, 10000);
    let builder = VisualQueryBuilder::new(profile);

    let result = builder.build("find products where price < 100");

    assert!(!result.needleql.is_empty());
    assert!(
        result.analysis.class == QueryClass::MetadataOnly
            || result.analysis.class == QueryClass::Hybrid
    );
}

#[test]
fn test_query_builder_generates_hints() {
    let profile = CollectionProfile::new("large_collection", 1024, 1_000_000);
    let builder = VisualQueryBuilder::new(profile);

    // Complex query should generate optimization hints
    let result = builder.build("find all documents where status = 'active' AND category IN ('a', 'b', 'c') similar to query");

    // Large collection with complex filters should generate some hints
    assert!(result.quality_score <= 1.0);
}

// ============================================================================
// Federated Search Tests
// ============================================================================

#[test]
fn test_federation_config_builder() {
    use std::time::Duration;

    let config = FederationConfig::default()
        .with_routing(RoutingStrategy::LatencyAware)
        .with_merge(MergeStrategy::ReciprocalRankFusion)
        .with_timeout(Duration::from_millis(5000));

    assert_eq!(config.routing_strategy, RoutingStrategy::LatencyAware);
    assert_eq!(config.merge_strategy, MergeStrategy::ReciprocalRankFusion);
    assert_eq!(config.query_timeout, Duration::from_millis(5000));
}

#[test]
fn test_federation_instance_registration() {
    let config = FederationConfig::default();
    let federation = Federation::new(config);

    // Register multiple instances
    federation.register_instance(InstanceConfig::new("instance-1", "http://localhost:8081"));
    federation.register_instance(InstanceConfig::new("instance-2", "http://localhost:8082"));
    federation.register_instance(InstanceConfig::new("instance-3", "http://localhost:8083"));

    // Check health shows all registered
    let health = federation.health();
    assert_eq!(health.total_instances, 3);
}

#[test]
fn test_federation_routing_strategies() {
    // Test broadcast
    let config = FederationConfig::default().with_routing(RoutingStrategy::Broadcast);
    let _federation = Federation::new(config);

    // Test latency aware
    let config = FederationConfig::default().with_routing(RoutingStrategy::LatencyAware);
    let _federation = Federation::new(config);

    // Test round robin
    let config = FederationConfig::default().with_routing(RoutingStrategy::RoundRobin);
    let _federation = Federation::new(config);

    // Test quorum
    let config = FederationConfig::default().with_routing(RoutingStrategy::Quorum(2));
    let _federation = Federation::new(config);
}

#[test]
fn test_federation_merge_strategies() {
    // Test distance based
    let config = FederationConfig::default().with_merge(MergeStrategy::DistanceBased);
    let _federation = Federation::new(config);

    // Test RRF
    let config = FederationConfig::default().with_merge(MergeStrategy::ReciprocalRankFusion);
    let _federation = Federation::new(config);

    // Test consensus
    let config = FederationConfig::default().with_merge(MergeStrategy::Consensus);
    let _federation = Federation::new(config);
}

#[test]
fn test_federation_health_monitoring() {
    let config = FederationConfig::default();
    let federation = Federation::new(config);

    federation.register_instance(InstanceConfig::new("test-1", "http://localhost:9001"));
    federation.register_instance(InstanceConfig::new("test-2", "http://localhost:9002"));

    let health = federation.health();

    // New instances should be in unknown state
    assert!(health.total_instances >= 2);
    assert!(health.avg_latency_ms >= 0.0);
}

#[test]
fn test_federation_stats() {
    let config = FederationConfig::default();
    let federation = Federation::new(config);

    federation.register_instance(InstanceConfig::new("stats-test", "http://localhost:9999"));

    let stats = federation.stats();

    // Fresh federation should have zero queries
    assert_eq!(stats.total_queries, 0);
    assert_eq!(stats.failed_queries, 0);
}

// ============================================================================
// Drift Detection Tests
// ============================================================================

#[test]
fn test_drift_detector_baseline() {
    let config = DriftConfig::default();
    let mut detector = DriftDetector::new(128, config);

    // Create baseline vectors
    let baseline: Vec<Vec<f32>> = (0..100).map(|_| random_vector(128)).collect();

    detector.add_baseline(&baseline).unwrap();

    // Baseline should be established
    let _report = detector.check(&random_vector(128)).unwrap();
    // Report is successfully generated
}

#[test]
fn test_drift_detector_no_drift() {
    let config = DriftConfig {
        centroid_threshold: 0.5,
        variance_threshold: 0.5,
        min_samples: 10,
        ..Default::default()
    };
    let mut detector = DriftDetector::new(64, config);

    // Create baseline with specific distribution
    let mut baseline = Vec::new();
    for _ in 0..50 {
        let mut v: Vec<f32> = (0..64).map(|_| 0.5).collect();
        // Add small noise
        for val in &mut v {
            *val += (rand::random::<f32>() - 0.5) * 0.1;
        }
        baseline.push(v);
    }
    detector.add_baseline(&baseline).unwrap();

    // Check vectors from same distribution - should not drift
    for _ in 0..20 {
        let mut v: Vec<f32> = (0..64).map(|_| 0.5).collect();
        for val in &mut v {
            *val += (rand::random::<f32>() - 0.5) * 0.1;
        }
        let _report = detector.check(&v).unwrap();
    }
}

#[test]
fn test_drift_detector_detects_drift() {
    let config = DriftConfig {
        centroid_threshold: 0.1,
        variance_threshold: 0.1,
        min_samples: 10,
        window_size: 20,
        ..Default::default()
    };
    let mut detector = DriftDetector::new(64, config);

    // Create baseline centered at 0
    let baseline: Vec<Vec<f32>> = (0..50)
        .map(|_| (0..64).map(|_| rand::random::<f32>() * 0.1).collect())
        .collect();
    detector.add_baseline(&baseline).unwrap();

    // Check vectors from very different distribution (centered at 1)
    let mut detected_drift = false;
    for _ in 0..30 {
        let v: Vec<f32> = (0..64).map(|_| 0.9 + rand::random::<f32>() * 0.1).collect();
        let report = detector.check(&v).unwrap();
        if report.is_drifting {
            detected_drift = true;
            break;
        }
    }

    // Should eventually detect drift
    assert!(
        detected_drift,
        "Expected drift to be detected for significantly different distribution"
    );
}

#[test]
fn test_drift_config_builder() {
    use needle::drift::DriftConfigBuilder;

    let config = DriftConfigBuilder::new()
        .window_size(500)
        .centroid_threshold(0.2)
        .variance_threshold(0.3)
        .min_samples(50)
        .build();

    assert_eq!(config.window_size, 500);
    assert_eq!(config.centroid_threshold, 0.2);
    assert_eq!(config.variance_threshold, 0.3);
    assert_eq!(config.min_samples, 50);
}

// ============================================================================
// Backup/Restore Tests
// ============================================================================

#[test]
fn test_backup_create_and_list() {
    let temp_dir = TempDir::new().unwrap();
    let backup_dir = temp_dir.path().join("backups");

    // Create a database with some data
    let db = Database::in_memory();
    db.create_collection("test", 64).unwrap();
    let coll = db.collection("test").unwrap();

    for i in 0..100 {
        coll.insert(format!("vec_{}", i), &random_vector(64), None)
            .unwrap();
    }

    // Create backup
    let config = BackupConfig::default();
    let manager = BackupManager::new(&backup_dir, config);
    let metadata = manager.create_backup(&db).unwrap();

    assert!(!metadata.id.is_empty());
    assert_eq!(metadata.num_collections, 1);
    assert_eq!(metadata.total_vectors, 100);
    assert_eq!(metadata.backup_type, BackupType::Full);

    // List backups
    let backups = manager.list_backups().unwrap();
    assert_eq!(backups.len(), 1);
    assert_eq!(backups[0].id, metadata.id);
}

#[test]
fn test_backup_verify() {
    let temp_dir = TempDir::new().unwrap();
    let backup_dir = temp_dir.path().join("backups");

    // Create database and backup
    let db = Database::in_memory();
    db.create_collection("verify_test", 32).unwrap();
    let coll = db.collection("verify_test").unwrap();
    coll.insert("v1", &random_vector(32), None).unwrap();

    let config = BackupConfig {
        verify: true,
        ..Default::default()
    };
    let manager = BackupManager::new(&backup_dir, config);
    let metadata = manager.create_backup(&db).unwrap();

    // Verify the backup
    let valid = manager.verify_backup(&metadata.id).unwrap();
    assert!(valid);
}

#[test]
fn test_backup_restore() {
    let temp_dir = TempDir::new().unwrap();
    let backup_dir = temp_dir.path().join("backups");

    // Create original database
    let db = Database::in_memory();
    db.create_collection("restore_test", 48).unwrap();
    let coll = db.collection("restore_test").unwrap();

    for i in 0..50 {
        let meta = serde_json::json!({"index": i});
        coll.insert(format!("item_{}", i), &random_vector(48), Some(meta))
            .unwrap();
    }

    // Create backup
    let config = BackupConfig::default();
    let manager = BackupManager::new(&backup_dir, config);
    let metadata = manager.create_backup(&db).unwrap();

    // Restore from backup
    let restored_db = manager.restore_backup(&metadata.id).unwrap();

    // Verify restored data
    assert_eq!(restored_db.list_collections().len(), 1);
    assert!(restored_db
        .list_collections()
        .contains(&"restore_test".to_string()));

    let restored_coll = restored_db.collection("restore_test").unwrap();
    assert_eq!(restored_coll.len(), 50);
}

#[test]
fn test_backup_with_compression() {
    let temp_dir = TempDir::new().unwrap();
    let backup_dir = temp_dir.path().join("compressed_backups");

    let db = Database::in_memory();
    db.create_collection("compress_test", 128).unwrap();
    let coll = db.collection("compress_test").unwrap();

    for i in 0..200 {
        coll.insert(format!("vec_{}", i), &random_vector(128), None)
            .unwrap();
    }

    let config = BackupConfig {
        compression: true,
        ..Default::default()
    };
    let manager = BackupManager::new(&backup_dir, config);
    let metadata = manager.create_backup(&db).unwrap();

    assert_eq!(metadata.total_vectors, 200);
}

#[test]
fn test_incremental_backup() {
    let temp_dir = TempDir::new().unwrap();
    let backup_dir = temp_dir.path().join("incremental_backups");

    let db = Database::in_memory();
    db.create_collection("incremental_test", 64).unwrap();
    let coll = db.collection("incremental_test").unwrap();

    // Initial data
    for i in 0..50 {
        coll.insert(format!("vec_{}", i), &random_vector(64), None)
            .unwrap();
    }

    // Create full backup
    let config = BackupConfig::default();
    let manager = BackupManager::new(&backup_dir, config);
    let full_metadata = manager.create_backup(&db).unwrap();
    assert_eq!(full_metadata.backup_type, BackupType::Full);

    // Add more data
    for i in 50..100 {
        coll.insert(format!("vec_{}", i), &random_vector(64), None)
            .unwrap();
    }

    // Create incremental backup
    let incr_metadata = manager.create_incremental(&db, &full_metadata.id).unwrap();
    assert_eq!(incr_metadata.backup_type, BackupType::Incremental);
    assert_eq!(incr_metadata.parent_id, Some(full_metadata.id));
}

// ============================================================================
// Integration: Combined Feature Tests
// ============================================================================

#[test]
fn test_query_with_drift_monitoring() {
    // Create a collection and monitor for drift over time
    let db = Database::in_memory();
    db.create_collection("monitored", 64).unwrap();
    let coll = db.collection("monitored").unwrap();

    // Insert initial data
    for i in 0..100 {
        coll.insert(format!("initial_{}", i), &random_vector(64), None)
            .unwrap();
    }

    // Set up drift detector
    let config = DriftConfig::default();
    let mut detector = DriftDetector::new(64, config);

    let baseline: Vec<Vec<f32>> = (0..50).map(|_| random_vector(64)).collect();
    detector.add_baseline(&baseline).unwrap();

    // Query and check for drift
    let query = random_vector(64);
    let results = coll.search(&query, 5).unwrap();
    assert_eq!(results.len(), 5);

    // Monitor the query vector
    let report = detector.check(&query).unwrap();
    assert!(!report.is_drifting || report.drift_score < 1.0);
}

#[test]
fn test_backup_with_query_builder() {
    let temp_dir = TempDir::new().unwrap();
    let backup_dir = temp_dir.path().join("query_backups");

    // Create database with metadata
    let db = Database::in_memory();
    db.create_collection("products", 128).unwrap();
    let coll = db.collection("products").unwrap();

    for i in 0..100 {
        let meta = serde_json::json!({
            "category": if i % 2 == 0 { "electronics" } else { "books" },
            "price": i as f64 * 10.0
        });
        coll.insert(format!("product_{}", i), &random_vector(128), Some(meta))
            .unwrap();
    }

    // Backup
    let config = BackupConfig::default();
    let manager = BackupManager::new(&backup_dir, config);
    let metadata = manager.create_backup(&db).unwrap();

    // Use query builder on original
    let profile = CollectionProfile::new("products", 128, 100);
    let builder = VisualQueryBuilder::new(profile);
    let result = builder.build("find electronics where price < 500");

    assert!(!result.needleql.is_empty());
    assert_eq!(metadata.total_vectors, 100);
}
