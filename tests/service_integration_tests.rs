//! Integration tests for high-level service modules through the Database API.
//!
//! Run with: cargo test --test service_integration_tests --features full

#![cfg(feature = "experimental")]

use rand::Rng;
use serde_json::json;

use needle::{
    AdaptiveIndexService, AdaptiveServiceConfig, BackpressureLevel, Database, IngestionService,
    IngestionServiceConfig, ModalInput, MultiModalService, MultiModalServiceConfig, PitrService,
    PitrServiceConfig, RecoveryTarget, TieredService, TieredServiceConfig,
};

fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

// ============================================================================
// IngestionService tests
// ============================================================================

#[test]
fn test_ingestion_service_ingest_and_flush() {
    let db = Database::in_memory();
    db.create_collection("docs", 128).unwrap();

    let config = IngestionServiceConfig::builder()
        .collection("docs")
        .batch_size(100)
        .enable_dedup(true)
        .build();

    let mut service = IngestionService::new(&db, config).unwrap();

    for i in 0..10 {
        let vec = random_vector(128);
        let level = service
            .ingest(format!("doc{i}"), &vec, Some(json!({"idx": i})))
            .unwrap();
        assert_eq!(level, BackpressureLevel::None);
    }

    assert_eq!(service.pending_count(), 10);

    let stats = service.flush().unwrap();
    assert_eq!(stats.total_ingested, 10);
    assert_eq!(service.pending_count(), 0);
    assert!(service.dead_letters().is_empty());
}

#[test]
fn test_ingestion_service_dedup_skips_duplicates() {
    let db = Database::in_memory();
    db.create_collection("dedup", 64).unwrap();

    let config = IngestionServiceConfig::builder()
        .collection("dedup")
        .batch_size(100)
        .enable_dedup(true)
        .build();

    let mut service = IngestionService::new(&db, config).unwrap();

    let vec = random_vector(64);
    service.ingest("same_id", &vec, None).unwrap();
    service.ingest("same_id", &vec, None).unwrap();

    let stats = service.flush().unwrap();
    assert_eq!(stats.duplicates_skipped, 1);
}

#[test]
fn test_ingestion_service_backpressure() {
    let db = Database::in_memory();
    db.create_collection("bp", 32).unwrap();

    let config = IngestionServiceConfig::builder()
        .collection("bp")
        .batch_size(5000)
        .max_buffer_size(50)
        .backpressure_threshold(0.5)
        .build();

    let mut service = IngestionService::new(&db, config).unwrap();

    // Fill past the threshold to trigger backpressure
    for i in 0..30 {
        service
            .ingest(format!("v{i}"), &random_vector(32), None)
            .unwrap();
    }

    let level = service.backpressure_level();
    assert_ne!(level, BackpressureLevel::None);

    service.flush().unwrap();
    assert_eq!(service.backpressure_level(), BackpressureLevel::None);
}

// ============================================================================
// PitrService tests
// ============================================================================

#[test]
fn test_pitr_create_snapshot_and_list() {
    let db = Database::in_memory();
    db.create_collection("pitr_coll", 64).unwrap();

    let coll = db.collection("pitr_coll").unwrap();
    for i in 0..5 {
        coll.insert(format!("v{i}"), &random_vector(64), None)
            .unwrap();
    }

    let config = PitrServiceConfig::builder()
        .max_snapshots(10)
        .enable_checksums(true)
        .build();

    let service = PitrService::new(&db, config).unwrap();

    let rp = service.create_snapshot("before-update").unwrap();
    assert_eq!(rp.label, "before-update");
    assert_eq!(rp.total_vectors, 5);
    assert!(rp.collections.contains(&"pitr_coll".to_string()));

    let points = service.list_restore_points();
    assert_eq!(points.len(), 1);
}

#[test]
fn test_pitr_recover_to_named_restore_point() {
    let db = Database::in_memory();
    db.create_collection("recovery", 64).unwrap();

    let coll = db.collection("recovery").unwrap();
    for i in 0..3 {
        coll.insert(format!("v{i}"), &random_vector(64), None)
            .unwrap();
    }

    let config = PitrServiceConfig::builder().enable_checksums(true).build();

    let service = PitrService::new(&db, config).unwrap();
    service.create_snapshot("checkpoint-1").unwrap();

    // Modify data after snapshot
    coll.insert("v_extra", &random_vector(64), None).unwrap();

    let result = service
        .recover_to(RecoveryTarget::Named("checkpoint-1".to_string()))
        .unwrap();
    assert!(!result.collections_restored.is_empty());
    assert!(result.verified);
}

#[test]
fn test_pitr_stats() {
    let db = Database::in_memory();
    db.create_collection("stats_coll", 32).unwrap();

    let config = PitrServiceConfig::builder().build();
    let service = PitrService::new(&db, config).unwrap();

    service.create_snapshot("snap1").unwrap();
    service.create_snapshot("snap2").unwrap();

    let stats = service.stats();
    assert_eq!(stats.total_snapshots, 2);
}

// ============================================================================
// TieredService tests
// ============================================================================

#[test]
fn test_tiered_service_insert_and_search() {
    let db = Database::in_memory();
    db.create_collection("tiered", 64).unwrap();

    let config = TieredServiceConfig::builder()
        .collection("tiered")
        .hot_capacity(100)
        .build();

    let service = TieredService::new(&db, config).unwrap();

    let target = random_vector(64);
    service.insert("t1", &target, None).unwrap();
    for i in 1..10 {
        service
            .insert(format!("t{}", i + 1), &random_vector(64), None)
            .unwrap();
    }

    assert_eq!(service.len(), 10);
    assert!(!service.is_empty());

    let results = service.search(&target, 5).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "t1");
}

#[test]
fn test_tiered_service_maintenance() {
    let db = Database::in_memory();
    db.create_collection("maint", 32).unwrap();

    let config = TieredServiceConfig::builder()
        .collection("maint")
        .hot_capacity(100)
        .build();

    let service = TieredService::new(&db, config).unwrap();

    for i in 0..5 {
        service
            .insert(format!("m{i}"), &random_vector(32), None)
            .unwrap();
    }

    let report = service.run_maintenance();
    let stats = service.stats();
    assert_eq!(stats.total_inserts, 5);
    // Maintenance should run without error; totals should sum correctly
    assert_eq!(report.total_hot + report.total_warm + report.total_cold, 5);
}

// ============================================================================
// AdaptiveIndexService tests
// ============================================================================

#[test]
fn test_adaptive_service_insert_and_search() {
    let db = Database::in_memory();
    db.create_collection("adaptive", 64).unwrap();

    let config = AdaptiveServiceConfig::builder()
        .collection("adaptive")
        .memory_budget_mb(128)
        .build();

    let mut service = AdaptiveIndexService::new(&db, config).unwrap();

    let query = random_vector(64);
    service.insert("a1", &query, None).unwrap();
    for i in 1..20 {
        service
            .insert(format!("a{}", i + 1), &random_vector(64), None)
            .unwrap();
    }

    let results = service.search(&query, 5).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "a1");
}

#[test]
fn test_adaptive_service_status_and_health() {
    let db = Database::in_memory();
    db.create_collection("adaptive_health", 32).unwrap();

    let config = AdaptiveServiceConfig::builder()
        .collection("adaptive_health")
        .enable_auto_migration(false)
        .build();

    let mut service = AdaptiveIndexService::new(&db, config).unwrap();

    for i in 0..5 {
        service
            .insert(format!("h{i}"), &random_vector(32), None)
            .unwrap();
    }

    let report = service.status_report();
    assert_eq!(report.total_inserts, 5);
    assert!(report.healthy);

    let health = service.health();
    // Verify we can read health fields without panic
    assert!(health.vector_count == 5);
}

#[test]
fn test_adaptive_service_active_index() {
    let db = Database::in_memory();
    db.create_collection("idx_type", 32).unwrap();

    let config = AdaptiveServiceConfig::builder()
        .collection("idx_type")
        .build();

    let service = AdaptiveIndexService::new(&db, config).unwrap();

    // The initial index type should be a valid variant
    let idx_type = service.active_index();
    // Just verify we can read it without panic
    let _ = format!("{:?}", idx_type);
}

// ============================================================================
// MultiModalService tests
// ============================================================================

#[test]
fn test_multimodal_service_insert_text_and_image() {
    let db = Database::in_memory();
    db.create_collection("mm", 128).unwrap();

    let config = MultiModalServiceConfig::builder()
        .name("mm")
        .text_dimension(128)
        .image_dimension(128)
        .build();

    let mut service = MultiModalService::new(&db, config).unwrap();

    service
        .insert_text("txt1", &random_vector(128), Some(json!({"type": "text"})))
        .unwrap();
    service
        .insert_image("img1", &random_vector(128), Some(json!({"type": "image"})))
        .unwrap();

    assert_eq!(service.len(), 2);
    assert!(!service.is_empty());
    assert_eq!(service.total_inserts(), 2);
}

#[test]
fn test_multimodal_service_search() {
    let db = Database::in_memory();
    db.create_collection("mm_search", 64).unwrap();

    let config = MultiModalServiceConfig::builder()
        .name("mm_search")
        .text_dimension(64)
        .image_dimension(64)
        .build();

    let mut service = MultiModalService::new(&db, config).unwrap();

    let target = random_vector(64);
    service.insert_text("t1", &target, None).unwrap();
    for i in 1..10 {
        service
            .insert_text(format!("t{}", i + 1), &random_vector(64), None)
            .unwrap();
    }

    let results = service.search(ModalInput::Text(target), 5).unwrap();
    assert!(!results.is_empty());
    assert_eq!(service.total_searches(), 1);
}

#[test]
fn test_multimodal_service_stats() {
    let db = Database::in_memory();
    db.create_collection("mm_stats", 32).unwrap();

    let config = MultiModalServiceConfig::builder()
        .name("mm_stats")
        .text_dimension(32)
        .image_dimension(32)
        .audio_dimension(32)
        .build();

    let mut service = MultiModalService::new(&db, config).unwrap();

    service.insert_text("t1", &random_vector(32), None).unwrap();
    service
        .insert_image("i1", &random_vector(32), None)
        .unwrap();
    service
        .insert_audio("a1", &random_vector(32), None)
        .unwrap();

    let stats = service.stats();
    assert_eq!(stats.total_documents, 3);
}
