#[cfg(feature = "observability")]
use needle::{Database, Result};
#[cfg(feature = "observability")]
use serde_json::json;

#[cfg(feature = "observability")]
use needle::drift::{DriftConfig, DriftDetector};

#[cfg(feature = "observability")]
use crate::cli::commands::DriftCommands;

#[cfg(feature = "observability")]
pub fn drift_command(cmd: DriftCommands) -> Result<()> {
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

    let dimensions = coll.dimensions().unwrap_or(0);
    if dimensions == 0 {
        return Err(needle::NeedleError::InvalidInput(
            "Cannot determine vector dimensions".to_string(),
        ));
    }

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

    let config = DriftConfig::default();
    let mut detector = DriftDetector::new(dimensions, config);
    detector.add_baseline(&sample)?;

    let centroid: Vec<f32> = (0..dimensions)
        .map(|d| sample.iter().map(|v| v[d]).sum::<f32>() / sample.len() as f32)
        .collect();

    let variance: Vec<f32> = (0..dimensions)
        .map(|d| {
            let mean = centroid[d];
            sample.iter().map(|v| (v[d] - mean).powi(2)).sum::<f32>() / sample.len() as f32
        })
        .collect();

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

    let baseline_content = std::fs::read_to_string(baseline_path)?;
    let baseline_data: serde_json::Value = serde_json::from_str(&baseline_content)?;

    let dimensions = baseline_data["dimensions"].as_u64().ok_or_else(|| {
        needle::NeedleError::InvalidInput("Invalid baseline: missing dimensions".to_string())
    })? as usize;

    let baseline_centroid: Vec<f32> = baseline_data["centroid"]
        .as_array()
        .ok_or_else(|| {
            needle::NeedleError::InvalidInput("Invalid baseline: missing centroid".to_string())
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

    let baseline_variance: Vec<f32> = baseline_data["variance"]
        .as_array()
        .map_or_else(|| vec![0.1; dimensions], |arr| {
            arr.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect()
        });

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

    let baseline_content = std::fs::read_to_string(baseline_path)?;
    let baseline_data: serde_json::Value = serde_json::from_str(&baseline_content)?;

    let dimensions = baseline_data["dimensions"].as_u64().ok_or_else(|| {
        needle::NeedleError::InvalidInput("Invalid baseline: missing dimensions".to_string())
    })? as usize;

    let baseline_centroid: Vec<f32> = baseline_data["centroid"]
        .as_array()
        .ok_or_else(|| {
            needle::NeedleError::InvalidInput("Invalid baseline: missing centroid".to_string())
        })?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();

    let baseline_variance: Vec<f32> = baseline_data["variance"]
        .as_array()
        .map_or_else(|| vec![0.1; dimensions], |arr| {
            arr.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect()
        });

    let vectors = coll.export_all()?;
    let current_vecs: Vec<Vec<f32>> = vectors.iter().map(|(_, v, _)| v.clone()).collect();

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

#[cfg(all(test, feature = "observability"))]
mod tests {
    use super::*;
    use serde_json::json;

    fn setup_db_with_vectors() -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle").to_str().unwrap().to_string();
        let mut db = Database::open(&path).unwrap();
        db.create_collection("embeddings", 4).unwrap();
        let coll = db.collection("embeddings").unwrap();
        for i in 0..20 {
            let vec = vec![
                (i as f32 * 0.1).sin(),
                (i as f32 * 0.2).cos(),
                (i as f32 * 0.3).sin(),
                (i as f32 * 0.4).cos(),
            ];
            coll.insert(format!("v{i}"), &vec, None).unwrap();
        }
        db.save().unwrap();
        (dir, path)
    }

    fn create_baseline_file(dir: &tempfile::TempDir, dims: usize) -> String {
        let centroid: Vec<f32> = (0..dims).map(|i| (i as f32 * 0.1).sin()).collect();
        let variance: Vec<f32> = vec![0.1; dims];
        let baseline = json!({
            "collection": "embeddings",
            "sample_size": 20,
            "dimensions": dims,
            "created_at": "2024-01-01T00:00:00Z",
            "centroid": centroid,
            "variance": variance
        });
        let path = dir.path().join("baseline.json");
        std::fs::write(&path, serde_json::to_string_pretty(&baseline).unwrap()).unwrap();
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn test_drift_baseline() {
        let (_dir, db_path) = setup_db_with_vectors();
        let output_dir = tempfile::tempdir().unwrap();
        let output = output_dir.path().join("baseline.json");
        let output = output.to_str().unwrap();

        assert!(drift_baseline(&db_path, "embeddings", output, 10).is_ok());

        let content = std::fs::read_to_string(output).unwrap();
        let data: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(data["dimensions"], 4);
        assert!(data["centroid"].as_array().is_some());
        assert!(data["variance"].as_array().is_some());
    }

    #[test]
    fn test_drift_baseline_full_sample() {
        let (_dir, db_path) = setup_db_with_vectors();
        let output_dir = tempfile::tempdir().unwrap();
        let output = output_dir.path().join("baseline.json");
        let output = output.to_str().unwrap();

        assert!(drift_baseline(&db_path, "embeddings", output, 0).is_ok());
    }

    #[test]
    fn test_drift_detect() {
        let (dir, db_path) = setup_db_with_vectors();
        let baseline_path = create_baseline_file(&dir, 4);

        let result = drift_detect(&db_path, "embeddings", &baseline_path, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_drift_report_text() {
        let (dir, db_path) = setup_db_with_vectors();
        let baseline_path = create_baseline_file(&dir, 4);

        let result = drift_report(&db_path, "embeddings", &baseline_path, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_drift_report_json() {
        let (dir, db_path) = setup_db_with_vectors();
        let baseline_path = create_baseline_file(&dir, 4);

        let result = drift_report(&db_path, "embeddings", &baseline_path, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_drift_detect_nonexistent_collection() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.needle");
        let db_path = db_path.to_str().unwrap();
        let _db = Database::open(db_path).unwrap();
        let baseline_path = create_baseline_file(&dir, 4);

        let result = drift_detect(db_path, "nonexistent", &baseline_path, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_drift_baseline_empty_collection() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.needle");
        let db_path_str = db_path.to_str().unwrap();
        let mut db = Database::open(db_path_str).unwrap();
        db.create_collection("empty", 4).unwrap();
        db.save().unwrap();

        let output = dir.path().join("baseline.json");
        let result = drift_baseline(db_path_str, "empty", output.to_str().unwrap(), 10);
        let _ = result;
    }
}
