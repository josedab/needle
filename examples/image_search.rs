//! Image Search Example
//!
//! This example demonstrates building an image search system using Needle.
//! Images are represented by their embeddings (e.g., from CLIP, ResNet).
//!
//! Run with: cargo run --example image_search

use needle::{CollectionConfig, Database, DistanceFunction, Filter, ScalarQuantizer, SearchResult};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Image metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImageInfo {
    id: String,
    filename: String,
    width: u32,
    height: u32,
    format: String,
    tags: Vec<String>,
    description: String,
    upload_date: String,
    album: Option<String>,
}

/// Image search engine
struct ImageSearchEngine {
    db: Database,
    collection_name: String,
    embedding_dim: usize,
    #[allow(dead_code)] // Reserved for quantized search implementation
    quantizer: Option<ScalarQuantizer>,
}

impl ImageSearchEngine {
    /// Create a new image search engine
    fn new(embedding_dim: usize, _use_quantization: bool) -> needle::Result<Self> {
        let db = Database::in_memory();
        let collection_name = "images".to_string();

        let config = CollectionConfig::new(&collection_name, embedding_dim)
            .with_distance(DistanceFunction::Cosine)
            .with_ef_construction(200)
            .with_m(32);

        db.create_collection_with_config(config)?;

        Ok(Self {
            db,
            collection_name,
            embedding_dim,
            quantizer: None,
        })
    }

    /// Train quantizer on existing embeddings (optional, for memory efficiency)
    #[allow(dead_code)] // Example method - quantized search not yet implemented
    fn train_quantizer(&mut self, sample_embeddings: &[Vec<f32>]) {
        let refs: Vec<&[f32]> = sample_embeddings.iter().map(|v| v.as_slice()).collect();
        self.quantizer = Some(ScalarQuantizer::train(&refs));
        println!("Quantizer trained on {} samples", sample_embeddings.len());
    }

    /// Index an image
    fn index_image(&self, embedding: &[f32], info: &ImageInfo) -> needle::Result<()> {
        let collection = self.db.collection(&self.collection_name)?;

        let metadata = json!({
            "filename": info.filename,
            "width": info.width,
            "height": info.height,
            "format": info.format,
            "tags": info.tags.join(","),
            "description": info.description,
            "upload_date": info.upload_date,
            "album": info.album,
        });

        collection.insert(&info.id, embedding, Some(metadata))?;
        Ok(())
    }

    /// Batch index images
    fn batch_index(&self, images: &[(Vec<f32>, ImageInfo)]) -> needle::Result<usize> {
        let mut count = 0;
        for (embedding, info) in images {
            self.index_image(embedding, info)?;
            count += 1;
        }
        Ok(count)
    }

    /// Search by embedding (image-to-image search)
    fn search_by_embedding(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> needle::Result<Vec<ImageSearchResult>> {
        let collection = self.db.collection(&self.collection_name)?;
        let results = collection.search(query_embedding, top_k)?;
        Ok(self.convert_results(results))
    }

    /// Search with filters
    fn search_with_filters(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filters: ImageFilters,
    ) -> needle::Result<Vec<ImageSearchResult>> {
        let collection = self.db.collection(&self.collection_name)?;

        let mut filter_conditions = Vec::new();

        if let Some(format) = filters.format {
            filter_conditions.push(Filter::eq("format", format));
        }

        if let Some(album) = filters.album {
            filter_conditions.push(Filter::eq("album", album));
        }

        if let Some(min_width) = filters.min_width {
            filter_conditions.push(Filter::gte("width", min_width as i64));
        }

        if let Some(min_height) = filters.min_height {
            filter_conditions.push(Filter::gte("height", min_height as i64));
        }

        let results = if filter_conditions.is_empty() {
            collection.search(query_embedding, top_k)?
        } else {
            let combined_filter = if filter_conditions.len() == 1 {
                filter_conditions.into_iter().next().unwrap()
            } else {
                Filter::and(filter_conditions)
            };
            collection.search_with_filter(query_embedding, top_k, &combined_filter)?
        };

        Ok(self.convert_results(results))
    }

    /// Search by tag (find images with similar tags)
    fn search_by_tag(&self, tag: &str, limit: usize) -> needle::Result<Vec<ImageSearchResult>> {
        let collection = self.db.collection(&self.collection_name)?;

        // Filter by tag (stored as comma-separated string, use $contains operator)
        let filter = Filter::parse(&serde_json::json!({
            "tags": { "$contains": tag }
        }))
        .map_err(needle::NeedleError::InvalidInput)?;

        // Get all matching images (we'll need a dummy embedding for filter-only search)
        // In practice, you'd use a dedicated metadata search or hybrid approach
        let dummy_embedding = vec![0.0; self.embedding_dim];
        let results = collection.search_with_filter(&dummy_embedding, limit * 10, &filter)?;

        Ok(self.convert_results(results.into_iter().take(limit).collect()))
    }

    /// Find duplicate/similar images by checking known image IDs
    /// Note: This is a simplified implementation that works with known IDs
    fn find_duplicates(
        &self,
        threshold: f32,
        image_ids: &[String],
    ) -> needle::Result<Vec<DuplicateGroup>> {
        let collection = self.db.collection(&self.collection_name)?;

        let mut duplicates: Vec<DuplicateGroup> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Check each image for duplicates
        for id in image_ids {
            if seen.contains(id) {
                continue;
            }

            // Get the vector for this image
            if let Some((vector, _)) = collection.get(id) {
                // Search for similar images
                let results = collection.search(&vector, 10)?;

                // Find those within threshold
                let similar: Vec<_> = results
                    .iter()
                    .filter(|r| &r.id != id && r.distance < threshold)
                    .collect();

                if !similar.is_empty() {
                    let mut group = DuplicateGroup {
                        primary_id: id.clone(),
                        duplicates: vec![],
                    };

                    for s in similar {
                        group.duplicates.push(DuplicateInfo {
                            id: s.id.clone(),
                            similarity: 1.0 - s.distance,
                        });
                        seen.insert(s.id.clone());
                    }

                    duplicates.push(group);
                }
            }
            seen.insert(id.clone());
        }

        Ok(duplicates)
    }

    /// Get collection statistics
    fn stats(&self) -> needle::Result<ImageCollectionStats> {
        let collection = self.db.collection(&self.collection_name)?;

        Ok(ImageCollectionStats {
            total_images: collection.len(),
            embedding_dim: self.embedding_dim,
            // Estimate: 4 bytes per f32 * dimensions * num_vectors + overhead
            index_size_bytes: collection.len() * self.embedding_dim * 4 + 1024,
        })
    }

    fn convert_results(&self, results: Vec<SearchResult>) -> Vec<ImageSearchResult> {
        results
            .into_iter()
            .map(|r| {
                let metadata = r.metadata.unwrap_or_default();
                ImageSearchResult {
                    id: r.id,
                    similarity: 1.0 - r.distance,
                    filename: metadata["filename"].as_str().unwrap_or("").to_string(),
                    tags: metadata["tags"]
                        .as_str()
                        .unwrap_or("")
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                        .collect(),
                    description: metadata["description"].as_str().unwrap_or("").to_string(),
                }
            })
            .collect()
    }
}

/// Search filters for images
#[derive(Default)]
struct ImageFilters {
    format: Option<String>,
    album: Option<String>,
    min_width: Option<u32>,
    min_height: Option<u32>,
}

/// Image search result
#[derive(Debug)]
struct ImageSearchResult {
    id: String,
    similarity: f32,
    filename: String,
    tags: Vec<String>,
    description: String,
}

/// Group of duplicate images
#[derive(Debug)]
struct DuplicateGroup {
    primary_id: String,
    duplicates: Vec<DuplicateInfo>,
}

/// Duplicate image info
#[derive(Debug)]
struct DuplicateInfo {
    id: String,
    similarity: f32,
}

/// Collection statistics
#[derive(Debug)]
struct ImageCollectionStats {
    total_images: usize,
    embedding_dim: usize,
    index_size_bytes: usize,
}

/// Generate mock CLIP-like embedding (512 dimensions)
fn mock_clip_embedding(seed: u64, dim: usize) -> Vec<f32> {
    let mut rng_state = seed;
    let embedding: Vec<f32> = (0..dim)
        .map(|_| {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng_state >> 16) as f32 / 32768.0) - 1.0
        })
        .collect();

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    embedding.into_iter().map(|x| x / norm).collect()
}

fn main() -> needle::Result<()> {
    println!("=== Image Search Example ===\n");

    // Create image search engine with 512-dim embeddings (CLIP-like)
    let embedding_dim = 512;
    let engine = ImageSearchEngine::new(embedding_dim, false)?;

    // Create sample image dataset
    let sample_images = vec![
        (
            mock_clip_embedding(1, embedding_dim),
            ImageInfo {
                id: "img001".to_string(),
                filename: "sunset_beach.jpg".to_string(),
                width: 1920,
                height: 1080,
                format: "jpeg".to_string(),
                tags: vec![
                    "sunset".to_string(),
                    "beach".to_string(),
                    "nature".to_string(),
                    "ocean".to_string(),
                ],
                description: "Beautiful sunset at the beach with orange sky".to_string(),
                upload_date: "2024-01-15".to_string(),
                album: Some("Vacation 2024".to_string()),
            },
        ),
        (
            mock_clip_embedding(2, embedding_dim),
            ImageInfo {
                id: "img002".to_string(),
                filename: "mountain_view.jpg".to_string(),
                width: 2560,
                height: 1440,
                format: "jpeg".to_string(),
                tags: vec![
                    "mountain".to_string(),
                    "nature".to_string(),
                    "landscape".to_string(),
                ],
                description: "Snow-capped mountains with clear blue sky".to_string(),
                upload_date: "2024-01-20".to_string(),
                album: Some("Vacation 2024".to_string()),
            },
        ),
        (
            mock_clip_embedding(3, embedding_dim),
            ImageInfo {
                id: "img003".to_string(),
                filename: "city_night.png".to_string(),
                width: 1920,
                height: 1080,
                format: "png".to_string(),
                tags: vec![
                    "city".to_string(),
                    "night".to_string(),
                    "lights".to_string(),
                    "urban".to_string(),
                ],
                description: "City skyline at night with illuminated buildings".to_string(),
                upload_date: "2024-02-01".to_string(),
                album: Some("Urban Photography".to_string()),
            },
        ),
        (
            mock_clip_embedding(4, embedding_dim),
            ImageInfo {
                id: "img004".to_string(),
                filename: "cat_portrait.jpg".to_string(),
                width: 1280,
                height: 1280,
                format: "jpeg".to_string(),
                tags: vec![
                    "cat".to_string(),
                    "pet".to_string(),
                    "animal".to_string(),
                    "portrait".to_string(),
                ],
                description: "Close-up portrait of a tabby cat".to_string(),
                upload_date: "2024-02-10".to_string(),
                album: Some("Pets".to_string()),
            },
        ),
        (
            mock_clip_embedding(5, embedding_dim),
            ImageInfo {
                id: "img005".to_string(),
                filename: "sunset_city.jpg".to_string(),
                width: 1920,
                height: 1080,
                format: "jpeg".to_string(),
                tags: vec![
                    "sunset".to_string(),
                    "city".to_string(),
                    "urban".to_string(),
                    "skyline".to_string(),
                ],
                description: "Sunset over the city skyline".to_string(),
                upload_date: "2024-02-15".to_string(),
                album: Some("Urban Photography".to_string()),
            },
        ),
        // Add a near-duplicate
        (
            mock_clip_embedding(1, embedding_dim), // Same as img001
            ImageInfo {
                id: "img006".to_string(),
                filename: "sunset_beach_copy.jpg".to_string(),
                width: 1920,
                height: 1080,
                format: "jpeg".to_string(),
                tags: vec!["sunset".to_string(), "beach".to_string()],
                description: "Beach sunset (duplicate)".to_string(),
                upload_date: "2024-02-20".to_string(),
                album: None,
            },
        ),
    ];

    // Index all images
    println!("Indexing {} images...", sample_images.len());
    let indexed = engine.batch_index(&sample_images)?;
    println!("Indexed {} images\n", indexed);

    // Show statistics
    let stats = engine.stats()?;
    println!("Collection stats:");
    println!("  Total images: {}", stats.total_images);
    println!("  Embedding dimensions: {}", stats.embedding_dim);
    println!("  Index size: {} bytes\n", stats.index_size_bytes);

    // Search by embedding (image-to-image search)
    println!("=== Search by Image (find similar to sunset_beach.jpg) ===");
    let query_embedding = mock_clip_embedding(1, embedding_dim);
    let results = engine.search_by_embedding(&query_embedding, 5)?;

    for result in &results {
        println!(
            "  {} (similarity: {:.3}) - {}",
            result.filename, result.similarity, result.description
        );
    }
    println!();

    // Search with filters
    println!("=== Search with Filters (format=jpeg, album=Vacation 2024) ===");
    let filters = ImageFilters {
        format: Some("jpeg".to_string()),
        album: Some("Vacation 2024".to_string()),
        ..Default::default()
    };
    let results = engine.search_with_filters(&query_embedding, 5, filters)?;

    for result in &results {
        println!(
            "  {} (similarity: {:.3}) - tags: {:?}",
            result.filename, result.similarity, result.tags
        );
    }
    println!();

    // Search by tag
    println!("=== Search by Tag (sunset) ===");
    let results = engine.search_by_tag("sunset", 5)?;

    for result in &results {
        println!(
            "  {} - {} - tags: {:?}",
            result.id, result.filename, result.tags
        );
    }
    println!();

    // Find duplicates
    println!("=== Duplicate Detection ===");
    let image_ids: Vec<String> = sample_images
        .iter()
        .map(|(_, info)| info.id.clone())
        .collect();
    let duplicates = engine.find_duplicates(0.1, &image_ids)?; // Very high similarity threshold

    if duplicates.is_empty() {
        println!("  No duplicates found");
    } else {
        for group in &duplicates {
            println!("  Primary: {}", group.primary_id);
            for dup in &group.duplicates {
                println!(
                    "    Duplicate: {} (similarity: {:.3})",
                    dup.id, dup.similarity
                );
            }
        }
    }
    println!();

    // Demonstrate quantization (memory savings)
    println!("=== Quantization Demo ===");
    let sample_embeddings: Vec<Vec<f32>> = sample_images.iter().map(|(e, _)| e.clone()).collect();

    let _quantizer = ScalarQuantizer::train(
        &sample_embeddings
            .iter()
            .map(|v| v.as_slice())
            .collect::<Vec<_>>(),
    );

    let original_size = embedding_dim * 4; // f32 = 4 bytes
    let quantized_size = embedding_dim; // u8 = 1 byte per element

    println!("  Original embedding size: {} bytes", original_size);
    println!("  Quantized embedding size: {} bytes", quantized_size);
    println!(
        "  Memory reduction: {:.1}x",
        original_size as f32 / quantized_size as f32
    );

    println!("\nImage search example complete!");
    Ok(())
}
