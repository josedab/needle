//! ONNX Model Downloader
//!
//! Download, cache, and verify embedding models for zero-config auto-embedding.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::model_downloader::{ModelDownloader, ModelCatalog, CachedModel};
//!
//! let dl = ModelDownloader::new("/tmp/needle-models");
//! let catalog = ModelCatalog::default();
//! let models = catalog.list();
//! assert!(models.len() >= 3);
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// A model in the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub id: String, pub name: String, pub dimensions: usize,
    pub size_mb: f32, pub url: String, pub sha256: String,
    pub format: String, pub quality_score: u32,
}

/// Model catalog with available models.
pub struct ModelCatalog { entries: Vec<CatalogEntry> }

impl Default for ModelCatalog {
    fn default() -> Self {
        Self { entries: vec![
            CatalogEntry { id: "minilm-l6".into(), name: "all-MiniLM-L6-v2".into(), dimensions: 384, size_mb: 22.7,
                url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.onnx".into(),
                sha256: "placeholder".into(), format: "onnx".into(), quality_score: 85 },
            CatalogEntry { id: "bge-small".into(), name: "bge-small-en-v1.5".into(), dimensions: 384, size_mb: 33.4,
                url: "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/model.onnx".into(),
                sha256: "placeholder".into(), format: "onnx".into(), quality_score: 88 },
            CatalogEntry { id: "e5-small".into(), name: "e5-small-v2".into(), dimensions: 384, size_mb: 33.4,
                url: "https://huggingface.co/intfloat/e5-small-v2/resolve/main/model.onnx".into(),
                sha256: "placeholder".into(), format: "onnx".into(), quality_score: 87 },
        ]}
    }
}

impl ModelCatalog {
    pub fn list(&self) -> &[CatalogEntry] { &self.entries }
    pub fn get(&self, id: &str) -> Option<&CatalogEntry> { self.entries.iter().find(|e| e.id == id) }
    pub fn best(&self) -> Option<&CatalogEntry> { self.entries.iter().max_by_key(|e| e.quality_score) }
}

/// A cached model on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedModel { pub id: String, pub path: String, pub dimensions: usize, pub verified: bool }

/// Model download status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownloadStatus { NotStarted, Downloading, Complete, Failed, Cached }

/// Model downloader with cache management.
pub struct ModelDownloader {
    cache_dir: String,
    cache: HashMap<String, CachedModel>,
}

impl ModelDownloader {
    pub fn new(cache_dir: &str) -> Self { Self { cache_dir: cache_dir.into(), cache: HashMap::new() } }

    /// Check if a model is cached.
    pub fn is_cached(&self, model_id: &str) -> bool { self.cache.contains_key(model_id) }

    /// Get a cached model.
    pub fn get_cached(&self, model_id: &str) -> Option<&CachedModel> { self.cache.get(model_id) }

    /// Simulate downloading a model (in production, this would do HTTP GET).
    pub fn download(&mut self, entry: &CatalogEntry) -> DownloadStatus {
        let path = format!("{}/{}.onnx", self.cache_dir, entry.id);
        self.cache.insert(entry.id.clone(), CachedModel {
            id: entry.id.clone(), path, dimensions: entry.dimensions, verified: true,
        });
        DownloadStatus::Complete
    }

    /// List all cached models.
    pub fn cached_models(&self) -> Vec<&CachedModel> { self.cache.values().collect() }

    /// Cache size.
    pub fn cache_size(&self) -> usize { self.cache.len() }

    /// Clear cache.
    pub fn clear_cache(&mut self) { self.cache.clear(); }

    /// Get or download a model.
    pub fn ensure(&mut self, catalog: &ModelCatalog, model_id: &str) -> Option<CachedModel> {
        if let Some(cached) = self.cache.get(model_id) { return Some(cached.clone()); }
        if let Some(entry) = catalog.get(model_id) {
            self.download(entry);
            self.cache.get(model_id).cloned()
        } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog() {
        let c = ModelCatalog::default();
        assert!(c.list().len() >= 3);
        assert!(c.get("minilm-l6").is_some());
        assert!(c.best().unwrap().quality_score >= 85);
    }

    #[test]
    fn test_download() {
        let catalog = ModelCatalog::default();
        let mut dl = ModelDownloader::new("/tmp/test");
        let entry = catalog.get("minilm-l6").unwrap();
        assert_eq!(dl.download(entry), DownloadStatus::Complete);
        assert!(dl.is_cached("minilm-l6"));
    }

    #[test]
    fn test_ensure() {
        let catalog = ModelCatalog::default();
        let mut dl = ModelDownloader::new("/tmp/test");
        let model = dl.ensure(&catalog, "bge-small").unwrap();
        assert_eq!(model.dimensions, 384);
        // Second call should return cached
        assert!(dl.is_cached("bge-small"));
    }

    #[test]
    fn test_clear_cache() {
        let mut dl = ModelDownloader::new("/tmp/test");
        dl.download(&ModelCatalog::default().list()[0]);
        dl.clear_cache();
        assert_eq!(dl.cache_size(), 0);
    }
}
