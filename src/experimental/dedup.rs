//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Vector Deduplication
//!
//! Detect and manage near-duplicate vectors in the database:
//! - Exact duplicate detection
//! - Near-duplicate detection using similarity thresholds
//! - Locality-Sensitive Hashing (LSH) for efficient detection
//! - Deduplication strategies (keep first, keep latest, merge)
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::dedup::{DuplicateDetector, DeduplicationConfig};
//!
//! let detector = DuplicateDetector::new(DeduplicationConfig::default());
//! detector.add("doc1", &vector1);
//!
//! if let Some(duplicate_of) = detector.find_duplicate(&new_vector) {
//!     println!("Near-duplicate of {}", duplicate_of);
//! }
//! ```

use crate::distance::DistanceFunction;
use ordered_float::OrderedFloat;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for deduplication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationConfig {
    /// Similarity threshold for considering vectors as duplicates (0.0-1.0)
    pub threshold: f32,
    /// Distance function to use
    pub distance: DistanceFunction,
    /// Number of hash tables for LSH
    pub num_tables: usize,
    /// Number of hash functions per table
    pub num_hashes: usize,
    /// Enable exact duplicate detection
    pub detect_exact: bool,
}

impl Default for DeduplicationConfig {
    fn default() -> Self {
        Self {
            threshold: 0.95,
            distance: DistanceFunction::Cosine,
            num_tables: 10,
            num_hashes: 5,
            detect_exact: true,
        }
    }
}

impl DeduplicationConfig {
    /// Create config with specific threshold
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }

    /// Set distance function
    pub fn with_distance(mut self, distance: DistanceFunction) -> Self {
        self.distance = distance;
        self
    }
}

/// Result of duplicate detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateResult {
    /// ID of the potential duplicate
    pub id: String,
    /// Similarity score (1 - distance)
    pub similarity: f32,
    /// Whether it's an exact duplicate
    pub exact: bool,
}

/// Duplicate detector using LSH and direct comparison
pub struct DuplicateDetector {
    config: DeduplicationConfig,
    /// Stored vectors with IDs
    vectors: HashMap<String, Vec<f32>>,
    /// LSH tables
    lsh: Option<Lsh>,
    /// Hash of exact vectors for quick lookup
    exact_hashes: HashMap<u64, Vec<String>>,
}

impl DuplicateDetector {
    /// Create a new duplicate detector
    pub fn new(config: DeduplicationConfig) -> Self {
        Self {
            config,
            vectors: HashMap::new(),
            lsh: None,
            exact_hashes: HashMap::new(),
        }
    }

    /// Initialize with dimensionality (required for LSH)
    pub fn init(&mut self, dimensions: usize) {
        self.lsh = Some(Lsh::new(
            dimensions,
            self.config.num_tables,
            self.config.num_hashes,
        ));
    }

    /// Add a vector to the detector
    pub fn add(&mut self, id: &str, vector: &[f32]) {
        // Add to exact hash
        if self.config.detect_exact {
            let hash = Self::hash_vector(vector);
            self.exact_hashes
                .entry(hash)
                .or_default()
                .push(id.to_string());
        }

        // Add to LSH
        if let Some(lsh) = &mut self.lsh {
            lsh.insert(id.to_string(), vector);
        } else {
            // Initialize if needed
            self.init(vector.len());
            if let Some(lsh) = &mut self.lsh {
                lsh.insert(id.to_string(), vector);
            }
        }

        // Store vector
        self.vectors.insert(id.to_string(), vector.to_vec());
    }

    /// Remove a vector from the detector
    pub fn remove(&mut self, id: &str) -> bool {
        if let Some(vector) = self.vectors.remove(id) {
            // Remove from exact hash
            let hash = Self::hash_vector(&vector);
            if let Some(ids) = self.exact_hashes.get_mut(&hash) {
                ids.retain(|i| i != id);
                if ids.is_empty() {
                    self.exact_hashes.remove(&hash);
                }
            }

            // Remove from LSH
            if let Some(lsh) = &mut self.lsh {
                lsh.remove(id);
            }

            true
        } else {
            false
        }
    }

    /// Find if a vector is a duplicate of any existing vector
    pub fn find_duplicate(&self, vector: &[f32]) -> Option<DuplicateResult> {
        // Check exact duplicates first
        if self.config.detect_exact {
            let hash = Self::hash_vector(vector);
            if let Some(ids) = self.exact_hashes.get(&hash) {
                for id in ids {
                    if let Some(stored) = self.vectors.get(id) {
                        if Self::vectors_equal(vector, stored) {
                            return Some(DuplicateResult {
                                id: id.clone(),
                                similarity: 1.0,
                                exact: true,
                            });
                        }
                    }
                }
            }
        }

        // Check near-duplicates using LSH
        let candidates = if let Some(lsh) = &self.lsh {
            lsh.query(vector)
        } else {
            // Fall back to brute force if LSH not initialized
            self.vectors.keys().cloned().collect()
        };

        let mut best_match: Option<DuplicateResult> = None;

        for id in candidates {
            if let Some(stored) = self.vectors.get(&id) {
                let distance = self.config.distance.compute(vector, stored);
                let similarity = 1.0 - distance.min(1.0);

                if similarity >= self.config.threshold
                    && (best_match.is_none()
                        || similarity
                            > best_match
                                .as_ref()
                                .expect("checked is_none above")
                                .similarity)
                {
                    best_match = Some(DuplicateResult {
                        id,
                        similarity,
                        exact: false,
                    });
                }
            }
        }

        best_match
    }

    /// Find all duplicates above threshold
    pub fn find_all_duplicates(&self, vector: &[f32]) -> Vec<DuplicateResult> {
        let mut results = Vec::new();

        // Check exact duplicates
        if self.config.detect_exact {
            let hash = Self::hash_vector(vector);
            if let Some(ids) = self.exact_hashes.get(&hash) {
                for id in ids {
                    if let Some(stored) = self.vectors.get(id) {
                        if Self::vectors_equal(vector, stored) {
                            results.push(DuplicateResult {
                                id: id.clone(),
                                similarity: 1.0,
                                exact: true,
                            });
                        }
                    }
                }
            }
        }

        // Check near-duplicates
        let candidates = if let Some(lsh) = &self.lsh {
            lsh.query(vector)
        } else {
            self.vectors.keys().cloned().collect()
        };

        for id in candidates {
            if results.iter().any(|r| r.id == id) {
                continue;
            }

            if let Some(stored) = self.vectors.get(&id) {
                let distance = self.config.distance.compute(vector, stored);
                let similarity = 1.0 - distance.min(1.0);

                if similarity >= self.config.threshold {
                    results.push(DuplicateResult {
                        id,
                        similarity,
                        exact: false,
                    });
                }
            }
        }

        // Sort by similarity
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Find all duplicate groups in the detector
    pub fn find_duplicate_groups(&self) -> Vec<DuplicateGroup> {
        let mut groups: Vec<DuplicateGroup> = Vec::new();
        let mut processed: HashSet<String> = HashSet::new();

        for (id, vector) in &self.vectors {
            if processed.contains(id) {
                continue;
            }

            let duplicates = self.find_all_duplicates(vector);
            let group_members: Vec<String> = duplicates
                .iter()
                .filter(|r| r.id != *id && !processed.contains(&r.id))
                .map(|r| r.id.clone())
                .collect();

            if !group_members.is_empty() {
                let mut all_members = vec![id.clone()];
                all_members.extend(group_members);

                let avg_similarity: f32 = if duplicates.is_empty() {
                    1.0
                } else {
                    duplicates.iter().map(|r| r.similarity).sum::<f32>() / duplicates.len() as f32
                };

                for member in &all_members {
                    processed.insert(member.clone());
                }

                groups.push(DuplicateGroup {
                    canonical: id.clone(),
                    members: all_members,
                    avg_similarity,
                });
            }
        }

        groups
    }

    /// Hash a vector for exact duplicate detection
    fn hash_vector(vector: &[f32]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        for &v in vector {
            OrderedFloat(v).hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Check if two vectors are exactly equal
    fn vectors_equal(a: &[f32], b: &[f32]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| x == y)
    }

    /// Number of vectors in detector
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// Duplicate group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    /// Canonical (first) member
    pub canonical: String,
    /// All members including canonical
    pub members: Vec<String>,
    /// Average similarity within group
    pub avg_similarity: f32,
}

/// Locality-Sensitive Hashing for approximate nearest neighbor
#[derive(Debug, Clone)]
struct Lsh {
    /// Random hyperplanes for each table
    hyperplanes: Vec<Vec<Vec<f32>>>,
    /// Hash tables: table_idx -> hash -> set of IDs
    tables: Vec<HashMap<u64, HashSet<String>>>,
    /// Vector ID to hashes mapping for removal
    id_hashes: HashMap<String, Vec<u64>>,
}

impl Lsh {
    fn new(dimensions: usize, num_tables: usize, num_hashes: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Generate random hyperplanes
        let hyperplanes: Vec<Vec<Vec<f32>>> = (0..num_tables)
            .map(|_| {
                (0..num_hashes)
                    .map(|_| {
                        let mut plane: Vec<f32> =
                            (0..dimensions).map(|_| rng.gen::<f32>() - 0.5).collect();
                        // Normalize
                        let norm: f32 = plane.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm > 0.0 {
                            for x in &mut plane {
                                *x /= norm;
                            }
                        }
                        plane
                    })
                    .collect()
            })
            .collect();

        let tables = vec![HashMap::new(); num_tables];

        Self {
            hyperplanes,
            tables,
            id_hashes: HashMap::new(),
        }
    }

    fn hash_vector(&self, vector: &[f32], table_idx: usize) -> u64 {
        let mut hash: u64 = 0;

        for (i, hyperplane) in self.hyperplanes[table_idx].iter().enumerate() {
            let dot: f32 = vector
                .iter()
                .zip(hyperplane.iter())
                .map(|(a, b)| a * b)
                .sum();

            if dot > 0.0 {
                hash |= 1 << i;
            }
        }

        hash
    }

    fn insert(&mut self, id: String, vector: &[f32]) {
        let mut hashes = Vec::with_capacity(self.tables.len());

        for i in 0..self.tables.len() {
            let hash = self.hash_vector(vector, i);
            hashes.push(hash);
            self.tables[i].entry(hash).or_default().insert(id.clone());
        }

        self.id_hashes.insert(id, hashes);
    }

    fn remove(&mut self, id: &str) {
        if let Some(hashes) = self.id_hashes.remove(id) {
            for (i, hash) in hashes.iter().enumerate() {
                if let Some(bucket) = self.tables[i].get_mut(hash) {
                    bucket.remove(id);
                    if bucket.is_empty() {
                        self.tables[i].remove(hash);
                    }
                }
            }
        }
    }

    fn query(&self, vector: &[f32]) -> HashSet<String> {
        let mut candidates = HashSet::new();

        for i in 0..self.tables.len() {
            let hash = self.hash_vector(vector, i);
            if let Some(bucket) = self.tables[i].get(&hash) {
                candidates.extend(bucket.iter().cloned());
            }
        }

        candidates
    }
}

/// Deduplication report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationReport {
    /// Total vectors analyzed
    pub total_vectors: usize,
    /// Number of duplicate groups found
    pub num_groups: usize,
    /// Total duplicate vectors (excludes canonical)
    pub num_duplicates: usize,
    /// Exact duplicates
    pub exact_duplicates: usize,
    /// Near duplicates
    pub near_duplicates: usize,
    /// Duplicate groups
    pub groups: Vec<DuplicateGroup>,
}

/// Generate a deduplication report for a set of vectors
pub fn generate_dedup_report(
    vectors: &[(&str, &[f32])],
    config: DeduplicationConfig,
) -> DeduplicationReport {
    let mut detector = DuplicateDetector::new(config);

    for (id, vector) in vectors {
        detector.add(id, vector);
    }

    let groups = detector.find_duplicate_groups();

    let total_duplicates: usize = groups.iter().map(|g| g.members.len() - 1).sum();
    let exact_count = groups
        .iter()
        .flat_map(|g| &g.members)
        .filter(|_| {
            // Would need to track this separately for accurate count
            false
        })
        .count();

    DeduplicationReport {
        total_vectors: vectors.len(),
        num_groups: groups.len(),
        num_duplicates: total_duplicates,
        exact_duplicates: exact_count,
        near_duplicates: total_duplicates - exact_count,
        groups,
    }
}

/// Deduplication strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DeduplicationStrategy {
    /// Keep the first occurrence
    KeepFirst,
    /// Keep the latest occurrence
    KeepLatest,
    /// Keep vector with most metadata
    KeepRichest,
}

/// Apply deduplication to get IDs to remove
pub fn get_ids_to_remove(
    groups: &[DuplicateGroup],
    strategy: DeduplicationStrategy,
) -> Vec<String> {
    let mut to_remove = Vec::new();

    for group in groups {
        match strategy {
            DeduplicationStrategy::KeepFirst => {
                // Keep canonical (first), remove rest
                to_remove.extend(group.members.iter().skip(1).cloned());
            }
            DeduplicationStrategy::KeepLatest => {
                // Keep last, remove rest
                if group.members.len() > 1 {
                    to_remove.extend(group.members.iter().take(group.members.len() - 1).cloned());
                }
            }
            DeduplicationStrategy::KeepRichest => {
                // Default to keeping first for now
                to_remove.extend(group.members.iter().skip(1).cloned());
            }
        }
    }

    to_remove
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_duplicate_detection() {
        let mut detector = DuplicateDetector::new(DeduplicationConfig::default());

        let vec1 = vec![1.0, 2.0, 3.0, 4.0];
        detector.add("doc1", &vec1);

        // Same vector should be detected as exact duplicate
        let result = detector.find_duplicate(&vec1);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.id, "doc1");
        assert!(result.exact);
        assert_eq!(result.similarity, 1.0);
    }

    #[test]
    fn test_near_duplicate_detection() {
        let config = DeduplicationConfig::with_threshold(0.9);
        let mut detector = DuplicateDetector::new(config);

        let vec1 = vec![1.0, 0.0, 0.0, 0.0];
        detector.add("doc1", &vec1);

        // Similar but not exact
        let vec2 = vec![0.99, 0.01, 0.0, 0.0];
        let result = detector.find_duplicate(&vec2);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(!result.exact);
        assert!(result.similarity > 0.9);
    }

    #[test]
    fn test_no_duplicate() {
        let config = DeduplicationConfig::with_threshold(0.95);
        let mut detector = DuplicateDetector::new(config);

        let vec1 = vec![1.0, 0.0, 0.0, 0.0];
        detector.add("doc1", &vec1);

        // Very different vector
        let vec2 = vec![0.0, 1.0, 0.0, 0.0];
        let result = detector.find_duplicate(&vec2);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_all_duplicates() {
        let config = DeduplicationConfig::with_threshold(0.9);
        let mut detector = DuplicateDetector::new(config);

        let base = vec![1.0, 0.0, 0.0, 0.0];
        detector.add("doc1", &base);

        // Add similar vectors
        detector.add("doc2", &[0.99, 0.01, 0.0, 0.0]);
        detector.add("doc3", &[0.98, 0.02, 0.0, 0.0]);

        // Add different vector
        detector.add("doc4", &[0.0, 1.0, 0.0, 0.0]);

        let duplicates = detector.find_all_duplicates(&base);

        // Should find doc1, doc2, doc3 but not doc4
        assert!(duplicates.len() >= 2);
        assert!(duplicates.iter().all(|r| r.id != "doc4"));
    }

    #[test]
    fn test_duplicate_groups() {
        let config = DeduplicationConfig::with_threshold(0.95);
        let mut detector = DuplicateDetector::new(config);

        // Group 1: similar vectors
        detector.add("a1", &[1.0, 0.0, 0.0, 0.0]);
        detector.add("a2", &[1.0, 0.0, 0.0, 0.0]); // Exact duplicate

        // Group 2: different vectors
        detector.add("b1", &[0.0, 1.0, 0.0, 0.0]);
        detector.add("b2", &[0.0, 1.0, 0.0, 0.0]); // Exact duplicate

        let groups = detector.find_duplicate_groups();
        assert!(!groups.is_empty());
    }

    #[test]
    fn test_remove_vector() {
        let mut detector = DuplicateDetector::new(DeduplicationConfig::default());

        detector.add("doc1", &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(detector.len(), 1);

        let removed = detector.remove("doc1");
        assert!(removed);
        assert_eq!(detector.len(), 0);

        // Now no duplicate should be found
        let result = detector.find_duplicate(&[1.0, 2.0, 3.0, 4.0]);
        assert!(result.is_none());
    }

    #[test]
    fn test_lsh_basic() {
        let mut lsh = Lsh::new(4, 5, 3);

        lsh.insert("v1".to_string(), &[1.0, 0.0, 0.0, 0.0]);
        lsh.insert("v2".to_string(), &[0.99, 0.01, 0.0, 0.0]);
        lsh.insert("v3".to_string(), &[0.0, 1.0, 0.0, 0.0]);

        // Query similar to v1/v2 should return them
        let candidates = lsh.query(&[1.0, 0.0, 0.0, 0.0]);
        assert!(candidates.contains("v1"));
    }

    #[test]
    fn test_dedup_report() {
        let vectors: Vec<(&str, Vec<f32>)> = vec![
            ("a", vec![1.0, 0.0, 0.0, 0.0]),
            ("b", vec![1.0, 0.0, 0.0, 0.0]), // Duplicate of a
            ("c", vec![0.0, 1.0, 0.0, 0.0]),
            ("d", vec![0.0, 1.0, 0.0, 0.0]), // Duplicate of c
        ];

        let refs: Vec<(&str, &[f32])> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        let report = generate_dedup_report(&refs, DeduplicationConfig::default());

        assert_eq!(report.total_vectors, 4);
        assert!(report.num_groups >= 1);
    }

    #[test]
    fn test_get_ids_to_remove() {
        let groups = vec![DuplicateGroup {
            canonical: "a".to_string(),
            members: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            avg_similarity: 0.98,
        }];

        // KeepFirst: remove b and c
        let to_remove = get_ids_to_remove(&groups, DeduplicationStrategy::KeepFirst);
        assert_eq!(to_remove.len(), 2);
        assert!(to_remove.contains(&"b".to_string()));
        assert!(to_remove.contains(&"c".to_string()));

        // KeepLatest: remove a and b
        let to_remove = get_ids_to_remove(&groups, DeduplicationStrategy::KeepLatest);
        assert_eq!(to_remove.len(), 2);
        assert!(to_remove.contains(&"a".to_string()));
        assert!(to_remove.contains(&"b".to_string()));
    }

    #[test]
    fn test_empty_detector() {
        let detector = DuplicateDetector::new(DeduplicationConfig::default());
        assert!(detector.is_empty());
        assert_eq!(detector.len(), 0);

        let result = detector.find_duplicate(&[1.0, 2.0, 3.0]);
        assert!(result.is_none());
    }
}
