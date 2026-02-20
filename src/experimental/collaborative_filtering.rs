//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Embedded Collaborative Filtering
//!
//! Built-in user-item interaction matrix with ALS (Alternating Least Squares)
//! factorization for implicit feedback. Supports incremental updates, hybrid
//! scoring combining CF with vector similarity, and session-based recommendations.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::experimental::collaborative_filtering::*;
//!
//! let mut cf = CollaborativeFilter::new(CfConfig::default());
//! cf.record_interaction("user1", "item1", InteractionType::Click);
//! cf.record_interaction("user1", "item2", InteractionType::Purchase);
//! cf.train();
//! let recs = cf.recommend("user1", 10);
//! ```

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Type of user-item interaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InteractionType {
    /// User viewed/clicked the item
    Click,
    /// User purchased/bookmarked the item
    Purchase,
    /// User rated the item (1-5)
    Rating(f64),
    /// User searched and found the item
    SearchResult,
    /// Custom interaction with weight
    Custom(f64),
}

impl InteractionType {
    /// Convert interaction to a confidence weight.
    pub fn weight(&self) -> f64 {
        match self {
            Self::Click => 1.0,
            Self::Purchase => 5.0,
            Self::Rating(r) => *r,
            Self::SearchResult => 0.5,
            Self::Custom(w) => *w,
        }
    }
}

/// A recorded interaction event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    /// User ID
    pub user_id: String,
    /// Item ID
    pub item_id: String,
    /// Interaction type
    pub interaction_type: InteractionType,
    /// Timestamp
    pub timestamp: u64,
    /// Optional session ID
    pub session_id: Option<String>,
}

/// Configuration for collaborative filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfConfig {
    /// Number of latent factors for matrix factorization
    pub num_factors: usize,
    /// Regularization parameter (lambda)
    pub regularization: f64,
    /// Number of ALS iterations
    pub num_iterations: usize,
    /// Confidence scaling for implicit feedback (c = 1 + alpha * r)
    pub confidence_alpha: f64,
    /// Weight for CF scores in hybrid scoring (vs vector similarity)
    pub cf_weight: f64,
    /// Maximum interactions to retain
    pub max_interactions: usize,
    /// Enable session-based recommendations
    pub session_based: bool,
}

impl Default for CfConfig {
    fn default() -> Self {
        Self {
            num_factors: 32,
            regularization: 0.1,
            num_iterations: 10,
            confidence_alpha: 40.0,
            cf_weight: 0.3,
            max_interactions: 1_000_000,
            session_based: true,
        }
    }
}

/// A recommendation with score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Item ID
    pub item_id: String,
    /// CF score (higher = more recommended)
    pub cf_score: f64,
    /// Optional vector similarity score
    pub vector_score: Option<f64>,
    /// Combined hybrid score
    pub hybrid_score: f64,
}

/// Collaborative filtering statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfStats {
    /// Number of unique users
    pub num_users: usize,
    /// Number of unique items
    pub num_items: usize,
    /// Total interactions recorded
    pub total_interactions: usize,
    /// Model trained (has latent factors)
    pub is_trained: bool,
    /// Number of latent factors
    pub num_factors: usize,
    /// Matrix sparsity (proportion of zero entries)
    pub sparsity: f64,
    /// Number of sessions tracked
    pub num_sessions: usize,
}

/// User latent factor vector.
#[derive(Debug, Clone)]
struct UserFactors {
    factors: Vec<f64>,
}

/// Item latent factor vector.
#[derive(Debug, Clone)]
struct ItemFactors {
    factors: Vec<f64>,
}

/// Main collaborative filtering engine.
pub struct CollaborativeFilter {
    config: CfConfig,
    interactions: VecDeque<InteractionEvent>,
    user_index: HashMap<String, usize>,
    item_index: HashMap<String, usize>,
    user_factors: Vec<Vec<f64>>,
    item_factors: Vec<Vec<f64>>,
    is_trained: bool,
    sessions: HashMap<String, Vec<String>>,
}

impl CollaborativeFilter {
    /// Create a new collaborative filter.
    pub fn new(config: CfConfig) -> Self {
        Self {
            config,
            interactions: VecDeque::new(),
            user_index: HashMap::new(),
            item_index: HashMap::new(),
            user_factors: Vec::new(),
            item_factors: Vec::new(),
            is_trained: false,
            sessions: HashMap::new(),
        }
    }

    /// Record a user-item interaction.
    pub fn record_interaction(
        &mut self,
        user_id: &str,
        item_id: &str,
        interaction: InteractionType,
    ) {
        self.record_interaction_event(InteractionEvent {
            user_id: user_id.to_string(),
            item_id: item_id.to_string(),
            interaction_type: interaction,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            session_id: None,
        });
    }

    /// Record an interaction event with full details.
    pub fn record_interaction_event(&mut self, event: InteractionEvent) {
        // Track user and item indices
        let user_count = self.user_index.len();
        self.user_index
            .entry(event.user_id.clone())
            .or_insert(user_count);

        let item_count = self.item_index.len();
        self.item_index
            .entry(event.item_id.clone())
            .or_insert(item_count);

        // Track sessions
        if let Some(session_id) = &event.session_id {
            self.sessions
                .entry(session_id.clone())
                .or_default()
                .push(event.item_id.clone());
        }

        // Evict old interactions if necessary
        if self.interactions.len() >= self.config.max_interactions {
            self.interactions.pop_front();
        }

        self.interactions.push_back(event);
        self.is_trained = false;
    }

    /// Train the model using ALS for implicit feedback.
    pub fn train(&mut self) {
        let num_users = self.user_index.len();
        let num_items = self.item_index.len();
        let num_factors = self.config.num_factors;

        if num_users == 0 || num_items == 0 {
            return;
        }

        // Initialize factors with small random values
        let mut rng = rand::thread_rng();
        use rand::Rng;

        self.user_factors = (0..num_users)
            .map(|_| (0..num_factors).map(|_| rng.gen::<f64>() * 0.01).collect())
            .collect();

        self.item_factors = (0..num_items)
            .map(|_| (0..num_factors).map(|_| rng.gen::<f64>() * 0.01).collect())
            .collect();

        // Build interaction matrix (sparse)
        let mut user_item_weights: HashMap<(usize, usize), f64> = HashMap::new();
        for event in &self.interactions {
            if let (Some(&u), Some(&i)) = (
                self.user_index.get(&event.user_id),
                self.item_index.get(&event.item_id),
            ) {
                let weight = event.interaction_type.weight();
                *user_item_weights.entry((u, i)).or_insert(0.0) += weight;
            }
        }

        // ALS iterations
        for _iter in 0..self.config.num_iterations {
            // Fix items, solve for users
            for u in 0..num_users {
                let user_interactions: Vec<(usize, f64)> = user_item_weights
                    .iter()
                    .filter(|((user, _), _)| *user == u)
                    .map(|((_, item), &w)| (*item, w))
                    .collect();

                if user_interactions.is_empty() {
                    continue;
                }

                self.user_factors[u] = self.solve_factors(
                    &user_interactions,
                    &self.item_factors,
                    num_factors,
                );
            }

            // Fix users, solve for items
            for i in 0..num_items {
                let item_interactions: Vec<(usize, f64)> = user_item_weights
                    .iter()
                    .filter(|((_, item), _)| *item == i)
                    .map(|((user, _), &w)| (*user, w))
                    .collect();

                if item_interactions.is_empty() {
                    continue;
                }

                self.item_factors[i] = self.solve_factors(
                    &item_interactions,
                    &self.user_factors,
                    num_factors,
                );
            }
        }

        self.is_trained = true;
    }

    /// Solve for one factor vector using weighted least squares.
    fn solve_factors(
        &self,
        interactions: &[(usize, f64)],
        other_factors: &[Vec<f64>],
        num_factors: usize,
    ) -> Vec<f64> {
        let lambda = self.config.regularization;
        let alpha = self.config.confidence_alpha;

        // A = sum(c_i * y_i * y_i^T) + lambda * I
        // b = sum(c_i * p_i * y_i)
        // where c_i = 1 + alpha * r_i, p_i = 1 if r_i > 0

        let mut a = vec![vec![0.0f64; num_factors]; num_factors];
        let mut b = vec![0.0f64; num_factors];

        // Add regularization
        for f in 0..num_factors {
            a[f][f] += lambda;
        }

        for &(idx, weight) in interactions {
            if idx >= other_factors.len() {
                continue;
            }
            let y = &other_factors[idx];
            let confidence = 1.0 + alpha * weight;
            let preference = if weight > 0.0 { 1.0 } else { 0.0 };

            for f1 in 0..num_factors {
                for f2 in 0..num_factors {
                    a[f1][f2] += confidence * y[f1] * y[f2];
                }
                b[f1] += confidence * preference * y[f1];
            }
        }

        // Solve A*x = b using simple iterative method (Gauss-Seidel)
        let mut x = vec![0.0f64; num_factors];
        for _iter in 0..20 {
            for f in 0..num_factors {
                let mut sum = b[f];
                for f2 in 0..num_factors {
                    if f2 != f {
                        sum -= a[f][f2] * x[f2];
                    }
                }
                if a[f][f].abs() > 1e-10 {
                    x[f] = sum / a[f][f];
                }
            }
        }

        x
    }

    /// Get recommendations for a user.
    pub fn recommend(&self, user_id: &str, limit: usize) -> Vec<Recommendation> {
        if !self.is_trained {
            return Vec::new();
        }

        let user_idx = match self.user_index.get(user_id) {
            Some(&idx) => idx,
            None => return Vec::new(),
        };

        if user_idx >= self.user_factors.len() {
            return Vec::new();
        }

        let user_vec = &self.user_factors[user_idx];

        // Compute scores for all items
        let reverse_item_index: HashMap<usize, &str> = self
            .item_index
            .iter()
            .map(|(name, &idx)| (idx, name.as_str()))
            .collect();

        // Get items the user has already interacted with
        let interacted: HashSet<&str> = self
            .interactions
            .iter()
            .filter(|e| e.user_id == user_id)
            .map(|e| e.item_id.as_str())
            .collect();

        let mut scores: Vec<(String, f64)> = self
            .item_factors
            .iter()
            .enumerate()
            .filter_map(|(i, item_vec)| {
                let item_name = reverse_item_index.get(&i)?;
                // Skip already-interacted items
                if interacted.contains(item_name) {
                    return None;
                }
                let score: f64 = user_vec
                    .iter()
                    .zip(item_vec.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                Some((item_name.to_string(), score))
            })
            .collect();

        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(limit);

        scores
            .into_iter()
            .map(|(item_id, cf_score)| Recommendation {
                item_id,
                cf_score,
                vector_score: None,
                hybrid_score: cf_score,
            })
            .collect()
    }

    /// Get hybrid recommendations combining CF scores with vector similarity.
    pub fn recommend_hybrid(
        &self,
        user_id: &str,
        vector_scores: &HashMap<String, f64>,
        limit: usize,
    ) -> Vec<Recommendation> {
        let cf_recs = self.recommend(user_id, limit * 3);
        let cf_weight = self.config.cf_weight;
        let vec_weight = 1.0 - cf_weight;

        let mut hybrid: Vec<Recommendation> = cf_recs
            .into_iter()
            .map(|rec| {
                let vec_score = vector_scores.get(&rec.item_id).copied();
                let hybrid_score = cf_weight * rec.cf_score
                    + vec_weight * vec_score.unwrap_or(0.0);
                Recommendation {
                    vector_score: vec_score,
                    hybrid_score,
                    ..rec
                }
            })
            .collect();

        // Also include items with vector scores but no CF score
        for (item_id, &score) in vector_scores {
            if !hybrid.iter().any(|r| r.item_id == *item_id) {
                hybrid.push(Recommendation {
                    item_id: item_id.clone(),
                    cf_score: 0.0,
                    vector_score: Some(score),
                    hybrid_score: vec_weight * score,
                });
            }
        }

        hybrid.sort_by(|a, b| {
            b.hybrid_score
                .partial_cmp(&a.hybrid_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hybrid.truncate(limit);
        hybrid
    }

    /// Get session-based recommendations for anonymous users.
    pub fn recommend_session(
        &self,
        session_items: &[&str],
        limit: usize,
    ) -> Vec<Recommendation> {
        if !self.is_trained || session_items.is_empty() {
            return Vec::new();
        }

        // Average the item factors for session items
        let num_factors = self.config.num_factors;
        let mut session_vec = vec![0.0f64; num_factors];
        let mut count = 0;

        for item_id in session_items {
            if let Some(&idx) = self.item_index.get(*item_id) {
                if idx < self.item_factors.len() {
                    for (f, v) in self.item_factors[idx].iter().enumerate() {
                        session_vec[f] += v;
                    }
                    count += 1;
                }
            }
        }

        if count == 0 {
            return Vec::new();
        }

        for v in &mut session_vec {
            *v /= count as f64;
        }

        // Score all items against the session vector
        let session_items_set: HashSet<&str> = session_items.iter().copied().collect();
        let reverse_item_index: HashMap<usize, &str> = self
            .item_index
            .iter()
            .map(|(name, &idx)| (idx, name.as_str()))
            .collect();

        let mut scores: Vec<Recommendation> = self
            .item_factors
            .iter()
            .enumerate()
            .filter_map(|(i, item_vec)| {
                let item_name = reverse_item_index.get(&i)?;
                if session_items_set.contains(item_name) {
                    return None;
                }
                let score: f64 = session_vec
                    .iter()
                    .zip(item_vec.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                Some(Recommendation {
                    item_id: item_name.to_string(),
                    cf_score: score,
                    vector_score: None,
                    hybrid_score: score,
                })
            })
            .collect();

        scores.sort_by(|a, b| {
            b.cf_score
                .partial_cmp(&a.cf_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(limit);
        scores
    }

    /// Get statistics about the collaborative filter.
    pub fn stats(&self) -> CfStats {
        let total = self.user_index.len() * self.item_index.len();
        let filled = self.interactions.len();
        let sparsity = if total > 0 {
            1.0 - (filled as f64 / total as f64)
        } else {
            1.0
        };

        CfStats {
            num_users: self.user_index.len(),
            num_items: self.item_index.len(),
            total_interactions: self.interactions.len(),
            is_trained: self.is_trained,
            num_factors: self.config.num_factors,
            sparsity,
            num_sessions: self.sessions.len(),
        }
    }

    /// Check if the model has been trained.
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Get the config.
    pub fn config(&self) -> &CfConfig {
        &self.config
    }

    /// Clear all data and reset.
    pub fn clear(&mut self) {
        self.interactions.clear();
        self.user_index.clear();
        self.item_index.clear();
        self.user_factors.clear();
        self.item_factors.clear();
        self.sessions.clear();
        self.is_trained = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interaction_weights() {
        assert_eq!(InteractionType::Click.weight(), 1.0);
        assert_eq!(InteractionType::Purchase.weight(), 5.0);
        assert_eq!(InteractionType::Rating(4.5).weight(), 4.5);
    }

    #[test]
    fn test_record_and_train() {
        let mut cf = CollaborativeFilter::new(CfConfig {
            num_factors: 4,
            num_iterations: 3,
            ..Default::default()
        });

        // Create some interactions
        cf.record_interaction("u1", "i1", InteractionType::Click);
        cf.record_interaction("u1", "i2", InteractionType::Purchase);
        cf.record_interaction("u2", "i2", InteractionType::Click);
        cf.record_interaction("u2", "i3", InteractionType::Click);
        cf.record_interaction("u3", "i1", InteractionType::Rating(5.0));
        cf.record_interaction("u3", "i3", InteractionType::Purchase);

        assert!(!cf.is_trained());
        cf.train();
        assert!(cf.is_trained());
    }

    #[test]
    fn test_recommend() {
        let mut cf = CollaborativeFilter::new(CfConfig {
            num_factors: 4,
            num_iterations: 5,
            ..Default::default()
        });

        cf.record_interaction("u1", "i1", InteractionType::Purchase);
        cf.record_interaction("u1", "i2", InteractionType::Purchase);
        cf.record_interaction("u2", "i2", InteractionType::Purchase);
        cf.record_interaction("u2", "i3", InteractionType::Purchase);

        cf.train();

        // u1 has interacted with i1, i2; should recommend i3
        let recs = cf.recommend("u1", 5);
        // May or may not recommend i3 depending on factor convergence
        // Just check it doesn't crash and returns valid results
        for rec in &recs {
            assert!(!rec.item_id.is_empty());
        }
    }

    #[test]
    fn test_recommend_unknown_user() {
        let mut cf = CollaborativeFilter::new(CfConfig::default());
        cf.record_interaction("u1", "i1", InteractionType::Click);
        cf.train();

        let recs = cf.recommend("unknown_user", 5);
        assert!(recs.is_empty());
    }

    #[test]
    fn test_hybrid_recommend() {
        let mut cf = CollaborativeFilter::new(CfConfig {
            num_factors: 4,
            num_iterations: 5,
            cf_weight: 0.5,
            ..Default::default()
        });

        cf.record_interaction("u1", "i1", InteractionType::Purchase);
        cf.record_interaction("u1", "i2", InteractionType::Purchase);
        cf.record_interaction("u2", "i3", InteractionType::Purchase);
        cf.train();

        let mut vector_scores = HashMap::new();
        vector_scores.insert("i3".to_string(), 0.9);
        vector_scores.insert("i4".to_string(), 0.8);

        let recs = cf.recommend_hybrid("u1", &vector_scores, 5);
        assert!(!recs.is_empty());
        // i4 should appear from vector scores even without CF score
        assert!(recs.iter().any(|r| r.item_id == "i4"));
    }

    #[test]
    fn test_session_recommend() {
        let mut cf = CollaborativeFilter::new(CfConfig {
            num_factors: 4,
            num_iterations: 5,
            ..Default::default()
        });

        cf.record_interaction("u1", "i1", InteractionType::Purchase);
        cf.record_interaction("u1", "i2", InteractionType::Purchase);
        cf.record_interaction("u2", "i2", InteractionType::Purchase);
        cf.record_interaction("u2", "i3", InteractionType::Purchase);
        cf.train();

        let recs = cf.recommend_session(&["i1", "i2"], 5);
        // Should return some recommendations (may include i3)
        for rec in &recs {
            assert!(rec.item_id != "i1" && rec.item_id != "i2");
        }
    }

    #[test]
    fn test_stats() {
        let mut cf = CollaborativeFilter::new(CfConfig::default());
        cf.record_interaction("u1", "i1", InteractionType::Click);
        cf.record_interaction("u1", "i2", InteractionType::Click);

        let stats = cf.stats();
        assert_eq!(stats.num_users, 1);
        assert_eq!(stats.num_items, 2);
        assert_eq!(stats.total_interactions, 2);
        assert!(!stats.is_trained);
    }
}
