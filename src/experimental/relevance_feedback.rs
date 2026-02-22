#![allow(clippy::unwrap_used)]
//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Closed-Loop Relevance Feedback
//!
//! Captures implicit (click-through, dwell time) and explicit (thumbs up/down) user signals
//! to improve search quality through online parameter adjustment and A/B testing.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::experimental::relevance_feedback::*;
//!
//! let mut engine = FeedbackEngine::new(FeedbackConfig::default());
//! engine.record(FeedbackEvent::explicit("q1", "v1", FeedbackSignal::ThumbsUp));
//! let adjustments = engine.compute_adjustments("docs");
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::SystemTime;

/// Type of feedback signal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeedbackSignal {
    /// User explicitly marked result as relevant
    ThumbsUp,
    /// User explicitly marked result as not relevant
    ThumbsDown,
    /// User clicked on a result
    Click,
    /// User spent time viewing a result (dwell time in milliseconds)
    Dwell { duration_ms: u64 },
    /// User skipped this result (implicit negative)
    Skip,
    /// Numeric rating (e.g., 1-5 stars)
    Rating(f64),
}

impl FeedbackSignal {
    /// Convert signal to a relevance score in [-1.0, 1.0].
    pub fn relevance_score(&self) -> f64 {
        match self {
            Self::ThumbsUp => 1.0,
            Self::ThumbsDown => -1.0,
            Self::Click => 0.5,
            Self::Dwell { duration_ms } => {
                // Dwell time > 5s is positive, < 1s is negative
                let secs = *duration_ms as f64 / 1000.0;
                ((secs - 3.0) / 5.0).clamp(-1.0, 1.0)
            }
            Self::Skip => -0.3,
            Self::Rating(r) => (r - 3.0) / 2.0, // Maps 1-5 to [-1, 1]
        }
    }
}

/// A feedback event from a user interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackEvent {
    /// Query identifier (hash or ID linking to the original search)
    pub query_id: String,
    /// Vector ID that received feedback
    pub vector_id: String,
    /// Collection name
    pub collection: String,
    /// The feedback signal
    pub signal: FeedbackSignal,
    /// Position of the result in the original result list (0-indexed)
    pub position: Option<usize>,
    /// Timestamp of the feedback
    pub timestamp: u64,
    /// Optional session/user identifier
    pub session_id: Option<String>,
    /// Optional A/B test variant
    pub ab_variant: Option<String>,
}

impl FeedbackEvent {
    /// Create an explicit feedback event.
    pub fn explicit(
        query_id: impl Into<String>,
        vector_id: impl Into<String>,
        collection: impl Into<String>,
        signal: FeedbackSignal,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            vector_id: vector_id.into(),
            collection: collection.into(),
            signal,
            position: None,
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            session_id: None,
            ab_variant: None,
        }
    }

    /// Set the result position.
    #[must_use]
    pub fn with_position(mut self, position: usize) -> Self {
        self.position = Some(position);
        self
    }

    /// Set the session ID.
    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set the A/B test variant.
    #[must_use]
    pub fn with_ab_variant(mut self, variant: impl Into<String>) -> Self {
        self.ab_variant = Some(variant.into());
        self
    }
}

/// Configuration for the feedback engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackConfig {
    /// Learning rate for online gradient descent (default: 0.01)
    pub learning_rate: f64,
    /// Maximum number of feedback events to retain per collection
    pub max_events_per_collection: usize,
    /// Minimum events required before computing adjustments
    pub min_events_for_adjustment: usize,
    /// Decay factor for older feedback (0.0 = ignore old, 1.0 = equal weight)
    pub temporal_decay: f64,
    /// Enable A/B testing
    pub ab_testing_enabled: bool,
    /// Rollback threshold: if avg relevance drops below this, revert adjustments
    pub rollback_threshold: f64,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_events_per_collection: 100_000,
            min_events_for_adjustment: 10,
            temporal_decay: 0.95,
            ab_testing_enabled: false,
            rollback_threshold: -0.2,
        }
    }
}

/// Computed parameter adjustments based on feedback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterAdjustment {
    /// Suggested ef_search adjustment factor (e.g., 1.2 = increase by 20%)
    pub ef_search_factor: f64,
    /// Suggested reranking weight adjustment
    pub rerank_weight: f64,
    /// Average relevance score from feedback
    pub avg_relevance: f64,
    /// Number of feedback events used
    pub event_count: usize,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

/// A/B test variant configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbTestVariant {
    /// Variant name (e.g., "control", "treatment_a")
    pub name: String,
    /// Variant-specific parameter adjustments
    pub adjustments: ParameterAdjustment,
    /// Total events for this variant
    pub event_count: usize,
    /// Average relevance for this variant
    pub avg_relevance: f64,
}

/// A/B test definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbTest {
    /// Test name
    pub name: String,
    /// Collection this test applies to
    pub collection: String,
    /// Variants with their results
    pub variants: Vec<AbTestVariant>,
    /// Whether the test is active
    pub active: bool,
    /// Start timestamp
    pub started_at: u64,
}

impl AbTest {
    /// Create a new A/B test.
    pub fn new(name: impl Into<String>, collection: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            collection: collection.into(),
            variants: Vec::new(),
            active: true,
            started_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Add a variant to the test.
    #[must_use]
    pub fn with_variant(mut self, name: impl Into<String>) -> Self {
        self.variants.push(AbTestVariant {
            name: name.into(),
            adjustments: ParameterAdjustment {
                ef_search_factor: 1.0,
                rerank_weight: 1.0,
                avg_relevance: 0.0,
                event_count: 0,
                confidence: 0.0,
            },
            event_count: 0,
            avg_relevance: 0.0,
        });
        self
    }

    /// Get the winning variant (highest avg relevance with sufficient data).
    pub fn winner(&self) -> Option<&AbTestVariant> {
        self.variants
            .iter()
            .filter(|v| v.event_count >= 10)
            .max_by(|a, b| {
                a.avg_relevance
                    .partial_cmp(&b.avg_relevance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// Feedback statistics for a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackStats {
    /// Total feedback events
    pub total_events: usize,
    /// Positive signals count
    pub positive_count: usize,
    /// Negative signals count
    pub negative_count: usize,
    /// Average relevance score
    pub avg_relevance: f64,
    /// Click-through rate (clicks / total)
    pub click_through_rate: f64,
    /// Active A/B tests
    pub active_ab_tests: usize,
}

/// Engine for processing feedback and computing adjustments.
pub struct FeedbackEngine {
    config: FeedbackConfig,
    events: HashMap<String, VecDeque<FeedbackEvent>>,
    ab_tests: Vec<AbTest>,
    previous_adjustments: HashMap<String, ParameterAdjustment>,
}

impl FeedbackEngine {
    /// Create a new feedback engine.
    pub fn new(config: FeedbackConfig) -> Self {
        Self {
            config,
            events: HashMap::new(),
            ab_tests: Vec::new(),
            previous_adjustments: HashMap::new(),
        }
    }

    /// Record a feedback event.
    pub fn record(&mut self, event: FeedbackEvent) {
        let queue = self
            .events
            .entry(event.collection.clone())
            .or_insert_with(VecDeque::new);

        if queue.len() >= self.config.max_events_per_collection {
            queue.pop_front();
        }
        queue.push_back(event);
    }

    /// Compute parameter adjustments for a collection using online gradient descent.
    pub fn compute_adjustments(&self, collection: &str) -> Option<ParameterAdjustment> {
        let events = self.events.get(collection)?;
        if events.len() < self.config.min_events_for_adjustment {
            return None;
        }

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut weighted_relevance = 0.0;
        let mut total_weight = 0.0;
        let mut positive_count = 0usize;

        for event in events.iter() {
            let age_secs = now.saturating_sub(event.timestamp) as f64;
            let weight = self.config.temporal_decay.powf(age_secs / 3600.0);
            let score = event.signal.relevance_score();

            weighted_relevance += score * weight;
            total_weight += weight;

            if score > 0.0 {
                positive_count += 1;
            }
        }

        if total_weight < 1e-10 {
            return None;
        }

        let avg_relevance = weighted_relevance / total_weight;
        let positive_ratio = positive_count as f64 / events.len() as f64;

        // Online gradient descent: adjust ef_search based on feedback
        // If results are poorly rated, increase ef_search for better recall
        let ef_adjustment = 1.0 + self.config.learning_rate * (1.0 - avg_relevance);
        let ef_search_factor = ef_adjustment.clamp(0.5, 2.0);

        let confidence = (events.len() as f64 / 100.0).min(1.0);

        Some(ParameterAdjustment {
            ef_search_factor,
            rerank_weight: positive_ratio,
            avg_relevance,
            event_count: events.len(),
            confidence,
        })
    }

    /// Check if adjustments should be rolled back based on guardrails.
    pub fn should_rollback(&self, collection: &str) -> bool {
        if let Some(adj) = self.compute_adjustments(collection) {
            adj.avg_relevance < self.config.rollback_threshold
        } else {
            false
        }
    }

    /// Create an A/B test for a collection.
    pub fn create_ab_test(&mut self, test: AbTest) {
        self.ab_tests.push(test);
    }

    /// List active A/B tests.
    pub fn active_ab_tests(&self) -> Vec<&AbTest> {
        self.ab_tests.iter().filter(|t| t.active).collect()
    }

    /// Get feedback statistics for a collection.
    pub fn stats(&self, collection: &str) -> FeedbackStats {
        let events = self.events.get(collection);
        let total = events.map_or(0, |e| e.len());

        let (pos, neg, click, total_rel) = events.map_or((0, 0, 0, 0.0), |evts| {
            let mut p = 0usize;
            let mut n = 0usize;
            let mut c = 0usize;
            let mut r = 0.0f64;
            for e in evts {
                let score = e.signal.relevance_score();
                r += score;
                if score > 0.0 { p += 1; }
                if score < 0.0 { n += 1; }
                if matches!(e.signal, FeedbackSignal::Click) { c += 1; }
            }
            (p, n, c, r)
        });

        FeedbackStats {
            total_events: total,
            positive_count: pos,
            negative_count: neg,
            avg_relevance: if total > 0 { total_rel / total as f64 } else { 0.0 },
            click_through_rate: if total > 0 { click as f64 / total as f64 } else { 0.0 },
            active_ab_tests: self.ab_tests.iter().filter(|t| t.active).count(),
        }
    }

    /// Get the config.
    pub fn config(&self) -> &FeedbackConfig {
        &self.config
    }

    /// Clear all feedback for a collection.
    pub fn clear(&mut self, collection: &str) {
        self.events.remove(collection);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_signal_scores() {
        assert_eq!(FeedbackSignal::ThumbsUp.relevance_score(), 1.0);
        assert_eq!(FeedbackSignal::ThumbsDown.relevance_score(), -1.0);
        assert_eq!(FeedbackSignal::Click.relevance_score(), 0.5);
        assert!(FeedbackSignal::Dwell { duration_ms: 10_000 }.relevance_score() > 0.0);
        assert!(FeedbackSignal::Dwell { duration_ms: 500 }.relevance_score() < 0.0);
    }

    #[test]
    fn test_record_and_compute() {
        let mut engine = FeedbackEngine::new(FeedbackConfig {
            min_events_for_adjustment: 2,
            ..Default::default()
        });

        for i in 0..10 {
            engine.record(FeedbackEvent::explicit(
                format!("q{}", i),
                format!("v{}", i),
                "docs",
                FeedbackSignal::ThumbsUp,
            ));
        }

        let adj = engine.compute_adjustments("docs");
        assert!(adj.is_some());
        let adj = adj.unwrap();
        assert!(adj.avg_relevance > 0.0);
        assert_eq!(adj.event_count, 10);
    }

    #[test]
    fn test_insufficient_events() {
        let engine = FeedbackEngine::new(FeedbackConfig::default());
        assert!(engine.compute_adjustments("docs").is_none());
    }

    #[test]
    fn test_ab_test_winner() {
        let mut test = AbTest::new("test1", "docs")
            .with_variant("control")
            .with_variant("treatment");

        // Need at least 10 events for winner
        assert!(test.winner().is_none());

        test.variants[0].event_count = 100;
        test.variants[0].avg_relevance = 0.5;
        test.variants[1].event_count = 100;
        test.variants[1].avg_relevance = 0.8;

        let winner = test.winner().unwrap();
        assert_eq!(winner.name, "treatment");
    }

    #[test]
    fn test_feedback_stats() {
        let mut engine = FeedbackEngine::new(FeedbackConfig {
            min_events_for_adjustment: 1,
            ..Default::default()
        });

        engine.record(FeedbackEvent::explicit("q1", "v1", "docs", FeedbackSignal::ThumbsUp));
        engine.record(FeedbackEvent::explicit("q2", "v2", "docs", FeedbackSignal::ThumbsDown));
        engine.record(FeedbackEvent::explicit("q3", "v3", "docs", FeedbackSignal::Click));

        let stats = engine.stats("docs");
        assert_eq!(stats.total_events, 3);
        assert_eq!(stats.positive_count, 2); // thumbs up + click
        assert_eq!(stats.negative_count, 1); // thumbs down
    }

    #[test]
    fn test_rollback_guardrail() {
        let mut engine = FeedbackEngine::new(FeedbackConfig {
            min_events_for_adjustment: 2,
            rollback_threshold: -0.2,
            ..Default::default()
        });

        // All negative feedback should trigger rollback
        for i in 0..10 {
            engine.record(FeedbackEvent::explicit(
                format!("q{}", i),
                format!("v{}", i),
                "docs",
                FeedbackSignal::ThumbsDown,
            ));
        }

        assert!(engine.should_rollback("docs"));
    }
}
