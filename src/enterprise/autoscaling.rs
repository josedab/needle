//! Intelligent Auto-Scaling Module
//!
//! Provides predictive load balancing and automatic scaling for vector databases
//! based on real-time metrics, historical patterns, and ML-based predictions.
//!
//! # Features
//!
//! - **Predictive Scaling**: ML-based load prediction using time series analysis
//! - **Auto-Scaling Policies**: Configurable scale-up/down triggers
//! - **Hot/Cold Tiering**: Automatic data placement based on access patterns
//! - **Load Balancing**: Query routing based on shard health and capacity
//! - **Resource Optimization**: Memory and CPU-aware scaling decisions
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::autoscaling::{AutoScaler, ScalingConfig, ScalingPolicy};
//!
//! // Create auto-scaler
//! let config = ScalingConfig::default()
//!     .with_min_shards(2)
//!     .with_max_shards(16)
//!     .with_target_utilization(0.7);
//!
//! let mut scaler = AutoScaler::new(config);
//!
//! // Record metrics
//! scaler.record_query_latency(15.0);
//! scaler.record_cpu_usage(0.8);
//!
//! // Get scaling recommendation
//! let action = scaler.recommend()?;
//! match action {
//!     ScalingAction::ScaleUp(n) => println!("Add {} shards", n),
//!     ScalingAction::ScaleDown(n) => println!("Remove {} shards", n),
//!     ScalingAction::Rebalance => println!("Rebalance shards"),
//!     ScalingAction::None => println!("No action needed"),
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::error::{NeedleError, Result};
use crate::shard::ShardId;

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    /// Minimum number of shards
    pub min_shards: usize,
    /// Maximum number of shards
    pub max_shards: usize,
    /// Target CPU utilization (0.0 - 1.0)
    pub target_cpu_utilization: f64,
    /// Target memory utilization (0.0 - 1.0)
    pub target_memory_utilization: f64,
    /// Target query latency in milliseconds
    pub target_latency_ms: f64,
    /// Scale-up threshold multiplier
    pub scale_up_threshold: f64,
    /// Scale-down threshold multiplier
    pub scale_down_threshold: f64,
    /// Cooldown period between scaling actions (seconds)
    pub cooldown_seconds: u64,
    /// Number of data points for predictions
    pub prediction_window: usize,
    /// Enable predictive scaling
    pub enable_prediction: bool,
    /// Scale-up increment (shards to add)
    pub scale_up_increment: usize,
    /// Scale-down increment (shards to remove)
    pub scale_down_increment: usize,
    /// Metrics retention period (seconds)
    pub metrics_retention_seconds: u64,
    /// Enable hot/cold tiering
    pub enable_tiering: bool,
    /// Hot tier threshold (access count per hour)
    pub hot_tier_threshold: usize,
    /// Cold tier threshold (hours since last access)
    pub cold_tier_hours: u64,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            min_shards: 1,
            max_shards: 32,
            target_cpu_utilization: 0.7,
            target_memory_utilization: 0.8,
            target_latency_ms: 50.0,
            scale_up_threshold: 1.3,
            scale_down_threshold: 0.5,
            cooldown_seconds: 300,
            prediction_window: 60,
            enable_prediction: true,
            scale_up_increment: 1,
            scale_down_increment: 1,
            metrics_retention_seconds: 3600,
            enable_tiering: true,
            hot_tier_threshold: 100,
            cold_tier_hours: 168, // 1 week
        }
    }
}

impl ScalingConfig {
    /// Set minimum shards
    pub fn with_min_shards(mut self, min: usize) -> Self {
        self.min_shards = min;
        self
    }

    /// Set maximum shards
    pub fn with_max_shards(mut self, max: usize) -> Self {
        self.max_shards = max;
        self
    }

    /// Set target CPU utilization
    pub fn with_target_cpu_utilization(mut self, util: f64) -> Self {
        self.target_cpu_utilization = util.clamp(0.0, 1.0);
        self
    }

    /// Set target latency
    pub fn with_target_latency_ms(mut self, latency: f64) -> Self {
        self.target_latency_ms = latency;
        self
    }

    /// Set cooldown period
    pub fn with_cooldown_seconds(mut self, seconds: u64) -> Self {
        self.cooldown_seconds = seconds;
        self
    }

    /// Enable or disable prediction
    pub fn with_prediction(mut self, enable: bool) -> Self {
        self.enable_prediction = enable;
        self
    }
}

/// Scaling action recommendation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalingAction {
    /// No scaling needed
    None,
    /// Scale up by adding shards
    ScaleUp(usize),
    /// Scale down by removing shards
    ScaleDown(usize),
    /// Rebalance existing shards
    Rebalance,
    /// Move data to hot tier
    PromoteToHot(Vec<String>),
    /// Move data to cold tier
    DemoteToCold(Vec<String>),
    /// Emergency scale (immediate action needed)
    EmergencyScale(usize),
}

/// Reason for scaling action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingReason {
    HighCpuUtilization,
    HighMemoryUtilization,
    HighLatency,
    LowUtilization,
    PredictedLoadIncrease,
    PredictedLoadDecrease,
    UnbalancedShards,
    HotDataDetected,
    ColdDataDetected,
    ScheduledScaling,
    Manual,
}

/// Metrics data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    /// Timestamp
    pub timestamp: u64,
    /// CPU utilization (0.0 - 1.0)
    pub cpu_utilization: f64,
    /// Memory utilization (0.0 - 1.0)
    pub memory_utilization: f64,
    /// Query latency in milliseconds
    pub query_latency_ms: f64,
    /// Queries per second
    pub qps: f64,
    /// Active connections
    pub connections: usize,
    /// Number of vectors
    pub vector_count: usize,
    /// Shard count
    pub shard_count: usize,
}

impl MetricPoint {
    pub fn new() -> Self {
        Self {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time before UNIX epoch")
                .as_secs(),
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            query_latency_ms: 0.0,
            qps: 0.0,
            connections: 0,
            vector_count: 0,
            shard_count: 1,
        }
    }
}

impl Default for MetricPoint {
    fn default() -> Self {
        Self::new()
    }
}

/// Shard metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardMetrics {
    /// Shard ID
    pub shard_id: ShardId,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory utilization
    pub memory_utilization: f64,
    /// Vector count
    pub vector_count: usize,
    /// Queries served
    pub query_count: u64,
    /// Average latency
    pub avg_latency_ms: f64,
    /// Health score (0-100)
    pub health_score: u32,
    /// Last updated
    pub last_updated: u64,
}

/// Scaling decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDecision {
    /// Recommended action
    pub action: ScalingAction,
    /// Primary reason
    pub reason: ScalingReason,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Additional context
    pub context: String,
    /// Timestamp
    pub timestamp: u64,
    /// Current metrics snapshot
    pub current_metrics: MetricPoint,
    /// Predicted metrics (if available)
    pub predicted_metrics: Option<MetricPoint>,
}

/// Time series predictor for load forecasting
pub struct LoadPredictor {
    /// Historical data points
    history: VecDeque<MetricPoint>,
    /// Maximum history size
    max_history: usize,
    /// Seasonality period (e.g., 24 hours = 86400 seconds)
    seasonality_period: u64,
}

impl LoadPredictor {
    pub fn new(max_history: usize, seasonality_period: u64) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
            seasonality_period,
        }
    }

    /// Add a data point
    pub fn add_point(&mut self, point: MetricPoint) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(point);
    }

    /// Predict next value using exponential smoothing with seasonality
    pub fn predict(&self, horizon_seconds: u64) -> Option<MetricPoint> {
        if self.history.len() < 3 {
            return None;
        }

        // Simple exponential smoothing for each metric
        let alpha = 0.3; // Smoothing factor
        let beta = 0.1; // Trend factor

        let mut smoothed_cpu = self.history.front()?.cpu_utilization;
        let mut trend_cpu = 0.0;
        let mut smoothed_mem = self.history.front()?.memory_utilization;
        let mut trend_mem = 0.0;
        let mut smoothed_lat = self.history.front()?.query_latency_ms;
        let mut trend_lat = 0.0;
        let mut smoothed_qps = self.history.front()?.qps;
        let mut trend_qps = 0.0;

        for point in self.history.iter().skip(1) {
            // CPU
            let prev_smoothed = smoothed_cpu;
            smoothed_cpu =
                alpha * point.cpu_utilization + (1.0 - alpha) * (smoothed_cpu + trend_cpu);
            trend_cpu = beta * (smoothed_cpu - prev_smoothed) + (1.0 - beta) * trend_cpu;

            // Memory
            let prev_smoothed = smoothed_mem;
            smoothed_mem =
                alpha * point.memory_utilization + (1.0 - alpha) * (smoothed_mem + trend_mem);
            trend_mem = beta * (smoothed_mem - prev_smoothed) + (1.0 - beta) * trend_mem;

            // Latency
            let prev_smoothed = smoothed_lat;
            smoothed_lat =
                alpha * point.query_latency_ms + (1.0 - alpha) * (smoothed_lat + trend_lat);
            trend_lat = beta * (smoothed_lat - prev_smoothed) + (1.0 - beta) * trend_lat;

            // QPS
            let prev_smoothed = smoothed_qps;
            smoothed_qps = alpha * point.qps + (1.0 - alpha) * (smoothed_qps + trend_qps);
            trend_qps = beta * (smoothed_qps - prev_smoothed) + (1.0 - beta) * trend_qps;
        }

        // Project forward
        let steps = (horizon_seconds / 60).max(1) as f64; // Assume 1-minute intervals

        let predicted = MetricPoint {
            timestamp: self.history.back()?.timestamp + horizon_seconds,
            cpu_utilization: (smoothed_cpu + steps * trend_cpu).clamp(0.0, 1.0),
            memory_utilization: (smoothed_mem + steps * trend_mem).clamp(0.0, 1.0),
            query_latency_ms: (smoothed_lat + steps * trend_lat).max(0.0),
            qps: (smoothed_qps + steps * trend_qps).max(0.0),
            connections: self.history.back()?.connections,
            vector_count: self.history.back()?.vector_count,
            shard_count: self.history.back()?.shard_count,
        };

        Some(predicted)
    }

    /// Detect seasonality patterns
    pub fn detect_seasonality(&self) -> Option<SeasonalityPattern> {
        if self.history.len() < 2 * self.seasonality_period as usize / 60 {
            return None;
        }

        // Compute average by time-of-day
        let mut hourly_avg: HashMap<u64, (f64, usize)> = HashMap::new();

        for point in &self.history {
            let hour = (point.timestamp % 86400) / 3600;
            let entry = hourly_avg.entry(hour).or_insert((0.0, 0));
            entry.0 += point.qps;
            entry.1 += 1;
        }

        let pattern: Vec<(u64, f64)> = hourly_avg
            .into_iter()
            .map(|(hour, (sum, count))| (hour, sum / count as f64))
            .collect();

        if pattern.is_empty() {
            return None;
        }

        let max_hour = pattern
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;
        let min_hour = pattern
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;

        Some(SeasonalityPattern {
            period_hours: 24,
            peak_hour: max_hour.0 as u8,
            peak_load: max_hour.1,
            trough_hour: min_hour.0 as u8,
            trough_load: min_hour.1,
        })
    }
}

/// Detected seasonality pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityPattern {
    /// Period in hours
    pub period_hours: u32,
    /// Hour of peak load (0-23)
    pub peak_hour: u8,
    /// Peak load value
    pub peak_load: f64,
    /// Hour of lowest load
    pub trough_hour: u8,
    /// Trough load value
    pub trough_load: f64,
}

/// Access pattern tracker for hot/cold tiering
pub struct AccessTracker {
    /// Vector access counts
    access_counts: HashMap<String, AccessInfo>,
    /// Time window for hot detection (seconds)
    hot_window: u64,
    /// Retention period for tracking
    retention_period: u64,
}

/// Access information for a vector
#[derive(Debug, Clone, Default)]
struct AccessInfo {
    /// Recent access timestamps
    accesses: VecDeque<u64>,
    /// Total access count
    total_count: u64,
    /// Last access timestamp
    last_access: u64,
}

impl AccessTracker {
    pub fn new(hot_window: u64, retention_period: u64) -> Self {
        Self {
            access_counts: HashMap::new(),
            hot_window,
            retention_period,
        }
    }

    /// Record an access
    pub fn record_access(&mut self, vector_id: &str) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before UNIX epoch")
            .as_secs();

        let info = self.access_counts.entry(vector_id.to_string()).or_default();
        info.accesses.push_back(now);
        info.total_count += 1;
        info.last_access = now;

        // Clean old accesses
        while let Some(&ts) = info.accesses.front() {
            if now - ts > self.retention_period {
                info.accesses.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get hot vectors (accessed frequently in recent window)
    pub fn get_hot_vectors(&self, threshold: usize) -> Vec<String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before UNIX epoch")
            .as_secs();

        self.access_counts
            .iter()
            .filter_map(|(id, info)| {
                let recent_count = info
                    .accesses
                    .iter()
                    .filter(|&&ts| now - ts < self.hot_window)
                    .count();
                if recent_count >= threshold {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get cold vectors (not accessed for a long time)
    pub fn get_cold_vectors(&self, cold_hours: u64) -> Vec<String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before UNIX epoch")
            .as_secs();
        let cold_threshold = cold_hours * 3600;

        self.access_counts
            .iter()
            .filter_map(|(id, info)| {
                if now - info.last_access > cold_threshold {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Clean up old tracking data
    pub fn cleanup(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before UNIX epoch")
            .as_secs();

        self.access_counts
            .retain(|_, info| now - info.last_access < self.retention_period);
    }
}

/// Main auto-scaler
pub struct AutoScaler {
    /// Configuration
    config: ScalingConfig,
    /// Load predictor
    predictor: LoadPredictor,
    /// Access tracker for tiering
    access_tracker: AccessTracker,
    /// Current metrics
    current_metrics: MetricPoint,
    /// Shard metrics
    shard_metrics: HashMap<ShardId, ShardMetrics>,
    /// Last scaling action time
    last_scaling_time: Option<Instant>,
    /// Scaling history
    scaling_history: VecDeque<ScalingDecision>,
    /// Max history entries
    max_history: usize,
    /// Scheduled scaling actions
    scheduled_actions: Vec<ScheduledScaling>,
}

/// Scheduled scaling action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledScaling {
    /// Action to take
    pub action: ScalingAction,
    /// Scheduled time (Unix timestamp)
    pub scheduled_time: u64,
    /// Repeat pattern (cron-like, optional)
    pub repeat: Option<String>,
    /// Is enabled
    pub enabled: bool,
    /// Description
    pub description: String,
}

impl AutoScaler {
    /// Create a new auto-scaler
    pub fn new(config: ScalingConfig) -> Self {
        let prediction_window = config.prediction_window;
        let hot_window = 3600; // 1 hour
        let retention = config.metrics_retention_seconds;

        Self {
            config,
            predictor: LoadPredictor::new(prediction_window, 86400),
            access_tracker: AccessTracker::new(hot_window, retention),
            current_metrics: MetricPoint::new(),
            shard_metrics: HashMap::new(),
            last_scaling_time: None,
            scaling_history: VecDeque::with_capacity(100),
            max_history: 100,
            scheduled_actions: Vec::new(),
        }
    }

    /// Record current metrics
    pub fn record_metrics(&mut self, metrics: MetricPoint) {
        self.current_metrics = metrics.clone();
        self.predictor.add_point(metrics);
    }

    /// Record query latency
    pub fn record_query_latency(&mut self, latency_ms: f64) {
        self.current_metrics.query_latency_ms = latency_ms;
    }

    /// Record CPU usage
    pub fn record_cpu_usage(&mut self, utilization: f64) {
        self.current_metrics.cpu_utilization = utilization.clamp(0.0, 1.0);
    }

    /// Record memory usage
    pub fn record_memory_usage(&mut self, utilization: f64) {
        self.current_metrics.memory_utilization = utilization.clamp(0.0, 1.0);
    }

    /// Record QPS
    pub fn record_qps(&mut self, qps: f64) {
        self.current_metrics.qps = qps;
    }

    /// Record vector access for tiering
    pub fn record_access(&mut self, vector_id: &str) {
        self.access_tracker.record_access(vector_id);
    }

    /// Update shard metrics
    pub fn update_shard_metrics(&mut self, shard_id: ShardId, metrics: ShardMetrics) {
        self.shard_metrics.insert(shard_id, metrics);
    }

    /// Get scaling recommendation
    pub fn recommend(&self) -> Result<ScalingDecision> {
        // Check cooldown
        if let Some(last_time) = self.last_scaling_time {
            if last_time.elapsed().as_secs() < self.config.cooldown_seconds {
                return Ok(ScalingDecision {
                    action: ScalingAction::None,
                    reason: ScalingReason::Manual, // Cooldown active
                    confidence: 1.0,
                    context: "Cooldown period active".to_string(),
                    timestamp: Self::now(),
                    current_metrics: self.current_metrics.clone(),
                    predicted_metrics: None,
                });
            }
        }

        // Check for scheduled scaling
        if let Some(scheduled) = self.check_scheduled_actions() {
            return Ok(scheduled);
        }

        // Analyze current metrics
        let current_analysis = self.analyze_current_metrics();

        // Get prediction if enabled
        let prediction = if self.config.enable_prediction {
            self.predictor.predict(300) // 5 minutes ahead
        } else {
            None
        };

        // Combine current and predicted analysis
        let decision = self.make_decision(current_analysis, prediction.as_ref());

        Ok(decision)
    }

    /// Acknowledge that scaling action was taken
    pub fn acknowledge_scaling(&mut self, decision: ScalingDecision) {
        self.last_scaling_time = Some(Instant::now());
        if self.scaling_history.len() >= self.max_history {
            self.scaling_history.pop_front();
        }
        self.scaling_history.push_back(decision);
    }

    /// Check for due scheduled actions
    fn check_scheduled_actions(&self) -> Option<ScalingDecision> {
        let now = Self::now();

        for scheduled in &self.scheduled_actions {
            if scheduled.enabled && scheduled.scheduled_time <= now {
                return Some(ScalingDecision {
                    action: scheduled.action.clone(),
                    reason: ScalingReason::ScheduledScaling,
                    confidence: 1.0,
                    context: scheduled.description.clone(),
                    timestamp: now,
                    current_metrics: self.current_metrics.clone(),
                    predicted_metrics: None,
                });
            }
        }

        None
    }

    /// Analyze current metrics
    fn analyze_current_metrics(&self) -> MetricAnalysis {
        let cpu_ratio = self.current_metrics.cpu_utilization / self.config.target_cpu_utilization;
        let mem_ratio =
            self.current_metrics.memory_utilization / self.config.target_memory_utilization;
        let lat_ratio = self.current_metrics.query_latency_ms / self.config.target_latency_ms;

        MetricAnalysis {
            cpu_pressure: cpu_ratio,
            memory_pressure: mem_ratio,
            latency_pressure: lat_ratio,
            needs_scale_up: cpu_ratio > self.config.scale_up_threshold
                || mem_ratio > self.config.scale_up_threshold
                || lat_ratio > self.config.scale_up_threshold,
            needs_scale_down: cpu_ratio < self.config.scale_down_threshold
                && mem_ratio < self.config.scale_down_threshold
                && lat_ratio < self.config.scale_down_threshold,
            is_balanced: self.check_shard_balance(),
        }
    }

    /// Check if shards are balanced
    fn check_shard_balance(&self) -> bool {
        if self.shard_metrics.len() < 2 {
            return true;
        }

        let loads: Vec<f64> = self
            .shard_metrics
            .values()
            .map(|m| m.cpu_utilization * 0.5 + (m.vector_count as f64 / 1_000_000.0) * 0.5)
            .collect();

        let avg: f64 = loads.iter().sum::<f64>() / loads.len() as f64;
        let max_deviation = loads.iter().map(|l| (l - avg).abs()).fold(0.0, f64::max);

        // Consider balanced if max deviation is less than 20%
        max_deviation < 0.2
    }

    /// Make scaling decision based on analysis
    fn make_decision(
        &self,
        current: MetricAnalysis,
        predicted: Option<&MetricPoint>,
    ) -> ScalingDecision {
        let now = Self::now();
        let current_shards = self.current_metrics.shard_count;

        // Check for emergency conditions
        if (self.current_metrics.cpu_utilization > 0.95
            || self.current_metrics.memory_utilization > 0.95)
            && current_shards < self.config.max_shards
        {
            return ScalingDecision {
                action: ScalingAction::EmergencyScale(2),
                reason: if self.current_metrics.cpu_utilization > 0.95 {
                    ScalingReason::HighCpuUtilization
                } else {
                    ScalingReason::HighMemoryUtilization
                },
                confidence: 0.95,
                context: "Emergency scaling due to critical resource pressure".to_string(),
                timestamp: now,
                current_metrics: self.current_metrics.clone(),
                predicted_metrics: predicted.cloned(),
            };
        }

        // Check for predicted load increase
        if let Some(pred) = predicted {
            if pred.cpu_utilization
                > self.config.target_cpu_utilization * self.config.scale_up_threshold
                && current_shards < self.config.max_shards
            {
                return ScalingDecision {
                    action: ScalingAction::ScaleUp(self.config.scale_up_increment),
                    reason: ScalingReason::PredictedLoadIncrease,
                    confidence: 0.8,
                    context: format!(
                        "Predicted CPU utilization: {:.1}%",
                        pred.cpu_utilization * 100.0
                    ),
                    timestamp: now,
                    current_metrics: self.current_metrics.clone(),
                    predicted_metrics: Some(pred.clone()),
                };
            }
        }

        // Scale up if needed
        if current.needs_scale_up && current_shards < self.config.max_shards {
            let (reason, context) = if current.latency_pressure > current.cpu_pressure
                && current.latency_pressure > current.memory_pressure
            {
                (
                    ScalingReason::HighLatency,
                    format!(
                        "Query latency {:.1}ms exceeds target {:.1}ms",
                        self.current_metrics.query_latency_ms, self.config.target_latency_ms
                    ),
                )
            } else if current.cpu_pressure > current.memory_pressure {
                (
                    ScalingReason::HighCpuUtilization,
                    format!(
                        "CPU utilization {:.1}% exceeds target {:.1}%",
                        self.current_metrics.cpu_utilization * 100.0,
                        self.config.target_cpu_utilization * 100.0
                    ),
                )
            } else {
                (
                    ScalingReason::HighMemoryUtilization,
                    format!(
                        "Memory utilization {:.1}% exceeds target {:.1}%",
                        self.current_metrics.memory_utilization * 100.0,
                        self.config.target_memory_utilization * 100.0
                    ),
                )
            };

            return ScalingDecision {
                action: ScalingAction::ScaleUp(self.config.scale_up_increment),
                reason,
                confidence: 0.85,
                context,
                timestamp: now,
                current_metrics: self.current_metrics.clone(),
                predicted_metrics: predicted.cloned(),
            };
        }

        // Scale down if utilization is low
        if current.needs_scale_down && current_shards > self.config.min_shards {
            return ScalingDecision {
                action: ScalingAction::ScaleDown(self.config.scale_down_increment),
                reason: ScalingReason::LowUtilization,
                confidence: 0.75,
                context: format!(
                    "All metrics below {:.0}% of targets",
                    self.config.scale_down_threshold * 100.0
                ),
                timestamp: now,
                current_metrics: self.current_metrics.clone(),
                predicted_metrics: predicted.cloned(),
            };
        }

        // Check for rebalance need
        if !current.is_balanced && self.shard_metrics.len() >= 2 {
            return ScalingDecision {
                action: ScalingAction::Rebalance,
                reason: ScalingReason::UnbalancedShards,
                confidence: 0.7,
                context: "Shard load imbalance detected".to_string(),
                timestamp: now,
                current_metrics: self.current_metrics.clone(),
                predicted_metrics: predicted.cloned(),
            };
        }

        // Check for tiering actions
        if self.config.enable_tiering {
            let hot_vectors = self
                .access_tracker
                .get_hot_vectors(self.config.hot_tier_threshold);
            if !hot_vectors.is_empty() {
                return ScalingDecision {
                    action: ScalingAction::PromoteToHot(hot_vectors),
                    reason: ScalingReason::HotDataDetected,
                    confidence: 0.8,
                    context: "Frequently accessed vectors detected".to_string(),
                    timestamp: now,
                    current_metrics: self.current_metrics.clone(),
                    predicted_metrics: predicted.cloned(),
                };
            }

            let cold_vectors = self
                .access_tracker
                .get_cold_vectors(self.config.cold_tier_hours);
            if cold_vectors.len() > 100 {
                // Only tier if significant
                return ScalingDecision {
                    action: ScalingAction::DemoteToCold(cold_vectors),
                    reason: ScalingReason::ColdDataDetected,
                    confidence: 0.75,
                    context: "Infrequently accessed vectors detected".to_string(),
                    timestamp: now,
                    current_metrics: self.current_metrics.clone(),
                    predicted_metrics: predicted.cloned(),
                };
            }
        }

        // No action needed
        ScalingDecision {
            action: ScalingAction::None,
            reason: ScalingReason::Manual,
            confidence: 1.0,
            context: "All metrics within targets".to_string(),
            timestamp: now,
            current_metrics: self.current_metrics.clone(),
            predicted_metrics: predicted.cloned(),
        }
    }

    /// Add a scheduled scaling action
    pub fn schedule_scaling(&mut self, scheduled: ScheduledScaling) {
        self.scheduled_actions.push(scheduled);
    }

    /// Remove a scheduled scaling action
    pub fn unschedule(&mut self, index: usize) -> Option<ScheduledScaling> {
        if index < self.scheduled_actions.len() {
            Some(self.scheduled_actions.remove(index))
        } else {
            None
        }
    }

    /// Get scaling history
    pub fn history(&self) -> &VecDeque<ScalingDecision> {
        &self.scaling_history
    }

    /// Get detected seasonality pattern
    pub fn seasonality(&self) -> Option<SeasonalityPattern> {
        self.predictor.detect_seasonality()
    }

    /// Get current config
    pub fn config(&self) -> &ScalingConfig {
        &self.config
    }

    /// Update config
    pub fn set_config(&mut self, config: ScalingConfig) {
        self.config = config;
    }

    /// Get current timestamp
    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before UNIX epoch")
            .as_secs()
    }

    /// Cleanup old data
    pub fn cleanup(&mut self) {
        self.access_tracker.cleanup();
    }
}

/// Internal metric analysis result
#[derive(Debug)]
struct MetricAnalysis {
    cpu_pressure: f64,
    memory_pressure: f64,
    latency_pressure: f64,
    needs_scale_up: bool,
    needs_scale_down: bool,
    is_balanced: bool,
}

/// Thread-safe auto-scaler wrapper
pub struct SharedAutoScaler {
    inner: Arc<RwLock<AutoScaler>>,
}

impl SharedAutoScaler {
    pub fn new(config: ScalingConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(AutoScaler::new(config))),
        }
    }

    pub fn record_metrics(&self, metrics: MetricPoint) {
        if let Ok(mut scaler) = self.inner.write() {
            scaler.record_metrics(metrics);
        }
    }

    pub fn record_access(&self, vector_id: &str) {
        if let Ok(mut scaler) = self.inner.write() {
            scaler.record_access(vector_id);
        }
    }

    pub fn recommend(&self) -> Result<ScalingDecision> {
        self.inner
            .read()
            .map_err(|_| NeedleError::LockError)?
            .recommend()
    }

    pub fn acknowledge_scaling(&self, decision: ScalingDecision) {
        if let Ok(mut scaler) = self.inner.write() {
            scaler.acknowledge_scaling(decision);
        }
    }
}

impl Clone for SharedAutoScaler {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_config_defaults() {
        let config = ScalingConfig::default();
        assert_eq!(config.min_shards, 1);
        assert_eq!(config.max_shards, 32);
        assert!((config.target_cpu_utilization - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_scaling_config_builder() {
        let config = ScalingConfig::default()
            .with_min_shards(2)
            .with_max_shards(16)
            .with_target_cpu_utilization(0.8);

        assert_eq!(config.min_shards, 2);
        assert_eq!(config.max_shards, 16);
        assert!((config.target_cpu_utilization - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_auto_scaler_no_action_when_healthy() {
        let config = ScalingConfig::default();
        let scaler = AutoScaler::new(config);

        let decision = scaler.recommend().unwrap();
        assert_eq!(decision.action, ScalingAction::None);
    }

    #[test]
    fn test_auto_scaler_scale_up_on_high_cpu() {
        let config = ScalingConfig {
            cooldown_seconds: 0,
            ..Default::default()
        };
        let mut scaler = AutoScaler::new(config);

        // Record high CPU
        scaler.current_metrics.cpu_utilization = 0.95;
        scaler.current_metrics.shard_count = 1;

        let decision = scaler.recommend().unwrap();
        assert!(matches!(
            decision.action,
            ScalingAction::ScaleUp(_) | ScalingAction::EmergencyScale(_)
        ));
    }

    #[test]
    fn test_auto_scaler_scale_down_on_low_usage() {
        let config = ScalingConfig {
            cooldown_seconds: 0,
            ..Default::default()
        };
        let mut scaler = AutoScaler::new(config);

        // Record low usage
        scaler.current_metrics.cpu_utilization = 0.1;
        scaler.current_metrics.memory_utilization = 0.1;
        scaler.current_metrics.query_latency_ms = 5.0;
        scaler.current_metrics.shard_count = 4;

        let decision = scaler.recommend().unwrap();
        assert!(matches!(decision.action, ScalingAction::ScaleDown(_)));
    }

    #[test]
    fn test_load_predictor() {
        let mut predictor = LoadPredictor::new(100, 86400);

        // Add some data points
        for i in 0..10 {
            let point = MetricPoint {
                timestamp: i * 60,
                cpu_utilization: 0.5 + (i as f64 * 0.02),
                memory_utilization: 0.4,
                query_latency_ms: 20.0,
                qps: 100.0 + i as f64 * 10.0,
                connections: 10,
                vector_count: 1000,
                shard_count: 2,
            };
            predictor.add_point(point);
        }

        let prediction = predictor.predict(300);
        assert!(prediction.is_some());

        let pred = prediction.unwrap();
        // Should predict increasing trend
        assert!(pred.cpu_utilization > 0.5);
    }

    #[test]
    fn test_access_tracker() {
        let mut tracker = AccessTracker::new(3600, 7200);

        // Record accesses
        for _ in 0..150 {
            tracker.record_access("hot_vector");
        }
        tracker.record_access("cold_vector");

        let hot = tracker.get_hot_vectors(100);
        assert!(hot.contains(&"hot_vector".to_string()));
        assert!(!hot.contains(&"cold_vector".to_string()));
    }

    #[test]
    fn test_shared_auto_scaler() {
        let config = ScalingConfig::default();
        let scaler = SharedAutoScaler::new(config);

        let metrics = MetricPoint {
            timestamp: 0,
            cpu_utilization: 0.5,
            memory_utilization: 0.4,
            query_latency_ms: 25.0,
            qps: 100.0,
            connections: 5,
            vector_count: 1000,
            shard_count: 2,
        };

        scaler.record_metrics(metrics);
        let decision = scaler.recommend().unwrap();
        assert_eq!(decision.action, ScalingAction::None);
    }

    #[test]
    fn test_cooldown_period() {
        let config = ScalingConfig {
            cooldown_seconds: 300,
            ..Default::default()
        };
        let mut scaler = AutoScaler::new(config);

        // Trigger scaling
        scaler.current_metrics.cpu_utilization = 0.95;
        scaler.current_metrics.shard_count = 1;

        let decision1 = scaler.recommend().unwrap();
        scaler.acknowledge_scaling(decision1);

        // Try again immediately
        let decision2 = scaler.recommend().unwrap();
        assert_eq!(decision2.action, ScalingAction::None);
    }
}
