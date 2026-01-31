#![allow(dead_code)]

//! Embedded Observability Dashboard
//!
//! Self-contained in-memory metrics aggregation and inline HTML dashboard
//! for real-time monitoring. No external CDN dependencies.
//!
//! # Features
//!
//! - **Rolling-window metrics store**: QPS, latency percentiles, memory usage
//! - **Inline HTML/JS dashboard**: Single-page app served from memory
//! - **Query explain visualization**: Breakdown of search query execution
//! - **Slow query log**: Track queries exceeding threshold
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────┐
//! │          MetricsAggregator                     │
//! │  ├── RollingWindow<LatencyObservation>        │
//! │  ├── RollingWindow<QpsObservation>            │
//! │  ├── RollingWindow<MemoryObservation>         │
//! │  └── SlowQueryLog                             │
//! ├──────────────────────────────────────────────┤
//! │          DashboardRenderer                     │
//! │  └── Inline HTML/JS (no external deps)        │
//! └──────────────────────────────────────────────┘
//! ```

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Configuration for the observability dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Rolling window size (number of observations).
    pub window_size: usize,
    /// Slow query threshold (microseconds).
    pub slow_query_threshold_us: u64,
    /// Maximum slow queries to retain.
    pub max_slow_queries: usize,
    /// Metrics aggregation interval (seconds).
    pub aggregation_interval_secs: u64,
    /// Enable query explain tracking.
    pub enable_query_explain: bool,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            window_size: 10_000,
            slow_query_threshold_us: 50_000, // 50ms
            max_slow_queries: 100,
            aggregation_interval_secs: 5,
            enable_query_explain: true,
        }
    }
}

/// A single latency observation.
#[derive(Debug, Clone)]
struct LatencyObservation {
    timestamp: Instant,
    operation: String,
    latency_us: u64,
    collection: String,
}

/// A single QPS observation window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QpsSnapshot {
    /// Timestamp.
    pub timestamp: u64,
    /// Queries per second.
    pub qps: f64,
    /// Inserts per second.
    pub insert_rate: f64,
    /// Deletes per second.
    pub delete_rate: f64,
}

/// Memory usage observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Timestamp.
    pub timestamp: u64,
    /// Total memory usage bytes.
    pub total_bytes: u64,
    /// Per-collection memory usage.
    pub per_collection: Vec<(String, u64)>,
}

/// A slow query entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowQuery {
    /// Timestamp.
    pub timestamp: u64,
    /// Collection searched.
    pub collection: String,
    /// Duration in microseconds.
    pub duration_us: u64,
    /// Number of results returned.
    pub result_count: usize,
    /// Query dimensions.
    pub dimensions: usize,
    /// Ef search parameter used.
    pub ef_search: Option<usize>,
    /// Whether filter was applied.
    pub had_filter: bool,
}

/// Query execution explain breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExplain {
    /// Total time (microseconds).
    pub total_time_us: u64,
    /// Index traversal time.
    pub index_time_us: u64,
    /// Filter evaluation time.
    pub filter_time_us: u64,
    /// Result enrichment time.
    pub enrich_time_us: u64,
    /// Candidates examined.
    pub candidates_examined: usize,
    /// Candidates after filter.
    pub candidates_after_filter: usize,
    /// Results returned.
    pub results_returned: usize,
    /// Collection name.
    pub collection: String,
    /// Timestamp.
    pub timestamp: u64,
}

/// Latency percentiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    /// Minimum latency (us).
    pub min_us: u64,
    /// Median (P50) latency (us).
    pub p50_us: u64,
    /// P90 latency (us).
    pub p90_us: u64,
    /// P95 latency (us).
    pub p95_us: u64,
    /// P99 latency (us).
    pub p99_us: u64,
    /// Maximum latency (us).
    pub max_us: u64,
    /// Average latency (us).
    pub avg_us: u64,
    /// Number of observations.
    pub count: usize,
}

/// Aggregated dashboard snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSnapshot {
    /// Current QPS metrics.
    pub qps: QpsSnapshot,
    /// Latency percentiles.
    pub latency: LatencyPercentiles,
    /// Memory usage.
    pub memory: MemorySnapshot,
    /// Recent slow queries.
    pub slow_queries: Vec<SlowQuery>,
    /// Recent query explains.
    pub recent_explains: Vec<QueryExplain>,
    /// Index health metrics.
    pub index_health: IndexHealth,
    /// Timestamp of this snapshot.
    pub timestamp: u64,
}

/// Index health status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexHealth {
    /// Number of collections.
    pub collection_count: usize,
    /// Total vectors across all collections.
    pub total_vectors: usize,
    /// Fragmentation ratio (0.0 = no fragmentation).
    pub fragmentation_ratio: f64,
    /// Whether compaction is recommended.
    pub needs_compaction: bool,
}

/// Rolling-window metrics aggregator.
pub struct MetricsAggregator {
    config: DashboardConfig,
    latencies: RwLock<VecDeque<LatencyObservation>>,
    query_count: RwLock<u64>,
    insert_count: RwLock<u64>,
    delete_count: RwLock<u64>,
    slow_queries: RwLock<VecDeque<SlowQuery>>,
    explains: RwLock<VecDeque<QueryExplain>>,
    last_snapshot_time: RwLock<Instant>,
}

impl MetricsAggregator {
    /// Create a new metrics aggregator.
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            latencies: RwLock::new(VecDeque::new()),
            query_count: RwLock::new(0),
            insert_count: RwLock::new(0),
            delete_count: RwLock::new(0),
            slow_queries: RwLock::new(VecDeque::new()),
            explains: RwLock::new(VecDeque::new()),
            last_snapshot_time: RwLock::new(Instant::now()),
        }
    }

    /// Record a query latency observation.
    pub fn record_query(&self, collection: &str, latency_us: u64, result_count: usize) {
        {
            let mut latencies = self.latencies.write();
            latencies.push_back(LatencyObservation {
                timestamp: Instant::now(),
                operation: "search".to_string(),
                latency_us,
                collection: collection.to_string(),
            });
            while latencies.len() > self.config.window_size {
                latencies.pop_front();
            }
        }

        *self.query_count.write() += 1;

        // Check for slow query
        if latency_us > self.config.slow_query_threshold_us {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let mut slow = self.slow_queries.write();
            slow.push_back(SlowQuery {
                timestamp: now,
                collection: collection.to_string(),
                duration_us: latency_us,
                result_count,
                dimensions: 0,
                ef_search: None,
                had_filter: false,
            });
            while slow.len() > self.config.max_slow_queries {
                slow.pop_front();
            }
        }
    }

    /// Record an insert operation.
    pub fn record_insert(&self, _collection: &str) {
        *self.insert_count.write() += 1;
    }

    /// Record a delete operation.
    pub fn record_delete(&self, _collection: &str) {
        *self.delete_count.write() += 1;
    }

    /// Record a query explain.
    pub fn record_explain(&self, explain: QueryExplain) {
        if self.config.enable_query_explain {
            let mut explains = self.explains.write();
            explains.push_back(explain);
            while explains.len() > 50 {
                explains.pop_front();
            }
        }
    }

    /// Compute latency percentiles from the rolling window.
    pub fn latency_percentiles(&self) -> LatencyPercentiles {
        let latencies = self.latencies.read();
        if latencies.is_empty() {
            return LatencyPercentiles {
                min_us: 0,
                p50_us: 0,
                p90_us: 0,
                p95_us: 0,
                p99_us: 0,
                max_us: 0,
                avg_us: 0,
                count: 0,
            };
        }

        let mut sorted: Vec<u64> = latencies.iter().map(|o| o.latency_us).collect();
        sorted.sort_unstable();

        let n = sorted.len();
        let sum: u64 = sorted.iter().sum();

        LatencyPercentiles {
            min_us: sorted[0],
            p50_us: sorted[n / 2],
            p90_us: sorted[(n as f64 * 0.9) as usize],
            p95_us: sorted[(n as f64 * 0.95) as usize],
            p99_us: sorted[((n as f64 * 0.99) as usize).min(n - 1)],
            max_us: sorted[n - 1],
            avg_us: sum / n as u64,
            count: n,
        }
    }

    /// Compute current QPS from the rolling window.
    pub fn current_qps(&self) -> QpsSnapshot {
        let latencies = self.latencies.read();
        let now = Instant::now();

        let window = Duration::from_secs(self.config.aggregation_interval_secs);
        let cutoff = now - window;

        let recent_queries = latencies
            .iter()
            .filter(|o| o.timestamp >= cutoff && o.operation == "search")
            .count();

        let window_secs = window.as_secs_f64();
        let qps = recent_queries as f64 / window_secs;

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        QpsSnapshot {
            timestamp: ts,
            qps,
            insert_rate: *self.insert_count.read() as f64 / window_secs.max(1.0),
            delete_rate: *self.delete_count.read() as f64 / window_secs.max(1.0),
        }
    }

    /// Get recent slow queries.
    pub fn slow_queries(&self) -> Vec<SlowQuery> {
        self.slow_queries.read().iter().cloned().collect()
    }

    /// Get recent query explains.
    pub fn recent_explains(&self) -> Vec<QueryExplain> {
        self.explains.read().iter().cloned().collect()
    }

    /// Build a full dashboard snapshot.
    pub fn snapshot(
        &self,
        collection_count: usize,
        total_vectors: usize,
        memory_bytes: u64,
    ) -> DashboardSnapshot {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        DashboardSnapshot {
            qps: self.current_qps(),
            latency: self.latency_percentiles(),
            memory: MemorySnapshot {
                timestamp: ts,
                total_bytes: memory_bytes,
                per_collection: Vec::new(),
            },
            slow_queries: self.slow_queries(),
            recent_explains: self.recent_explains(),
            index_health: IndexHealth {
                collection_count,
                total_vectors,
                fragmentation_ratio: 0.0,
                needs_compaction: false,
            },
            timestamp: ts,
        }
    }
}

/// Generates the inline HTML/JS dashboard (no external CDN dependencies).
pub fn generate_dashboard_html(snapshot: &DashboardSnapshot) -> String {
    let snapshot_json = serde_json::to_string(snapshot).unwrap_or_default();

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Needle Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #1a1a2e; color: #eee; padding: 20px; }}
  h1 {{ color: #e94560; margin-bottom: 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
  .card {{ background: #16213e; border-radius: 8px; padding: 20px; border: 1px solid #0f3460; }}
  .card h2 {{ color: #e94560; font-size: 14px; text-transform: uppercase; margin-bottom: 12px; }}
  .metric {{ font-size: 32px; font-weight: bold; color: #53d8fb; }}
  .label {{ font-size: 12px; color: #888; margin-top: 4px; }}
  .table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
  .table th, .table td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #0f3460; }}
  .table th {{ color: #e94560; font-size: 12px; text-transform: uppercase; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; }}
  .badge-ok {{ background: #2d6a4f; color: #95d5b2; }}
  .badge-warn {{ background: #7f5539; color: #e6ccb2; }}
</style>
</head>
<body>
<h1>🪡 Needle Observability Dashboard</h1>
<div class="grid">
  <div class="card">
    <h2>Queries Per Second</h2>
    <div class="metric" id="qps">0</div>
    <div class="label">queries/sec</div>
  </div>
  <div class="card">
    <h2>Latency (P50 / P99)</h2>
    <div class="metric" id="latency">0 / 0</div>
    <div class="label">microseconds</div>
  </div>
  <div class="card">
    <h2>Memory Usage</h2>
    <div class="metric" id="memory">0</div>
    <div class="label">bytes</div>
  </div>
  <div class="card">
    <h2>Index Health</h2>
    <div class="metric" id="health">-</div>
    <div class="label">collections / vectors</div>
  </div>
</div>
<div class="card" style="margin-top: 16px;">
  <h2>Slow Queries</h2>
  <table class="table">
    <thead><tr><th>Time</th><th>Collection</th><th>Duration</th><th>Results</th></tr></thead>
    <tbody id="slow-queries"></tbody>
  </table>
</div>
<script>
  const data = {snapshot_json};
  document.getElementById('qps').textContent = data.qps.qps.toFixed(1);
  document.getElementById('latency').textContent = data.latency.p50_us + ' / ' + data.latency.p99_us;
  document.getElementById('memory').textContent = (data.memory.total_bytes / 1048576).toFixed(1) + ' MB';
  document.getElementById('health').textContent = data.index_health.collection_count + ' / ' + data.index_health.total_vectors;
  const tbody = document.getElementById('slow-queries');
  data.slow_queries.slice(-10).reverse().forEach(sq => {{
    const tr = document.createElement('tr');
    tr.innerHTML = '<td>' + new Date(sq.timestamp * 1000).toLocaleTimeString() + '</td>'
      + '<td>' + sq.collection + '</td>'
      + '<td>' + (sq.duration_us / 1000).toFixed(1) + 'ms</td>'
      + '<td>' + sq.result_count + '</td>';
    tbody.appendChild(tr);
  }});
</script>
</body>
</html>"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_aggregator() {
        let agg = MetricsAggregator::new(DashboardConfig::default());

        for i in 0..100 {
            agg.record_query("test", (i * 100) as u64, 10);
        }
        agg.record_insert("test");
        agg.record_delete("test");

        let percentiles = agg.latency_percentiles();
        assert_eq!(percentiles.count, 100);
        assert!(percentiles.p50_us > 0);
        assert!(percentiles.p99_us > percentiles.p50_us);
    }

    #[test]
    fn test_slow_query_tracking() {
        let config = DashboardConfig {
            slow_query_threshold_us: 100, // Very low threshold for testing
            ..Default::default()
        };
        let agg = MetricsAggregator::new(config);

        agg.record_query("test", 50, 5);   // Not slow
        agg.record_query("test", 200, 10);  // Slow
        agg.record_query("test", 500, 15);  // Slow

        let slow = agg.slow_queries();
        assert_eq!(slow.len(), 2);
    }

    #[test]
    fn test_dashboard_html_generation() {
        let agg = MetricsAggregator::new(DashboardConfig::default());
        let snapshot = agg.snapshot(3, 10000, 50_000_000);
        let html = generate_dashboard_html(&snapshot);

        assert!(html.contains("Needle Observability Dashboard"));
        assert!(html.contains("<script>"));
        assert!(!html.contains("cdn")); // No external CDN
    }

    #[test]
    fn test_query_explain_tracking() {
        let agg = MetricsAggregator::new(DashboardConfig::default());

        agg.record_explain(QueryExplain {
            total_time_us: 1000,
            index_time_us: 800,
            filter_time_us: 100,
            enrich_time_us: 100,
            candidates_examined: 500,
            candidates_after_filter: 50,
            results_returned: 10,
            collection: "test".to_string(),
            timestamp: 0,
        });

        let explains = agg.recent_explains();
        assert_eq!(explains.len(), 1);
        assert_eq!(explains[0].total_time_us, 1000);
    }

    #[test]
    fn test_snapshot() {
        let agg = MetricsAggregator::new(DashboardConfig::default());
        for _ in 0..10 {
            agg.record_query("coll", 500, 5);
        }

        let snap = agg.snapshot(2, 5000, 10_000_000);
        assert_eq!(snap.index_health.collection_count, 2);
        assert_eq!(snap.index_health.total_vectors, 5000);
        assert!(snap.timestamp > 0);
    }
}
