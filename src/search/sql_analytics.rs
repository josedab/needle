//! Embedded SQL Analytics Engine
//!
//! Provides a lightweight SQL-compatible analytics engine over vector metadata.
//! Supports aggregations (COUNT, SUM, AVG, MIN, MAX), GROUP BY, HAVING,
//! and ORDER BY without requiring an external database.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::services::search::sql_analytics::*;
//!
//! let engine = AnalyticsEngine::new();
//! let query = AnalyticsQuery::parse(
//!     "SELECT category, COUNT(*), AVG(price) FROM docs GROUP BY category HAVING COUNT(*) > 5"
//! )?;
//! let result = engine.execute(&query, &metadata_rows)?;
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::error::{NeedleError, Result};

// ============================================================================
// Query AST
// ============================================================================

/// Aggregate functions supported by the analytics engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggFunc {
    /// COUNT(*) — count all rows.
    Count,
    /// COUNT(field) — count non-null values of a field.
    CountField(String),
    /// SUM(field)
    Sum(String),
    /// AVG(field)
    Avg(String),
    /// MIN(field)
    Min(String),
    /// MAX(field)
    Max(String),
}

/// A column in the SELECT clause.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectColumn {
    /// A raw field reference (used in GROUP BY keys).
    Field(String),
    /// An aggregate function.
    Aggregate(AggFunc),
    /// An aggregate with an alias.
    AliasedAggregate { func: AggFunc, alias: String },
}

/// Comparison operators for WHERE/HAVING conditions.
#[derive(Debug, Clone, PartialEq)]
pub enum CmpOp {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
}

/// A simple condition (field op value).
#[derive(Debug, Clone)]
pub struct Condition {
    pub left: SelectColumn,
    pub op: CmpOp,
    pub value: Value,
}

/// Sort direction.
#[derive(Debug, Clone, PartialEq)]
pub enum SortDir {
    Asc,
    Desc,
}

/// An analytics query over metadata.
#[derive(Debug, Clone)]
pub struct AnalyticsQuery {
    /// Columns to select.
    pub select: Vec<SelectColumn>,
    /// Collection to query (FROM clause).
    pub from: String,
    /// WHERE conditions (applied before grouping).
    pub where_clauses: Vec<Condition>,
    /// GROUP BY field names.
    pub group_by: Vec<String>,
    /// HAVING conditions (applied after grouping).
    pub having: Vec<Condition>,
    /// ORDER BY clauses.
    pub order_by: Vec<(String, SortDir)>,
    /// LIMIT
    pub limit: Option<usize>,
}

impl AnalyticsQuery {
    /// Create a simple query builder.
    pub fn builder(from: impl Into<String>) -> AnalyticsQueryBuilder {
        AnalyticsQueryBuilder {
            query: AnalyticsQuery {
                select: Vec::new(),
                from: from.into(),
                where_clauses: Vec::new(),
                group_by: Vec::new(),
                having: Vec::new(),
                order_by: Vec::new(),
                limit: None,
            },
        }
    }
}

/// Builder for constructing analytics queries programmatically.
pub struct AnalyticsQueryBuilder {
    query: AnalyticsQuery,
}

impl AnalyticsQueryBuilder {
    /// Add a field to SELECT.
    #[must_use]
    pub fn select_field(mut self, name: impl Into<String>) -> Self {
        self.query.select.push(SelectColumn::Field(name.into()));
        self
    }

    /// Add an aggregate to SELECT.
    #[must_use]
    pub fn select_agg(mut self, func: AggFunc) -> Self {
        self.query.select.push(SelectColumn::Aggregate(func));
        self
    }

    /// Add an aggregate with alias.
    #[must_use]
    pub fn select_agg_as(mut self, func: AggFunc, alias: impl Into<String>) -> Self {
        self.query.select.push(SelectColumn::AliasedAggregate {
            func,
            alias: alias.into(),
        });
        self
    }

    /// Add GROUP BY field.
    #[must_use]
    pub fn group_by(mut self, field: impl Into<String>) -> Self {
        self.query.group_by.push(field.into());
        self
    }

    /// Add a HAVING condition.
    #[must_use]
    pub fn having(mut self, cond: Condition) -> Self {
        self.query.having.push(cond);
        self
    }

    /// Add WHERE condition.
    #[must_use]
    pub fn where_clause(mut self, cond: Condition) -> Self {
        self.query.where_clauses.push(cond);
        self
    }

    /// Set ORDER BY.
    #[must_use]
    pub fn order_by(mut self, field: impl Into<String>, dir: SortDir) -> Self {
        self.query.order_by.push((field.into(), dir));
        self
    }

    /// Set LIMIT.
    #[must_use]
    pub fn limit(mut self, n: usize) -> Self {
        self.query.limit = Some(n);
        self
    }

    /// Build the query.
    pub fn build(self) -> AnalyticsQuery {
        self.query
    }
}

// ============================================================================
// Result Types
// ============================================================================

/// A single row in the analytics result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsRow {
    /// Column values (aligned with column_names in AnalyticsResult).
    pub values: Vec<Value>,
}

/// Result of an analytics query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsResult {
    /// Column names.
    pub columns: Vec<String>,
    /// Result rows.
    pub rows: Vec<AnalyticsRow>,
    /// Number of input rows processed.
    pub rows_scanned: usize,
    /// Number of groups produced.
    pub groups: usize,
}

// ============================================================================
// Execution Engine
// ============================================================================

/// Metadata row: a single document's metadata as a JSON object.
pub type MetadataRow = Value;

/// The analytics execution engine.
pub struct AnalyticsEngine;

impl AnalyticsEngine {
    /// Create a new engine.
    pub fn new() -> Self {
        Self
    }

    /// Execute an analytics query against a set of metadata rows.
    pub fn execute(
        &self,
        query: &AnalyticsQuery,
        rows: &[MetadataRow],
    ) -> Result<AnalyticsResult> {
        // 1. Filter (WHERE)
        let filtered: Vec<&Value> = rows
            .iter()
            .filter(|row| self.evaluate_where(row, &query.where_clauses))
            .collect();
        let rows_scanned = filtered.len();

        // 2. Group
        let groups = self.group_rows(&filtered, &query.group_by);

        // 3. Compute aggregates per group
        let mut result_rows: Vec<(HashMap<String, Value>, Vec<Value>)> = Vec::new();
        for (group_key, group_rows) in &groups {
            let mut row_values = Vec::new();
            for col in &query.select {
                let val = match col {
                    SelectColumn::Field(name) => group_key
                        .get(name)
                        .cloned()
                        .unwrap_or(Value::Null),
                    SelectColumn::Aggregate(func) | SelectColumn::AliasedAggregate { func, .. } => {
                        self.compute_aggregate(func, group_rows)
                    }
                };
                row_values.push(val);
            }
            result_rows.push((group_key.clone(), row_values));
        }

        // 4. HAVING filter
        let result_rows: Vec<Vec<Value>> = result_rows
            .into_iter()
            .filter(|(_, row_vals)| self.evaluate_having(row_vals, &query.select, &query.having))
            .map(|(_, vals)| vals)
            .collect();

        // 5. ORDER BY
        let mut result_rows = result_rows;
        if !query.order_by.is_empty() {
            let col_names: Vec<String> = query.select.iter().map(|c| self.column_name(c)).collect();
            result_rows.sort_by(|a, b| {
                for (field, dir) in &query.order_by {
                    if let Some(idx) = col_names.iter().position(|n| n == field) {
                        let cmp = compare_values(&a[idx], &b[idx]);
                        let cmp = match dir {
                            SortDir::Asc => cmp,
                            SortDir::Desc => cmp.reverse(),
                        };
                        if cmp != std::cmp::Ordering::Equal {
                            return cmp;
                        }
                    }
                }
                std::cmp::Ordering::Equal
            });
        }

        // 6. LIMIT
        if let Some(limit) = query.limit {
            result_rows.truncate(limit);
        }

        let group_count = groups.len();
        let columns: Vec<String> = query.select.iter().map(|c| self.column_name(c)).collect();

        Ok(AnalyticsResult {
            columns,
            rows: result_rows
                .into_iter()
                .map(|values| AnalyticsRow { values })
                .collect(),
            rows_scanned,
            groups: group_count,
        })
    }

    fn column_name(&self, col: &SelectColumn) -> String {
        match col {
            SelectColumn::Field(name) => name.clone(),
            SelectColumn::Aggregate(func) => self.agg_name(func),
            SelectColumn::AliasedAggregate { alias, .. } => alias.clone(),
        }
    }

    fn agg_name(&self, func: &AggFunc) -> String {
        match func {
            AggFunc::Count => "COUNT(*)".to_string(),
            AggFunc::CountField(f) => format!("COUNT({f})"),
            AggFunc::Sum(f) => format!("SUM({f})"),
            AggFunc::Avg(f) => format!("AVG({f})"),
            AggFunc::Min(f) => format!("MIN({f})"),
            AggFunc::Max(f) => format!("MAX({f})"),
        }
    }

    fn evaluate_where(&self, row: &Value, conditions: &[Condition]) -> bool {
        conditions.iter().all(|cond| {
            let field_name = match &cond.left {
                SelectColumn::Field(f) => f.as_str(),
                _ => return true, // aggregates not valid in WHERE
            };
            let val = row.get(field_name).unwrap_or(&Value::Null);
            compare_with_op(val, &cond.op, &cond.value)
        })
    }

    fn group_rows<'a>(
        &self,
        rows: &[&'a Value],
        group_by: &[String],
    ) -> Vec<(HashMap<String, Value>, Vec<&'a Value>)> {
        if group_by.is_empty() {
            // No grouping: all rows in one group
            return vec![(HashMap::new(), rows.to_vec())];
        }

        let mut groups: Vec<(HashMap<String, Value>, Vec<&'a Value>)> = Vec::new();
        for row in rows {
            let key: HashMap<String, Value> = group_by
                .iter()
                .map(|f| (f.clone(), row.get(f).cloned().unwrap_or(Value::Null)))
                .collect();
            if let Some(group) = groups.iter_mut().find(|(k, _)| *k == key) {
                group.1.push(row);
            } else {
                groups.push((key, vec![row]));
            }
        }
        groups
    }

    fn compute_aggregate(&self, func: &AggFunc, rows: &[&Value]) -> Value {
        match func {
            AggFunc::Count => Value::from(rows.len() as u64),
            AggFunc::CountField(field) => {
                let count = rows
                    .iter()
                    .filter(|r| r.get(field).is_some_and(|v| !v.is_null()))
                    .count();
                Value::from(count as u64)
            }
            AggFunc::Sum(field) => {
                let sum: f64 = rows
                    .iter()
                    .filter_map(|r| r.get(field).and_then(|v| v.as_f64()))
                    .sum();
                Value::from(sum)
            }
            AggFunc::Avg(field) => {
                let vals: Vec<f64> = rows
                    .iter()
                    .filter_map(|r| r.get(field).and_then(|v| v.as_f64()))
                    .collect();
                if vals.is_empty() {
                    Value::Null
                } else {
                    Value::from(vals.iter().sum::<f64>() / vals.len() as f64)
                }
            }
            AggFunc::Min(field) => rows
                .iter()
                .filter_map(|r| r.get(field))
                .filter(|v| !v.is_null())
                .min_by(|a, b| compare_values(a, b))
                .cloned()
                .unwrap_or(Value::Null),
            AggFunc::Max(field) => rows
                .iter()
                .filter_map(|r| r.get(field))
                .filter(|v| !v.is_null())
                .max_by(|a, b| compare_values(a, b))
                .cloned()
                .unwrap_or(Value::Null),
        }
    }

    fn evaluate_having(
        &self,
        row_values: &[Value],
        select: &[SelectColumn],
        having: &[Condition],
    ) -> bool {
        having.iter().all(|cond| {
            let col_name = match &cond.left {
                SelectColumn::Field(f) => f.clone(),
                SelectColumn::Aggregate(func) => self.agg_name(func),
                SelectColumn::AliasedAggregate { alias, .. } => alias.clone(),
            };
            let col_names: Vec<String> = select.iter().map(|c| self.column_name(c)).collect();
            if let Some(idx) = col_names.iter().position(|n| *n == col_name) {
                compare_with_op(&row_values[idx], &cond.op, &cond.value)
            } else {
                true
            }
        })
    }
}

impl Default for AnalyticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    match (a.as_f64(), b.as_f64()) {
        (Some(fa), Some(fb)) => fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal),
        _ => {
            let sa = a.as_str().unwrap_or("");
            let sb = b.as_str().unwrap_or("");
            sa.cmp(sb)
        }
    }
}

fn compare_with_op(val: &Value, op: &CmpOp, target: &Value) -> bool {
    let ord = compare_values(val, target);
    match op {
        CmpOp::Eq => ord == std::cmp::Ordering::Equal,
        CmpOp::Ne => ord != std::cmp::Ordering::Equal,
        CmpOp::Gt => ord == std::cmp::Ordering::Greater,
        CmpOp::Gte => ord != std::cmp::Ordering::Less,
        CmpOp::Lt => ord == std::cmp::Ordering::Less,
        CmpOp::Lte => ord != std::cmp::Ordering::Greater,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_data() -> Vec<Value> {
        vec![
            json!({"category": "books", "price": 10.0, "rating": 4.5}),
            json!({"category": "books", "price": 20.0, "rating": 3.0}),
            json!({"category": "books", "price": 15.0, "rating": 5.0}),
            json!({"category": "electronics", "price": 100.0, "rating": 4.0}),
            json!({"category": "electronics", "price": 200.0, "rating": 4.5}),
            json!({"category": "clothing", "price": 30.0, "rating": 3.5}),
        ]
    }

    #[test]
    fn test_count_all() {
        let engine = AnalyticsEngine::new();
        let query = AnalyticsQuery::builder("products")
            .select_agg(AggFunc::Count)
            .build();
        let result = engine.execute(&query, &sample_data()).expect("execute");
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].values[0], json!(6));
    }

    #[test]
    fn test_group_by_count() {
        let engine = AnalyticsEngine::new();
        let query = AnalyticsQuery::builder("products")
            .select_field("category")
            .select_agg(AggFunc::Count)
            .group_by("category")
            .build();
        let result = engine.execute(&query, &sample_data()).expect("execute");
        assert_eq!(result.rows.len(), 3); // 3 categories
        assert_eq!(result.groups, 3);
    }

    #[test]
    fn test_sum_avg() {
        let engine = AnalyticsEngine::new();
        let query = AnalyticsQuery::builder("products")
            .select_agg(AggFunc::Sum("price".into()))
            .select_agg(AggFunc::Avg("price".into()))
            .build();
        let result = engine.execute(&query, &sample_data()).expect("execute");
        let sum = result.rows[0].values[0].as_f64().expect("sum");
        let avg = result.rows[0].values[1].as_f64().expect("avg");
        assert!((sum - 375.0).abs() < 0.01);
        assert!((avg - 62.5).abs() < 0.01);
    }

    #[test]
    fn test_min_max() {
        let engine = AnalyticsEngine::new();
        let query = AnalyticsQuery::builder("products")
            .select_agg(AggFunc::Min("price".into()))
            .select_agg(AggFunc::Max("price".into()))
            .build();
        let result = engine.execute(&query, &sample_data()).expect("execute");
        assert_eq!(result.rows[0].values[0].as_f64(), Some(10.0));
        assert_eq!(result.rows[0].values[1].as_f64(), Some(200.0));
    }

    #[test]
    fn test_where_clause() {
        let engine = AnalyticsEngine::new();
        let query = AnalyticsQuery::builder("products")
            .select_agg(AggFunc::Count)
            .where_clause(Condition {
                left: SelectColumn::Field("category".into()),
                op: CmpOp::Eq,
                value: json!("books"),
            })
            .build();
        let result = engine.execute(&query, &sample_data()).expect("execute");
        assert_eq!(result.rows[0].values[0], json!(3));
    }

    #[test]
    fn test_having_clause() {
        let engine = AnalyticsEngine::new();
        let query = AnalyticsQuery::builder("products")
            .select_field("category")
            .select_agg_as(AggFunc::Count, "cnt")
            .group_by("category")
            .having(Condition {
                left: SelectColumn::AliasedAggregate {
                    func: AggFunc::Count,
                    alias: "cnt".into(),
                },
                op: CmpOp::Gte,
                value: json!(3),
            })
            .build();
        let result = engine.execute(&query, &sample_data()).expect("execute");
        // Only "books" has count >= 3
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].values[0], json!("books"));
    }

    #[test]
    fn test_order_by_and_limit() {
        let engine = AnalyticsEngine::new();
        let query = AnalyticsQuery::builder("products")
            .select_field("category")
            .select_agg_as(AggFunc::Avg("price".into()), "avg_price")
            .group_by("category")
            .order_by("avg_price", SortDir::Desc)
            .limit(2)
            .build();
        let result = engine.execute(&query, &sample_data()).expect("execute");
        assert_eq!(result.rows.len(), 2);
        // Electronics has highest avg price (150), should be first
        assert_eq!(result.rows[0].values[0], json!("electronics"));
    }

    #[test]
    fn test_empty_data() {
        let engine = AnalyticsEngine::new();
        let query = AnalyticsQuery::builder("products")
            .select_agg(AggFunc::Count)
            .build();
        let result = engine.execute(&query, &[]).expect("execute");
        assert_eq!(result.rows[0].values[0], json!(0));
    }

    #[test]
    fn test_group_by_with_multiple_aggs() {
        let engine = AnalyticsEngine::new();
        let query = AnalyticsQuery::builder("products")
            .select_field("category")
            .select_agg(AggFunc::Count)
            .select_agg(AggFunc::Avg("rating".into()))
            .select_agg(AggFunc::Sum("price".into()))
            .group_by("category")
            .build();
        let result = engine.execute(&query, &sample_data()).expect("execute");
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.columns.len(), 4);
        assert_eq!(result.columns[0], "category");
        assert_eq!(result.columns[1], "COUNT(*)");
    }
}
