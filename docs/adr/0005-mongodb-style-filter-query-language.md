# ADR-0005: MongoDB-Style Filter Query Language

## Status

Accepted

## Context

Vector search alone is often insufficient for production applications. Users need to filter results based on metadata attributes:

- "Find similar images **where category = 'products'**"
- "Find similar documents **where date > '2024-01-01' AND author IN ['alice', 'bob']**"
- "Find similar embeddings **where NOT (status = 'archived')**"

This requires a query language for expressing filter predicates. Several options were considered:

| Approach | Familiarity | Expressiveness | Implementation Complexity |
|----------|-------------|----------------|---------------------------|
| MongoDB-style JSON | High (widely known) | High | Medium |
| SQL WHERE clauses | High (universal) | Very High | High (requires parser) |
| Custom DSL | Low (learning curve) | Variable | Variable |
| Method chaining | Medium (Rust-specific) | Medium | Low |
| No filtering | N/A | None | None |

## Decision

Implement **MongoDB-style JSON filter syntax** with support for comparison, logical, and set operators.

### Supported Operators

**Comparison Operators:**
```json
{"field": {"$eq": value}}   // Equal (implicit when value is scalar)
{"field": {"$ne": value}}   // Not equal
{"field": {"$gt": value}}   // Greater than
{"field": {"$gte": value}}  // Greater than or equal
{"field": {"$lt": value}}   // Less than
{"field": {"$lte": value}}  // Less than or equal
```

**Set Operators:**
```json
{"field": {"$in": [v1, v2, v3]}}   // Field value in set
{"field": {"$nin": [v1, v2, v3]}}  // Field value not in set
```

**Logical Operators:**
```json
{"$and": [filter1, filter2]}  // All conditions must match
{"$or": [filter1, filter2]}   // Any condition must match
{"$not": filter}              // Negation
```

### Usage Examples

```rust
use needle::Filter;
use serde_json::json;

// Simple equality (implicit $eq)
let filter = Filter::parse(&json!({"category": "books"}))?;

// Range query
let filter = Filter::parse(&json!({
    "price": {"$gte": 10, "$lte": 50}
}))?;

// Complex logical query
let filter = Filter::parse(&json!({
    "$or": [
        {"category": "electronics"},
        {"$and": [
            {"category": "books"},
            {"price": {"$lt": 20}}
        ]}
    ]
}))?;

// Search with filter
let results = collection.search_with_filter(&query_vector, 10, Some(&filter))?;
```

### Two-Tier Filter Evaluation

Filters can be applied at two stages:

1. **Pre-filter (during search)** — Applied during HNSW traversal, skipping non-matching vectors
2. **Post-filter (after search)** — Applied to search results, may return fewer than `k` results

```rust
let builder = SearchBuilder::new(&query_vector, 10)
    .with_pre_filter(category_filter)   // Efficient, integrated with search
    .with_post_filter(price_filter);    // Flexible, applied after

let results = collection.search_with_builder(&builder)?;
```

**Pre-filter** is more efficient but requires careful HNSW parameter tuning. **Post-filter** is simpler but may require over-fetching.

### Code References

- `src/metadata.rs:1-58` — Filter syntax documentation
- `src/metadata.rs` — Filter enum and parse() implementation
- `src/collection.rs:240-247` — SearchBuilder with pre/post filter support
- `src/database.rs:760-774` — search_with_filter implementation

## Consequences

### Benefits

1. **Familiar syntax** — MongoDB's query language is widely known and documented
2. **JSON-native** — Filters are valid JSON, easy to serialize/deserialize
3. **Composable** — Logical operators enable arbitrarily complex queries
4. **Type-flexible** — Works with strings, numbers, booleans, arrays
5. **Two-tier evaluation** — Users choose efficiency vs flexibility tradeoff
6. **API consistency** — Same filter syntax works in CLI, HTTP API, and library

### Tradeoffs

1. **Not SQL** — Developers expecting SQL syntax need to learn MongoDB conventions
2. **Limited expressiveness** — No JOINs, subqueries, or aggregations (not needed for filtering)
3. **No index optimization** — Filters are evaluated per-vector, not via secondary indices
4. **JSON verbosity** — Complex filters can be verbose compared to SQL

### What This Enabled

- Metadata-aware search in a single API call
- CLI filtering: `needle search db.needle -c docs -f '{"type":"article"}'`
- HTTP API filtering with JSON request bodies
- Programmatic filter construction via serde_json

### What This Prevented

- SQL query interface (would require full parser)
- Secondary index support (filters scan metadata linearly)
- Complex predicates like regex or full-text within filters

### Filter Evaluation Implementation

```rust
impl Filter {
    pub fn matches(&self, metadata: &serde_json::Value) -> bool {
        match self {
            Filter::Eq(field, value) => metadata.get(field) == Some(value),
            Filter::Lt(field, value) => {
                metadata.get(field)
                    .and_then(|v| compare_values(v, value))
                    .map(|ord| ord == Ordering::Less)
                    .unwrap_or(false)
            }
            Filter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            Filter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
            Filter::Not(filter) => !filter.matches(metadata),
            // ... other operators
        }
    }
}
```

### Example: E-Commerce Product Search

```rust
// Find products similar to a reference image
// Category: electronics OR (clothing AND price < $50)
// Exclude out-of-stock items

let filter = Filter::parse(&json!({
    "$and": [
        {"in_stock": true},
        {"$or": [
            {"category": "electronics"},
            {"$and": [
                {"category": "clothing"},
                {"price": {"$lt": 50}}
            ]}
        ]}
    ]
}))?;

let results = products.search_with_filter(&image_embedding, 20, Some(&filter))?;
```
