---
sidebar_position: 5
---

# Metadata Filtering

Needle supports rich metadata filtering using a MongoDB-style query language. This allows you to combine vector similarity search with attribute-based filtering.

## Storing Metadata

Every vector can have associated JSON metadata:

```rust
use serde_json::json;

collection.insert(
    "doc1",
    &embedding,
    json!({
        "title": "Introduction to Rust",
        "author": "Jane Smith",
        "year": 2024,
        "tags": ["rust", "programming", "tutorial"],
        "rating": 4.5,
        "published": true,
        "word_count": 2500
    })
)?;
```

## Basic Filtering

### Equality

```rust
use needle::Filter;

// Exact match
let filter = Filter::parse(&json!({
    "author": "Jane Smith"
}))?;

let results = collection.search_with_filter(&query, 10, &filter)?;
```

### Multiple Conditions (AND)

```rust
// All conditions must match (implicit AND)
let filter = Filter::parse(&json!({
    "author": "Jane Smith",
    "year": 2024
}))?;
```

## Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equals | `{"year": {"$eq": 2024}}` |
| `$ne` | Not equals | `{"year": {"$ne": 2023}}` |
| `$gt` | Greater than | `{"year": {"$gt": 2020}}` |
| `$gte` | Greater than or equal | `{"year": {"$gte": 2020}}` |
| `$lt` | Less than | `{"year": {"$lt": 2025}}` |
| `$lte` | Less than or equal | `{"year": {"$lte": 2024}}` |

### Examples

```rust
// Documents from 2023 or later
let filter = Filter::parse(&json!({
    "year": { "$gte": 2023 }
}))?;

// Documents with rating above 4.0
let filter = Filter::parse(&json!({
    "rating": { "$gt": 4.0 }
}))?;

// Documents between 1000-5000 words
let filter = Filter::parse(&json!({
    "word_count": { "$gte": 1000, "$lte": 5000 }
}))?;
```

## Array Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$in` | Value in array | `{"year": {"$in": [2023, 2024]}}` |
| `$nin` | Value not in array | `{"year": {"$nin": [2020, 2021]}}` |

### Examples

```rust
// Documents from specific years
let filter = Filter::parse(&json!({
    "year": { "$in": [2023, 2024] }
}))?;

// Documents with specific tags
let filter = Filter::parse(&json!({
    "tags": { "$in": ["rust", "python"] }
}))?;

// Exclude certain categories
let filter = Filter::parse(&json!({
    "category": { "$nin": ["draft", "archived"] }
}))?;
```

## Logical Operators

| Operator | Description |
|----------|-------------|
| `$and` | All conditions must match |
| `$or` | Any condition must match |
| `$not` | Negates a condition |

### AND

```rust
// Explicit AND (same as implicit)
let filter = Filter::parse(&json!({
    "$and": [
        { "year": { "$gte": 2023 } },
        { "rating": { "$gt": 4.0 } }
    ]
}))?;
```

### OR

```rust
// Either condition matches
let filter = Filter::parse(&json!({
    "$or": [
        { "author": "Jane Smith" },
        { "author": "John Doe" }
    ]
}))?;

// Combine OR with other conditions
let filter = Filter::parse(&json!({
    "year": 2024,
    "$or": [
        { "category": "tutorial" },
        { "category": "guide" }
    ]
}))?;
```

### NOT

```rust
// Negate a condition
let filter = Filter::parse(&json!({
    "$not": { "category": "draft" }
}))?;

// Complex negation
let filter = Filter::parse(&json!({
    "$not": {
        "$or": [
            { "status": "deleted" },
            { "status": "archived" }
        ]
    }
}))?;
```

## Nested Conditions

```rust
// Complex query combining multiple operators
let filter = Filter::parse(&json!({
    "$and": [
        { "year": { "$gte": 2023 } },
        {
            "$or": [
                { "tags": { "$in": ["rust", "go"] } },
                { "rating": { "$gt": 4.5 } }
            ]
        },
        { "$not": { "status": "draft" } }
    ]
}))?;
```

## Filtering on Array Fields

When a field contains an array, filters check if any element matches:

```rust
// Stored metadata:
// "tags": ["rust", "programming", "tutorial"]

// This matches because "rust" is in the tags array
let filter = Filter::parse(&json!({
    "tags": "rust"
}))?;

// Check for multiple possible values
let filter = Filter::parse(&json!({
    "tags": { "$in": ["rust", "python", "go"] }
}))?;
```

## Null and Existence Checks

```rust
// Check for null values
let filter = Filter::parse(&json!({
    "optional_field": null
}))?;

// Check for non-null values
let filter = Filter::parse(&json!({
    "optional_field": { "$ne": null }
}))?;
```

## Performance Considerations

### Index Usage

Metadata filtering happens after the initial HNSW search, so:
1. HNSW finds candidate vectors
2. Metadata filter is applied to candidates
3. Top-k filtered results are returned

### Optimization Tips

1. **Filter early**: More restrictive filters are faster:
   ```rust
   // Good: Specific filter reduces candidates quickly
   let filter = Filter::parse(&json!({
       "category": "tutorial",
       "year": 2024
   }))?;
   ```

2. **Avoid expensive operations**: `$or` with many conditions is slower than `$in`:
   ```rust
   // Slower
   let filter = Filter::parse(&json!({
       "$or": [
           { "year": 2020 },
           { "year": 2021 },
           { "year": 2022 }
       ]
   }))?;

   // Faster
   let filter = Filter::parse(&json!({
       "year": { "$in": [2020, 2021, 2022] }
   }))?;
   ```

3. **Increase ef_search for heavy filtering**: When many candidates are filtered out:
   ```rust
   // If 90% of vectors are filtered, increase ef_search
   let results = collection.search_with_params(
       &query,
       10,        // k
       Some(&filter),
       500        // higher ef_search for heavy filtering
   )?;
   ```

## Common Patterns

### Multi-Tenant Search

```rust
// Always filter by tenant
let filter = Filter::parse(&json!({
    "tenant_id": current_tenant_id
}))?;

let results = collection.search_with_filter(&query, 10, &filter)?;
```

### Date Range Queries

```rust
// Documents from last 30 days
let thirty_days_ago = chrono::Utc::now() - chrono::Duration::days(30);
let filter = Filter::parse(&json!({
    "created_at": { "$gte": thirty_days_ago.to_rfc3339() }
}))?;
```

### Access Control

```rust
// User can see public docs or docs they own
let filter = Filter::parse(&json!({
    "$or": [
        { "visibility": "public" },
        { "owner_id": current_user_id }
    ]
}))?;
```

### Category + Quality

```rust
// High-quality programming articles
let filter = Filter::parse(&json!({
    "$and": [
        { "category": "programming" },
        { "rating": { "$gte": 4.0 } },
        { "status": "published" }
    ]
}))?;
```

## Error Handling

```rust
// Invalid filter syntax returns an error
let result = Filter::parse(&json!({
    "year": { "$invalid": 2024 }
}));

match result {
    Ok(filter) => { /* use filter */ },
    Err(e) => eprintln!("Invalid filter: {}", e),
}
```

## Next Steps

- [Hybrid Search](/docs/guides/hybrid-search) - Combine with text search
- [Semantic Search](/docs/guides/semantic-search) - Build search applications
- [API Reference](/docs/api-reference) - Complete filter documentation
