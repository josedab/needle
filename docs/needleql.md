# NeedleQL Query Language

NeedleQL is a SQL-like query language for vector operations in Needle. It provides a familiar, text-based interface for vector similarity search, metadata filtering, and query analysis.

> **Background:** See [ADR-0026](adr/0026-needleql-query-language.md) for the design rationale.

## Quick Example

```sql
SELECT id, title, distance FROM documents
WHERE vector SIMILAR TO $query
  AND category = 'electronics'
  AND price < 100
LIMIT 20;
```

## Getting Started

### CLI

```bash
needle sql mydata.needle \
  --query "SELECT * FROM docs WHERE vector SIMILAR TO \$q LIMIT 10" \
  --vector "0.1,0.2,0.3" \
  --format json
```

### REST API

```bash
curl -X POST http://127.0.0.1:8080/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT * FROM docs WHERE vector SIMILAR TO $q LIMIT 5", "vector": [0.1, 0.2, 0.3]}'
```

## Syntax Reference

### SELECT

```sql
SELECT select_list FROM collection_name
[WHERE where_clause]
[ORDER BY order_clause]
[LIMIT number];
```

- `select_list` — `*` or comma-separated column names (`id`, `distance`, `score`, or metadata field names)
- `collection_name` — the target collection
- `WHERE` — optional vector search and/or metadata filters
- `ORDER BY` — optional ordering (column `ASC`/`DESC` or function calls)
- `LIMIT` — maximum number of results (defaults to 10)

### Vector Similarity Search

Use the `SIMILAR TO` clause to perform approximate nearest neighbor search:

```sql
SELECT * FROM documents
WHERE vector SIMILAR TO $query
LIMIT 10;
```

The `$query` parameter is bound to a vector value at execution time.

### Metadata Filtering

Combine vector search with metadata filters using `AND`:

```sql
SELECT * FROM products
WHERE vector SIMILAR TO $query
  AND category = 'electronics'
  AND price < 100
  AND status != 'discontinued'
LIMIT 20;
```

#### Supported Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `=` | Equal | `category = 'books'` |
| `!=` | Not equal | `status != 'draft'` |
| `<` | Less than | `price < 50` |
| `<=` | Less than or equal | `score <= 0.9` |
| `>` | Greater than | `count > 0` |
| `>=` | Greater than or equal | `rating >= 4.0` |

### Hybrid Search (Vector + Text)

Combine vector similarity with BM25 full-text search:

```sql
SELECT * FROM articles
WHERE vector SIMILAR TO $query
  AND text MATCH 'machine learning'
ORDER BY hybrid_score(vector_score, text_score, 0.7)
LIMIT 10;
```

The `hybrid_score` function fuses vector and text scores using Reciprocal Rank Fusion (RRF). The third argument controls the vector weight (0.0–1.0).

### Time-Decayed Relevance

Boost recent results using the `time_decay` function:

```sql
SELECT * FROM news
WHERE vector SIMILAR TO $query
ORDER BY time_decay(distance, published_at, '7d')
LIMIT 10;
```

### EXPLAIN / EXPLAIN ANALYZE

Inspect query execution plans without (or with) running the query:

```sql
-- Show the query plan
EXPLAIN
SELECT * FROM documents
WHERE vector SIMILAR TO $query
LIMIT 10;

-- Execute and show actual performance stats
EXPLAIN ANALYZE
SELECT * FROM products
WHERE vector SIMILAR TO $query AND category = 'electronics'
LIMIT 10;
```

#### Example EXPLAIN ANALYZE Output

```
Query Plan:
├── VectorScan (collection=products, k=100, ef_search=50)
│   └── Estimated candidates: 100
├── FilterApply (category = 'electronics')
│   └── Selectivity: 0.15 (estimated 15 rows)
├── Sort (distance ASC, limit=10)
└── Project (*)

Execution Stats:
├── Planning time: 0.2ms
├── Execution time: 12.5ms
├── Rows scanned: 89
├── Rows after filter: 14
└── Rows returned: 10
```

### INSERT

```sql
INSERT INTO collection_name (id, vector, metadata)
VALUES ('doc1', $vec, '{"key": "value"}');
```

### DELETE

```sql
DELETE FROM collection_name WHERE id = 'doc1';
```

## Grammar (Simplified BNF)

```
query           := select_stmt | explain_stmt | insert_stmt | delete_stmt

select_stmt     := SELECT select_list FROM table_name
                   [WHERE where_clause]
                   [ORDER BY order_clause]
                   [LIMIT number]

select_list     := '*' | column_list
column_list     := column_name (',' column_name)*
column_name     := IDENTIFIER | 'distance' | 'score'

where_clause    := vector_clause [AND filter_clause]*
vector_clause   := 'vector' SIMILAR TO parameter
filter_clause   := comparison | text_match | in_list
comparison      := column_name op value
op              := '=' | '!=' | '<' | '<=' | '>' | '>='

order_clause    := order_expr (',' order_expr)*
order_expr      := column_name [ASC | DESC]
                 | function_call [ASC | DESC]

explain_stmt    := EXPLAIN [ANALYZE] select_stmt

parameter       := '$' IDENTIFIER
```

## Output Formats

The `sql` CLI command supports three output formats via `--format`:

| Format | Description |
|--------|-------------|
| `json` | JSON array of result objects (default) |
| `table` | Human-readable table |
| `csv` | Comma-separated values |

## Common Patterns

### Similarity Search with Metadata Filter

Find similar products in a specific category under a price threshold:

```sql
SELECT id, title, price, distance FROM products
WHERE vector SIMILAR TO $query
  AND category = 'electronics'
  AND price < 200
  AND status != 'discontinued'
LIMIT 10;
```

### Hybrid Search (Vector + BM25 with RRF)

Combine vector similarity with full-text search for better relevance — useful when queries contain specific keywords:

```sql
SELECT id, title, distance FROM articles
WHERE vector SIMILAR TO $query
  AND text MATCH 'distributed systems consensus'
ORDER BY hybrid_score(vector_score, text_score, 0.6)
LIMIT 15;
```

The `0.6` weight favors vector similarity. Use `0.3`–`0.4` when keyword precision matters more.

### Time-Decay Boosted Search

Surface recent news articles while still matching by relevance — the `7d` half-life means articles older than 7 days are progressively down-ranked:

```sql
SELECT id, headline, published_at, distance FROM news
WHERE vector SIMILAR TO $query
ORDER BY time_decay(distance, published_at, '7d')
LIMIT 10;
```

### Batch Insert and Delete

Insert a new document and clean up old entries in a single session:

```sql
INSERT INTO knowledge_base (id, vector, metadata)
VALUES ('kb-042', $vec, '{"source": "docs", "version": "2.1"}');

DELETE FROM knowledge_base WHERE id = 'kb-017';
DELETE FROM knowledge_base WHERE id = 'kb-003';
```

### Diagnose a Slow Query

Use `EXPLAIN ANALYZE` to inspect filter selectivity and scan counts:

```sql
EXPLAIN ANALYZE
SELECT id, distance FROM logs
WHERE vector SIMILAR TO $query
  AND level = 'error'
  AND timestamp > '2025-01-01'
LIMIT 20;
```

## See Also

- [ADR-0026: NeedleQL Design Decision](adr/0026-needleql-query-language.md)
- [API Reference](api-reference.md)
- [How-To Guides](how-to-guides.md)
