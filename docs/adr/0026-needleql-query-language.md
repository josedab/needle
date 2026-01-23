# ADR-0026: NeedleQL - SQL-Like Query Language

## Status

Accepted

## Context

Vector databases traditionally expose programmatic APIs:

```rust
// Typical vector database API
let results = collection.search(&query_vector, k)
    .filter(Filter::eq("category", "electronics"))
    .with_metadata(true)
    .execute()?;
```

While flexible, this approach has limitations:

1. **Language lock-in** — API must be reimplemented for each language binding
2. **No query introspection** — Hard to log, audit, or analyze queries
3. **Limited composability** — Complex queries require imperative chaining
4. **No query planning** — Optimizer can't see the full query upfront

SQL has proven its value as a universal query language for decades. A SQL-like language for vectors would provide:
- Familiar syntax for developers
- Text-based queries (loggable, auditable)
- Query plan optimization
- Language-agnostic interface

### Alternatives Considered

1. **GraphQL-style schema** — Overkill for vector search, designed for graphs
2. **JSON query language (like MongoDB)** — Verbose, doesn't compose well
3. **Pure REST with query params** — Limited expressiveness

## Decision

Needle implements **NeedleQL**, a SQL-like domain-specific language for vector operations:

### Syntax Overview

```sql
-- Basic vector similarity search
SELECT * FROM documents
WHERE vector SIMILAR TO $query
LIMIT 10;

-- With metadata filtering
SELECT id, title, distance FROM products
WHERE vector SIMILAR TO $query
  AND category = 'electronics'
  AND price < 100
LIMIT 20;

-- Hybrid search (vector + text)
SELECT * FROM articles
WHERE vector SIMILAR TO $query
  AND text MATCH 'machine learning'
ORDER BY hybrid_score(vector_score, text_score, 0.7)
LIMIT 10;

-- Time-decayed relevance
SELECT * FROM news
WHERE vector SIMILAR TO $query
ORDER BY time_decay(distance, published_at, '7d')
LIMIT 10;

-- Explain query execution
EXPLAIN ANALYZE
SELECT * FROM documents
WHERE vector SIMILAR TO $query
LIMIT 10;
```

### Grammar Definition

```
// src/query_lang.rs (simplified BNF)

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

### Query Parser

```rust
// src/query_lang.rs
pub struct QueryParser {
    tokens: Vec<Token>,
    position: usize,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Select(SelectQuery),
    Insert(InsertQuery),
    Delete(DeleteQuery),
    Explain { analyze: bool, query: Box<Statement> },
}

#[derive(Debug, Clone)]
pub struct SelectQuery {
    pub columns: SelectColumns,
    pub from: String,
    pub vector_search: Option<VectorSearch>,
    pub filters: Vec<Filter>,
    pub order_by: Option<OrderBy>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct VectorSearch {
    pub column: String,
    pub query_param: String,
    pub distance_function: Option<DistanceFunction>,
}

impl QueryParser {
    pub fn parse(sql: &str) -> Result<Statement> {
        let tokens = Lexer::tokenize(sql)?;
        let mut parser = Self { tokens, position: 0 };
        parser.parse_statement()
    }

    fn parse_select(&mut self) -> Result<SelectQuery> {
        self.expect(Token::Select)?;

        let columns = self.parse_select_list()?;

        self.expect(Token::From)?;
        let from = self.parse_identifier()?;

        let mut vector_search = None;
        let mut filters = Vec::new();

        if self.match_token(Token::Where) {
            // Parse vector similarity clause
            if self.check_ahead(&[Token::Identifier("vector"), Token::Similar]) {
                vector_search = Some(self.parse_vector_search()?);
            }

            // Parse additional filters
            while self.match_token(Token::And) {
                filters.push(self.parse_filter()?);
            }
        }

        let order_by = if self.match_token(Token::Order) {
            self.expect(Token::By)?;
            Some(self.parse_order_by()?)
        } else {
            None
        };

        let limit = if self.match_token(Token::Limit) {
            Some(self.parse_number()? as usize)
        } else {
            None
        };

        Ok(SelectQuery { columns, from, vector_search, filters, order_by, limit })
    }
}
```

### Query Executor

```rust
// src/query_lang.rs
pub struct QueryExecutor<'a> {
    database: &'a Database,
    parameters: HashMap<String, ParameterValue>,
}

pub enum ParameterValue {
    Vector(Vec<f32>),
    String(String),
    Number(f64),
    Bool(bool),
}

impl<'a> QueryExecutor<'a> {
    pub fn execute(&self, query: &Statement) -> Result<QueryResult> {
        match query {
            Statement::Select(select) => self.execute_select(select),
            Statement::Explain { analyze, query } => self.execute_explain(*analyze, query),
            Statement::Insert(insert) => self.execute_insert(insert),
            Statement::Delete(delete) => self.execute_delete(delete),
        }
    }

    fn execute_select(&self, query: &SelectQuery) -> Result<QueryResult> {
        let collection = self.database.collection(&query.from)?;

        // Get query vector from parameters
        let query_vector = if let Some(vs) = &query.vector_search {
            match self.parameters.get(&vs.query_param) {
                Some(ParameterValue::Vector(v)) => v.clone(),
                _ => return Err(NeedleError::MissingParameter(vs.query_param.clone())),
            }
        } else {
            return Err(NeedleError::InvalidQuery("Vector search required".into()));
        };

        // Build filter from SQL conditions
        let filter = self.build_filter(&query.filters)?;

        // Execute search
        let k = query.limit.unwrap_or(10);
        let results = collection.search_with_filter(&query_vector, k, filter)?;

        // Project columns
        let projected = self.project_columns(&results, &query.columns)?;

        Ok(QueryResult::Rows(projected))
    }
}
```

### Query Plan and EXPLAIN

```rust
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub steps: Vec<PlanStep>,
    pub estimated_cost: f64,
}

#[derive(Debug, Clone)]
pub enum PlanStep {
    VectorScan {
        collection: String,
        k: usize,
        ef_search: usize,
        estimated_candidates: usize,
    },
    FilterApply {
        filter: String,
        selectivity: f64,
    },
    Sort {
        columns: Vec<String>,
        limit: Option<usize>,
    },
    Project {
        columns: Vec<String>,
    },
}

impl QueryExecutor<'_> {
    fn execute_explain(&self, analyze: bool, query: &Statement) -> Result<QueryResult> {
        let plan = self.plan_query(query)?;

        if analyze {
            // Actually execute and collect stats
            let start = Instant::now();
            let result = self.execute(query)?;
            let elapsed = start.elapsed();

            Ok(QueryResult::ExplainAnalyze {
                plan,
                actual_time_ms: elapsed.as_millis() as u64,
                rows_returned: result.row_count(),
            })
        } else {
            Ok(QueryResult::Explain { plan })
        }
    }
}
```

### Example EXPLAIN Output

```sql
EXPLAIN ANALYZE SELECT * FROM products
WHERE vector SIMILAR TO $query AND category = 'electronics'
LIMIT 10;
```

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

### Type Checking

```rust
impl QueryParser {
    pub fn validate(&self, query: &Statement, schema: &Schema) -> Result<()> {
        match query {
            Statement::Select(select) => {
                // Verify collection exists
                if !schema.has_collection(&select.from) {
                    return Err(NeedleError::UnknownCollection(select.from.clone()));
                }

                // Verify columns exist
                for col in select.columns.iter() {
                    if !schema.has_column(&select.from, col) {
                        return Err(NeedleError::UnknownColumn(col.clone()));
                    }
                }

                // Verify filter types
                for filter in &select.filters {
                    self.validate_filter(filter, &select.from, schema)?;
                }

                Ok(())
            }
            // ... other statements
        }
    }
}
```

## Consequences

### Benefits

1. **Familiar syntax** — SQL developers are immediately productive
2. **Language agnostic** — Same queries work from any language/binding
3. **Auditable** — Queries are strings that can be logged
4. **Optimizable** — Query planner can reorder operations
5. **Composable** — Complex queries expressed declaratively

### Tradeoffs

1. **Parser complexity** — Full SQL-like parser is non-trivial
2. **Performance overhead** — Parsing adds latency vs direct API calls
3. **Feature parity** — Must keep query language in sync with API

### What This Enabled

- **SQL tools integration** — Use existing SQL clients and BI tools
- **Query logging** — Audit trail of all searches
- **Query optimization** — Future: cost-based optimizer

### What This Prevented

- **Language-specific APIs** — Don't need separate query builders per language
- **Opaque queries** — All queries are human-readable

## References

- Query language implementation: `src/query_lang.rs` (2091 lines)
- Parser: `src/query_lang.rs:100-500`
- Executor: `src/query_lang.rs:500-900`
- Query planning: `src/query_lang.rs:900-1200`
