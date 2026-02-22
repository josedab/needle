#![allow(clippy::unwrap_used)]
#![allow(dead_code)]

//! Native GraphQL API
//!
//! Provides GraphQL schema definitions, resolver types, and subscription support
//! for the Needle vector database. Designed to work alongside the existing REST API
//! in `src/server.rs`.
//!
//! # Architecture
//!
//! The GraphQL API maps Needle's existing Rust types to a GraphQL schema:
//!
//! - **Queries**: `collections`, `collection(name)`, `vector(collection, id)`, `search`
//! - **Mutations**: `createCollection`, `insertVector`, `deleteVector`, `deleteCollection`
//! - **Subscriptions**: `searchStream` for real-time result streaming
//!
//! # Schema
//!
//! ```graphql
//! type Query {
//!   collections: [Collection!]!
//!   collection(name: String!): Collection
//!   vector(collection: String!, id: String!): Vector
//!   search(input: SearchInput!): SearchResponse!
//!   health: HealthStatus!
//! }
//!
//! type Mutation {
//!   createCollection(input: CreateCollectionInput!): Collection!
//!   insertVector(input: InsertVectorInput!): Vector!
//!   deleteVector(collection: String!, id: String!): Boolean!
//!   deleteCollection(name: String!): Boolean!
//!   save: Boolean!
//! }
//!
//! type Subscription {
//!   searchStream(input: SearchInput!): SearchResult!
//! }
//! ```
//!
//! # Feature Flag
//!
//! This module provides the schema types and resolver logic. Integration with
//! `async-graphql` and the HTTP server requires the `server` feature.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::search::graphql_api::{GraphqlSchema, QueryRoot, MutationRoot};
//!
//! let schema = GraphqlSchema::build(db.clone());
//! // Serve via /graphql endpoint
//! ```

use crate::database::Database;
use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ── GraphQL Type Definitions ────────────────────────────────────────────────

/// GraphQL representation of a Collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GqlCollection {
    /// Collection name.
    pub name: String,
    /// Vector dimensions.
    pub dimensions: usize,
    /// Distance function.
    pub distance_function: String,
    /// Number of vectors.
    pub vector_count: usize,
    /// Whether the collection is empty.
    pub is_empty: bool,
}

/// GraphQL representation of a Vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GqlVector {
    /// Vector ID.
    pub id: String,
    /// Vector data.
    pub vector: Vec<f32>,
    /// Metadata as JSON string.
    pub metadata: Option<String>,
    /// Collection this vector belongs to.
    pub collection: String,
}

/// GraphQL search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GqlSearchResult {
    /// Vector ID.
    pub id: String,
    /// Distance from query.
    pub distance: f32,
    /// Metadata as JSON string.
    pub metadata: Option<String>,
    /// Source collection.
    pub collection: String,
}

/// GraphQL search response with stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GqlSearchResponse {
    /// Search results.
    pub results: Vec<GqlSearchResult>,
    /// Total results found.
    pub total: usize,
    /// Query time in microseconds.
    pub query_time_us: u64,
    /// Collection searched.
    pub collection: String,
}

/// GraphQL health status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GqlHealthStatus {
    /// Whether the database is healthy.
    pub healthy: bool,
    /// Database version.
    pub version: String,
    /// Number of collections.
    pub collection_count: usize,
    /// Total vectors across all collections.
    pub total_vectors: usize,
    /// Uptime description.
    pub uptime: String,
}

// ── Input Types ─────────────────────────────────────────────────────────────

/// Input for creating a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCollectionInput {
    /// Collection name.
    pub name: String,
    /// Vector dimensions.
    pub dimensions: usize,
    /// Distance function (optional, defaults to "cosine").
    pub distance_function: Option<String>,
}

/// Input for inserting a vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertVectorInput {
    /// Collection name.
    pub collection: String,
    /// Vector ID.
    pub id: String,
    /// Vector data.
    pub vector: Vec<f32>,
    /// Metadata as JSON string.
    pub metadata: Option<String>,
}

/// Input for search queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchInput {
    /// Collection to search.
    pub collection: String,
    /// Query vector.
    pub vector: Vec<f32>,
    /// Number of results (default: 10).
    pub k: Option<usize>,
    /// Metadata filter as JSON string.
    pub filter: Option<String>,
    /// Include metadata in results.
    pub include_metadata: Option<bool>,
    /// Ef search parameter for HNSW.
    pub ef_search: Option<usize>,
}

// ── Resolver Logic ──────────────────────────────────────────────────────────

/// GraphQL query resolvers.
pub struct QueryResolver {
    db: Arc<Database>,
}

impl QueryResolver {
    /// Create a new query resolver.
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Resolve: list all collections.
    pub fn collections(&self) -> Result<Vec<GqlCollection>> {
        let names = self.db.list_collections();
        let mut collections = Vec::new();

        for name in names {
            if let Ok(coll) = self.db.collection(&name) {
                collections.push(GqlCollection {
                    name: name.clone(),
                    dimensions: coll.dimensions().unwrap_or(0),
                    distance_function: coll
                        .stats()
                        .map(|s| format!("{:?}", s.distance_function))
                        .unwrap_or_default(),
                    vector_count: coll.len(),
                    is_empty: coll.is_empty(),
                });
            }
        }

        Ok(collections)
    }

    /// Resolve: get a single collection by name.
    pub fn collection(&self, name: &str) -> Result<GqlCollection> {
        let coll = self.db.collection(name)?;
        Ok(GqlCollection {
            name: name.to_string(),
            dimensions: coll.dimensions().unwrap_or(0),
            distance_function: coll
                .stats()
                .map(|s| format!("{:?}", s.distance_function))
                .unwrap_or_default(),
            vector_count: coll.len(),
            is_empty: coll.is_empty(),
        })
    }

    /// Resolve: get a vector by ID.
    pub fn vector(&self, collection: &str, id: &str) -> Result<GqlVector> {
        let coll = self.db.collection(collection)?;
        let (vec, meta) = coll
            .get(id)
            .ok_or_else(|| NeedleError::VectorNotFound(id.to_string()))?;

        Ok(GqlVector {
            id: id.to_string(),
            vector: vec,
            metadata: meta.map(|m| m.to_string()),
            collection: collection.to_string(),
        })
    }

    /// Resolve: search for vectors.
    pub fn search(&self, input: &SearchInput) -> Result<GqlSearchResponse> {
        let start = std::time::Instant::now();
        let coll = self.db.collection(&input.collection)?;

        let k = input.k.unwrap_or(10);
        let results = if let Some(ref filter_str) = input.filter {
            let filter_json: serde_json::Value = serde_json::from_str(filter_str)
                .map_err(|e| NeedleError::InvalidInput(format!("Invalid filter JSON: {}", e)))?;
            let filter = crate::metadata::Filter::parse(&filter_json)
                .map_err(|e| NeedleError::InvalidInput(format!("Invalid filter: {}", e)))?;
            coll.search_with_filter(&input.vector, k, &filter)?
        } else {
            coll.search(&input.vector, k)?
        };

        let query_time_us = start.elapsed().as_micros() as u64;

        let gql_results: Vec<GqlSearchResult> = results
            .iter()
            .map(|r| GqlSearchResult {
                id: r.id.clone(),
                distance: r.distance,
                metadata: r.metadata.as_ref().map(|m| m.to_string()),
                collection: input.collection.clone(),
            })
            .collect();

        let total = gql_results.len();

        Ok(GqlSearchResponse {
            results: gql_results,
            total,
            query_time_us,
            collection: input.collection.clone(),
        })
    }

    /// Resolve: health check.
    pub fn health(&self) -> GqlHealthStatus {
        let collections = self.db.list_collections();
        let total_vectors: usize = collections
            .iter()
            .filter_map(|name| {
                self.db.collection(name).ok().map(|c| c.len())
            })
            .sum();

        GqlHealthStatus {
            healthy: true,
            version: env!("CARGO_PKG_VERSION").to_string(),
            collection_count: collections.len(),
            total_vectors,
            uptime: "unknown".to_string(),
        }
    }
}

/// GraphQL mutation resolvers.
pub struct MutationResolver {
    db: Arc<Database>,
}

impl MutationResolver {
    /// Create a new mutation resolver.
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Resolve: create a collection.
    pub fn create_collection(&self, input: &CreateCollectionInput) -> Result<GqlCollection> {
        self.db.create_collection(&input.name, input.dimensions)?;

        Ok(GqlCollection {
            name: input.name.clone(),
            dimensions: input.dimensions,
            distance_function: input
                .distance_function
                .clone()
                .unwrap_or_else(|| "Cosine".to_string()),
            vector_count: 0,
            is_empty: true,
        })
    }

    /// Resolve: insert a vector.
    pub fn insert_vector(&self, input: &InsertVectorInput) -> Result<GqlVector> {
        let coll = self.db.collection(&input.collection)?;

        let metadata = input
            .metadata
            .as_ref()
            .map(|s| serde_json::from_str(s))
            .transpose()
            .map_err(|e| NeedleError::InvalidInput(format!("Invalid metadata JSON: {}", e)))?;

        coll.insert(&input.id, &input.vector, metadata)?;

        Ok(GqlVector {
            id: input.id.clone(),
            vector: input.vector.clone(),
            metadata: input.metadata.clone(),
            collection: input.collection.clone(),
        })
    }

    /// Resolve: delete a vector.
    pub fn delete_vector(&self, collection: &str, id: &str) -> Result<bool> {
        let coll = self.db.collection(collection)?;
        coll.delete(id)
    }

    /// Resolve: delete a collection.
    pub fn delete_collection(&self, name: &str) -> Result<bool> {
        self.db.drop_collection(name)?;
        Ok(true)
    }

    /// Resolve: batch insert vectors.
    pub fn batch_insert(
        &self,
        collection: &str,
        inputs: &[InsertVectorInput],
    ) -> Result<usize> {
        let coll = self.db.collection(collection)?;
        let mut count = 0;

        for input in inputs {
            let metadata = input
                .metadata
                .as_ref()
                .map(|s| serde_json::from_str(s))
                .transpose()
                .map_err(|e| NeedleError::InvalidInput(format!("Invalid metadata JSON: {}", e)))?;

            coll.insert(&input.id, &input.vector, metadata)?;
            count += 1;
        }

        Ok(count)
    }

    /// Resolve: save the database.
    ///
    /// Note: requires mutable access to the database. In production, this would
    /// be coordinated through the server's write lock.
    pub fn save(&self) -> Result<bool> {
        // Save requires &mut self on Database. In a real server, this would
        // be coordinated through the write path. Return an error explaining this.
        Err(NeedleError::InvalidOperation(
            "Save must be coordinated through the server's write path".to_string(),
        ))
    }
}

/// Subscription event for real-time search result streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchStreamEvent {
    /// Search result.
    pub result: GqlSearchResult,
    /// Sequence number.
    pub sequence: usize,
    /// Whether this is the last result.
    pub is_last: bool,
}

/// GraphQL schema container holding query and mutation resolvers.
pub struct GraphqlSchema {
    pub query: QueryResolver,
    pub mutation: MutationResolver,
}

impl GraphqlSchema {
    /// Build a GraphQL schema from a database reference.
    pub fn build(db: Arc<Database>) -> Self {
        Self {
            query: QueryResolver::new(db.clone()),
            mutation: MutationResolver::new(db),
        }
    }
}

/// Generate the GraphQL SDL (Schema Definition Language) string.
pub fn generate_sdl() -> &'static str {
    r#"
type Query {
  collections: [Collection!]!
  collection(name: String!): Collection
  vector(collection: String!, id: String!): Vector
  search(input: SearchInput!): SearchResponse!
  health: HealthStatus!
}

type Mutation {
  createCollection(input: CreateCollectionInput!): Collection!
  insertVector(input: InsertVectorInput!): Vector!
  deleteVector(collection: String!, id: String!): Boolean!
  deleteCollection(name: String!): Boolean!
  save: Boolean!
}

type Subscription {
  searchStream(input: SearchInput!): SearchResult!
}

type Collection {
  name: String!
  dimensions: Int!
  distanceFunction: String!
  vectorCount: Int!
  isEmpty: Boolean!
}

type Vector {
  id: String!
  vector: [Float!]!
  metadata: String
  collection: String!
}

type SearchResult {
  id: String!
  distance: Float!
  metadata: String
  collection: String!
}

type SearchResponse {
  results: [SearchResult!]!
  total: Int!
  queryTimeUs: Int!
  collection: String!
}

type HealthStatus {
  healthy: Boolean!
  version: String!
  collectionCount: Int!
  totalVectors: Int!
  uptime: String!
}

input CreateCollectionInput {
  name: String!
  dimensions: Int!
  distanceFunction: String
}

input InsertVectorInput {
  collection: String!
  id: String!
  vector: [Float!]!
  metadata: String
}

input SearchInput {
  collection: String!
  vector: [Float!]!
  k: Int
  filter: String
  includeMetadata: Boolean
  efSearch: Int
}
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_build() {
        let db = Arc::new(Database::in_memory());
        let schema = GraphqlSchema::build(db);

        let health = schema.query.health();
        assert!(health.healthy);
        assert_eq!(health.collection_count, 0);
    }

    #[test]
    fn test_collection_crud() {
        let db = Arc::new(Database::in_memory());
        let schema = GraphqlSchema::build(db);

        // Create
        let input = CreateCollectionInput {
            name: "test".to_string(),
            dimensions: 128,
            distance_function: None,
        };
        let coll = schema.mutation.create_collection(&input).unwrap();
        assert_eq!(coll.name, "test");
        assert_eq!(coll.dimensions, 128);

        // List
        let collections = schema.query.collections().unwrap();
        assert_eq!(collections.len(), 1);

        // Get
        let coll = schema.query.collection("test").unwrap();
        assert_eq!(coll.vector_count, 0);

        // Delete
        schema.mutation.delete_collection("test").unwrap();
        assert!(schema.query.collection("test").is_err());
    }

    #[test]
    fn test_vector_operations() {
        let db = Arc::new(Database::in_memory());
        let schema = GraphqlSchema::build(db);

        schema
            .mutation
            .create_collection(&CreateCollectionInput {
                name: "docs".to_string(),
                dimensions: 4,
                distance_function: None,
            })
            .unwrap();

        // Insert
        let input = InsertVectorInput {
            collection: "docs".to_string(),
            id: "v1".to_string(),
            vector: vec![1.0, 0.0, 0.0, 0.0],
            metadata: Some(r#"{"title": "test"}"#.to_string()),
        };
        schema.mutation.insert_vector(&input).unwrap();

        // Get
        let vec = schema.query.vector("docs", "v1").unwrap();
        assert_eq!(vec.id, "v1");
        assert!(vec.metadata.is_some());

        // Search
        let search_input = SearchInput {
            collection: "docs".to_string(),
            vector: vec![1.0, 0.0, 0.0, 0.0],
            k: Some(5),
            filter: None,
            include_metadata: Some(true),
            ef_search: None,
        };
        let response = schema.query.search(&search_input).unwrap();
        assert_eq!(response.total, 1);
        assert_eq!(response.results[0].id, "v1");

        // Delete
        schema.mutation.delete_vector("docs", "v1").unwrap();
    }

    #[test]
    fn test_sdl_generation() {
        let sdl = generate_sdl();
        assert!(sdl.contains("type Query"));
        assert!(sdl.contains("type Mutation"));
        assert!(sdl.contains("type Subscription"));
    }
}
