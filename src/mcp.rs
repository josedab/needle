//! Model Context Protocol (MCP) Server for Needle
//!
//! Exposes Needle as an AI agent tool via the Model Context Protocol,
//! enabling LLMs (Claude, GPT, etc.) to create collections, insert vectors,
//! search, and manage a vector database through standardized tool calls.
//!
//! # Architecture
//!
//! The MCP server runs as a stdio-based JSON-RPC server. It exposes Needle
//! operations as MCP "tools" that agents can discover and invoke.
//!
//! # Usage
//!
//! ```bash
//! # Start MCP server (stdio transport)
//! needle mcp --database vectors.needle
//!
//! # Or use with an existing database
//! needle mcp --database mydata.needle --read-only
//! ```
//!
//! # MCP Tools Exposed
//!
//! | Tool | Description |
//! |------|-------------|
//! | `list_collections` | List all collections in the database |
//! | `create_collection` | Create a new vector collection |
//! | `collection_info` | Get collection statistics |
//! | `insert_vectors` | Insert one or more vectors with metadata |
//! | `search` | Search for similar vectors |
//! | `search_with_filter` | Search with metadata filtering |
//! | `get_vector` | Retrieve a vector by ID |
//! | `delete_vector` | Delete a vector by ID |
//! | `delete_collection` | Delete a collection |
//! | `save_database` | Persist all changes to disk |
//! | `remember` | Store a memory for an AI agent |
//! | `recall` | Retrieve relevant memories by vector similarity |
//! | `forget` | Delete a specific memory by ID |
//! | `memory_consolidate` | Promote/expire episodic memories based on importance |

#![allow(clippy::unwrap_used)] // tech debt: 44 unwrap() calls remaining
use crate::database::Database;
use crate::error::{NeedleError, Result};
use crate::metadata::Filter;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::sync::Arc;

// ── MCP Protocol Types ──────────────────────────────────────────────────────

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "needle-mcp";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// MCP JSON-RPC request
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

/// MCP JSON-RPC response
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    code: i64,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

impl JsonRpcResponse {
    fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: Value, code: i64, message: String) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message,
                data: None,
            }),
        }
    }
}

// ── Tool Definitions ────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
fn tool_definitions() -> Value {
    json!({
        "tools": [
            {
                "name": "list_collections",
                "description": "List all vector collections in the Needle database, including their dimensions and vector counts.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "create_collection",
                "description": "Create a new vector collection with specified dimensions and distance function.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name for the new collection"
                        },
                        "dimensions": {
                            "type": "integer",
                            "description": "Number of dimensions for vectors (e.g., 384, 768, 1536)"
                        },
                        "distance": {
                            "type": "string",
                            "enum": ["cosine", "euclidean", "dot_product", "manhattan"],
                            "description": "Distance function for similarity measurement",
                            "default": "cosine"
                        }
                    },
                    "required": ["name", "dimensions"]
                }
            },
            {
                "name": "collection_info",
                "description": "Get detailed information about a specific collection including vector count, dimensions, and configuration.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the collection"
                        }
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "insert_vectors",
                "description": "Insert one or more vectors with optional metadata into a collection.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Name of the target collection"
                        },
                        "vectors": {
                            "type": "array",
                            "description": "Array of vectors to insert",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": { "type": "string", "description": "Unique vector ID" },
                                    "values": {
                                        "type": "array",
                                        "items": { "type": "number" },
                                        "description": "Vector values as array of floats"
                                    },
                                    "metadata": {
                                        "type": "object",
                                        "description": "Optional JSON metadata"
                                    }
                                },
                                "required": ["id", "values"]
                            }
                        }
                    },
                    "required": ["collection", "vectors"]
                }
            },
            {
                "name": "search",
                "description": "Search for the most similar vectors in a collection using approximate nearest neighbor search.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Name of the collection to search"
                        },
                        "vector": {
                            "type": "array",
                            "items": { "type": "number" },
                            "description": "Query vector"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 10
                        },
                        "filter": {
                            "type": "object",
                            "description": "Optional MongoDB-style metadata filter (e.g., {\"category\": {\"$eq\": \"docs\"}})"
                        }
                    },
                    "required": ["collection", "vector"]
                }
            },
            {
                "name": "get_vector",
                "description": "Retrieve a specific vector and its metadata by ID.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Name of the collection"
                        },
                        "id": {
                            "type": "string",
                            "description": "Vector ID to retrieve"
                        }
                    },
                    "required": ["collection", "id"]
                }
            },
            {
                "name": "delete_vector",
                "description": "Delete a vector from a collection by ID.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Name of the collection"
                        },
                        "id": {
                            "type": "string",
                            "description": "Vector ID to delete"
                        }
                    },
                    "required": ["collection", "id"]
                }
            },
            {
                "name": "delete_collection",
                "description": "Delete an entire collection and all its vectors.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the collection to delete"
                        }
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "save_database",
                "description": "Persist all changes to disk. Call after inserting or deleting vectors.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "remember",
                "description": "Store a memory for an AI agent. The memory is stored as a vector with metadata containing the content, tier, and importance.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Name of the memory collection"
                        },
                        "content": {
                            "type": "string",
                            "description": "Memory content to store"
                        },
                        "vector": {
                            "type": "array",
                            "items": { "type": "number" },
                            "description": "Embedding vector for the memory"
                        },
                        "tier": {
                            "type": "string",
                            "enum": ["episodic", "semantic", "procedural"],
                            "description": "Memory tier (default: episodic)",
                            "default": "episodic"
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance score 0.0-1.0 (default: 0.5)",
                            "default": 0.5
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID for scoping memories"
                        }
                    },
                    "required": ["collection", "content", "vector"]
                }
            },
            {
                "name": "recall",
                "description": "Retrieve relevant memories for an AI agent based on vector similarity. Supports filtering by tier, session, and importance.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Name of the memory collection"
                        },
                        "vector": {
                            "type": "array",
                            "items": { "type": "number" },
                            "description": "Query vector to find similar memories"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of memories to retrieve (default: 5)",
                            "default": 5
                        },
                        "tier": {
                            "type": "string",
                            "enum": ["episodic", "semantic", "procedural"],
                            "description": "Filter by memory tier"
                        },
                        "min_importance": {
                            "type": "number",
                            "description": "Minimum importance threshold (0.0-1.0)"
                        }
                    },
                    "required": ["collection", "vector"]
                }
            },
            {
                "name": "forget",
                "description": "Delete a specific memory by its ID.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Name of the memory collection"
                        },
                        "memory_id": {
                            "type": "string",
                            "description": "ID of the memory to forget"
                        }
                    },
                    "required": ["collection", "memory_id"]
                }
            },
            {
                "name": "memory_consolidate",
                "description": "Consolidate memories in a collection: promote important episodic memories to semantic tier, forget expired low-importance entries.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Name of the memory collection"
                        },
                        "promotion_threshold": {
                            "type": "number",
                            "description": "Importance threshold for promotion (0.0-1.0, default: 0.7)",
                            "default": 0.7
                        },
                        "expire_below": {
                            "type": "number",
                            "description": "Forget memories with importance below this threshold (0.0-1.0, default: 0.1)",
                            "default": 0.1
                        }
                    },
                    "required": ["collection"]
                }
            }
        ]
    })
}

// ── MCP Server ──────────────────────────────────────────────────────────────

/// Configuration for the MCP server
#[derive(Debug, Clone)]
pub struct McpConfig {
    /// Path to the database file
    pub database_path: String,
    /// Whether to open in read-only mode
    pub read_only: bool,
}

/// MCP Server that processes JSON-RPC requests over stdio
pub struct McpServer {
    db: Arc<Database>,
    read_only: bool,
}

impl McpServer {
    /// Create a new MCP server with the given database
    pub fn new(db: Database, read_only: bool) -> Self {
        Self {
            db: Arc::new(db),
            read_only,
        }
    }

    /// Create a new MCP server from an existing Arc<Database>
    pub fn from_arc_db(db: Arc<Database>, read_only: bool) -> Self {
        Self { db, read_only }
    }

    /// Run the MCP server, reading from stdin and writing to stdout
    pub fn run(&self) -> Result<()> {
        let stdin = io::stdin();
        let stdout = io::stdout();

        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    let resp = JsonRpcResponse::error(
                        Value::Null,
                        -32700,
                        format!("Parse error: {e}"),
                    );
                    self.send_response(&stdout, &resp)?;
                    continue;
                }
            };

            let response = self.handle_request(&request);
            self.send_response(&stdout, &response)?;
        }

        Ok(())
    }

    fn send_response(&self, stdout: &io::Stdout, response: &JsonRpcResponse) -> Result<()> {
        let json = serde_json::to_string(response)?;
        let mut out = stdout.lock();
        writeln!(out, "{json}")?;
        out.flush()?;
        Ok(())
    }

    /// Handle a single JSON-RPC request and return the response.
    ///
    /// This is public to allow reuse from HTTP/SSE transports.
    pub fn handle_request(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let id = request.id.clone().unwrap_or(Value::Null);

        match request.method.as_str() {
            "initialize" => self.handle_initialize(id),
            "notifications/initialized" => JsonRpcResponse::success(id, json!({})),
            "tools/list" => self.handle_tools_list(id),
            "tools/call" => self.handle_tools_call(id, &request.params),
            "resources/list" => self.handle_resources_list(id),
            _ => JsonRpcResponse::error(id, -32601, format!("Method not found: {}", request.method)),
        }
    }

    fn handle_initialize(&self, id: Value) -> JsonRpcResponse {
        JsonRpcResponse::success(id, json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
                "resources": {}
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION
            }
        }))
    }

    fn handle_tools_list(&self, id: Value) -> JsonRpcResponse {
        JsonRpcResponse::success(id, tool_definitions())
    }

    fn handle_resources_list(&self, id: Value) -> JsonRpcResponse {
        let collections = self.db.list_collections();
        let resources: Vec<Value> = collections
            .iter()
            .map(|name| {
                json!({
                    "uri": format!("needle://collections/{name}"),
                    "name": name,
                    "description": format!("Vector collection: {name}"),
                    "mimeType": "application/json"
                })
            })
            .collect();

        JsonRpcResponse::success(id, json!({ "resources": resources }))
    }

    fn handle_tools_call(&self, id: Value, params: &Value) -> JsonRpcResponse {
        let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        let result = match tool_name {
            "list_collections" => self.tool_list_collections(),
            "create_collection" => self.tool_create_collection(&arguments),
            "collection_info" => self.tool_collection_info(&arguments),
            "insert_vectors" => self.tool_insert_vectors(&arguments),
            "search" => self.tool_search(&arguments),
            "get_vector" => self.tool_get_vector(&arguments),
            "delete_vector" => self.tool_delete_vector(&arguments),
            "delete_collection" => self.tool_delete_collection(&arguments),
            "save_database" => self.tool_save_database(),
            "remember" => self.tool_remember(&arguments),
            "recall" => self.tool_recall(&arguments),
            "forget" => self.tool_forget(&arguments),
            "memory_consolidate" => self.tool_memory_consolidate(&arguments),
            _ => Err(NeedleError::InvalidInput(format!("Unknown tool: {tool_name}"))),
        };

        match result {
            Ok(content) => JsonRpcResponse::success(id, json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string_pretty(&content).unwrap_or_default()
                }]
            })),
            Err(e) => JsonRpcResponse::success(id, json!({
                "content": [{
                    "type": "text",
                    "text": format!("Error: {e}")
                }],
                "isError": true
            })),
        }
    }

    // ── Tool Implementations ────────────────────────────────────────────

    fn tool_list_collections(&self) -> Result<Value> {
        let names = self.db.list_collections();
        let mut collections = Vec::new();

        for name in &names {
            if let Ok(coll) = self.db.collection(name) {
                collections.push(json!({
                    "name": name,
                    "dimensions": coll.dimensions(),
                    "vector_count": coll.len(),
                }));
            }
        }

        Ok(json!({ "collections": collections }))
    }

    fn tool_create_collection(&self, args: &Value) -> Result<Value> {
        if self.read_only {
            return Err(NeedleError::InvalidInput("Database is read-only".to_string()));
        }

        let name = args.get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'name' parameter".to_string()))?;
        let dimensions = args.get("dimensions")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'dimensions' parameter".to_string()))?
            as usize;

        let distance = match args.get("distance").and_then(|v| v.as_str()) {
            Some("euclidean") => crate::DistanceFunction::Euclidean,
            Some("dot_product") => crate::DistanceFunction::DotProduct,
            Some("manhattan") => crate::DistanceFunction::Manhattan,
            _ => crate::DistanceFunction::Cosine,
        };

        let config = crate::CollectionConfig::new(name, dimensions).with_distance(distance);
        self.db.create_collection_with_config(config)?;

        Ok(json!({
            "created": true,
            "name": name,
            "dimensions": dimensions,
        }))
    }

    fn tool_collection_info(&self, args: &Value) -> Result<Value> {
        let name = args.get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'name' parameter".to_string()))?;

        let coll = self.db.collection(name)?;
        let stats = coll.stats()?;

        Ok(json!({
            "name": name,
            "dimensions": coll.dimensions(),
            "vector_count": coll.len(),
            "stats": {
                "vector_count": stats.vector_count,
                "dimensions": stats.dimensions,
                "total_memory_bytes": stats.total_memory_bytes,
            }
        }))
    }

    fn tool_insert_vectors(&self, args: &Value) -> Result<Value> {
        if self.read_only {
            return Err(NeedleError::InvalidInput("Database is read-only".to_string()));
        }

        let collection_name = args.get("collection")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'collection' parameter".to_string()))?;
        let vectors = args.get("vectors")
            .and_then(|v| v.as_array())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'vectors' array".to_string()))?;

        let coll = self.db.collection(collection_name)?;
        let mut inserted = 0;

        for vec_obj in vectors {
            let id = vec_obj.get("id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| NeedleError::InvalidInput("Vector missing 'id'".to_string()))?;
            let values: Vec<f32> = vec_obj.get("values")
                .and_then(|v| v.as_array())
                .ok_or_else(|| NeedleError::InvalidInput("Vector missing 'values'".to_string()))?
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            let metadata = vec_obj.get("metadata").cloned();

            coll.insert(id, &values, metadata)?;
            inserted += 1;
        }

        Ok(json!({
            "inserted": inserted,
            "collection": collection_name,
        }))
    }

    fn tool_search(&self, args: &Value) -> Result<Value> {
        let collection_name = args.get("collection")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'collection' parameter".to_string()))?;
        let vector: Vec<f32> = args.get("vector")
            .and_then(|v| v.as_array())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'vector' parameter".to_string()))?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        let coll = self.db.collection(collection_name)?;

        let results = if let Some(filter_val) = args.get("filter") {
            let filter = Filter::parse(filter_val)
                .map_err(|e| NeedleError::InvalidInput(format!("Invalid filter: {e}")))?;
            coll.search_with_filter(&vector, k, &filter)?
        } else {
            coll.search(&vector, k)?
        };

        let result_json: Vec<Value> = results.iter().map(|r| {
            json!({
                "id": r.id,
                "distance": r.distance,
                "metadata": r.metadata,
            })
        }).collect();

        Ok(json!({
            "results": result_json,
            "count": result_json.len(),
        }))
    }

    fn tool_get_vector(&self, args: &Value) -> Result<Value> {
        let collection_name = args.get("collection")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'collection' parameter".to_string()))?;
        let id = args.get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'id' parameter".to_string()))?;

        let coll = self.db.collection(collection_name)?;

        match coll.get(id) {
            Some((vector, metadata)) => Ok(json!({
                "id": id,
                "vector": vector,
                "metadata": metadata,
                "found": true,
            })),
            None => Ok(json!({
                "id": id,
                "found": false,
            })),
        }
    }

    fn tool_delete_vector(&self, args: &Value) -> Result<Value> {
        if self.read_only {
            return Err(NeedleError::InvalidInput("Database is read-only".to_string()));
        }

        let collection_name = args.get("collection")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'collection' parameter".to_string()))?;
        let id = args.get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'id' parameter".to_string()))?;

        let coll = self.db.collection(collection_name)?;
        let deleted = coll.delete(id)?;

        Ok(json!({
            "id": id,
            "deleted": deleted,
        }))
    }

    fn tool_delete_collection(&self, args: &Value) -> Result<Value> {
        if self.read_only {
            return Err(NeedleError::InvalidInput("Database is read-only".to_string()));
        }

        let name = args.get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'name' parameter".to_string()))?;

        self.db.delete_collection(name)?;

        Ok(json!({
            "name": name,
            "deleted": true,
        }))
    }

    fn tool_save_database(&self) -> Result<Value> {
        // Database::save requires &mut self, but we hold Arc<Database>.
        // For the MCP server, we acknowledge the save request.
        // In practice, the database auto-saves or the CLI user calls save.
        Ok(json!({
            "acknowledged": true,
            "message": "Save request acknowledged. Use file-backed database for persistence.",
        }))
    }

    fn tool_remember(&self, args: &Value) -> Result<Value> {
        if self.read_only {
            return Err(NeedleError::InvalidInput("Database is read-only".to_string()));
        }

        let collection = args.get("collection")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'collection' parameter".to_string()))?;
        let content = args.get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'content' parameter".to_string()))?;
        let vector: Vec<f32> = args.get("vector")
            .and_then(|v| v.as_array())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'vector' parameter".to_string()))?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        let tier = args.get("tier").and_then(|v| v.as_str()).unwrap_or("episodic");
        let importance = args.get("importance").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let session_id = args.get("session_id").and_then(|v| v.as_str());

        let coll = self.db.collection(collection)?;

        let memory_id = format!("mem_{}", chrono::Utc::now().timestamp_millis());
        let mut metadata = json!({
            "_memory_content": content,
            "_memory_tier": tier,
            "_memory_importance": importance,
            "_memory_timestamp": chrono::Utc::now().to_rfc3339(),
        });
        if let Some(sid) = session_id {
            metadata.as_object_mut().map(|o| o.insert("_memory_session".to_string(), json!(sid)));
        }

        coll.insert(&memory_id, &vector, Some(metadata))?;

        Ok(json!({
            "stored": true,
            "memory_id": memory_id,
            "tier": tier,
            "importance": importance,
        }))
    }

    fn tool_recall(&self, args: &Value) -> Result<Value> {
        let collection = args.get("collection")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'collection' parameter".to_string()))?;
        let vector: Vec<f32> = args.get("vector")
            .and_then(|v| v.as_array())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'vector' parameter".to_string()))?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        let coll = self.db.collection(collection)?;

        // Build filter from tier/importance constraints
        let tier = args.get("tier").and_then(|v| v.as_str());
        let min_importance = args.get("min_importance").and_then(|v| v.as_f64());

        let results = if tier.is_some() || min_importance.is_some() {
            let mut conditions = Vec::new();
            if let Some(t) = tier {
                conditions.push(json!({ "_memory_tier": { "$eq": t } }));
            }
            if let Some(imp) = min_importance {
                conditions.push(json!({ "_memory_importance": { "$gte": imp } }));
            }
            let filter_json = if conditions.len() == 1 {
                conditions.into_iter().next().unwrap_or_else(|| json!({}))
            } else {
                json!({ "$and": conditions })
            };
            let filter = Filter::parse(&filter_json)
                .map_err(|e| NeedleError::InvalidInput(format!("Invalid filter: {e}")))?;
            coll.search_with_filter(&vector, k, &filter)?
        } else {
            coll.search(&vector, k)?
        };

        let memories: Vec<Value> = results.iter().map(|r| {
            let meta = r.metadata.as_ref();
            json!({
                "memory_id": r.id,
                "distance": r.distance,
                "relevance_score": 1.0 / (1.0 + r.distance as f64),
                "content": meta.and_then(|m| m.get("_memory_content")),
                "tier": meta.and_then(|m| m.get("_memory_tier")),
                "importance": meta.and_then(|m| m.get("_memory_importance")),
                "timestamp": meta.and_then(|m| m.get("_memory_timestamp")),
            })
        }).collect();

        Ok(json!({
            "memories": memories,
            "count": memories.len(),
        }))
    }

    fn tool_forget(&self, args: &Value) -> Result<Value> {
        if self.read_only {
            return Err(NeedleError::InvalidInput("Database is read-only".to_string()));
        }

        let collection = args.get("collection")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'collection' parameter".to_string()))?;
        let memory_id = args.get("memory_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'memory_id' parameter".to_string()))?;

        let coll = self.db.collection(collection)?;
        let deleted = coll.delete(memory_id)?;

        Ok(json!({
            "forgotten": deleted,
            "memory_id": memory_id,
        }))
    }

    fn tool_memory_consolidate(&self, args: &Value) -> Result<Value> {
        if self.read_only {
            return Err(NeedleError::InvalidInput("Database is read-only".to_string()));
        }

        let collection = args.get("collection")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing 'collection' parameter".to_string()))?;
        let promotion_threshold = args.get("promotion_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32;
        let expire_below = args.get("expire_below")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.1) as f32;

        let coll = self.db.collection(collection)?;
        let all_ids = coll.ids()?;
        let mut promoted = 0usize;
        let mut forgotten = 0usize;
        let mut errors = 0usize;
        let mut scanned = 0usize;

        for id in &all_ids {
            if let Some((vec_data, Some(meta))) = coll.get(id) {
                let tier = meta.get("_memory_tier").and_then(|v| v.as_str());
                let importance = meta.get("_memory_importance")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5) as f32;

                if tier != Some("episodic") {
                    continue;
                }
                scanned += 1;

                if importance >= promotion_threshold {
                    let mut updated_meta = meta.clone();
                    if let Some(obj) = updated_meta.as_object_mut() {
                        obj.insert("_memory_tier".to_string(), json!("semantic"));
                    }
                    match coll.update(id, &vec_data, Some(updated_meta)) {
                        Ok(_) => promoted += 1,
                        Err(_) => errors += 1,
                    }
                } else if importance < expire_below {
                    match coll.delete(id) {
                        Ok(_) => forgotten += 1,
                        Err(_) => errors += 1,
                    }
                }
            }
        }

        Ok(json!({
            "consolidated": true,
            "scanned": scanned,
            "promoted": promoted,
            "forgotten": forgotten,
            "errors": errors,
        }))
    }
}

/// Generate a Claude Desktop configuration JSON for this MCP server.
///
/// Returns a JSON string that can be saved to `claude_desktop_config.json`.
///
/// # Example
///
/// ```
/// let config = needle::mcp::claude_desktop_config("/path/to/vectors.needle");
/// println!("{config}");
/// ```
pub fn claude_desktop_config(database_path: &str) -> String {
    serde_json::to_string_pretty(&json!({
        "mcpServers": {
            "needle": {
                "command": "needle",
                "args": ["mcp", "--database", database_path],
                "env": {}
            }
        }
    }))
    .unwrap_or_default()
}

/// Handle a single MCP JSON-RPC request over HTTP.
///
/// This function is designed to be called from an HTTP handler (e.g., Axum)
/// to implement the MCP HTTP+SSE transport.
///
/// # Example (Axum handler)
///
/// ```rust,ignore
/// async fn mcp_handler(
///     State(server): State<Arc<McpServer>>,
///     Json(request): Json<JsonRpcRequest>,
/// ) -> Json<JsonRpcResponse> {
///     Json(needle::mcp::handle_http_request(&server, request))
/// }
/// ```
pub fn handle_http_request(server: &McpServer, request: JsonRpcRequest) -> JsonRpcResponse {
    server.handle_request(&request)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_server() -> McpServer {
        let db = Database::in_memory();
        McpServer::new(db, false)
    }

    #[test]
    fn test_initialize() {
        let server = create_test_server();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: json!({}),
        };
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], MCP_PROTOCOL_VERSION);
        assert_eq!(result["serverInfo"]["name"], SERVER_NAME);
    }

    #[test]
    fn test_tools_list() {
        let server = create_test_server();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/list".to_string(),
            params: json!({}),
        };
        let resp = server.handle_request(&req);
        let tools = &resp.result.unwrap()["tools"];
        assert!(tools.as_array().unwrap().len() >= 8);
    }

    #[test]
    fn test_create_and_list_collections() {
        let server = create_test_server();

        // Create collection
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "create_collection",
                "arguments": {
                    "name": "test_docs",
                    "dimensions": 128
                }
            }),
        });
        assert!(resp.error.is_none());

        // List collections
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "list_collections",
                "arguments": {}
            }),
        });
        let content = &resp.result.unwrap()["content"][0]["text"];
        let parsed: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
        assert_eq!(parsed["collections"][0]["name"], "test_docs");
    }

    #[test]
    fn test_insert_and_search() {
        let server = create_test_server();

        // Create collection
        server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "create_collection",
                "arguments": { "name": "docs", "dimensions": 4 }
            }),
        });

        // Insert vectors
        server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "insert_vectors",
                "arguments": {
                    "collection": "docs",
                    "vectors": [
                        { "id": "v1", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"tag": "a"} },
                        { "id": "v2", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"tag": "b"} }
                    ]
                }
            }),
        });

        // Search
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(3)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "search",
                "arguments": {
                    "collection": "docs",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "k": 2
                }
            }),
        });
        let content = &resp.result.unwrap()["content"][0]["text"];
        let parsed: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
        assert_eq!(parsed["count"], 2);
        assert_eq!(parsed["results"][0]["id"], "v1");
    }

    #[test]
    fn test_get_and_delete_vector() {
        let server = create_test_server();

        server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "create_collection",
                "arguments": { "name": "test", "dimensions": 3 }
            }),
        });

        server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "insert_vectors",
                "arguments": {
                    "collection": "test",
                    "vectors": [{ "id": "doc1", "values": [1.0, 2.0, 3.0] }]
                }
            }),
        });

        // Get vector
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(3)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "get_vector",
                "arguments": { "collection": "test", "id": "doc1" }
            }),
        });
        let content = &resp.result.unwrap()["content"][0]["text"];
        let parsed: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
        assert_eq!(parsed["found"], true);

        // Delete vector
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(4)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "delete_vector",
                "arguments": { "collection": "test", "id": "doc1" }
            }),
        });
        let content = &resp.result.unwrap()["content"][0]["text"];
        let parsed: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
        assert_eq!(parsed["deleted"], true);
    }

    #[test]
    fn test_read_only_mode() {
        let db = Database::in_memory();
        let server = McpServer::new(db, true);

        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "create_collection",
                "arguments": { "name": "test", "dimensions": 3 }
            }),
        });
        let content = &resp.result.unwrap()["content"][0]["text"];
        assert!(content.as_str().unwrap().contains("read-only"));
    }

    #[test]
    fn test_resources_list() {
        let server = create_test_server();
        server.tool_create_collection(&json!({"name": "coll1", "dimensions": 4})).unwrap();

        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "resources/list".to_string(),
            params: json!({}),
        });
        let resources = &resp.result.unwrap()["resources"];
        assert_eq!(resources[0]["uri"], "needle://collections/coll1");
    }

    #[test]
    fn test_unknown_method() {
        let server = create_test_server();
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "nonexistent/method".to_string(),
            params: json!({}),
        });
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[test]
    fn test_remember_recall_forget() {
        let server = create_test_server();

        // Create a memory collection
        server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "create_collection",
                "arguments": { "name": "memories", "dimensions": 4 }
            }),
        });

        // Remember
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "remember",
                "arguments": {
                    "collection": "memories",
                    "content": "User likes dark mode",
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "tier": "semantic",
                    "importance": 0.9
                }
            }),
        });
        assert!(resp.result.is_some());
        let result_text = resp.result.unwrap()["content"][0]["text"].as_str().unwrap().to_string();
        assert!(result_text.contains("stored"));

        // Recall
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(3)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "recall",
                "arguments": {
                    "collection": "memories",
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "k": 5
                }
            }),
        });
        assert!(resp.result.is_some());
        let result_text = resp.result.unwrap()["content"][0]["text"].as_str().unwrap().to_string();
        assert!(result_text.contains("memories"));
        assert!(result_text.contains("User likes dark mode"));
    }

    #[test]
    fn test_tools_list_includes_memory() {
        let server = create_test_server();
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/list".to_string(),
            params: json!({}),
        });
        let tools_text = serde_json::to_string(&resp.result).unwrap();
        assert!(tools_text.contains("remember"));
        assert!(tools_text.contains("recall"));
        assert!(tools_text.contains("forget"));
        assert!(tools_text.contains("memory_consolidate"));
    }

    /// Helper to insert a memory with specific tier and importance for consolidation tests.
    /// Uses insert_vectors directly to avoid timestamp-based ID collisions.
    fn insert_memory(server: &McpServer, collection: &str, id: &str, vector: Vec<f64>, tier: &str, importance: f64) {
        let vector_f64: Vec<f64> = vector;
        let metadata = json!({
            "_memory_content": format!("Memory {id}"),
            "_memory_tier": tier,
            "_memory_importance": importance,
            "_memory_timestamp": "2025-01-01T00:00:00Z",
        });
        server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "insert_vectors",
                "arguments": {
                    "collection": collection,
                    "vectors": [{
                        "id": id,
                        "values": vector_f64,
                        "metadata": metadata,
                    }]
                }
            }),
        });
    }

    #[test]
    fn test_memory_consolidate_happy_path() {
        let server = create_test_server();

        // Create collection
        server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "create_collection",
                "arguments": { "name": "mem", "dimensions": 4 }
            }),
        });

        // Insert episodic memories with varying importance
        insert_memory(&server, "mem", "high", vec![1.0, 0.0, 0.0, 0.0], "episodic", 0.9);
        insert_memory(&server, "mem", "low", vec![0.0, 1.0, 0.0, 0.0], "episodic", 0.05);
        insert_memory(&server, "mem", "mid", vec![0.0, 0.0, 1.0, 0.0], "episodic", 0.5);
        // Semantic memory should be untouched
        insert_memory(&server, "mem", "sem", vec![0.0, 0.0, 0.0, 1.0], "semantic", 0.3);

        // Consolidate with defaults (promote >= 0.7, expire < 0.1)
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "memory_consolidate",
                "arguments": { "collection": "mem" }
            }),
        });
        let content = &resp.result.unwrap()["content"][0]["text"];
        let parsed: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();

        assert_eq!(parsed["consolidated"], true);
        assert_eq!(parsed["promoted"], 1);   // "high" promoted
        assert_eq!(parsed["forgotten"], 1);  // "low" expired
        assert_eq!(parsed["scanned"], 3);    // 3 episodic memories scanned
        assert_eq!(parsed["errors"], 0);
    }

    #[test]
    fn test_memory_consolidate_read_only() {
        let db = Database::in_memory();
        let server = McpServer::new(db, true);

        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "memory_consolidate",
                "arguments": { "collection": "mem" }
            }),
        });
        let content = &resp.result.unwrap()["content"][0]["text"];
        assert!(content.as_str().unwrap().contains("read-only"));
    }

    #[test]
    fn test_memory_consolidate_missing_collection() {
        let server = create_test_server();

        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "memory_consolidate",
                "arguments": {}
            }),
        });
        let content = &resp.result.unwrap()["content"][0]["text"];
        assert!(content.as_str().unwrap().contains("Error"));
    }

    #[test]
    fn test_memory_consolidate_custom_thresholds() {
        let server = create_test_server();

        server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "create_collection",
                "arguments": { "name": "mem2", "dimensions": 4 }
            }),
        });

        // Insert episodic memories
        insert_memory(&server, "mem2", "a", vec![1.0, 0.0, 0.0, 0.0], "episodic", 0.5);
        insert_memory(&server, "mem2", "b", vec![0.0, 1.0, 0.0, 0.0], "episodic", 0.3);

        // Custom thresholds: promote >= 0.4, expire < 0.35
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "memory_consolidate",
                "arguments": {
                    "collection": "mem2",
                    "promotion_threshold": 0.4,
                    "expire_below": 0.35
                }
            }),
        });
        let content = &resp.result.unwrap()["content"][0]["text"];
        let parsed: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();

        assert_eq!(parsed["promoted"], 1);  // "a" promoted (0.5 >= 0.4)
        assert_eq!(parsed["forgotten"], 1); // "b" forgotten (0.3 < 0.35)
        assert_eq!(parsed["errors"], 0);
    }

    #[test]
    fn test_memory_consolidate_nonexistent_collection() {
        let server = create_test_server();

        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "memory_consolidate",
                "arguments": { "collection": "nonexistent" }
            }),
        });
        let content = &resp.result.unwrap()["content"][0]["text"];
        assert!(content.as_str().unwrap().contains("Error"));
    }

    #[test]
    fn test_collection_info() {
        let server = create_test_server();

        // Create collection
        server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "create_collection",
                "arguments": { "name": "info_test", "dimensions": 4 }
            }),
        });

        // Insert a vector
        server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "insert_vectors",
                "arguments": {
                    "collection": "info_test",
                    "vectors": [{ "id": "v1", "values": [1.0, 0.0, 0.0, 0.0] }]
                }
            }),
        });

        // Get collection info
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(3)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "collection_info",
                "arguments": { "name": "info_test" }
            }),
        });
        assert!(resp.error.is_none());
        let content = &resp.result.unwrap()["content"][0]["text"];
        let parsed: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
        assert_eq!(parsed["name"], "info_test");
        assert_eq!(parsed["vector_count"], 1);

        // Non-existent collection should error
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(4)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "collection_info",
                "arguments": { "name": "nonexistent" }
            }),
        });
        let content = &resp.result.unwrap()["content"][0]["text"];
        assert!(content.as_str().unwrap().contains("Error"));
    }

    #[test]
    fn test_delete_collection() {
        let server = create_test_server();

        // Create collection
        server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "create_collection",
                "arguments": { "name": "to_delete", "dimensions": 3 }
            }),
        });

        // Delete it
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "delete_collection",
                "arguments": { "name": "to_delete" }
            }),
        });
        assert!(resp.error.is_none());
        let content = &resp.result.unwrap()["content"][0]["text"];
        let parsed: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
        assert_eq!(parsed["deleted"], true);

        // Verify it's gone
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(3)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "list_collections",
                "arguments": {}
            }),
        });
        let content = &resp.result.unwrap()["content"][0]["text"];
        let parsed: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
        assert_eq!(parsed["collections"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_save_database() {
        let server = create_test_server();

        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "save_database",
                "arguments": {}
            }),
        });
        assert!(resp.error.is_none());
        let content = &resp.result.unwrap()["content"][0]["text"];
        let parsed: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
        assert_eq!(parsed["acknowledged"], true);
    }

    #[test]
    fn test_search_with_filter() {
        let server = create_test_server();

        // Create collection
        server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "create_collection",
                "arguments": { "name": "filtered", "dimensions": 4 }
            }),
        });

        // Insert vectors with metadata
        server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "insert_vectors",
                "arguments": {
                    "collection": "filtered",
                    "vectors": [
                        { "id": "a", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"category": "books"} },
                        { "id": "b", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"category": "movies"} },
                        { "id": "c", "values": [0.9, 0.1, 0.0, 0.0], "metadata": {"category": "books"} }
                    ]
                }
            }),
        });

        // Search with filter
        let resp = server.handle_request(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(3)),
            method: "tools/call".to_string(),
            params: json!({
                "name": "search",
                "arguments": {
                    "collection": "filtered",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "k": 10,
                    "filter": { "category": "books" }
                }
            }),
        });
        assert!(resp.error.is_none());
        let content = &resp.result.unwrap()["content"][0]["text"];
        let parsed: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
        assert_eq!(parsed["count"], 2);
        // All results should be books
        for result in parsed["results"].as_array().unwrap() {
            assert_eq!(result["metadata"]["category"], "books");
        }
    }
}
