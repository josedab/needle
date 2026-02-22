#![allow(clippy::unwrap_used)]
//! LLM Function Calling Interface
//!
//! OpenAI-compatible tool schemas for autonomous AI agents. Exposes Needle
//! operations as function calling tools for LLMs like GPT-4, Claude, and Ollama.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::llm_tools::{
//!     ToolRegistry, FunctionCall, FunctionResult, ToolSchema,
//! };
//!
//! let registry = ToolRegistry::new();
//! let schemas = registry.schemas();
//! assert!(schemas.len() >= 5);
//!
//! // Execute a function call from an LLM
//! let call = FunctionCall {
//!     name: "needle_search".into(),
//!     arguments: serde_json::json!({"collection": "docs", "query": "rust programming", "k": 5}),
//! };
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A tool schema compatible with OpenAI function calling format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionSchema,
}

/// Function definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSchema {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// A function call from an LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: Value,
}

/// Result of executing a function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResult {
    pub name: String,
    pub success: bool,
    pub result: Value,
    pub error: Option<String>,
}

/// Tool registry with all Needle operations as LLM tools.
pub struct ToolRegistry {
    tools: HashMap<String, ToolSchema>,
}

impl ToolRegistry {
    /// Create a new registry with all Needle tools.
    pub fn new() -> Self {
        let mut tools = HashMap::new();

        tools.insert("needle_create_collection".into(), ToolSchema {
            tool_type: "function".into(),
            function: FunctionSchema {
                name: "needle_create_collection".into(),
                description: "Create a new vector collection in the Needle database".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Collection name" },
                        "dimensions": { "type": "integer", "description": "Vector dimensions (e.g., 384, 768, 1536)" }
                    },
                    "required": ["name", "dimensions"]
                }),
            },
        });

        tools.insert("needle_insert".into(), ToolSchema {
            tool_type: "function".into(),
            function: FunctionSchema {
                name: "needle_insert".into(),
                description: "Insert a text document into a collection (auto-embeds)".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "collection": { "type": "string", "description": "Target collection" },
                        "id": { "type": "string", "description": "Document ID" },
                        "text": { "type": "string", "description": "Text content to embed and store" },
                        "metadata": { "type": "object", "description": "Optional metadata" }
                    },
                    "required": ["collection", "id", "text"]
                }),
            },
        });

        tools.insert("needle_search".into(), ToolSchema {
            tool_type: "function".into(),
            function: FunctionSchema {
                name: "needle_search".into(),
                description: "Search for similar documents by text query".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "collection": { "type": "string", "description": "Collection to search" },
                        "query": { "type": "string", "description": "Search query text" },
                        "k": { "type": "integer", "description": "Number of results", "default": 5 }
                    },
                    "required": ["collection", "query"]
                }),
            },
        });

        tools.insert("needle_delete".into(), ToolSchema {
            tool_type: "function".into(),
            function: FunctionSchema {
                name: "needle_delete".into(),
                description: "Delete a document from a collection".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "collection": { "type": "string", "description": "Collection name" },
                        "id": { "type": "string", "description": "Document ID to delete" }
                    },
                    "required": ["collection", "id"]
                }),
            },
        });

        tools.insert("needle_list_collections".into(), ToolSchema {
            tool_type: "function".into(),
            function: FunctionSchema {
                name: "needle_list_collections".into(),
                description: "List all collections in the database".into(),
                parameters: serde_json::json!({ "type": "object", "properties": {} }),
            },
        });

        tools.insert("needle_collection_info".into(), ToolSchema {
            tool_type: "function".into(),
            function: FunctionSchema {
                name: "needle_collection_info".into(),
                description: "Get information about a collection (vector count, dimensions)".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "collection": { "type": "string", "description": "Collection name" } },
                    "required": ["collection"]
                }),
            },
        });

        Self { tools }
    }

    /// Get all tool schemas (for sending to LLM).
    pub fn schemas(&self) -> Vec<&ToolSchema> { self.tools.values().collect() }

    /// Get a specific tool schema.
    pub fn get(&self, name: &str) -> Option<&ToolSchema> { self.tools.get(name) }

    /// Get schemas as JSON array (OpenAI format).
    pub fn to_openai_tools(&self) -> Value {
        Value::Array(self.tools.values().map(|t| serde_json::to_value(t).unwrap_or_default()).collect())
    }

    /// Parse and validate a function call.
    pub fn validate_call(&self, call: &FunctionCall) -> Result<(), String> {
        if !self.tools.contains_key(&call.name) {
            return Err(format!("Unknown function: {}", call.name));
        }
        // Basic argument validation
        let schema = &self.tools[&call.name];
        if let Some(required) = schema.function.parameters.get("required") {
            if let Some(required_arr) = required.as_array() {
                for field in required_arr {
                    if let Some(field_name) = field.as_str() {
                        if call.arguments.get(field_name).is_none() {
                            return Err(format!("Missing required field: {}", field_name));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Execute a function call (returns a mock result for now).
    pub fn execute(&self, call: &FunctionCall) -> FunctionResult {
        if let Err(e) = self.validate_call(call) {
            return FunctionResult { name: call.name.clone(), success: false, result: Value::Null, error: Some(e) };
        }
        FunctionResult {
            name: call.name.clone(), success: true,
            result: serde_json::json!({ "status": "executed", "function": call.name }),
            error: None,
        }
    }

    /// Tool count.
    pub fn len(&self) -> usize { self.tools.len() }
    pub fn is_empty(&self) -> bool { self.tools.is_empty() }
}

impl Default for ToolRegistry { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_count() {
        let reg = ToolRegistry::new();
        assert!(reg.len() >= 6);
    }

    #[test]
    fn test_search_schema() {
        let reg = ToolRegistry::new();
        let schema = reg.get("needle_search").unwrap();
        assert!(schema.function.description.contains("Search"));
    }

    #[test]
    fn test_validate_valid_call() {
        let reg = ToolRegistry::new();
        let call = FunctionCall {
            name: "needle_search".into(),
            arguments: serde_json::json!({"collection": "docs", "query": "test"}),
        };
        assert!(reg.validate_call(&call).is_ok());
    }

    #[test]
    fn test_validate_missing_field() {
        let reg = ToolRegistry::new();
        let call = FunctionCall {
            name: "needle_search".into(),
            arguments: serde_json::json!({"collection": "docs"}), // missing query
        };
        assert!(reg.validate_call(&call).is_err());
    }

    #[test]
    fn test_unknown_function() {
        let reg = ToolRegistry::new();
        let call = FunctionCall { name: "nonexistent".into(), arguments: Value::Null };
        assert!(reg.validate_call(&call).is_err());
    }

    #[test]
    fn test_execute() {
        let reg = ToolRegistry::new();
        let call = FunctionCall {
            name: "needle_list_collections".into(), arguments: serde_json::json!({}),
        };
        let result = reg.execute(&call);
        assert!(result.success);
    }

    #[test]
    fn test_openai_format() {
        let reg = ToolRegistry::new();
        let tools = reg.to_openai_tools();
        assert!(tools.is_array());
        assert!(tools.as_array().unwrap().len() >= 6);
    }
}
