//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Universal SDK Generator
//!
//! Auto-generates type-safe SDKs from an OpenAPI spec derived from the Needle
//! HTTP API. Supports Go, Java, C#, Ruby, PHP, Dart templates.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::experimental::sdk_generator::*;
//!
//! let spec = OpenApiSpec::from_needle_api();
//! let generator = SdkGenerator::new(SdkConfig::default());
//! let output = generator.generate(&spec, Language::Go);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported target languages for SDK generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    /// Go
    Go,
    /// Java
    Java,
    /// C#
    CSharp,
    /// Ruby
    Ruby,
    /// PHP
    Php,
    /// Dart
    Dart,
    /// TypeScript
    TypeScript,
    /// Python
    Python,
}

impl Language {
    /// Get the file extension for this language.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Go => "go",
            Self::Java => "java",
            Self::CSharp => "cs",
            Self::Ruby => "rb",
            Self::Php => "php",
            Self::Dart => "dart",
            Self::TypeScript => "ts",
            Self::Python => "py",
        }
    }

    /// Get the language name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Go => "Go",
            Self::Java => "Java",
            Self::CSharp => "C#",
            Self::Ruby => "Ruby",
            Self::Php => "PHP",
            Self::Dart => "Dart",
            Self::TypeScript => "TypeScript",
            Self::Python => "Python",
        }
    }

    /// All supported languages.
    pub fn all() -> &'static [Language] {
        &[
            Self::Go,
            Self::Java,
            Self::CSharp,
            Self::Ruby,
            Self::Php,
            Self::Dart,
            Self::TypeScript,
            Self::Python,
        ]
    }
}

/// HTTP method for an API endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HttpMethod {
    /// GET
    Get,
    /// POST
    Post,
    /// PUT
    Put,
    /// DELETE
    Delete,
    /// PATCH
    Patch,
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Get => write!(f, "GET"),
            Self::Post => write!(f, "POST"),
            Self::Put => write!(f, "PUT"),
            Self::Delete => write!(f, "DELETE"),
            Self::Patch => write!(f, "PATCH"),
        }
    }
}

/// Type of an API parameter or field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApiType {
    /// String type
    String,
    /// Integer type
    Integer,
    /// Float type
    Float,
    /// Boolean type
    Boolean,
    /// Array of another type
    Array(Box<ApiType>),
    /// Map/Object with string keys
    Map(Box<ApiType>),
    /// Reference to a named schema
    Ref(String),
    /// Optional/nullable wrapper
    Optional(Box<ApiType>),
    /// Any/dynamic JSON value
    Any,
}

impl ApiType {
    /// Convert to a Go type string.
    pub fn to_go(&self) -> String {
        match self {
            Self::String => "string".to_string(),
            Self::Integer => "int64".to_string(),
            Self::Float => "float64".to_string(),
            Self::Boolean => "bool".to_string(),
            Self::Array(inner) => format!("[]{}", inner.to_go()),
            Self::Map(inner) => format!("map[string]{}", inner.to_go()),
            Self::Ref(name) => name.clone(),
            Self::Optional(inner) => format!("*{}", inner.to_go()),
            Self::Any => "interface{}".to_string(),
        }
    }

    /// Convert to a TypeScript type string.
    pub fn to_typescript(&self) -> String {
        match self {
            Self::String => "string".to_string(),
            Self::Integer | Self::Float => "number".to_string(),
            Self::Boolean => "boolean".to_string(),
            Self::Array(inner) => format!("{}[]", inner.to_typescript()),
            Self::Map(inner) => format!("Record<string, {}>", inner.to_typescript()),
            Self::Ref(name) => name.clone(),
            Self::Optional(inner) => format!("{} | null", inner.to_typescript()),
            Self::Any => "any".to_string(),
        }
    }

    /// Convert to a Java type string.
    pub fn to_java(&self) -> String {
        match self {
            Self::String => "String".to_string(),
            Self::Integer => "Long".to_string(),
            Self::Float => "Double".to_string(),
            Self::Boolean => "Boolean".to_string(),
            Self::Array(inner) => format!("List<{}>", inner.to_java()),
            Self::Map(inner) => format!("Map<String, {}>", inner.to_java()),
            Self::Ref(name) => name.clone(),
            Self::Optional(inner) => inner.to_java(),
            Self::Any => "Object".to_string(),
        }
    }
}

/// A field in a schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiField {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: ApiType,
    /// Whether the field is required
    pub required: bool,
    /// Description
    pub description: Option<String>,
}

/// A schema definition (like a struct/class).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiSchema {
    /// Schema name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Fields
    pub fields: Vec<ApiField>,
}

/// An API endpoint definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEndpoint {
    /// HTTP method
    pub method: HttpMethod,
    /// URL path (e.g., "/collections/{name}/search")
    pub path: String,
    /// Operation ID
    pub operation_id: String,
    /// Description
    pub description: Option<String>,
    /// Request body schema name (if any)
    pub request_body: Option<String>,
    /// Response schema name
    pub response_schema: Option<String>,
    /// Path parameters
    pub path_params: Vec<ApiField>,
    /// Query parameters
    pub query_params: Vec<ApiField>,
    /// Tags for grouping
    pub tags: Vec<String>,
}

/// OpenAPI specification derived from Needle's HTTP API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiSpec {
    /// API title
    pub title: String,
    /// API version
    pub version: String,
    /// Base URL
    pub base_url: String,
    /// Description
    pub description: String,
    /// Endpoints
    pub endpoints: Vec<ApiEndpoint>,
    /// Schema definitions
    pub schemas: HashMap<String, ApiSchema>,
}

impl OpenApiSpec {
    /// Generate an OpenAPI spec from Needle's built-in API endpoints.
    pub fn from_needle_api() -> Self {
        let mut schemas = HashMap::new();

        // Core schemas
        schemas.insert(
            "CreateCollectionRequest".to_string(),
            ApiSchema {
                name: "CreateCollectionRequest".to_string(),
                description: Some("Request to create a new collection".to_string()),
                fields: vec![
                    ApiField {
                        name: "name".to_string(),
                        field_type: ApiType::String,
                        required: true,
                        description: Some("Collection name".to_string()),
                    },
                    ApiField {
                        name: "dimensions".to_string(),
                        field_type: ApiType::Integer,
                        required: true,
                        description: Some("Vector dimensions".to_string()),
                    },
                    ApiField {
                        name: "distance".to_string(),
                        field_type: ApiType::Optional(Box::new(ApiType::String)),
                        required: false,
                        description: Some("Distance function (cosine, euclidean, dot)".to_string()),
                    },
                ],
            },
        );

        schemas.insert(
            "InsertRequest".to_string(),
            ApiSchema {
                name: "InsertRequest".to_string(),
                description: Some("Request to insert vectors".to_string()),
                fields: vec![
                    ApiField {
                        name: "vectors".to_string(),
                        field_type: ApiType::Array(Box::new(ApiType::Ref("VectorEntry".to_string()))),
                        required: true,
                        description: Some("Vectors to insert".to_string()),
                    },
                ],
            },
        );

        schemas.insert(
            "VectorEntry".to_string(),
            ApiSchema {
                name: "VectorEntry".to_string(),
                description: Some("A vector with ID and optional metadata".to_string()),
                fields: vec![
                    ApiField {
                        name: "id".to_string(),
                        field_type: ApiType::String,
                        required: true,
                        description: Some("Vector ID".to_string()),
                    },
                    ApiField {
                        name: "vector".to_string(),
                        field_type: ApiType::Array(Box::new(ApiType::Float)),
                        required: true,
                        description: Some("Vector data".to_string()),
                    },
                    ApiField {
                        name: "metadata".to_string(),
                        field_type: ApiType::Optional(Box::new(ApiType::Any)),
                        required: false,
                        description: Some("JSON metadata".to_string()),
                    },
                ],
            },
        );

        schemas.insert(
            "SearchRequest".to_string(),
            ApiSchema {
                name: "SearchRequest".to_string(),
                description: Some("Search request".to_string()),
                fields: vec![
                    ApiField {
                        name: "vector".to_string(),
                        field_type: ApiType::Array(Box::new(ApiType::Float)),
                        required: true,
                        description: Some("Query vector".to_string()),
                    },
                    ApiField {
                        name: "k".to_string(),
                        field_type: ApiType::Optional(Box::new(ApiType::Integer)),
                        required: false,
                        description: Some("Number of results (default: 10)".to_string()),
                    },
                    ApiField {
                        name: "filter".to_string(),
                        field_type: ApiType::Optional(Box::new(ApiType::Any)),
                        required: false,
                        description: Some("MongoDB-style metadata filter".to_string()),
                    },
                ],
            },
        );

        schemas.insert(
            "SearchResult".to_string(),
            ApiSchema {
                name: "SearchResult".to_string(),
                description: Some("Search result entry".to_string()),
                fields: vec![
                    ApiField {
                        name: "id".to_string(),
                        field_type: ApiType::String,
                        required: true,
                        description: Some("Vector ID".to_string()),
                    },
                    ApiField {
                        name: "distance".to_string(),
                        field_type: ApiType::Float,
                        required: true,
                        description: Some("Distance to query".to_string()),
                    },
                    ApiField {
                        name: "metadata".to_string(),
                        field_type: ApiType::Optional(Box::new(ApiType::Any)),
                        required: false,
                        description: Some("Vector metadata".to_string()),
                    },
                ],
            },
        );

        schemas.insert(
            "CollectionInfo".to_string(),
            ApiSchema {
                name: "CollectionInfo".to_string(),
                description: Some("Collection information".to_string()),
                fields: vec![
                    ApiField {
                        name: "name".to_string(),
                        field_type: ApiType::String,
                        required: true,
                        description: Some("Collection name".to_string()),
                    },
                    ApiField {
                        name: "dimensions".to_string(),
                        field_type: ApiType::Integer,
                        required: true,
                        description: Some("Vector dimensions".to_string()),
                    },
                    ApiField {
                        name: "count".to_string(),
                        field_type: ApiType::Integer,
                        required: true,
                        description: Some("Number of vectors".to_string()),
                    },
                ],
            },
        );

        let endpoints = vec![
            ApiEndpoint {
                method: HttpMethod::Get,
                path: "/health".to_string(),
                operation_id: "health_check".to_string(),
                description: Some("Health check".to_string()),
                request_body: None,
                response_schema: None,
                path_params: Vec::new(),
                query_params: Vec::new(),
                tags: vec!["system".to_string()],
            },
            ApiEndpoint {
                method: HttpMethod::Get,
                path: "/collections".to_string(),
                operation_id: "list_collections".to_string(),
                description: Some("List all collections".to_string()),
                request_body: None,
                response_schema: Some("CollectionInfo".to_string()),
                path_params: Vec::new(),
                query_params: Vec::new(),
                tags: vec!["collections".to_string()],
            },
            ApiEndpoint {
                method: HttpMethod::Post,
                path: "/collections".to_string(),
                operation_id: "create_collection".to_string(),
                description: Some("Create a new collection".to_string()),
                request_body: Some("CreateCollectionRequest".to_string()),
                response_schema: Some("CollectionInfo".to_string()),
                path_params: Vec::new(),
                query_params: Vec::new(),
                tags: vec!["collections".to_string()],
            },
            ApiEndpoint {
                method: HttpMethod::Post,
                path: "/collections/{name}/vectors".to_string(),
                operation_id: "insert_vectors".to_string(),
                description: Some("Insert vectors into a collection".to_string()),
                request_body: Some("InsertRequest".to_string()),
                response_schema: None,
                path_params: vec![ApiField {
                    name: "name".to_string(),
                    field_type: ApiType::String,
                    required: true,
                    description: Some("Collection name".to_string()),
                }],
                query_params: Vec::new(),
                tags: vec!["vectors".to_string()],
            },
            ApiEndpoint {
                method: HttpMethod::Post,
                path: "/collections/{name}/search".to_string(),
                operation_id: "search".to_string(),
                description: Some("Search for similar vectors".to_string()),
                request_body: Some("SearchRequest".to_string()),
                response_schema: Some("SearchResult".to_string()),
                path_params: vec![ApiField {
                    name: "name".to_string(),
                    field_type: ApiType::String,
                    required: true,
                    description: Some("Collection name".to_string()),
                }],
                query_params: Vec::new(),
                tags: vec!["search".to_string()],
            },
            ApiEndpoint {
                method: HttpMethod::Delete,
                path: "/collections/{name}/vectors/{id}".to_string(),
                operation_id: "delete_vector".to_string(),
                description: Some("Delete a vector by ID".to_string()),
                request_body: None,
                response_schema: None,
                path_params: vec![
                    ApiField {
                        name: "name".to_string(),
                        field_type: ApiType::String,
                        required: true,
                        description: Some("Collection name".to_string()),
                    },
                    ApiField {
                        name: "id".to_string(),
                        field_type: ApiType::String,
                        required: true,
                        description: Some("Vector ID".to_string()),
                    },
                ],
                query_params: Vec::new(),
                tags: vec!["vectors".to_string()],
            },
        ];

        Self {
            title: "Needle Vector Database API".to_string(),
            version: "0.1.0".to_string(),
            base_url: "http://localhost:8080".to_string(),
            description: "REST API for the Needle embedded vector database".to_string(),
            endpoints,
            schemas,
        }
    }

    /// Serialize to OpenAPI 3.1 JSON.
    pub fn to_openapi_json(&self) -> serde_json::Value {
        serde_json::json!({
            "openapi": "3.1.0",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": self.description,
            },
            "servers": [{"url": self.base_url}],
            "paths": self.paths_json(),
            "components": {
                "schemas": self.schemas_json(),
            }
        })
    }

    fn paths_json(&self) -> serde_json::Value {
        let mut paths = serde_json::Map::new();
        for endpoint in &self.endpoints {
            let method = match endpoint.method {
                HttpMethod::Get => "get",
                HttpMethod::Post => "post",
                HttpMethod::Put => "put",
                HttpMethod::Delete => "delete",
                HttpMethod::Patch => "patch",
            };

            let mut op = serde_json::Map::new();
            op.insert("operationId".to_string(), serde_json::json!(endpoint.operation_id));
            if let Some(desc) = &endpoint.description {
                op.insert("description".to_string(), serde_json::json!(desc));
            }
            op.insert("tags".to_string(), serde_json::json!(endpoint.tags));

            let path_entry = paths
                .entry(&endpoint.path)
                .or_insert_with(|| serde_json::json!({}));

            if let serde_json::Value::Object(path_map) = path_entry {
                path_map.insert(method.to_string(), serde_json::Value::Object(op));
            }
        }
        serde_json::Value::Object(paths)
    }

    fn schemas_json(&self) -> serde_json::Value {
        let mut result = serde_json::Map::new();
        for (name, schema) in &self.schemas {
            let mut props = serde_json::Map::new();
            let mut required = Vec::new();

            for field in &schema.fields {
                props.insert(field.name.clone(), serde_json::json!({"type": "string"}));
                if field.required {
                    required.push(serde_json::json!(field.name));
                }
            }

            result.insert(
                name.clone(),
                serde_json::json!({
                    "type": "object",
                    "properties": props,
                    "required": required,
                }),
            );
        }
        serde_json::Value::Object(result)
    }
}

/// Configuration for SDK generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdkConfig {
    /// Package/module name for generated code
    pub package_name: String,
    /// Version string for the generated SDK
    pub version: String,
    /// Whether to generate async/await code (where applicable)
    pub async_mode: bool,
    /// Whether to include inline documentation
    pub generate_docs: bool,
}

impl Default for SdkConfig {
    fn default() -> Self {
        Self {
            package_name: "needle_sdk".to_string(),
            version: "0.1.0".to_string(),
            async_mode: true,
            generate_docs: true,
        }
    }
}

/// A generated SDK file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedFile {
    /// File path relative to output directory
    pub path: String,
    /// File content
    pub content: String,
    /// Language this file is for
    pub language: Language,
}

/// SDK code generator.
pub struct SdkGenerator {
    config: SdkConfig,
}

impl SdkGenerator {
    /// Create a new SDK generator.
    pub fn new(config: SdkConfig) -> Self {
        Self { config }
    }

    /// Generate SDK files for a target language.
    pub fn generate(&self, spec: &OpenApiSpec, language: Language) -> Vec<GeneratedFile> {
        match language {
            Language::TypeScript => self.generate_typescript(spec),
            Language::Go => self.generate_go(spec),
            Language::Python => self.generate_python(spec),
            _ => self.generate_generic(spec, language),
        }
    }

    /// Generate for all supported languages.
    pub fn generate_all(&self, spec: &OpenApiSpec) -> Vec<GeneratedFile> {
        Language::all()
            .iter()
            .flat_map(|lang| self.generate(spec, *lang))
            .collect()
    }

    fn generate_typescript(&self, spec: &OpenApiSpec) -> Vec<GeneratedFile> {
        let mut content = String::new();
        content.push_str("// Auto-generated Needle SDK for TypeScript\n");
        content.push_str(&format!("// Version: {}\n\n", self.config.version));

        // Generate interfaces for schemas
        for (name, schema) in &spec.schemas {
            content.push_str(&format!("export interface {} {{\n", name));
            for field in &schema.fields {
                let ts_type = field.field_type.to_typescript();
                let optional = if field.required { "" } else { "?" };
                if let Some(desc) = &field.description {
                    content.push_str(&format!("  /** {} */\n", desc));
                }
                content.push_str(&format!("  {}{}: {};\n", field.name, optional, ts_type));
            }
            content.push_str("}\n\n");
        }

        // Generate client class
        content.push_str("export class NeedleClient {\n");
        content.push_str("  private baseUrl: string;\n\n");
        content.push_str("  constructor(baseUrl: string = 'http://localhost:8080') {\n");
        content.push_str("    this.baseUrl = baseUrl;\n");
        content.push_str("  }\n\n");

        for endpoint in &spec.endpoints {
            let method_name = endpoint.operation_id.replace('-', "_");
            content.push_str(&format!(
                "  async {}(",
                to_camel_case(&method_name)
            ));

            // Parameters
            let mut params = Vec::new();
            for p in &endpoint.path_params {
                params.push(format!("{}: {}", p.name, p.field_type.to_typescript()));
            }
            if endpoint.request_body.is_some() {
                params.push("body: any".to_string());
            }
            content.push_str(&params.join(", "));
            content.push_str("): Promise<any> {\n");

            let mut path = endpoint.path.clone();
            for p in &endpoint.path_params {
                path = path.replace(&format!("{{{}}}", p.name), &format!("${{{}}}", p.name));
            }

            content.push_str(&format!(
                "    const response = await fetch(`${{this.baseUrl}}{}`",
                path
            ));

            if endpoint.request_body.is_some() {
                content.push_str(&format!(
                    ", {{\n      method: '{}',\n      headers: {{ 'Content-Type': 'application/json' }},\n      body: JSON.stringify(body),\n    }}",
                    endpoint.method
                ));
            }
            content.push_str(");\n");
            content.push_str("    return response.json();\n");
            content.push_str("  }\n\n");
        }

        content.push_str("}\n");

        vec![GeneratedFile {
            path: format!("{}/index.ts", self.config.package_name),
            content,
            language: Language::TypeScript,
        }]
    }

    fn generate_go(&self, spec: &OpenApiSpec) -> Vec<GeneratedFile> {
        let mut content = String::new();
        content.push_str(&format!("// Auto-generated Needle SDK for Go\n"));
        content.push_str(&format!("// Version: {}\n\n", self.config.version));
        content.push_str(&format!("package {}\n\n", self.config.package_name));

        content.push_str("import (\n");
        content.push_str("\t\"bytes\"\n");
        content.push_str("\t\"encoding/json\"\n");
        content.push_str("\t\"fmt\"\n");
        content.push_str("\t\"net/http\"\n");
        content.push_str(")\n\n");

        // Generate structs for schemas
        for (name, schema) in &spec.schemas {
            content.push_str(&format!("type {} struct {{\n", name));
            for field in &schema.fields {
                let go_type = field.field_type.to_go();
                let json_tag = if field.required {
                    format!("`json:\"{}\"`", field.name)
                } else {
                    format!("`json:\"{},omitempty\"`", field.name)
                };
                content.push_str(&format!(
                    "\t{} {} {}\n",
                    to_pascal_case(&field.name),
                    go_type,
                    json_tag
                ));
            }
            content.push_str("}\n\n");
        }

        // Generate client struct
        content.push_str("type Client struct {\n");
        content.push_str("\tBaseURL    string\n");
        content.push_str("\tHTTPClient *http.Client\n");
        content.push_str("}\n\n");

        content.push_str("func NewClient(baseURL string) *Client {\n");
        content.push_str("\treturn &Client{\n");
        content.push_str("\t\tBaseURL:    baseURL,\n");
        content.push_str("\t\tHTTPClient: http.DefaultClient,\n");
        content.push_str("\t}\n");
        content.push_str("}\n\n");

        vec![GeneratedFile {
            path: format!("{}/client.go", self.config.package_name),
            content,
            language: Language::Go,
        }]
    }

    fn generate_python(&self, spec: &OpenApiSpec) -> Vec<GeneratedFile> {
        let mut content = String::new();
        content.push_str("# Auto-generated Needle SDK for Python\n");
        content.push_str(&format!("# Version: {}\n\n", self.config.version));
        content.push_str("from dataclasses import dataclass, field\n");
        content.push_str("from typing import Any, Dict, List, Optional\n");
        content.push_str("import json\n");
        content.push_str("import urllib.request\n\n");

        // Generate dataclasses for schemas
        for (name, schema) in &spec.schemas {
            content.push_str("@dataclass\n");
            content.push_str(&format!("class {}:\n", name));
            if let Some(desc) = &schema.description {
                content.push_str(&format!("    \"\"\"{}\"\"\"\n", desc));
            }
            for field in &schema.fields {
                let py_type = match &field.field_type {
                    ApiType::String => "str".to_string(),
                    ApiType::Integer => "int".to_string(),
                    ApiType::Float => "float".to_string(),
                    ApiType::Boolean => "bool".to_string(),
                    ApiType::Array(_) => "List[Any]".to_string(),
                    ApiType::Map(_) => "Dict[str, Any]".to_string(),
                    ApiType::Optional(inner) => format!("Optional[Any]"),
                    ApiType::Ref(name) => format!("'{}'", name),
                    ApiType::Any => "Any".to_string(),
                };
                if field.required {
                    content.push_str(&format!("    {}: {}\n", field.name, py_type));
                } else {
                    content.push_str(&format!("    {}: {} = None\n", field.name, py_type));
                }
            }
            content.push_str("\n\n");
        }

        // Generate client class
        content.push_str("class NeedleClient:\n");
        content.push_str("    def __init__(self, base_url: str = 'http://localhost:8080'):\n");
        content.push_str("        self.base_url = base_url\n\n");

        for endpoint in &spec.endpoints {
            let method_name = to_snake_case(&endpoint.operation_id);
            let mut params = vec!["self".to_string()];
            for p in &endpoint.path_params {
                params.push(format!("{}: str", p.name));
            }
            if endpoint.request_body.is_some() {
                params.push("body: dict = None".to_string());
            }

            content.push_str(&format!("    def {}({}):\n", method_name, params.join(", ")));
            if let Some(desc) = &endpoint.description {
                content.push_str(&format!("        \"\"\"{}\"\"\"\n", desc));
            }

            let path = endpoint.path.replace('{', "' + str(").replace('}', ") + '");
            content.push_str(&format!(
                "        url = self.base_url + '{}'\n",
                path
            ));
            content.push_str("        # Implementation: use urllib.request or requests\n");
            content.push_str("        pass\n\n");
        }

        vec![GeneratedFile {
            path: format!("{}/client.py", self.config.package_name),
            content,
            language: Language::Python,
        }]
    }

    fn generate_generic(&self, spec: &OpenApiSpec, language: Language) -> Vec<GeneratedFile> {
        let mut content = String::new();
        content.push_str(&format!(
            "// Auto-generated Needle SDK for {}\n",
            language.name()
        ));
        content.push_str(&format!("// Version: {}\n", self.config.version));
        content.push_str(&format!("// Endpoints: {}\n", spec.endpoints.len()));
        content.push_str(&format!("// Schemas: {}\n\n", spec.schemas.len()));

        content.push_str("// Schemas:\n");
        for (name, schema) in &spec.schemas {
            content.push_str(&format!("// - {} ({} fields)\n", name, schema.fields.len()));
        }

        content.push_str("\n// Endpoints:\n");
        for endpoint in &spec.endpoints {
            content.push_str(&format!(
                "// - {} {} ({})\n",
                endpoint.method, endpoint.path, endpoint.operation_id
            ));
        }

        vec![GeneratedFile {
            path: format!(
                "{}/client.{}",
                self.config.package_name,
                language.extension()
            ),
            content,
            language,
        }]
    }
}

fn to_camel_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;
    for (i, c) in s.chars().enumerate() {
        if c == '_' || c == '-' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else if i == 0 {
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    result
}

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    let mut result = first.to_ascii_uppercase().to_string();
                    result.extend(chars);
                    result
                }
            }
        })
        .collect()
}

fn to_snake_case(s: &str) -> String {
    s.replace('-', "_").to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openapi_spec_generation() {
        let spec = OpenApiSpec::from_needle_api();
        assert!(!spec.endpoints.is_empty());
        assert!(!spec.schemas.is_empty());
        assert!(spec.schemas.contains_key("SearchRequest"));
        assert!(spec.schemas.contains_key("SearchResult"));
    }

    #[test]
    fn test_openapi_json() {
        let spec = OpenApiSpec::from_needle_api();
        let json = spec.to_openapi_json();
        assert_eq!(json["openapi"], "3.1.0");
        assert!(json["paths"].is_object());
        assert!(json["components"]["schemas"].is_object());
    }

    #[test]
    fn test_generate_typescript() {
        let spec = OpenApiSpec::from_needle_api();
        let generator = SdkGenerator::new(SdkConfig::default());
        let files = generator.generate(&spec, Language::TypeScript);

        assert!(!files.is_empty());
        assert!(files[0].content.contains("export interface"));
        assert!(files[0].content.contains("NeedleClient"));
    }

    #[test]
    fn test_generate_go() {
        let spec = OpenApiSpec::from_needle_api();
        let generator = SdkGenerator::new(SdkConfig::default());
        let files = generator.generate(&spec, Language::Go);

        assert!(!files.is_empty());
        assert!(files[0].content.contains("package needle_sdk"));
        assert!(files[0].content.contains("type Client struct"));
    }

    #[test]
    fn test_generate_python() {
        let spec = OpenApiSpec::from_needle_api();
        let generator = SdkGenerator::new(SdkConfig::default());
        let files = generator.generate(&spec, Language::Python);

        assert!(!files.is_empty());
        assert!(files[0].content.contains("class NeedleClient"));
        assert!(files[0].content.contains("@dataclass"));
    }

    #[test]
    fn test_generate_all() {
        let spec = OpenApiSpec::from_needle_api();
        let generator = SdkGenerator::new(SdkConfig::default());
        let files = generator.generate_all(&spec);

        assert!(files.len() >= Language::all().len());
    }

    #[test]
    fn test_api_type_conversion() {
        let t = ApiType::Array(Box::new(ApiType::Float));
        assert_eq!(t.to_go(), "[]float64");
        assert_eq!(t.to_typescript(), "number[]");
        assert_eq!(t.to_java(), "List<Double>");
    }

    #[test]
    fn test_case_conversion() {
        assert_eq!(to_camel_case("list_collections"), "listCollections");
        assert_eq!(to_pascal_case("field_name"), "FieldName");
        assert_eq!(to_snake_case("list-collections"), "list_collections");
    }
}
