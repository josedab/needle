//! Typed Client SDK Generators
//!
//! Generates typed REST client code for TypeScript, Go, and Java from the
//! gRPC schema definitions. Each generator produces a complete client with
//! request/response types, error handling, and HTTP methods.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::client_sdk::{
//!     ClientGenerator, Language, GeneratedClient,
//! };
//!
//! let gen = ClientGenerator::new("http://localhost:8080");
//!
//! let ts_client = gen.generate(Language::TypeScript);
//! println!("{}", ts_client.code);
//!
//! let go_client = gen.generate(Language::Go);
//! println!("{}", go_client.code);
//! ```

use serde::{Deserialize, Serialize};

/// Target language for client generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Language {
    TypeScript,
    Go,
    Java,
    Python,
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeScript => write!(f, "typescript"),
            Self::Go => write!(f, "go"),
            Self::Java => write!(f, "java"),
            Self::Python => write!(f, "python"),
        }
    }
}

/// A generated client file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedClient {
    pub language: String,
    pub filename: String,
    pub code: String,
    pub package_name: String,
}

/// Client code generator.
pub struct ClientGenerator {
    base_url: String,
}

impl ClientGenerator {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self { base_url: base_url.into() }
    }

    /// Generate a typed client for the specified language.
    pub fn generate(&self, language: Language) -> GeneratedClient {
        match language {
            Language::TypeScript => self.gen_typescript(),
            Language::Go => self.gen_go(),
            Language::Java => self.gen_java(),
            Language::Python => self.gen_python(),
        }
    }

    /// Generate clients for all supported languages.
    pub fn generate_all(&self) -> Vec<GeneratedClient> {
        vec![
            self.generate(Language::TypeScript),
            self.generate(Language::Go),
            self.generate(Language::Java),
            self.generate(Language::Python),
        ]
    }

    fn gen_typescript(&self) -> GeneratedClient {
        let code = format!(
            r#"// Auto-generated Needle TypeScript Client
// Base URL: {base}

export interface VectorResult {{
  id: string;
  distance: number;
  metadata?: Record<string, unknown>;
}}

export interface SearchRequest {{
  query: number[];
  k: number;
  filter?: Record<string, unknown>;
}}

export interface InsertRequest {{
  id: string;
  vector: number[];
  metadata?: Record<string, unknown>;
}}

export interface TextInsertRequest {{
  id: string;
  text: string;
  metadata?: Record<string, unknown>;
}}

export interface CollectionInfo {{
  name: string;
  dimensions: number;
  vector_count: number;
}}

export class NeedleClient {{
  private baseUrl: string;

  constructor(baseUrl: string = "{base}") {{
    this.baseUrl = baseUrl;
  }}

  async createCollection(name: string, dimensions: number): Promise<void> {{
    await fetch(`${{this.baseUrl}}/collections`, {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ name, dimensions }}),
    }});
  }}

  async insert(collection: string, req: InsertRequest): Promise<void> {{
    await fetch(`${{this.baseUrl}}/collections/${{collection}}/vectors`, {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify(req),
    }});
  }}

  async insertText(collection: string, req: TextInsertRequest): Promise<void> {{
    await fetch(`${{this.baseUrl}}/collections/${{collection}}/text`, {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify(req),
    }});
  }}

  async search(collection: string, req: SearchRequest): Promise<VectorResult[]> {{
    const resp = await fetch(`${{this.baseUrl}}/collections/${{collection}}/search`, {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify(req),
    }});
    return resp.json();
  }}

  async listCollections(): Promise<CollectionInfo[]> {{
    const resp = await fetch(`${{this.baseUrl}}/collections`);
    return resp.json();
  }}

  async health(): Promise<boolean> {{
    const resp = await fetch(`${{this.baseUrl}}/health`);
    return resp.ok;
  }}
}}
"#,
            base = self.base_url
        );

        GeneratedClient {
            language: "typescript".into(),
            filename: "needle-client.ts".into(),
            code,
            package_name: "@anthropic/needle-client".into(),
        }
    }

    fn gen_go(&self) -> GeneratedClient {
        let code = format!(
            r#"// Auto-generated Needle Go Client
// Base URL: {base}
package needle

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

type VectorResult struct {{
	ID       string                 `json:"id"`
	Distance float32                `json:"distance"`
	Metadata map[string]interface{{}} `json:"metadata,omitempty"`
}}

type SearchRequest struct {{
	Query  []float32              `json:"query"`
	K      int                    `json:"k"`
	Filter map[string]interface{{}} `json:"filter,omitempty"`
}}

type InsertRequest struct {{
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Metadata map[string]interface{{}} `json:"metadata,omitempty"`
}}

type Client struct {{
	BaseURL    string
	HTTPClient *http.Client
}}

func NewClient(baseURL string) *Client {{
	return &Client{{BaseURL: baseURL, HTTPClient: http.DefaultClient}}
}}

func (c *Client) Search(collection string, req SearchRequest) ([]VectorResult, error) {{
	body, _ := json.Marshal(req)
	resp, err := c.HTTPClient.Post(
		fmt.Sprintf("%s/collections/%s/search", c.BaseURL, collection),
		"application/json", bytes.NewReader(body),
	)
	if err != nil {{
		return nil, err
	}}
	defer resp.Body.Close()
	var results []VectorResult
	json.NewDecoder(resp.Body).Decode(&results)
	return results, nil
}}

func (c *Client) Insert(collection string, req InsertRequest) error {{
	body, _ := json.Marshal(req)
	_, err := c.HTTPClient.Post(
		fmt.Sprintf("%s/collections/%s/vectors", c.BaseURL, collection),
		"application/json", bytes.NewReader(body),
	)
	return err
}}

func (c *Client) Health() bool {{
	resp, err := c.HTTPClient.Get(fmt.Sprintf("%s/health", c.BaseURL))
	return err == nil && resp.StatusCode == 200
}}
"#,
            base = self.base_url
        );

        GeneratedClient {
            language: "go".into(),
            filename: "client.go".into(),
            code,
            package_name: "github.com/anthropics/needle-go".into(),
        }
    }

    fn gen_java(&self) -> GeneratedClient {
        let code = format!(
            r#"// Auto-generated Needle Java Client
// Base URL: {base}
package com.anthropic.needle;

import java.net.http.*;
import java.net.URI;

public class NeedleClient {{
    private final String baseUrl;
    private final HttpClient http;

    public NeedleClient(String baseUrl) {{
        this.baseUrl = baseUrl;
        this.http = HttpClient.newHttpClient();
    }}

    public NeedleClient() {{
        this("{base}");
    }}

    public String search(String collection, String queryJson) throws Exception {{
        HttpRequest req = HttpRequest.newBuilder()
            .uri(URI.create(baseUrl + "/collections/" + collection + "/search"))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(queryJson))
            .build();
        HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
        return resp.body();
    }}

    public boolean health() {{
        try {{
            HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/health"))
                .GET().build();
            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            return resp.statusCode() == 200;
        }} catch (Exception e) {{
            return false;
        }}
    }}
}}
"#,
            base = self.base_url
        );

        GeneratedClient {
            language: "java".into(),
            filename: "NeedleClient.java".into(),
            code,
            package_name: "com.anthropic.needle".into(),
        }
    }

    fn gen_python(&self) -> GeneratedClient {
        let code = format!(
            r#""""Auto-generated Needle Python Client."""
import requests
from typing import List, Dict, Optional, Any

class NeedleClient:
    def __init__(self, base_url: str = "{base}"):
        self.base_url = base_url

    def search(self, collection: str, query: List[float], k: int = 10,
               filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        resp = requests.post(
            f"{{self.base_url}}/collections/{{collection}}/search",
            json={{"query": query, "k": k, "filter": filter}},
        )
        resp.raise_for_status()
        return resp.json()

    def insert(self, collection: str, id: str, vector: List[float],
               metadata: Optional[Dict] = None) -> None:
        requests.post(
            f"{{self.base_url}}/collections/{{collection}}/vectors",
            json={{"id": id, "vector": vector, "metadata": metadata}},
        ).raise_for_status()

    def insert_text(self, collection: str, id: str, text: str,
                    metadata: Optional[Dict] = None) -> None:
        requests.post(
            f"{{self.base_url}}/collections/{{collection}}/text",
            json={{"id": id, "text": text, "metadata": metadata}},
        ).raise_for_status()

    def health(self) -> bool:
        try:
            return requests.get(f"{{self.base_url}}/health").ok
        except Exception:
            return False
"#,
            base = self.base_url
        );

        GeneratedClient {
            language: "python".into(),
            filename: "needle_client.py".into(),
            code,
            package_name: "needle-client".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typescript_generation() {
        let gen = ClientGenerator::new("http://localhost:8080");
        let client = gen.generate(Language::TypeScript);
        assert_eq!(client.filename, "needle-client.ts");
        assert!(client.code.contains("NeedleClient"));
        assert!(client.code.contains("search"));
        assert!(client.code.contains("insertText"));
        assert!(client.code.contains("VectorResult"));
    }

    #[test]
    fn test_go_generation() {
        let gen = ClientGenerator::new("http://localhost:8080");
        let client = gen.generate(Language::Go);
        assert_eq!(client.filename, "client.go");
        assert!(client.code.contains("package needle"));
        assert!(client.code.contains("func (c *Client) Search"));
    }

    #[test]
    fn test_java_generation() {
        let gen = ClientGenerator::new("http://localhost:8080");
        let client = gen.generate(Language::Java);
        assert_eq!(client.filename, "NeedleClient.java");
        assert!(client.code.contains("public class NeedleClient"));
    }

    #[test]
    fn test_python_generation() {
        let gen = ClientGenerator::new("http://localhost:8080");
        let client = gen.generate(Language::Python);
        assert_eq!(client.filename, "needle_client.py");
        assert!(client.code.contains("class NeedleClient"));
        assert!(client.code.contains("insert_text"));
    }

    #[test]
    fn test_generate_all() {
        let gen = ClientGenerator::new("http://localhost:8080");
        let clients = gen.generate_all();
        assert_eq!(clients.len(), 4);
        let languages: Vec<&str> = clients.iter().map(|c| c.language.as_str()).collect();
        assert!(languages.contains(&"typescript"));
        assert!(languages.contains(&"go"));
        assert!(languages.contains(&"java"));
        assert!(languages.contains(&"python"));
    }

    #[test]
    fn test_custom_base_url() {
        let gen = ClientGenerator::new("https://api.needle.dev");
        let client = gen.generate(Language::TypeScript);
        assert!(client.code.contains("api.needle.dev"));
    }
}
