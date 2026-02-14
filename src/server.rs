//! HTTP REST API server for Needle
//!
//! Provides a REST API for vector database operations, making Needle
//! accessible from any language or tool that can make HTTP requests.
//!
//! # Authentication
//!
//! The server supports multiple authentication methods:
//!
//! ## API Key Authentication
//!
//! ```rust,ignore
//! use needle::server::{ServerConfig, AuthConfig, ApiKey};
//!
//! let auth_config = AuthConfig::new()
//!     .with_api_key(ApiKey::new("my-api-key").with_role("admin"))
//!     .require_auth(true);
//!
//! let config = ServerConfig::default()
//!     .with_auth(auth_config);
//! ```
//!
//! API keys are passed via the `X-API-Key` header:
//! ```bash
//! curl -H "X-API-Key: my-api-key" http://localhost:8080/collections
//! ```
//!
//! ## JWT Authentication
//!
//! ```rust,ignore
//! let auth_config = AuthConfig::new()
//!     .with_jwt_secret("your-secret-key")
//!     .require_auth(true);
//! ```
//!
//! JWT tokens are passed via the `Authorization: Bearer <token>` header.

use crate::database::Database;
use crate::error::NeedleError;
use crate::metadata::Filter;
use crate::security::{Role, User};
use axum::{
    body::Body,
    extract::{ConnectInfo, Path, Query, State},
    http::{header, Method, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Json, Router,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use governor::{clock::DefaultClock, state::keyed::DashMapStateStore, Quota, RateLimiter};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::Sha256;
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};

#[cfg(feature = "metrics")]
use crate::metrics::{http_metrics, metrics};

// ============ Authentication Types ============

/// API key configuration for authentication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// The API key value (should be kept secret)
    pub key: String,
    /// Optional name/description for the key
    pub name: Option<String>,
    /// Roles assigned to this API key
    pub roles: Vec<String>,
    /// Whether the key is active
    pub active: bool,
    /// Optional expiration timestamp (Unix epoch seconds)
    pub expires_at: Option<u64>,
}

impl ApiKey {
    /// Create a new API key with default reader role.
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            name: None,
            roles: vec!["reader".to_string()],
            active: true,
            expires_at: None,
        }
    }

    /// Set a name/description for this key.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the roles for this API key.
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.roles = vec![role.into()];
        self
    }

    /// Add multiple roles to this API key.
    pub fn with_roles(mut self, roles: Vec<String>) -> Self {
        self.roles = roles;
        self
    }

    /// Set an expiration time for this key (Unix epoch seconds).
    pub fn expires_at(mut self, timestamp: u64) -> Self {
        self.expires_at = Some(timestamp);
        self
    }

    /// Check if the key is expired.
    pub fn is_expired(&self) -> bool {
        if let Some(exp) = self.expires_at {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            exp < now
        } else {
            false
        }
    }

    /// Check if the key is valid (active and not expired).
    pub fn is_valid(&self) -> bool {
        self.active && !self.is_expired()
    }

    /// Convert this API key to a User for RBAC checks.
    pub fn to_user(&self) -> User {
        let mut user = User::new(format!(
            "apikey:{}",
            self.key.chars().take(8).collect::<String>()
        ));
        if let Some(name) = &self.name {
            user = user.with_name(name.clone());
        }
        for role_name in &self.roles {
            let role = match role_name.as_str() {
                "admin" => Role::admin(),
                "writer" => Role::writer(),
                _ => Role::reader(),
            };
            user = user.with_role(role);
        }
        user
    }
}

/// JWT claims for token validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtClaims {
    /// Subject (user ID)
    pub sub: String,
    /// Expiration time (Unix timestamp)
    pub exp: u64,
    /// Issued at (Unix timestamp)
    pub iat: u64,
    /// Roles assigned to this token
    #[serde(default)]
    pub roles: Vec<String>,
    /// Additional custom claims
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl JwtClaims {
    /// Create new claims for a user.
    pub fn new(subject: impl Into<String>, expires_in_secs: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            sub: subject.into(),
            exp: now + expires_in_secs,
            iat: now,
            roles: vec!["reader".to_string()],
            extra: HashMap::new(),
        }
    }

    /// Set roles for this token.
    pub fn with_roles(mut self, roles: Vec<String>) -> Self {
        self.roles = roles;
        self
    }

    /// Check if the token is expired.
    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.exp < now
    }

    /// Convert claims to a User for RBAC checks.
    pub fn to_user(&self) -> User {
        let mut user = User::new(&self.sub);
        for role_name in &self.roles {
            let role = match role_name.as_str() {
                "admin" => Role::admin(),
                "writer" => Role::writer(),
                _ => Role::reader(),
            };
            user = user.with_role(role);
        }
        user
    }
}

/// Authentication configuration for the server.
#[derive(Debug, Clone, Default)]
pub struct AuthConfig {
    /// Whether authentication is required for all endpoints
    pub require_auth: bool,
    /// API keys for authentication
    pub api_keys: Vec<ApiKey>,
    /// JWT secret for token validation (HS256)
    pub jwt_secret: Option<String>,
    /// Endpoints that don't require authentication (e.g., "/health")
    pub public_endpoints: Vec<String>,
}

impl AuthConfig {
    /// Create a new authentication configuration.
    pub fn new() -> Self {
        Self {
            require_auth: false,
            api_keys: Vec::new(),
            jwt_secret: None,
            public_endpoints: vec!["/health".to_string(), "/".to_string()],
        }
    }

    /// Require authentication for all endpoints except public ones.
    pub fn require_auth(mut self, require: bool) -> Self {
        self.require_auth = require;
        self
    }

    /// Add an API key for authentication.
    pub fn with_api_key(mut self, key: ApiKey) -> Self {
        self.api_keys.push(key);
        self
    }

    /// Add multiple API keys.
    pub fn with_api_keys(mut self, keys: Vec<ApiKey>) -> Self {
        self.api_keys.extend(keys);
        self
    }

    /// Set the JWT secret for token validation.
    pub fn with_jwt_secret(mut self, secret: impl Into<String>) -> Self {
        self.jwt_secret = Some(secret.into());
        self
    }

    /// Add a public endpoint that doesn't require authentication.
    pub fn with_public_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.public_endpoints.push(endpoint.into());
        self
    }

    /// Check if an endpoint is public (doesn't require auth).
    pub fn is_public_endpoint(&self, path: &str) -> bool {
        self.public_endpoints.iter().any(|e| {
            // Exact match or path starts with endpoint followed by / or ?
            path == e || (e != "/" && path.starts_with(e))
        })
    }

    /// Validate an API key and return the associated user.
    pub fn validate_api_key(&self, key: &str) -> Option<User> {
        self.api_keys
            .iter()
            .find(|k| k.key == key && k.is_valid())
            .map(|k| k.to_user())
    }

    /// Validate a JWT token and return the claims.
    pub fn validate_jwt(&self, token: &str) -> Result<JwtClaims, AuthError> {
        let secret = self.jwt_secret.as_ref().ok_or(AuthError::NoJwtSecret)?;

        // Split token into parts
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err(AuthError::InvalidToken("Invalid token format".into()));
        }

        let header = parts[0];
        let payload = parts[1];
        let signature = parts[2];

        // Verify signature (HS256)
        type HmacSha256 = Hmac<Sha256>;
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
            .map_err(|_| AuthError::InvalidToken("Invalid secret".into()))?;
        mac.update(format!("{}.{}", header, payload).as_bytes());

        let sig_bytes = URL_SAFE_NO_PAD
            .decode(signature)
            .map_err(|_| AuthError::InvalidToken("Invalid signature encoding".into()))?;

        mac.verify_slice(&sig_bytes)
            .map_err(|_| AuthError::InvalidSignature)?;

        // Decode payload
        let payload_bytes = URL_SAFE_NO_PAD
            .decode(payload)
            .map_err(|_| AuthError::InvalidToken("Invalid payload encoding".into()))?;
        let claims: JwtClaims = serde_json::from_slice(&payload_bytes)
            .map_err(|e| AuthError::InvalidToken(format!("Invalid claims: {}", e)))?;

        // Check expiration
        if claims.is_expired() {
            return Err(AuthError::TokenExpired);
        }

        Ok(claims)
    }

    /// Generate a JWT token for the given claims.
    pub fn generate_jwt(&self, claims: &JwtClaims) -> Result<String, AuthError> {
        let secret = self.jwt_secret.as_ref().ok_or(AuthError::NoJwtSecret)?;

        // Header (always HS256)
        let header = r#"{"alg":"HS256","typ":"JWT"}"#;
        let header_b64 = URL_SAFE_NO_PAD.encode(header.as_bytes());

        // Payload
        let payload = serde_json::to_string(claims)
            .map_err(|e| AuthError::InvalidToken(format!("Failed to serialize claims: {}", e)))?;
        let payload_b64 = URL_SAFE_NO_PAD.encode(payload.as_bytes());

        // Signature
        type HmacSha256 = Hmac<Sha256>;
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
            .map_err(|_| AuthError::InvalidToken("Invalid secret".into()))?;
        mac.update(format!("{}.{}", header_b64, payload_b64).as_bytes());
        let sig = mac.finalize().into_bytes();
        let sig_b64 = URL_SAFE_NO_PAD.encode(sig);

        Ok(format!("{}.{}.{}", header_b64, payload_b64, sig_b64))
    }
}

/// Authentication errors.
#[derive(Debug, Clone)]
pub enum AuthError {
    /// No authentication credentials provided
    MissingCredentials,
    /// Invalid API key
    InvalidApiKey,
    /// Invalid JWT token
    InvalidToken(String),
    /// JWT signature verification failed
    InvalidSignature,
    /// JWT token has expired
    TokenExpired,
    /// No JWT secret configured
    NoJwtSecret,
    /// Insufficient permissions
    Forbidden(String),
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::MissingCredentials => write!(f, "Authentication required"),
            AuthError::InvalidApiKey => write!(f, "Invalid API key"),
            AuthError::InvalidToken(msg) => write!(f, "Invalid token: {}", msg),
            AuthError::InvalidSignature => write!(f, "Invalid token signature"),
            AuthError::TokenExpired => write!(f, "Token has expired"),
            AuthError::NoJwtSecret => write!(f, "JWT authentication not configured"),
            AuthError::Forbidden(msg) => write!(f, "Access denied: {}", msg),
        }
    }
}

impl std::error::Error for AuthError {}

/// Authenticated user context available in request extensions.
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// The authenticated user
    pub user: User,
    /// The authentication method used
    pub method: AuthMethod,
}

/// How the user was authenticated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthMethod {
    /// API key authentication
    ApiKey,
    /// JWT bearer token
    Jwt,
    /// No authentication (public endpoint or auth not required)
    None,
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to bind to
    pub addr: SocketAddr,
    /// CORS configuration
    pub cors_config: CorsConfig,
    /// Database path (None for in-memory)
    pub db_path: Option<String>,
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
    /// Authentication configuration
    pub auth: AuthConfig,
    /// Maximum request body size in bytes (default: 100MB)
    pub max_body_size: usize,
    /// Maximum number of items in a batch operation (default: 10000)
    /// Prevents memory exhaustion from large batch requests
    pub max_batch_size: usize,
    /// Request timeout in seconds (default: 30)
    /// Requests exceeding this duration will be terminated
    pub request_timeout_secs: u64,
    /// Trusted proxies for honoring X-Forwarded-For/X-Real-IP headers.
    /// Only requests from these IPs may override the client IP.
    pub trusted_proxies: Vec<IpAddr>,
}

/// CORS configuration
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Enable CORS
    pub enabled: bool,
    /// Allowed origins (None = allow all, Some([]) = deny all external)
    pub allowed_origins: Option<Vec<String>>,
    /// Allow credentials
    pub allow_credentials: bool,
    /// Max age for preflight cache (in seconds)
    pub max_age_secs: u64,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // Default to localhost only for security
            allowed_origins: Some(vec![
                "http://localhost:3000".to_string(),
                "http://localhost:8080".to_string(),
                "http://127.0.0.1:3000".to_string(),
                "http://127.0.0.1:8080".to_string(),
            ]),
            allow_credentials: false,
            max_age_secs: 3600,
        }
    }
}

impl CorsConfig {
    /// Create a permissive CORS config (not recommended for production)
    ///
    /// Note: Credentials are NOT enabled by default even in permissive mode.
    /// Combining allow-all-origins with credentials is a CSRF vulnerability.
    /// Use `with_credentials(true)` only with specific origins for secure apps.
    pub fn permissive() -> Self {
        Self {
            enabled: true,
            allowed_origins: None,    // Allow all
            allow_credentials: false, // SECURITY: Never combine wildcard origins with credentials
            max_age_secs: 3600,
        }
    }

    /// Enable credentials (cookies, auth headers).
    ///
    /// WARNING: Only enable credentials with specific allowed_origins, not with
    /// wildcard origins. Combining both creates a CSRF vulnerability where any
    /// website can make authenticated requests on behalf of users.
    pub fn with_credentials(mut self, allow: bool) -> Self {
        self.allow_credentials = allow;
        self
    }

    /// Create a restrictive CORS config
    pub fn restrictive() -> Self {
        Self {
            enabled: true,
            allowed_origins: Some(vec![]), // No external origins
            allow_credentials: false,
            max_age_secs: 0,
        }
    }

    /// Add an allowed origin
    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        let origins = self.allowed_origins.get_or_insert_with(Vec::new);
        origins.push(origin.into());
        self
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Requests per second (global limit)
    pub requests_per_second: u32,
    /// Burst size (allows short bursts above the rate)
    pub burst_size: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_second: 100,
            burst_size: 50,
        }
    }
}

impl RateLimitConfig {
    /// Disable rate limiting
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            requests_per_second: 0,
            burst_size: 0,
        }
    }

    /// Set requests per second
    pub fn with_rate(mut self, rps: u32) -> Self {
        self.requests_per_second = rps;
        self
    }

    /// Set burst size
    pub fn with_burst(mut self, burst: u32) -> Self {
        self.burst_size = burst;
        self
    }
}

fn default_trusted_proxies() -> Vec<IpAddr> {
    vec![
        IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(Ipv6Addr::LOCALHOST),
    ]
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:8080"
                .parse()
                .unwrap_or_else(|_| SocketAddr::from(([127, 0, 0, 1], 8080))),
            cors_config: CorsConfig::default(),
            db_path: None,
            rate_limit: RateLimitConfig::default(),
            auth: AuthConfig::default(),
            max_body_size: 100 * 1024 * 1024, // 100MB
            max_batch_size: 10_000,           // 10k items max per batch
            request_timeout_secs: 30,         // 30 seconds default timeout
            trusted_proxies: default_trusted_proxies(),
        }
    }
}

impl ServerConfig {
    pub fn new(addr: &str) -> Result<Self, std::net::AddrParseError> {
        Ok(Self {
            addr: addr.parse()?,
            ..Default::default()
        })
    }

    pub fn with_db_path(mut self, path: impl Into<String>) -> Self {
        self.db_path = Some(path.into());
        self
    }

    pub fn with_cors(mut self, config: CorsConfig) -> Self {
        self.cors_config = config;
        self
    }

    pub fn with_rate_limit(mut self, config: RateLimitConfig) -> Self {
        self.rate_limit = config;
        self
    }

    pub fn with_max_body_size(mut self, bytes: usize) -> Self {
        self.max_body_size = bytes;
        self
    }

    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    pub fn with_request_timeout(mut self, secs: u64) -> Self {
        self.request_timeout_secs = secs;
        self
    }

    /// Add a trusted proxy IP for forwarded header handling.
    pub fn with_trusted_proxy(mut self, proxy: IpAddr) -> Self {
        self.trusted_proxies.push(proxy);
        self
    }

    /// Replace the trusted proxy list. Use an empty list to disable trust.
    pub fn with_trusted_proxies(mut self, proxies: Vec<IpAddr>) -> Self {
        self.trusted_proxies = proxies;
        self
    }

    /// Set authentication configuration.
    pub fn with_auth(mut self, config: AuthConfig) -> Self {
        self.auth = config;
        self
    }
}

/// Per-IP rate limiter type
type PerIpRateLimiter = RateLimiter<IpAddr, DashMapStateStore<IpAddr>, DefaultClock>;

/// Shared application state
pub struct AppState {
    db: RwLock<Database>,
    rate_limiter: Option<Arc<PerIpRateLimiter>>,
    /// Authentication configuration
    auth: AuthConfig,
    /// Maximum items allowed in batch operations
    max_batch_size: usize,
    /// Trusted proxies for forwarded header handling
    trusted_proxies: Vec<IpAddr>,
}

impl AppState {
    /// Create a new AppState with the given database and no rate limiting
    pub fn new(db: Database) -> Self {
        Self {
            db: RwLock::new(db),
            rate_limiter: None,
            auth: AuthConfig::default(),
            max_batch_size: 10_000, // default
            trusted_proxies: default_trusted_proxies(),
        }
    }

    /// Create a new AppState with the given database and rate limiting config
    pub fn with_rate_limit(db: Database, config: &RateLimitConfig) -> Self {
        Self {
            db: RwLock::new(db),
            rate_limiter: create_rate_limiter(config),
            auth: AuthConfig::default(),
            max_batch_size: 10_000, // default
            trusted_proxies: default_trusted_proxies(),
        }
    }

    /// Create a new AppState with full configuration
    pub fn with_config(db: Database, config: &ServerConfig) -> Self {
        Self {
            db: RwLock::new(db),
            rate_limiter: create_rate_limiter(&config.rate_limit),
            auth: config.auth.clone(),
            max_batch_size: config.max_batch_size,
            trusted_proxies: config.trusted_proxies.clone(),
        }
    }
}

/// API error response
#[derive(Debug, Serialize)]
struct ApiError {
    error: String,
    code: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    help: String,
}

impl ApiError {
    fn new(error: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            code: code.into(),
            help: String::new(),
        }
    }
}

impl From<NeedleError> for (StatusCode, Json<ApiError>) {
    fn from(err: NeedleError) -> Self {
        let (status, code) = match &err {
            NeedleError::CollectionNotFound(_) => (StatusCode::NOT_FOUND, "COLLECTION_NOT_FOUND"),
            NeedleError::VectorNotFound(_) => (StatusCode::NOT_FOUND, "VECTOR_NOT_FOUND"),
            NeedleError::CollectionAlreadyExists(_) => (StatusCode::CONFLICT, "COLLECTION_EXISTS"),
            NeedleError::VectorAlreadyExists(_) => (StatusCode::CONFLICT, "VECTOR_EXISTS"),
            NeedleError::DimensionMismatch { .. } => {
                (StatusCode::BAD_REQUEST, "DIMENSION_MISMATCH")
            }
            NeedleError::InvalidVector(_) => (StatusCode::BAD_REQUEST, "INVALID_VECTOR"),
            NeedleError::InvalidConfig(_) => (StatusCode::BAD_REQUEST, "INVALID_CONFIG"),
            NeedleError::AliasNotFound(_) => (StatusCode::NOT_FOUND, "ALIAS_NOT_FOUND"),
            NeedleError::AliasAlreadyExists(_) => (StatusCode::CONFLICT, "ALIAS_EXISTS"),
            NeedleError::CollectionHasAliases(_) => {
                (StatusCode::CONFLICT, "COLLECTION_HAS_ALIASES")
            }
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR"),
        };

        let help = err.help();

        (
            status,
            Json(ApiError {
                error: err.to_string(),
                code: code.to_string(),
                help,
            }),
        )
    }
}

// ============ Request/Response Types ============

#[derive(Debug, Deserialize)]
struct CreateCollectionRequest {
    name: String,
    dimensions: usize,
    #[serde(default)]
    distance: Option<String>,
    #[serde(default)]
    m: Option<usize>,
    #[serde(default)]
    ef_construction: Option<usize>,
}

#[derive(Debug, Serialize)]
struct CollectionInfo {
    name: String,
    dimensions: usize,
    count: usize,
    deleted_count: usize,
}

#[derive(Debug, Deserialize)]
struct InsertRequest {
    id: String,
    vector: Vec<f32>,
    #[serde(default)]
    metadata: Option<Value>,
    /// Optional TTL in seconds; if not provided, uses collection default
    #[serde(default)]
    ttl_seconds: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct BatchInsertRequest {
    vectors: Vec<InsertRequest>,
}

#[derive(Debug, Deserialize)]
struct UpsertRequest {
    id: String,
    vector: Vec<f32>,
    #[serde(default)]
    metadata: Option<Value>,
    /// Optional TTL in seconds; if not provided, uses collection default
    #[serde(default)]
    ttl_seconds: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct SearchRequest {
    vector: Vec<f32>,
    #[serde(default = "default_k")]
    k: usize,
    /// Pre-filter: applied during ANN search (efficient, reduces candidates)
    #[serde(default)]
    filter: Option<Value>,
    /// Post-filter: applied after ANN search (guarantees k candidates before filtering)
    #[serde(default)]
    post_filter: Option<Value>,
    /// Over-fetch factor for post-filtering (default: 3)
    /// Search fetches k * post_filter_factor candidates before post-filtering
    #[serde(default = "default_post_filter_factor")]
    post_filter_factor: usize,
    #[serde(default)]
    include_vectors: bool,
    #[serde(default)]
    explain: bool,
    /// Override distance function for this query ("cosine", "euclidean", "dot", "manhattan")
    /// When different from the collection's index, uses brute-force search
    #[serde(default)]
    distance: Option<String>,
}

fn default_k() -> usize {
    10
}

fn default_post_filter_factor() -> usize {
    3
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    results: Vec<SearchResultResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    explanation: Option<SearchExplanation>,
}

#[derive(Debug, Serialize)]
struct SearchResultResponse {
    id: String,
    distance: f32,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vector: Option<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct SearchExplanation {
    query_norm: f32,
    distance_metric: String,
    top_dimensions: Vec<DimensionContribution>,
    /// Detailed profiling data (when available)
    #[serde(skip_serializing_if = "Option::is_none")]
    profiling: Option<ProfilingData>,
}

#[derive(Debug, Serialize)]
struct ProfilingData {
    /// Total search time in microseconds
    total_time_us: u64,
    /// Time spent in HNSW index traversal (microseconds)
    index_time_us: u64,
    /// Time spent evaluating metadata filters (microseconds)
    filter_time_us: u64,
    /// Time spent enriching results with metadata (microseconds)
    enrich_time_us: u64,
    /// Number of candidates before filtering
    candidates_before_filter: usize,
    /// Number of candidates after filtering
    candidates_after_filter: usize,
    /// HNSW index statistics
    hnsw_stats: HnswStatsResponse,
    /// Collection dimensions
    dimensions: usize,
    /// Collection vector count
    collection_size: usize,
    /// Requested k value
    requested_k: usize,
    /// Effective k (clamped to collection size)
    effective_k: usize,
    /// ef_search parameter used
    ef_search: usize,
    /// Whether a filter was applied
    filter_applied: bool,
}

#[derive(Debug, Serialize)]
struct HnswStatsResponse {
    /// Number of nodes visited during the search
    visited_nodes: usize,
    /// Number of layers traversed (including layer 0)
    layers_traversed: usize,
    /// Number of distance computations performed
    distance_computations: usize,
    /// Time spent in HNSW traversal (microseconds)
    traversal_time_us: u64,
}

#[derive(Debug, Serialize)]
struct DimensionContribution {
    dimension: usize,
    query_value: f32,
    contribution: f32,
}

#[derive(Debug, Deserialize)]
struct BatchSearchRequest {
    vectors: Vec<Vec<f32>>,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    filter: Option<Value>,
}

/// Request for radius-based (range) search
#[derive(Debug, Deserialize)]
struct RadiusSearchRequest {
    /// Query vector
    vector: Vec<f32>,
    /// Maximum distance from query (all vectors within this distance are returned)
    max_distance: f32,
    /// Maximum number of results to return
    #[serde(default = "default_radius_limit")]
    limit: usize,
    /// Optional metadata filter
    #[serde(default)]
    filter: Option<Value>,
    /// Include vector data in response
    #[serde(default)]
    include_vectors: bool,
}

fn default_radius_limit() -> usize {
    1000
}

#[derive(Debug, Serialize)]
struct VectorResponse {
    id: String,
    vector: Vec<f32>,
    metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct UpdateMetadataRequest {
    metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct QueryParams {
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

// ============ Alias Request/Response Types ============

#[derive(Debug, Deserialize)]
struct CreateAliasRequest {
    alias: String,
    collection: String,
}

#[derive(Debug, Deserialize)]
struct UpdateAliasRequest {
    collection: String,
}

#[derive(Debug, Serialize)]
struct AliasInfo {
    alias: String,
    collection: String,
}

// ============ Handlers ============

/// Health check endpoint
async fn health() -> impl IntoResponse {
    Json(json!({"status": "healthy", "version": env!("CARGO_PKG_VERSION")}))
}

/// Get database info
async fn get_info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = state.db.read().await;
    let collections = db.list_collections();

    Json(json!({
        "collections": collections.len(),
        "total_vectors": db.total_vectors(),
    }))
}

/// List all collections
async fn list_collections(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = state.db.read().await;
    let collections: Vec<CollectionInfo> = db
        .list_collections()
        .into_iter()
        .filter_map(|name| {
            let coll = db.collection(&name).ok()?;
            Some(CollectionInfo {
                name,
                dimensions: coll.dimensions().unwrap_or(0),
                count: coll.len(),
                deleted_count: coll.deleted_count(),
            })
        })
        .collect();

    Json(json!({"collections": collections}))
}

/// Create a new collection
async fn create_collection(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateCollectionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;

    let mut config = crate::CollectionConfig::new(&req.name, req.dimensions);

    if let Some(dist) = &req.distance {
        config = config.with_distance(match dist.to_lowercase().as_str() {
            "euclidean" | "l2" => crate::DistanceFunction::Euclidean,
            "dot" | "dotproduct" => crate::DistanceFunction::DotProduct,
            "manhattan" | "l1" => crate::DistanceFunction::Manhattan,
            _ => crate::DistanceFunction::Cosine,
        });
    }

    if let Some(m) = req.m {
        config = config.with_m(m);
    }

    if let Some(ef) = req.ef_construction {
        config = config.with_ef_construction(ef);
    }

    db.create_collection_with_config(config)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok((StatusCode::CREATED, Json(json!({"created": req.name}))))
}

/// Get collection info
async fn get_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&name)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({
        "name": name,
        "dimensions": coll.dimensions(),
        "count": coll.len(),
        "deleted_count": coll.deleted_count(),
        "needs_compaction": coll.needs_compaction(0.2),
    })))
}

/// Delete a collection
async fn delete_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let dropped = db
        .delete_collection(&name)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    if dropped {
        Ok(Json(json!({"deleted": name})))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ApiError::new(
                format!("Collection '{}' not found", name),
                "COLLECTION_NOT_FOUND".to_string(),
            )),
        ))
    }
}

/// Insert a vector
async fn insert_vector(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<InsertRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    coll.insert_with_ttl(&req.id, &req.vector, req.metadata, req.ttl_seconds)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok((StatusCode::CREATED, Json(json!({"inserted": req.id}))))
}

/// Batch insert vectors
async fn batch_insert(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<BatchInsertRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    // Validate batch size to prevent memory exhaustion
    if req.vectors.len() > state.max_batch_size {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ApiError::new(
                format!(
                    "Batch size {} exceeds maximum allowed {}",
                    req.vectors.len(),
                    state.max_batch_size
                ),
                "BATCH_TOO_LARGE".to_string(),
            )),
        ));
    }

    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let mut inserted = 0;
    let mut errors = Vec::new();

    for item in req.vectors {
        match coll.insert_with_ttl(&item.id, &item.vector, item.metadata, item.ttl_seconds) {
            Ok(_) => inserted += 1,
            Err(e) => errors.push(json!({"id": item.id, "error": e.to_string()})),
        }
    }

    Ok(Json(json!({
        "inserted": inserted,
        "errors": errors,
    })))
}

/// Upsert a vector
async fn upsert_vector(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<UpsertRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    // Check if exists and update or insert
    let existed = if coll.get(&req.id).is_some() {
        coll.delete(&req.id)
            .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
        true
    } else {
        false
    };

    coll.insert_with_ttl(&req.id, &req.vector, req.metadata, req.ttl_seconds)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({
        "id": req.id,
        "updated": existed,
    })))
}

/// Get a vector by ID
async fn get_vector(
    State(state): State<Arc<AppState>>,
    Path((collection, id)): Path<(String, String)>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    match coll.get(&id) {
        Some((vector, metadata)) => Ok(Json(VectorResponse {
            id,
            vector,
            metadata,
        })),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(ApiError::new(
                format!("Vector '{}' not found", id),
                "VECTOR_NOT_FOUND".to_string(),
            )),
        )),
    }
}

/// Delete a vector
async fn delete_vector(
    State(state): State<Arc<AppState>>,
    Path((collection, id)): Path<(String, String)>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let deleted = coll
        .delete(&id)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    if deleted {
        Ok(Json(json!({"deleted": id})))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ApiError::new(
                format!("Vector '{}' not found", id),
                "VECTOR_NOT_FOUND".to_string(),
            )),
        ))
    }
}

/// Update vector metadata
async fn update_metadata(
    State(state): State<Arc<AppState>>,
    Path((collection, id)): Path<(String, String)>,
    Json(req): Json<UpdateMetadataRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    // Get existing vector
    let (vector, _) = coll.get(&id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ApiError::new(
                format!("Vector '{}' not found", id),
                "VECTOR_NOT_FOUND".to_string(),
            )),
        )
    })?;

    // Delete and re-insert with new metadata
    coll.delete(&id)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
    coll.insert(&id, &vector, req.metadata)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({"updated": id})))
}

/// Search for similar vectors
async fn search(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<SearchRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    use crate::DistanceFunction;

    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    // Parse filters once
    let pre_filter = if let Some(filter_value) = &req.filter {
        Some(Filter::parse(filter_value).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(
                    format!("Invalid pre-filter: {}", e),
                    "INVALID_FILTER",
                )),
            )
        })?)
    } else {
        None
    };

    let post_filter = if let Some(filter_value) = &req.post_filter {
        Some(Filter::parse(filter_value).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(
                    format!("Invalid post-filter: {}", e),
                    "INVALID_POST_FILTER",
                )),
            )
        })?)
    } else {
        None
    };

    // Parse distance override if provided
    let distance_override = if let Some(ref dist_str) = req.distance {
        match dist_str.to_lowercase().as_str() {
            "cosine" => Some(DistanceFunction::Cosine),
            "euclidean" => Some(DistanceFunction::Euclidean),
            "dot" | "dotproduct" => Some(DistanceFunction::DotProduct),
            "manhattan" => Some(DistanceFunction::Manhattan),
            _ => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ApiError::new(
                        format!(
                        "Invalid distance function: '{}'. Use: cosine, euclidean, dot, manhattan",
                        dist_str
                    ),
                        "INVALID_DISTANCE",
                    )),
                ))
            }
        }
    } else {
        None
    };

    // Perform search - use explain variants when profiling is requested
    // Note: explain mode doesn't support post-filter or distance override (uses direct methods)
    let (raw_results, profiling_data) = if req.explain {
        // Explain mode: use direct methods (post-filter and distance override not supported in explain)
        if distance_override.is_some() {
            tracing::warn!("Distance override ignored in explain mode");
        }
        if let Some(ref filter) = pre_filter {
            let (results, explain) = coll
                .search_with_filter_explain(&req.vector, req.k, filter)
                .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
            (results, Some(explain))
        } else {
            let (results, explain) = coll
                .search_explain(&req.vector, req.k)
                .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
            (results, Some(explain))
        }
    } else if distance_override.is_some() || pre_filter.is_some() || post_filter.is_some() {
        // Use search_with_options for distance override, pre-filter, or post-filter
        let results = coll
            .search_with_options(
                &req.vector,
                req.k,
                distance_override,
                pre_filter.as_ref(),
                post_filter.as_ref(),
                req.post_filter_factor,
            )
            .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
        (results, None)
    } else {
        // Standard search without filters or distance override
        let results = coll
            .search(&req.vector, req.k)
            .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
        (results, None)
    };

    // Convert to response format with optional vectors
    let results: Vec<SearchResultResponse> = raw_results
        .into_iter()
        .map(|r| {
            let vector = if req.include_vectors {
                coll.get(&r.id).map(|(v, _)| v)
            } else {
                None
            };

            SearchResultResponse {
                id: r.id,
                distance: r.distance,
                score: 1.0 / (1.0 + r.distance), // Convert distance to similarity score
                metadata: r.metadata,
                vector,
            }
        })
        .collect();

    // Generate explanation if requested
    let explanation = if req.explain {
        let query_norm: f32 = req.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Find dimensions that contribute most to similarity
        let mut contributions: Vec<(usize, f32, f32)> = req
            .vector
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v, v.abs()))
            .collect();
        contributions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let top_dims: Vec<DimensionContribution> = contributions
            .into_iter()
            .take(10)
            .map(|(dim, val, contrib)| DimensionContribution {
                dimension: dim,
                query_value: val,
                contribution: contrib / query_norm,
            })
            .collect();

        // Extract distance metric before moving profiling_data
        let distance_metric = profiling_data
            .as_ref()
            .map(|p| p.distance_function.clone())
            .unwrap_or_else(|| "cosine".to_string());

        // Convert profiling data if available
        let profiling = profiling_data.map(|p| ProfilingData {
            total_time_us: p.total_time_us,
            index_time_us: p.index_time_us,
            filter_time_us: p.filter_time_us,
            enrich_time_us: p.enrich_time_us,
            candidates_before_filter: p.candidates_before_filter,
            candidates_after_filter: p.candidates_after_filter,
            hnsw_stats: HnswStatsResponse {
                visited_nodes: p.hnsw_stats.visited_nodes,
                layers_traversed: p.hnsw_stats.layers_traversed,
                distance_computations: p.hnsw_stats.distance_computations,
                traversal_time_us: p.hnsw_stats.traversal_time_us,
            },
            dimensions: p.dimensions,
            collection_size: p.collection_size,
            requested_k: p.requested_k,
            effective_k: p.effective_k,
            ef_search: p.ef_search,
            filter_applied: p.filter_applied,
        });

        Some(SearchExplanation {
            query_norm,
            distance_metric,
            top_dimensions: top_dims,
            profiling,
        })
    } else {
        None
    };

    Ok(Json(SearchResponse {
        results,
        explanation,
    }))
}

/// Batch search
async fn batch_search(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<BatchSearchRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    // Validate batch size to prevent memory exhaustion
    if req.vectors.len() > state.max_batch_size {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ApiError::new(
                format!(
                    "Batch size {} exceeds maximum allowed {}",
                    req.vectors.len(),
                    state.max_batch_size
                ),
                "BATCH_TOO_LARGE".to_string(),
            )),
        ));
    }

    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let all_results: Vec<Vec<SearchResultResponse>> = if let Some(filter_value) = &req.filter {
        let filter = Filter::parse(filter_value).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(
                    format!("Invalid filter: {}", e),
                    "INVALID_FILTER".to_string(),
                )),
            )
        })?;

        let mut results = Vec::new();
        for query in &req.vectors {
            let r = coll
                .search_with_filter(query, req.k, &filter)
                .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
            results.push(r);
        }
        results
    } else {
        req.vectors
            .iter()
            .map(|q| coll.search(q, req.k))
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?
    }
    .into_iter()
    .map(|results| {
        results
            .into_iter()
            .map(|r| SearchResultResponse {
                id: r.id,
                distance: r.distance,
                score: 1.0 / (1.0 + r.distance),
                metadata: r.metadata,
                vector: None,
            })
            .collect()
    })
    .collect();

    Ok(Json(json!({"results": all_results})))
}

/// Radius-based search (range search)
/// Returns all vectors within max_distance from the query vector
async fn radius_search(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<RadiusSearchRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    // Perform radius search with optional filter
    let raw_results = if let Some(filter_value) = &req.filter {
        let filter = Filter::parse(filter_value).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(
                    format!("Invalid filter: {}", e),
                    "INVALID_FILTER",
                )),
            )
        })?;
        coll.search_radius_with_filter(&req.vector, req.max_distance, req.limit, &filter)
    } else {
        coll.search_radius(&req.vector, req.max_distance, req.limit)
    }
    .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    // Convert to response format with optional vectors
    let results: Vec<SearchResultResponse> = raw_results
        .into_iter()
        .map(|r| {
            let vector = if req.include_vectors {
                coll.get(&r.id).map(|(v, _)| v)
            } else {
                None
            };

            SearchResultResponse {
                id: r.id,
                distance: r.distance,
                score: 1.0 / (1.0 + r.distance),
                metadata: r.metadata,
                vector,
            }
        })
        .collect();

    Ok(Json(json!({
        "results": results,
        "max_distance": req.max_distance,
        "count": results.len(),
    })))
}

/// Compact a collection
async fn compact_collection(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let removed = coll
        .compact()
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({
        "compacted": collection,
        "removed": removed,
    })))
}

/// List vector IDs in a collection
async fn list_vectors(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Query(params): Query<QueryParams>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let ids = coll
        .ids()
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let offset = params.offset.unwrap_or(0);
    let limit = params.limit.unwrap_or(100).min(1000);

    let page: Vec<String> = ids.into_iter().skip(offset).take(limit).collect();

    Ok(Json(json!({
        "ids": page,
        "offset": offset,
        "limit": limit,
        "total": coll.len(),
    })))
}

/// Export collection
async fn export_collection(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let vectors = coll
        .export_all()
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let export: Vec<Value> = vectors
        .into_iter()
        .map(|(id, vec, meta)| {
            json!({
                "id": id,
                "vector": vec,
                "metadata": meta,
            })
        })
        .collect();

    Ok(Json(json!({
        "collection": collection,
        "dimensions": coll.dimensions(),
        "count": export.len(),
        "vectors": export,
    })))
}

/// Save database to disk
async fn save_database(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let mut db = state.db.write().await;
    db.save()
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
    Ok(Json(json!({"saved": true})))
}

// ============ Alias Handlers ============

/// Create a new alias for a collection
async fn create_alias_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateAliasRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    db.create_alias(&req.alias, &req.collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "created": true,
            "alias": req.alias,
            "collection": req.collection
        })),
    ))
}

/// List all aliases
async fn list_aliases_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = state.db.read().await;
    let aliases: Vec<AliasInfo> = db
        .list_aliases()
        .into_iter()
        .map(|(alias, collection)| AliasInfo { alias, collection })
        .collect();

    Json(json!({"aliases": aliases}))
}

/// Get (resolve) an alias to its canonical collection name
async fn get_alias_handler(
    State(state): State<Arc<AppState>>,
    Path(alias): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;

    match db.get_canonical_name(&alias) {
        Some(collection) => Ok(Json(AliasInfo { alias, collection })),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(ApiError::new(
                format!("Alias '{}' not found", alias),
                "ALIAS_NOT_FOUND".to_string(),
            )),
        )),
    }
}

/// Delete an alias
async fn delete_alias_handler(
    State(state): State<Arc<AppState>>,
    Path(alias): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let deleted = db
        .delete_alias(&alias)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    if deleted {
        Ok(Json(json!({"deleted": alias})))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ApiError::new(
                format!("Alias '{}' not found", alias),
                "ALIAS_NOT_FOUND".to_string(),
            )),
        ))
    }
}

/// Update an alias to point to a different collection
async fn update_alias_handler(
    State(state): State<Arc<AppState>>,
    Path(alias): Path<String>,
    Json(req): Json<UpdateAliasRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    db.update_alias(&alias, &req.collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({
        "updated": true,
        "alias": alias,
        "collection": req.collection
    })))
}

// ============ TTL Handlers ============

/// Sweep and delete expired vectors from a collection
async fn expire_vectors_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let expired = coll
        .expire_vectors()
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({
        "collection": collection,
        "expired_count": expired
    })))
}

/// Get TTL statistics for a collection
async fn ttl_stats_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let (total_with_ttl, expired_count, earliest_expiration, latest_expiration) = coll.ttl_stats();

    Ok(Json(json!({
        "collection": collection,
        "vectors_with_ttl": total_with_ttl,
        "expired_count": expired_count,
        "earliest_expiration": earliest_expiration,
        "latest_expiration": latest_expiration,
        "needs_sweep": coll.needs_expiration_sweep(0.1)
    })))
}

/// Build CORS layer from configuration
fn build_cors_layer(config: &CorsConfig) -> CorsLayer {
    if !config.enabled {
        return CorsLayer::new();
    }

    let mut cors = CorsLayer::new()
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::DELETE,
            Method::PUT,
            Method::OPTIONS,
        ])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION, header::ACCEPT])
        .max_age(std::time::Duration::from_secs(config.max_age_secs));

    // Set allowed origins
    cors = match &config.allowed_origins {
        None => cors.allow_origin(AllowOrigin::any()),
        Some(origins) if origins.is_empty() => cors,
        Some(origins) => {
            let origins: Vec<_> = origins.iter().filter_map(|o| o.parse().ok()).collect();
            cors.allow_origin(origins)
        }
    };

    if config.allow_credentials {
        cors = cors.allow_credentials(true);
    }

    cors
}

/// Build the router with configuration
#[allow(deprecated)]
pub fn create_router_with_config(state: Arc<AppState>, config: &ServerConfig) -> Router {
    let cors_layer = build_cors_layer(&config.cors_config);
    let timeout_layer = TimeoutLayer::new(Duration::from_secs(config.request_timeout_secs));

    #[allow(unused_mut)]
    let mut router = Router::new()
        // Health & Info
        .route("/health", get(health))
        .route("/", get(get_info))
        .route("/info", get(get_info))
        // Collections
        .route("/collections", get(list_collections))
        .route("/collections", post(create_collection))
        .route("/collections/:name", get(get_collection))
        .route("/collections/:name", delete(delete_collection))
        .route("/collections/:name/compact", post(compact_collection))
        .route("/collections/:name/export", get(export_collection))
        // Vectors
        .route("/collections/:collection/vectors", get(list_vectors))
        .route("/collections/:collection/vectors", post(insert_vector))
        .route("/collections/:collection/vectors/batch", post(batch_insert))
        .route("/collections/:collection/vectors/upsert", post(upsert_vector))
        .route("/collections/:collection/vectors/:id", get(get_vector))
        .route("/collections/:collection/vectors/:id", delete(delete_vector))
        .route("/collections/:collection/vectors/:id/metadata", post(update_metadata))
        // Search
        .route("/collections/:collection/search", post(search))
        .route("/collections/:collection/search/batch", post(batch_search))
        .route("/collections/:collection/search/radius", post(radius_search))
        // Database operations
        .route("/save", post(save_database))
        // Aliases
        .route("/aliases", post(create_alias_handler))
        .route("/aliases", get(list_aliases_handler))
        .route("/aliases/:alias", get(get_alias_handler))
        .route("/aliases/:alias", delete(delete_alias_handler))
        .route("/aliases/:alias", put(update_alias_handler))
        // TTL endpoints
        .route("/collections/:name/expire", post(expire_vectors_handler))
        .route("/collections/:name/ttl-stats", get(ttl_stats_handler))
        // OpenAPI spec
        .route("/openapi.json", get(serve_openapi_spec));

    // Add metrics endpoint when metrics feature is enabled
    #[cfg(feature = "metrics")]
    {
        router = router.route("/metrics", get(get_metrics));
    }

    // Apply authentication and rate limiting (auth runs first, then rate limiting)
    let router = router
        .layer(middleware::from_fn_with_state(
            state.clone(),
            rate_limit_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    // Build final router with all layers
    // Note: metrics middleware is applied here (outermost) so it captures all requests
    #[cfg(feature = "metrics")]
    let router = router.layer(middleware::from_fn(metrics_middleware));

    router
        .layer(TraceLayer::new_for_http())
        .layer(timeout_layer)
        .layer(RequestBodyLimitLayer::new(config.max_body_size))
        .layer(cors_layer)
}

/// Build the router (legacy, uses permissive CORS for backwards compatibility)
pub fn create_router(state: Arc<AppState>) -> Router {
    let config = ServerConfig {
        cors_config: CorsConfig::permissive(),
        ..Default::default()
    };
    create_router_with_config(state, &config)
}

/// Create a per-IP rate limiter from configuration
fn create_rate_limiter(config: &RateLimitConfig) -> Option<Arc<PerIpRateLimiter>> {
    if !config.enabled || config.requests_per_second == 0 {
        return None;
    }

    // Create quota: requests_per_second with burst_size burst capacity
    let quota = Quota::per_second(
        NonZeroU32::new(config.requests_per_second)
            .unwrap_or(NonZeroU32::new(100).expect("100 is non-zero")),
    )
    .allow_burst(
        NonZeroU32::new(config.burst_size).unwrap_or(NonZeroU32::new(1).expect("1 is non-zero")),
    );

    Some(Arc::new(RateLimiter::dashmap(quota)))
}

/// Extract client IP from request. Forwarded headers are trusted only for known proxies.
fn extract_client_ip(request: &Request<Body>, trusted_proxies: &[IpAddr]) -> IpAddr {
    let remote_ip = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|info| info.0.ip());
    let trust_headers = remote_ip
        .map(|ip| trusted_proxies.contains(&ip))
        .unwrap_or(false);

    if trust_headers {
        // Check X-Forwarded-For header first (for proxied requests)
        if let Some(forwarded_for) = request.headers().get("x-forwarded-for") {
            if let Ok(value) = forwarded_for.to_str() {
                // X-Forwarded-For can contain multiple IPs, take the first one
                if let Some(first_ip) = value.split(',').next() {
                    if let Ok(ip) = first_ip.trim().parse::<IpAddr>() {
                        return ip;
                    }
                }
            }
        }

        // Check X-Real-IP header
        if let Some(real_ip) = request.headers().get("x-real-ip") {
            if let Ok(value) = real_ip.to_str() {
                if let Ok(ip) = value.trim().parse::<IpAddr>() {
                    return ip;
                }
            }
        }
    }

    remote_ip.unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST))
}

/// Rate limiting middleware - returns 429 if rate limit exceeded (per-IP)
async fn rate_limit_middleware(
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if let Some(limiter) = &state.rate_limiter {
        let client_ip = extract_client_ip(&request, &state.trusted_proxies);
        match limiter.check_key(&client_ip) {
            Ok(_) => next.run(request).await,
            Err(_) => {
                warn!(
                    client_ip = %client_ip,
                    "Rate limit exceeded"
                );
                let error = ApiError::new(
                    "Rate limit exceeded. Please slow down.".to_string(),
                    "RATE_LIMIT_EXCEEDED".to_string(),
                );
                (StatusCode::TOO_MANY_REQUESTS, Json(error)).into_response()
            }
        }
    } else {
        next.run(request).await
    }
}

/// Authentication middleware - validates API keys and JWT tokens.
///
/// Checks for authentication credentials in this order:
/// 1. `X-API-Key` header for API key authentication
/// 2. `Authorization: Bearer <token>` header for JWT authentication
///
/// If authentication is required and no valid credentials are provided,
/// returns 401 Unauthorized. If credentials are invalid, returns 401.
/// Public endpoints (as configured in AuthConfig) bypass authentication.
async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    let path = request.uri().path().to_string();
    let auth_config = &state.auth;

    // Check if this is a public endpoint
    if auth_config.is_public_endpoint(&path) {
        // Still try to extract auth context if credentials are present
        if let Some(context) = try_authenticate(&request, auth_config) {
            request.extensions_mut().insert(context);
        }
        return next.run(request).await;
    }

    // If auth is not required, proceed without validation
    if !auth_config.require_auth {
        // Still try to extract auth context if credentials are present
        if let Some(context) = try_authenticate(&request, auth_config) {
            request.extensions_mut().insert(context);
        }
        return next.run(request).await;
    }

    // Auth is required - try to authenticate
    match try_authenticate(&request, auth_config) {
        Some(context) => {
            request.extensions_mut().insert(context);
            next.run(request).await
        }
        None => {
            warn!(path = %path, "Authentication required but no valid credentials provided");
            let error = ApiError::new(
                "Authentication required".to_string(),
                "UNAUTHORIZED".to_string(),
            );
            (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "Bearer, ApiKey")],
                Json(error),
            )
                .into_response()
        }
    }
}

/// Try to authenticate a request using available methods.
fn try_authenticate(request: &Request<Body>, auth_config: &AuthConfig) -> Option<AuthContext> {
    // Try API key first (X-API-Key header)
    if let Some(api_key) = request.headers().get("x-api-key") {
        if let Ok(key_str) = api_key.to_str() {
            if let Some(user) = auth_config.validate_api_key(key_str) {
                return Some(AuthContext {
                    user,
                    method: AuthMethod::ApiKey,
                });
            }
        }
    }

    // Try Bearer token (Authorization header)
    if let Some(auth_header) = request.headers().get(header::AUTHORIZATION) {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                if let Ok(claims) = auth_config.validate_jwt(token.trim()) {
                    return Some(AuthContext {
                        user: claims.to_user(),
                        method: AuthMethod::Jwt,
                    });
                }
            }
        }
    }

    None
}

/// Metrics middleware - records HTTP request metrics (when metrics feature is enabled)
#[cfg(feature = "metrics")]
async fn metrics_middleware(request: Request<Body>, next: Next) -> Response {
    let method = request.method().to_string();
    let path = request.uri().path().to_string();

    let mut timer = http_metrics().start_request(&method, &path);
    let response = next.run(request).await;
    timer.set_status(response.status().as_u16());

    response
}

/// Handler for /metrics endpoint - exports Prometheus metrics
#[cfg(feature = "metrics")]
async fn get_metrics() -> impl IntoResponse {
    let output = metrics().export();
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        output,
    )
}

/// Start the HTTP server
pub async fn serve(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber with environment filter
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    info!(
        addr = %config.addr,
        timeout_secs = config.request_timeout_secs,
        rate_limit_enabled = config.rate_limit.enabled,
        "Starting Needle server"
    );

    let db = if let Some(path) = &config.db_path {
        info!(path = %path, "Opening database file");
        Database::open(path)?
    } else {
        info!("Using in-memory database");
        Database::in_memory()
    };

    let state = Arc::new(AppState::with_config(db, &config));
    let app = create_router_with_config(state.clone(), &config);

    info!("Listening on http://{}", config.addr);

    let listener = tokio::net::TcpListener::bind(&config.addr).await?;

    // Graceful shutdown handling
    let shutdown_signal = async {
        let ctrl_c = async {
            tokio::signal::ctrl_c()
                .await
                .expect("Failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("Failed to install SIGTERM handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }

        info!("Shutdown signal received, starting graceful shutdown");
    };

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal)
    .await?;

    // Save database on shutdown
    info!("Saving database before shutdown");
    {
        let mut db = state.db.write().await;
        if let Err(e) = db.save() {
            error!(error = %e, "Failed to save database during shutdown");
        } else {
            info!("Database saved successfully");
        }
    }

    info!("Server shutdown complete");
    Ok(())
}

/// Start server with default config
pub async fn serve_default() -> Result<(), Box<dyn std::error::Error>> {
    serve(ServerConfig::default()).await
}

// ---------------------------------------------------------------------------
// OpenAPI Specification Generator
// ---------------------------------------------------------------------------

/// Generates an OpenAPI 3.1 specification for the Needle REST API.
pub fn generate_openapi_spec() -> serde_json::Value {
    serde_json::json!({
        "openapi": "3.1.0",
        "info": {
            "title": "Needle Vector Database API",
            "description": "REST API for the Needle embedded vector database",
            "version": "0.1.0",
            "license": { "name": "MIT" },
            "contact": { "name": "Needle Team" }
        },
        "servers": [
            { "url": "http://localhost:8080", "description": "Local development" }
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "operationId": "healthCheck",
                    "tags": ["System"],
                    "responses": {
                        "200": {
                            "description": "Server is healthy",
                            "content": { "application/json": { "schema": { "$ref": "#/components/schemas/HealthResponse" } } }
                        }
                    }
                }
            },
            "/collections": {
                "get": {
                    "summary": "List all collections",
                    "operationId": "listCollections",
                    "tags": ["Collections"],
                    "responses": {
                        "200": {
                            "description": "List of collections",
                            "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/CollectionInfo" } } } }
                        }
                    }
                },
                "post": {
                    "summary": "Create a new collection",
                    "operationId": "createCollection",
                    "tags": ["Collections"],
                    "requestBody": {
                        "required": true,
                        "content": { "application/json": { "schema": { "$ref": "#/components/schemas/CreateCollectionRequest" } } }
                    },
                    "responses": {
                        "201": { "description": "Collection created" },
                        "409": { "description": "Collection already exists" }
                    }
                }
            },
            "/collections/{name}": {
                "get": {
                    "summary": "Get collection info",
                    "operationId": "getCollection",
                    "tags": ["Collections"],
                    "parameters": [{ "name": "name", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "Collection info", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/CollectionInfo" } } } },
                        "404": { "description": "Collection not found" }
                    }
                },
                "delete": {
                    "summary": "Delete a collection",
                    "operationId": "deleteCollection",
                    "tags": ["Collections"],
                    "parameters": [{ "name": "name", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "Collection deleted" },
                        "404": { "description": "Collection not found" }
                    }
                }
            },
            "/collections/{collection}/vectors": {
                "get": {
                    "summary": "List vectors in collection",
                    "operationId": "listVectors",
                    "tags": ["Vectors"],
                    "parameters": [
                        { "name": "collection", "in": "path", "required": true, "schema": { "type": "string" } },
                        { "name": "limit", "in": "query", "schema": { "type": "integer", "default": 100 } },
                        { "name": "offset", "in": "query", "schema": { "type": "integer", "default": 0 } }
                    ],
                    "responses": { "200": { "description": "List of vectors" } }
                },
                "post": {
                    "summary": "Insert a vector",
                    "operationId": "insertVector",
                    "tags": ["Vectors"],
                    "parameters": [{ "name": "collection", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "requestBody": {
                        "required": true,
                        "content": { "application/json": { "schema": { "$ref": "#/components/schemas/InsertVectorRequest" } } }
                    },
                    "responses": {
                        "201": { "description": "Vector inserted" },
                        "400": { "description": "Invalid input" }
                    }
                }
            },
            "/collections/{collection}/vectors/{id}": {
                "get": {
                    "summary": "Get a vector by ID",
                    "operationId": "getVector",
                    "tags": ["Vectors"],
                    "parameters": [
                        { "name": "collection", "in": "path", "required": true, "schema": { "type": "string" } },
                        { "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }
                    ],
                    "responses": {
                        "200": { "description": "Vector data", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/VectorResponse" } } } },
                        "404": { "description": "Vector not found" }
                    }
                },
                "delete": {
                    "summary": "Delete a vector",
                    "operationId": "deleteVector",
                    "tags": ["Vectors"],
                    "parameters": [
                        { "name": "collection", "in": "path", "required": true, "schema": { "type": "string" } },
                        { "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }
                    ],
                    "responses": { "200": { "description": "Vector deleted" }, "404": { "description": "Not found" } }
                }
            },
            "/collections/{collection}/search": {
                "post": {
                    "summary": "Search for similar vectors",
                    "operationId": "searchVectors",
                    "tags": ["Search"],
                    "parameters": [{ "name": "collection", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "requestBody": {
                        "required": true,
                        "content": { "application/json": { "schema": { "$ref": "#/components/schemas/SearchRequest" } } }
                    },
                    "responses": {
                        "200": {
                            "description": "Search results",
                            "content": { "application/json": { "schema": { "$ref": "#/components/schemas/SearchResponse" } } }
                        }
                    }
                }
            },
            "/save": {
                "post": {
                    "summary": "Save database to disk",
                    "operationId": "saveDatabase",
                    "tags": ["System"],
                    "responses": { "200": { "description": "Database saved" } }
                }
            }
        },
        "components": {
            "schemas": {
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": { "type": "string", "example": "healthy" }
                    }
                },
                "CollectionInfo": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "dimensions": { "type": "integer" },
                        "vector_count": { "type": "integer" },
                        "distance_function": { "type": "string", "enum": ["cosine", "euclidean", "dot_product", "manhattan"] }
                    }
                },
                "CreateCollectionRequest": {
                    "type": "object",
                    "required": ["name", "dimensions"],
                    "properties": {
                        "name": { "type": "string" },
                        "dimensions": { "type": "integer", "minimum": 1 },
                        "distance": { "type": "string", "enum": ["cosine", "euclidean", "dot_product", "manhattan"], "default": "cosine" },
                        "m": { "type": "integer", "default": 16 },
                        "ef_construction": { "type": "integer", "default": 200 }
                    }
                },
                "InsertVectorRequest": {
                    "type": "object",
                    "required": ["id", "vector"],
                    "properties": {
                        "id": { "type": "string" },
                        "vector": { "type": "array", "items": { "type": "number", "format": "float" } },
                        "metadata": { "type": "object", "additionalProperties": true }
                    }
                },
                "VectorResponse": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "vector": { "type": "array", "items": { "type": "number" } },
                        "metadata": { "type": "object", "nullable": true }
                    }
                },
                "SearchRequest": {
                    "type": "object",
                    "required": ["vector"],
                    "properties": {
                        "vector": { "type": "array", "items": { "type": "number", "format": "float" } },
                        "k": { "type": "integer", "default": 10, "minimum": 1 },
                        "filter": { "type": "object", "description": "MongoDB-style metadata filter" },
                        "ef_search": { "type": "integer" }
                    }
                },
                "SearchResponse": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": { "type": "string" },
                                    "distance": { "type": "number" },
                                    "metadata": { "type": "object", "nullable": true }
                                }
                            }
                        }
                    }
                }
            },
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                },
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                }
            }
        },
        "security": [
            { "ApiKeyAuth": [] },
            { "BearerAuth": [] }
        ]
    })
}

/// Return the OpenAPI spec as a formatted JSON string.
pub fn openapi_spec_json() -> String {
    serde_json::to_string_pretty(&generate_openapi_spec()).unwrap_or_default()
}

/// Axum handler that serves the OpenAPI spec as JSON.
async fn serve_openapi_spec() -> impl IntoResponse {
    Json(generate_openapi_spec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    // RateLimitConfig tests
    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert!(config.enabled);
        assert_eq!(config.requests_per_second, 100);
        assert_eq!(config.burst_size, 50);
    }

    #[test]
    fn test_rate_limit_config_disabled() {
        let config = RateLimitConfig::disabled();
        assert!(!config.enabled);
        assert_eq!(config.requests_per_second, 0);
        assert_eq!(config.burst_size, 0);
    }

    #[test]
    fn test_rate_limit_config_with_rate() {
        let config = RateLimitConfig::default().with_rate(200);
        assert_eq!(config.requests_per_second, 200);
        assert_eq!(config.burst_size, 50); // unchanged
    }

    #[test]
    fn test_rate_limit_config_with_burst() {
        let config = RateLimitConfig::default().with_burst(100);
        assert_eq!(config.requests_per_second, 100); // unchanged
        assert_eq!(config.burst_size, 100);
    }

    #[test]
    fn test_rate_limit_config_chained() {
        let config = RateLimitConfig::default().with_rate(500).with_burst(250);
        assert!(config.enabled);
        assert_eq!(config.requests_per_second, 500);
        assert_eq!(config.burst_size, 250);
    }

    // CorsConfig tests
    #[test]
    fn test_cors_config_default() {
        let config = CorsConfig::default();
        assert!(config.enabled);
        assert!(!config.allow_credentials);
        assert_eq!(config.max_age_secs, 3600);
        let origins = config.allowed_origins.unwrap();
        assert!(origins.contains(&"http://localhost:3000".to_string()));
        assert!(origins.contains(&"http://localhost:8080".to_string()));
    }

    #[test]
    fn test_cors_config_permissive() {
        let config = CorsConfig::permissive();
        assert!(config.enabled);
        assert!(!config.allow_credentials); // SECURITY: credentials disabled even in permissive mode
        assert!(config.allowed_origins.is_none()); // Allow all
        assert_eq!(config.max_age_secs, 3600);
    }

    #[test]
    fn test_cors_config_with_credentials() {
        // Credentials should only be used with specific origins
        let config = CorsConfig::default().with_credentials(true);
        assert!(config.allow_credentials);
    }

    #[test]
    fn test_cors_config_restrictive() {
        let config = CorsConfig::restrictive();
        assert!(config.enabled);
        assert!(!config.allow_credentials);
        assert_eq!(config.max_age_secs, 0);
        let origins = config.allowed_origins.unwrap();
        assert!(origins.is_empty()); // No external origins
    }

    #[test]
    fn test_cors_config_with_origin() {
        let config = CorsConfig::restrictive().with_origin("https://example.com");
        let origins = config.allowed_origins.unwrap();
        assert_eq!(origins.len(), 1);
        assert!(origins.contains(&"https://example.com".to_string()));
    }

    #[test]
    fn test_cors_config_with_multiple_origins() {
        let config = CorsConfig::restrictive()
            .with_origin("https://example.com")
            .with_origin("https://api.example.com");
        let origins = config.allowed_origins.unwrap();
        assert_eq!(origins.len(), 2);
    }

    // Error response formatting tests
    #[test]
    fn test_error_collection_not_found() {
        let err = NeedleError::CollectionNotFound("test".to_string());
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(json.code, "COLLECTION_NOT_FOUND");
    }

    #[test]
    fn test_error_vector_not_found() {
        let err = NeedleError::VectorNotFound("vec1".to_string());
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(json.code, "VECTOR_NOT_FOUND");
    }

    #[test]
    fn test_error_collection_already_exists() {
        let err = NeedleError::CollectionAlreadyExists("test".to_string());
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(json.code, "COLLECTION_EXISTS");
    }

    #[test]
    fn test_error_vector_already_exists() {
        let err = NeedleError::VectorAlreadyExists("vec1".to_string());
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(json.code, "VECTOR_EXISTS");
    }

    #[test]
    fn test_error_dimension_mismatch() {
        let err = NeedleError::DimensionMismatch {
            expected: 384,
            got: 128,
        };
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json.code, "DIMENSION_MISMATCH");
        assert!(json.error.contains("384"));
        assert!(json.error.contains("128"));
    }

    #[test]
    fn test_error_invalid_vector() {
        let err = NeedleError::InvalidVector("contains NaN".to_string());
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json.code, "INVALID_VECTOR");
    }

    #[test]
    fn test_error_invalid_config() {
        let err = NeedleError::InvalidConfig("bad M value".to_string());
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json.code, "INVALID_CONFIG");
    }

    #[test]
    fn test_error_internal_fallback() {
        let err = NeedleError::Corruption("data corrupted".to_string());
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(json.code, "INTERNAL_ERROR");
    }

    // ServerConfig tests
    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.max_body_size, 100 * 1024 * 1024); // 100MB
        assert!(config.db_path.is_none());
        assert!(config.cors_config.enabled);
        assert!(config.rate_limit.enabled);
    }

    // Authentication tests
    #[test]
    fn test_api_key_creation() {
        let key = ApiKey::new("test-key-123");
        assert_eq!(key.key, "test-key-123");
        assert_eq!(key.roles, vec!["reader".to_string()]);
        assert!(key.active);
        assert!(key.expires_at.is_none());
    }

    #[test]
    fn test_api_key_with_role() {
        let key = ApiKey::new("admin-key").with_role("admin");
        assert_eq!(key.roles, vec!["admin".to_string()]);
    }

    #[test]
    fn test_api_key_with_name() {
        let key = ApiKey::new("test-key").with_name("Production API Key");
        assert_eq!(key.name, Some("Production API Key".to_string()));
    }

    #[test]
    fn test_api_key_expiration() {
        // Key expires in the past
        let expired_key = ApiKey::new("expired").expires_at(0); // Unix epoch = expired
        assert!(expired_key.is_expired());
        assert!(!expired_key.is_valid());

        // Key expires in the future
        let future_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3600; // 1 hour from now
        let valid_key = ApiKey::new("valid").expires_at(future_ts);
        assert!(!valid_key.is_expired());
        assert!(valid_key.is_valid());
    }

    #[test]
    fn test_api_key_to_user() {
        let key = ApiKey::new("test-key")
            .with_name("Test Key")
            .with_role("writer");
        let user = key.to_user();
        assert!(user.id.starts_with("apikey:"));
        assert!(user.name.is_some());
        assert!(!user.roles.is_empty());
    }

    #[test]
    fn test_auth_config_creation() {
        let config = AuthConfig::new();
        assert!(!config.require_auth);
        assert!(config.api_keys.is_empty());
        assert!(config.jwt_secret.is_none());
        // Health and root are public by default
        assert!(config.is_public_endpoint("/health"));
        assert!(config.is_public_endpoint("/"));
    }

    #[test]
    fn test_auth_config_with_api_key() {
        let config = AuthConfig::new().with_api_key(ApiKey::new("test-key"));
        assert_eq!(config.api_keys.len(), 1);
    }

    #[test]
    fn test_auth_config_validate_api_key() {
        let config = AuthConfig::new().with_api_key(ApiKey::new("valid-key").with_role("admin"));

        // Valid key
        let user = config.validate_api_key("valid-key");
        assert!(user.is_some());

        // Invalid key
        let invalid = config.validate_api_key("wrong-key");
        assert!(invalid.is_none());
    }

    #[test]
    fn test_auth_config_require_auth() {
        let config = AuthConfig::new().require_auth(true);
        assert!(config.require_auth);
    }

    #[test]
    fn test_auth_config_public_endpoints() {
        let config = AuthConfig::new().with_public_endpoint("/metrics");
        assert!(config.is_public_endpoint("/health"));
        assert!(config.is_public_endpoint("/metrics"));
        assert!(!config.is_public_endpoint("/collections"));
    }

    #[test]
    fn test_jwt_claims_creation() {
        let claims = JwtClaims::new("user123", 3600);
        assert_eq!(claims.sub, "user123");
        assert!(!claims.is_expired());
        assert_eq!(claims.roles, vec!["reader".to_string()]);
    }

    #[test]
    fn test_jwt_claims_with_roles() {
        let claims = JwtClaims::new("admin", 3600)
            .with_roles(vec!["admin".to_string(), "writer".to_string()]);
        assert_eq!(claims.roles.len(), 2);
        assert!(claims.roles.contains(&"admin".to_string()));
    }

    #[test]
    fn test_jwt_claims_to_user() {
        let claims =
            JwtClaims::new("user@example.com", 3600).with_roles(vec!["writer".to_string()]);
        let user = claims.to_user();
        assert_eq!(user.id, "user@example.com");
        assert!(!user.roles.is_empty());
    }

    #[test]
    fn test_jwt_generate_and_validate() {
        let config = AuthConfig::new().with_jwt_secret("super-secret-key-for-testing");

        let claims = JwtClaims::new("testuser", 3600).with_roles(vec!["admin".to_string()]);

        // Generate token
        let token = config.generate_jwt(&claims).expect("should generate token");
        assert!(!token.is_empty());
        assert_eq!(token.split('.').count(), 3); // header.payload.signature

        // Validate token
        let validated = config.validate_jwt(&token).expect("should validate token");
        assert_eq!(validated.sub, "testuser");
        assert!(validated.roles.contains(&"admin".to_string()));
    }

    #[test]
    fn test_jwt_invalid_signature() {
        let config = AuthConfig::new().with_jwt_secret("correct-secret");

        let claims = JwtClaims::new("user", 3600);
        let token = config.generate_jwt(&claims).unwrap();

        // Try validating with a different secret
        let wrong_config = AuthConfig::new().with_jwt_secret("wrong-secret");

        let result = wrong_config.validate_jwt(&token);
        assert!(result.is_err());
    }

    #[test]
    fn test_jwt_expired_token() {
        let config = AuthConfig::new().with_jwt_secret("secret");

        // Create claims that are already expired (negative expiration)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut claims = JwtClaims::new("user", 0);
        claims.exp = now - 100; // Expired 100 seconds ago

        let token = config.generate_jwt(&claims).unwrap();
        let result = config.validate_jwt(&token);
        assert!(matches!(result, Err(AuthError::TokenExpired)));
    }

    #[test]
    fn test_jwt_no_secret_configured() {
        let config = AuthConfig::new(); // No JWT secret
        let claims = JwtClaims::new("user", 3600);

        let result = config.generate_jwt(&claims);
        assert!(matches!(result, Err(AuthError::NoJwtSecret)));
    }

    #[test]
    fn test_auth_method_equality() {
        assert_eq!(AuthMethod::ApiKey, AuthMethod::ApiKey);
        assert_eq!(AuthMethod::Jwt, AuthMethod::Jwt);
        assert_eq!(AuthMethod::None, AuthMethod::None);
        assert_ne!(AuthMethod::ApiKey, AuthMethod::Jwt);
    }

    #[test]
    fn test_auth_error_display() {
        assert_eq!(
            AuthError::MissingCredentials.to_string(),
            "Authentication required"
        );
        assert_eq!(AuthError::InvalidApiKey.to_string(), "Invalid API key");
        assert_eq!(AuthError::TokenExpired.to_string(), "Token has expired");
        assert_eq!(
            AuthError::InvalidSignature.to_string(),
            "Invalid token signature"
        );
    }

    #[test]
    fn test_server_config_with_auth() {
        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new("test-key"))
            .require_auth(true);

        let server_config = ServerConfig::default().with_auth(auth_config);

        assert!(server_config.auth.require_auth);
        assert_eq!(server_config.auth.api_keys.len(), 1);
    }

    #[test]
    fn test_openapi_spec_generation() {
        let spec = generate_openapi_spec();
        assert_eq!(spec["openapi"], "3.1.0");
        assert_eq!(spec["info"]["title"], "Needle Vector Database API");
        assert!(spec["paths"]["/health"].is_object());
        assert!(spec["paths"]["/collections"].is_object());
        assert!(spec["paths"]["/collections/{collection}/search"].is_object());
        assert!(spec["components"]["schemas"]["SearchRequest"].is_object());
        assert!(spec["components"]["schemas"]["SearchResponse"].is_object());
    }

    #[test]
    fn test_openapi_spec_json_string() {
        let json = openapi_spec_json();
        assert!(json.contains("Needle Vector Database API"));
        assert!(json.contains("searchVectors"));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_object());
    }
}
