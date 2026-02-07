//! Authentication types for the Needle HTTP server.
//!
//! Provides API key and JWT-based authentication with role-based access control.

use crate::security::{Role, User};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Sha256;
use std::collections::HashMap;
use subtle::ConstantTimeEq;

/// Default clock skew leeway for JWT expiration checks (in seconds).
const JWT_CLOCK_LEEWAY_SECS: u64 = 60;

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

    /// Check if the token is expired (with clock skew leeway).
    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.exp + JWT_CLOCK_LEEWAY_SECS < now
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
    ///
    /// The secret must be at least 32 bytes to prevent brute-force attacks on HS256.
    /// Shorter secrets will cause [`validate`](Self::validate) to return an error.
    pub fn with_jwt_secret(mut self, secret: impl Into<String>) -> Self {
        self.jwt_secret = Some(secret.into());
        self
    }

    /// Validate the authentication configuration.
    ///
    /// Returns an error if the JWT secret is configured but shorter than 32 bytes.
    pub fn validate(&self) -> std::result::Result<(), AuthError> {
        if let Some(secret) = &self.jwt_secret {
            if secret.len() < 32 {
                return Err(AuthError::InvalidToken(
                    "JWT secret must be at least 32 bytes to prevent brute-force attacks".into(),
                ));
            }
        }
        Ok(())
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
            path == e
                || (e != "/"
                    && (path.starts_with(&format!("{}/", e))
                        || path.starts_with(&format!("{}?", e))))
        })
    }

    /// Validate an API key and return the associated user.
    pub fn validate_api_key(&self, key: &str) -> Option<User> {
        self.api_keys
            .iter()
            .find(|k| k.key.as_bytes().ct_eq(key.as_bytes()).into() && k.is_valid())
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

        // Decode and validate header algorithm
        let header_bytes = URL_SAFE_NO_PAD
            .decode(header)
            .map_err(|_| AuthError::InvalidToken("Invalid header encoding".into()))?;
        let header_json: Value = serde_json::from_slice(&header_bytes)
            .map_err(|_| AuthError::InvalidToken("Invalid header JSON".into()))?;
        match header_json.get("alg").and_then(|v| v.as_str()) {
            Some("HS256") => {}
            Some(alg) => {
                return Err(AuthError::InvalidToken(format!(
                    "Unsupported algorithm: {}. Only HS256 is supported",
                    alg
                )));
            }
            None => {
                return Err(AuthError::InvalidToken(
                    "Missing algorithm in header".into(),
                ));
            }
        }

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
