//! Authentication types for the Needle HTTP server.
//!
//! Provides API key and JWT-based authentication with role-based access control.

use crate::security::{Role, User};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use subtle::ConstantTimeEq;

/// Default clock skew leeway for JWT expiration checks (in seconds).
const JWT_CLOCK_LEEWAY_SECS: u64 = 60;

#[derive(Clone, Serialize, Deserialize)]
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
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the roles for this API key.
    #[must_use]
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.roles = vec![role.into()];
        self
    }

    /// Add multiple roles to this API key.
    #[must_use]
    pub fn with_roles(mut self, roles: Vec<String>) -> Self {
        self.roles = roles;
        self
    }

    /// Set an expiration time for this key (Unix epoch seconds).
    #[must_use]
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

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKey")
            .field("key", &"[REDACTED]")
            .field("name", &self.name)
            .field("roles", &self.roles)
            .field("active", &self.active)
            .field("expires_at", &self.expires_at)
            .finish()
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
#[derive(Clone, Default)]
pub struct AuthConfig {
    /// Whether authentication is required for all endpoints
    pub require_auth: bool,
    /// API keys for authentication
    pub api_keys: Vec<ApiKey>,
    /// JWT secret for token validation (HS256) — the current/primary signing key
    pub jwt_secret: Option<String>,
    /// Previous JWT secrets still accepted for validation during key rotation.
    /// Tokens signed with these keys will be accepted but new tokens are always
    /// signed with `jwt_secret`.
    pub jwt_previous_secrets: Vec<String>,
    /// Endpoints that don't require authentication (e.g., "/health")
    pub public_endpoints: Vec<String>,
}

impl std::fmt::Debug for AuthConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthConfig")
            .field("require_auth", &self.require_auth)
            .field("api_keys", &format!("[{} keys]", self.api_keys.len()))
            .field("jwt_secret", &self.jwt_secret.as_ref().map(|_| "[REDACTED]"))
            .field("jwt_previous_secrets", &format!("[{} keys]", self.jwt_previous_secrets.len()))
            .field("public_endpoints", &self.public_endpoints)
            .finish()
    }
}

impl AuthConfig {
    /// Create a new authentication configuration.
    pub fn new() -> Self {
        Self {
            require_auth: false,
            api_keys: Vec::new(),
            jwt_secret: None,
            jwt_previous_secrets: Vec::new(),
            public_endpoints: vec!["/health".to_string(), "/".to_string()],
        }
    }

    /// Require authentication for all endpoints except public ones.
    #[must_use]
    pub fn require_auth(mut self, require: bool) -> Self {
        self.require_auth = require;
        self
    }

    /// Add an API key for authentication.
    #[must_use]
    pub fn with_api_key(mut self, key: ApiKey) -> Self {
        self.api_keys.push(key);
        self
    }

    /// Add multiple API keys.
    #[must_use]
    pub fn with_api_keys(mut self, keys: Vec<ApiKey>) -> Self {
        self.api_keys.extend(keys);
        self
    }

    /// Set the JWT secret for token validation.
    ///
    /// The secret must be at least 32 bytes to prevent brute-force attacks on HS256.
    /// Shorter secrets will cause [`validate`](Self::validate) to return an error.
    #[must_use]
    pub fn with_jwt_secret(mut self, secret: impl Into<String>) -> Self {
        self.jwt_secret = Some(secret.into());
        self
    }

    /// Add a previous JWT secret for key rotation.
    ///
    /// During validation, if the current secret fails, previous secrets are tried
    /// in order. New tokens are always signed with the current `jwt_secret`.
    #[must_use]
    pub fn with_previous_jwt_secret(mut self, secret: impl Into<String>) -> Self {
        self.jwt_previous_secrets.push(secret.into());
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
        for secret in &self.jwt_previous_secrets {
            if secret.len() < 32 {
                return Err(AuthError::InvalidToken(
                    "JWT previous secret must be at least 32 bytes".into(),
                ));
            }
        }
        Ok(())
    }

    /// Add a public endpoint that doesn't require authentication.
    ///
    /// Public endpoints are matched using prefix semantics: a path is considered
    /// public if it exactly matches the endpoint, or if the path starts with the
    /// endpoint followed by `/` or `?`. For example, adding `/api` would make
    /// `/api/foo` and `/api?bar` public as well.
    ///
    /// **Warning:** Adding short prefixes like `/api` or `/collections` may
    /// unintentionally expose more endpoints than intended. Prefer specific paths
    /// (e.g., `/health`, `/collections/public-data`) to minimize risk.
    #[must_use]
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
    ///
    /// Keys are hashed before comparison to ensure constant-time equality
    /// regardless of input length differences.
    pub fn validate_api_key(&self, key: &str) -> Option<User> {
        let input_hash = Sha256::digest(key.as_bytes());
        self.api_keys
            .iter()
            .find(|k| {
                let stored_hash = Sha256::digest(k.key.as_bytes());
                stored_hash.ct_eq(&input_hash).into() && k.is_valid()
            })
            .map(|k| k.to_user())
    }

    /// Validate a JWT token and return the claims.
    ///
    /// Tries the current secret first. If validation fails with an invalid signature,
    /// previous secrets are tried in order to support key rotation.
    pub fn validate_jwt(&self, token: &str) -> Result<JwtClaims, AuthError> {
        let secret = self.jwt_secret.as_ref().ok_or(AuthError::NoJwtSecret)?;

        // Try the current secret first
        match self.validate_jwt_with_secret(token, secret) {
            Ok(claims) => return Ok(claims),
            Err(AuthError::InvalidSignature) => {
                // Signature mismatch — try previous secrets for key rotation
            }
            Err(e) => return Err(e), // Non-signature errors (expired, malformed) fail immediately
        }

        // Try previous secrets in order
        for prev_secret in &self.jwt_previous_secrets {
            if let Ok(claims) = self.validate_jwt_with_secret(token, prev_secret) {
                return Ok(claims);
            }
        }

        Err(AuthError::InvalidSignature)
    }

    /// Validate a JWT token against a specific secret.
    fn validate_jwt_with_secret(&self, token: &str, secret: &str) -> Result<JwtClaims, AuthError> {
        let mut validation = Validation::new(Algorithm::HS256);
        validation.leeway = JWT_CLOCK_LEEWAY_SECS;
        validation.validate_exp = true;
        validation.required_spec_claims = std::collections::HashSet::from(["exp".to_string()]);

        let key = DecodingKey::from_secret(secret.as_bytes());
        let token_data = decode::<JwtClaims>(token, &key, &validation).map_err(|e| {
            match e.kind() {
                jsonwebtoken::errors::ErrorKind::ExpiredSignature => AuthError::TokenExpired,
                jsonwebtoken::errors::ErrorKind::InvalidSignature => AuthError::InvalidSignature,
                jsonwebtoken::errors::ErrorKind::InvalidAlgorithm => {
                    AuthError::InvalidToken("Unsupported algorithm. Only HS256 is supported".into())
                }
                _ => AuthError::InvalidToken(format!("JWT validation failed: {}", e)),
            }
        })?;

        Ok(token_data.claims)
    }

    /// Generate a JWT token for the given claims.
    pub fn generate_jwt(&self, claims: &JwtClaims) -> Result<String, AuthError> {
        let secret = self.jwt_secret.as_ref().ok_or(AuthError::NoJwtSecret)?;

        let key = EncodingKey::from_secret(secret.as_bytes());
        encode(&Header::new(Algorithm::HS256), claims, &key)
            .map_err(|e| AuthError::InvalidToken(format!("Failed to generate JWT: {}", e)))
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

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_API_KEY: &str = "test-key";
    const TEST_SECRET: &str = "this-is-a-test-secret-at-least-32-bytes-long!!";

    fn auth_config_with_key(key: &str) -> AuthConfig {
        AuthConfig::new()
            .require_auth(true)
            .with_api_key(ApiKey::new(key))
    }

    fn auth_config_with_jwt() -> AuthConfig {
        AuthConfig::new()
            .require_auth(true)
            .with_jwt_secret(TEST_SECRET)
    }

    // ── ApiKey ───────────────────────────────────────────────────────────

    #[test]
    fn test_api_key_new_defaults() {
        let key = ApiKey::new(TEST_API_KEY);
        assert_eq!(key.key, TEST_API_KEY);
        assert!(key.active);
        assert!(key.expires_at.is_none());
        assert_eq!(key.roles, vec!["reader".to_string()]);
        assert!(key.name.is_none());
    }

    #[test]
    fn test_api_key_builder() {
        let key = ApiKey::new("k")
            .with_name("My Key")
            .with_role("admin");
        assert_eq!(key.name, Some("My Key".to_string()));
        assert_eq!(key.roles, vec!["admin".to_string()]);
    }

    #[test]
    fn test_api_key_with_roles() {
        let key = ApiKey::new("k")
            .with_roles(vec!["admin".into(), "writer".into()]);
        assert_eq!(key.roles.len(), 2);
    }

    #[test]
    fn test_api_key_not_expired() {
        let key = ApiKey::new("k");
        assert!(!key.is_expired());
        assert!(key.is_valid());
    }

    #[test]
    fn test_api_key_expired() {
        let key = ApiKey::new("k").expires_at(0); // expired in 1970
        assert!(key.is_expired());
        assert!(!key.is_valid());
    }

    #[test]
    fn test_api_key_inactive() {
        let mut key = ApiKey::new("k");
        key.active = false;
        assert!(!key.is_valid());
    }

    #[test]
    fn test_api_key_to_user_reader() {
        let key = ApiKey::new("test-key-12345678");
        let user = key.to_user();
        assert!(user.id.starts_with("apikey:"));
    }

    #[test]
    fn test_api_key_to_user_admin() {
        let key = ApiKey::new("k").with_role("admin");
        let user = key.to_user();
        assert!(!user.id.is_empty());
    }

    #[test]
    fn test_api_key_debug_redacts() {
        let key = ApiKey::new("super-secret-key");
        let debug = format!("{:?}", key);
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("super-secret-key"));
    }

    // ── AuthConfig ──────────────────────────────────────────────────────

    #[test]
    fn test_auth_config_defaults() {
        let config = AuthConfig::new();
        assert!(!config.require_auth);
        assert!(config.api_keys.is_empty());
        assert!(config.jwt_secret.is_none());
        assert!(config.public_endpoints.contains(&"/health".to_string()));
    }

    #[test]
    fn test_auth_config_debug_redacts_secret() {
        let config = auth_config_with_jwt();
        let debug = format!("{:?}", config);
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains(TEST_SECRET));
    }

    #[test]
    fn test_public_endpoint_exact_match() {
        let config = AuthConfig::new().with_public_endpoint("/metrics");
        assert!(config.is_public_endpoint("/metrics"));
        assert!(!config.is_public_endpoint("/metrics2"));
    }

    #[test]
    fn test_public_endpoint_prefix_match() {
        let config = AuthConfig::new().with_public_endpoint("/api/v1");
        assert!(config.is_public_endpoint("/api/v1"));
        assert!(config.is_public_endpoint("/api/v1/foo"));
        assert!(config.is_public_endpoint("/api/v1?key=val"));
    }

    #[test]
    fn test_health_is_public_by_default() {
        let config = AuthConfig::new();
        assert!(config.is_public_endpoint("/health"));
        assert!(config.is_public_endpoint("/"));
    }

    // ── API Key validation ──────────────────────────────────────────────

    #[test]
    fn test_validate_api_key_valid() {
        let config = auth_config_with_key("my-secret-key");
        let user = config.validate_api_key("my-secret-key");
        assert!(user.is_some());
    }

    #[test]
    fn test_validate_api_key_invalid() {
        let config = auth_config_with_key("my-secret-key");
        let user = config.validate_api_key("wrong-key");
        assert!(user.is_none());
    }

    #[test]
    fn test_validate_api_key_expired_key() {
        let config = AuthConfig::new()
            .with_api_key(ApiKey::new("expired-key").expires_at(0));
        let user = config.validate_api_key("expired-key");
        assert!(user.is_none());
    }

    #[test]
    fn test_validate_api_key_inactive() {
        let mut key = ApiKey::new("inactive-key");
        key.active = false;
        let config = AuthConfig::new().with_api_key(key);
        let user = config.validate_api_key("inactive-key");
        assert!(user.is_none());
    }

    // ── AuthConfig validation ───────────────────────────────────────────

    #[test]
    fn test_validate_config_no_jwt() {
        let config = AuthConfig::new();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_config_jwt_secret_too_short() {
        let config = AuthConfig::new().with_jwt_secret("short");
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_config_jwt_secret_valid() {
        let config = auth_config_with_jwt();
        assert!(config.validate().is_ok());
    }

    // ── JWT generation & validation ─────────────────────────────────────

    #[test]
    fn test_jwt_roundtrip() {
        let config = auth_config_with_jwt();
        let claims = JwtClaims::new("user1", 3600)
            .with_roles(vec!["admin".into()]);
        let token = config.generate_jwt(&claims).expect("should generate JWT");
        let validated = config.validate_jwt(&token).expect("should validate JWT");
        assert_eq!(validated.sub, "user1");
        assert_eq!(validated.roles, vec!["admin".to_string()]);
    }

    #[test]
    fn test_jwt_expired_token() {
        let config = auth_config_with_jwt();
        let mut claims = JwtClaims::new("user1", 0);
        // Force expiration far in the past
        claims.exp = 0;
        claims.iat = 0;
        let token = config.generate_jwt(&claims).expect("should generate JWT");
        let result = config.validate_jwt(&token);
        assert!(matches!(result, Err(AuthError::TokenExpired)));
    }

    #[test]
    fn test_jwt_invalid_format() {
        let config = auth_config_with_jwt();
        let result = config.validate_jwt("not.a.valid.jwt.token");
        assert!(result.is_err());
    }

    #[test]
    fn test_jwt_two_parts_only() {
        let config = auth_config_with_jwt();
        let result = config.validate_jwt("part1.part2");
        assert!(matches!(result, Err(AuthError::InvalidToken(_))));
    }

    #[test]
    fn test_jwt_tampered_payload() {
        let config = auth_config_with_jwt();
        let claims = JwtClaims::new("user1", 3600);
        let token = config.generate_jwt(&claims).expect("should generate JWT");
        // Tamper with the payload
        let parts: Vec<&str> = token.split('.').collect();
        let tampered = format!("{}.{}.{}", parts[0], "dGFtcGVyZWQ", parts[2]);
        let result = config.validate_jwt(&tampered);
        assert!(matches!(result, Err(AuthError::InvalidSignature)));
    }

    #[test]
    fn test_jwt_wrong_secret() {
        let config1 = AuthConfig::new()
            .with_jwt_secret("secret-one-that-is-at-least-32-bytes-long!!");
        let config2 = AuthConfig::new()
            .with_jwt_secret("secret-two-that-is-at-least-32-bytes-long!!");
        let claims = JwtClaims::new("user1", 3600);
        let token = config1.generate_jwt(&claims).expect("should generate JWT");
        let result = config2.validate_jwt(&token);
        assert!(result.is_err());
    }

    #[test]
    fn test_jwt_no_secret_configured() {
        let config = AuthConfig::new(); // no JWT secret
        let result = config.validate_jwt("any.token.here");
        assert!(matches!(result, Err(AuthError::NoJwtSecret)));
    }

    #[test]
    fn test_generate_jwt_no_secret() {
        let config = AuthConfig::new();
        let claims = JwtClaims::new("user1", 3600);
        let result = config.generate_jwt(&claims);
        assert!(matches!(result, Err(AuthError::NoJwtSecret)));
    }

    // ── JwtClaims ───────────────────────────────────────────────────────

    #[test]
    fn test_jwt_claims_new() {
        let claims = JwtClaims::new("user1", 3600);
        assert_eq!(claims.sub, "user1");
        assert!(claims.exp > claims.iat);
        assert_eq!(claims.roles, vec!["reader".to_string()]);
    }

    #[test]
    fn test_jwt_claims_not_expired() {
        let claims = JwtClaims::new("user1", 3600);
        assert!(!claims.is_expired());
    }

    #[test]
    fn test_jwt_claims_to_user() {
        let claims = JwtClaims::new("user1", 3600)
            .with_roles(vec!["writer".into()]);
        let user = claims.to_user();
        assert_eq!(user.id, "user1");
    }

    // ── AuthError Display ───────────────────────────────────────────────

    #[test]
    fn test_auth_error_display() {
        assert_eq!(AuthError::MissingCredentials.to_string(), "Authentication required");
        assert_eq!(AuthError::InvalidApiKey.to_string(), "Invalid API key");
        assert_eq!(AuthError::InvalidSignature.to_string(), "Invalid token signature");
        assert_eq!(AuthError::TokenExpired.to_string(), "Token has expired");
        assert_eq!(AuthError::NoJwtSecret.to_string(), "JWT authentication not configured");
        assert!(AuthError::Forbidden("test".into()).to_string().contains("test"));
        assert!(AuthError::InvalidToken("bad".into()).to_string().contains("bad"));
    }

    // ── AuthContext & AuthMethod ─────────────────────────────────────────

    #[test]
    fn test_auth_method_equality() {
        assert_eq!(AuthMethod::ApiKey, AuthMethod::ApiKey);
        assert_eq!(AuthMethod::Jwt, AuthMethod::Jwt);
        assert_eq!(AuthMethod::None, AuthMethod::None);
        assert_ne!(AuthMethod::ApiKey, AuthMethod::Jwt);
    }

    #[test]
    fn test_auth_context_debug() {
        let ctx = AuthContext {
            user: User::new("test"),
            method: AuthMethod::ApiKey,
        };
        let debug = format!("{:?}", ctx);
        assert!(debug.contains("ApiKey"));
    }

    #[test]
    fn test_jwt_key_rotation() {
        let old_secret = "old-secret-key-that-is-at-least-32-bytes-long!!";
        let new_secret = "new-secret-key-that-is-at-least-32-bytes-long!!";

        // Generate token with old secret
        let old_config = AuthConfig::new().with_jwt_secret(old_secret);
        let claims = JwtClaims::new("user1", 3600);
        let token = old_config.generate_jwt(&claims).expect("should generate JWT");

        // New config with rotated key — old secret as previous
        let new_config = AuthConfig::new()
            .with_jwt_secret(new_secret)
            .with_previous_jwt_secret(old_secret);

        // Token signed with old key should still validate
        let validated = new_config.validate_jwt(&token);
        assert!(validated.is_ok());
        assert_eq!(validated.expect("should validate JWT").sub, "user1");

        // New tokens use the new secret
        let new_token = new_config.generate_jwt(&claims).expect("should generate JWT");
        assert!(new_config.validate_jwt(&new_token).is_ok());

        // Old config can't validate new tokens
        assert!(old_config.validate_jwt(&new_token).is_err());
    }
}
