//! Lightweight OIDC integration for JWT validation against external identity providers.
//!
//! This module fetches JWKS from an OIDC issuer and validates tokens using RS256/ES256.
//! Requires the `server` feature.

use super::auth::{AuthError, JwtClaims};
use jsonwebtoken::{decode, decode_header, jwk::JwkSet, Algorithm, DecodingKey, Validation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use parking_lot::RwLock;

/// Default JWKS cache TTL (1 hour).
const DEFAULT_JWKS_TTL_SECS: u64 = 3600;

/// Default clock skew leeway for OIDC JWT expiration checks (in seconds).
const OIDC_CLOCK_LEEWAY_SECS: u64 = 60;

/// OIDC provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OidcConfig {
    /// OIDC issuer URL (e.g., `"https://accounts.google.com"`)
    pub issuer_url: String,
    /// Expected audience (`client_id`)
    pub audience: String,
    /// Mapping from OIDC claim values to Needle roles.
    /// e.g., `{"admin": "admin", "editor": "writer", "*": "reader"}`
    #[serde(default)]
    pub role_mapping: HashMap<String, String>,
    /// OIDC claim to use for role extraction (default: `"role"`)
    #[serde(default = "default_role_claim")]
    pub role_claim: String,
    /// JWKS cache TTL in seconds (default: 3600)
    #[serde(default = "default_jwks_ttl_secs")]
    pub jwks_ttl_secs: u64,
}

fn default_role_claim() -> String {
    "role".to_string()
}

fn default_jwks_ttl_secs() -> u64 {
    DEFAULT_JWKS_TTL_SECS
}

/// Cached JWKS with TTL.
pub struct JwksCache {
    jwks: Option<JwkSet>,
    fetched_at: Option<std::time::Instant>,
    ttl: std::time::Duration,
}

impl JwksCache {
    pub fn new(ttl: std::time::Duration) -> Self {
        Self {
            jwks: None,
            fetched_at: None,
            ttl,
        }
    }

    pub fn is_expired(&self) -> bool {
        self.fetched_at.map_or(true, |t| t.elapsed() > self.ttl)
    }

    pub fn get(&self) -> Option<&JwkSet> {
        if self.is_expired() {
            None
        } else {
            self.jwks.as_ref()
        }
    }

    pub fn set(&mut self, jwks: JwkSet) {
        self.jwks = Some(jwks);
        self.fetched_at = Some(std::time::Instant::now());
    }
}

/// OIDC validator that caches JWKS and validates tokens.
pub struct OidcValidator {
    config: OidcConfig,
    cache: RwLock<JwksCache>,
}

impl OidcValidator {
    /// Create a new OIDC validator with the given configuration.
    pub fn new(config: OidcConfig) -> Self {
        let ttl = std::time::Duration::from_secs(config.jwks_ttl_secs);
        Self {
            config,
            cache: RwLock::new(JwksCache::new(ttl)),
        }
    }

    /// Load a JWKS document into the cache (e.g., fetched from the issuer's JWKS URI).
    ///
    /// Call this after fetching the JWKS JSON from `{issuer_url}/.well-known/openid-configuration`
    /// → `jwks_uri`. When `reqwest` is available (via the `embedding-providers` feature), the
    /// HTTP fetch can be wired in; otherwise the caller is responsible for providing the JWKS.
    pub fn load_jwks(&self, jwks: JwkSet) {
        self.cache.write().set(jwks);
    }

    /// Parse a raw JSON string into a [`JwkSet`] and load it into the cache.
    pub fn load_jwks_from_json(&self, json: &str) -> Result<(), AuthError> {
        let jwks: JwkSet = serde_json::from_str(json).map_err(|e| {
            AuthError::InvalidToken(format!("Failed to parse JWKS: {e}"))
        })?;
        self.load_jwks(jwks);
        Ok(())
    }

    /// Validate an OIDC JWT token using the cached JWKS.
    ///
    /// 1. Decodes the JWT header to extract `kid` and `alg`.
    /// 2. Looks up the matching key in the cached JWKS.
    /// 3. Validates issuer, audience, and expiration.
    /// 4. Maps OIDC claims to Needle roles via `role_mapping`.
    pub fn validate_token(&self, token: &str) -> Result<JwtClaims, AuthError> {
        let header = decode_header(token).map_err(|e| {
            AuthError::InvalidToken(format!("Failed to decode JWT header: {e}"))
        })?;

        // Only RS256 and ES256 are supported for OIDC
        let algorithm = header.alg;
        if !matches!(algorithm, Algorithm::RS256 | Algorithm::ES256) {
            return Err(AuthError::InvalidToken(format!(
                "Unsupported OIDC algorithm: {algorithm:?}. Only RS256 and ES256 are supported"
            )));
        }

        let kid = header.kid.as_deref().ok_or_else(|| {
            AuthError::InvalidToken("OIDC token missing 'kid' header claim".into())
        })?;

        // Look up key in cached JWKS
        let decoding_key = self.find_decoding_key(kid)?;

        // Build validation with issuer and audience checks
        let mut validation = Validation::new(algorithm);
        validation.leeway = OIDC_CLOCK_LEEWAY_SECS;
        validation.validate_exp = true;
        validation.set_issuer(&[&self.config.issuer_url]);
        validation.set_audience(&[&self.config.audience]);

        let token_data = decode::<JwtClaims>(token, &decoding_key, &validation).map_err(|e| {
            match e.kind() {
                jsonwebtoken::errors::ErrorKind::ExpiredSignature => AuthError::TokenExpired,
                jsonwebtoken::errors::ErrorKind::InvalidSignature => AuthError::InvalidSignature,
                jsonwebtoken::errors::ErrorKind::InvalidIssuer => {
                    AuthError::InvalidToken("Token issuer does not match OIDC configuration".into())
                }
                jsonwebtoken::errors::ErrorKind::InvalidAudience => {
                    AuthError::InvalidToken("Token audience does not match OIDC configuration".into())
                }
                jsonwebtoken::errors::ErrorKind::InvalidAlgorithm => {
                    AuthError::InvalidToken("Algorithm mismatch".into())
                }
                _ => AuthError::InvalidToken(format!("OIDC token validation failed: {e}")),
            }
        })?;

        let mut claims = token_data.claims;

        // Map OIDC claims to Needle roles
        claims.roles = self.map_roles(&claims);

        Ok(claims)
    }

    /// Find a [`DecodingKey`] for the given `kid` in the cached JWKS.
    fn find_decoding_key(&self, kid: &str) -> Result<DecodingKey, AuthError> {
        let cache = self.cache.read();

        let jwks = cache.get().ok_or_else(|| {
            AuthError::InvalidToken(
                "JWKS not loaded. Fetch the provider's JWKS and call load_jwks() first".into(),
            )
        })?;

        let jwk = jwks.find(kid).ok_or_else(|| {
            AuthError::InvalidToken(format!("No key found in JWKS for kid '{kid}'"))
        })?;

        DecodingKey::from_jwk(jwk).map_err(|e| {
            AuthError::InvalidToken(format!("Failed to build decoding key from JWK: {e}"))
        })
    }

    /// Map OIDC token claims to Needle roles using the configured role mapping.
    fn map_roles(&self, claims: &JwtClaims) -> Vec<String> {
        if self.config.role_mapping.is_empty() {
            // No mapping configured — preserve existing roles
            return claims.roles.clone();
        }

        let mut roles = Vec::new();

        // Extract the role claim value from extra claims
        if let Some(claim_value) = claims.extra.get(&self.config.role_claim) {
            let claim_values: Vec<String> = match claim_value {
                serde_json::Value::String(s) => vec![s.clone()],
                serde_json::Value::Array(arr) => arr
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect(),
                _ => Vec::new(),
            };

            for val in &claim_values {
                if let Some(needle_role) = self.config.role_mapping.get(val) {
                    roles.push(needle_role.clone());
                }
            }
        }

        // Apply wildcard mapping if no specific mapping matched
        if roles.is_empty() {
            if let Some(default_role) = self.config.role_mapping.get("*") {
                roles.push(default_role.clone());
            }
        }

        // Fall back to reader if nothing mapped
        if roles.is_empty() {
            roles.push("reader".to_string());
        }

        roles
    }

    /// Returns a reference to the OIDC configuration.
    pub fn config(&self) -> &OidcConfig {
        &self.config
    }
}

// --- Feature-gated JWKS fetching ---

/// When the `embedding-providers` feature is enabled (which brings `reqwest`),
/// the OIDC validator can fetch JWKS over HTTP from the identity provider.
#[cfg(feature = "embedding-providers")]
impl OidcValidator {
    /// Fetch JWKS from the OIDC provider.
    ///
    /// First fetches `.well-known/openid-configuration` to get the `jwks_uri`,
    /// then fetches the actual JWKS.
    pub async fn fetch_jwks(&self) -> std::result::Result<JwkSet, String> {
        let client = reqwest::Client::new();

        // Fetch OIDC discovery document
        let discovery_url = format!(
            "{}/.well-known/openid-configuration",
            self.config.issuer_url.trim_end_matches('/')
        );
        let discovery: serde_json::Value = client
            .get(&discovery_url)
            .send()
            .await
            .map_err(|e| format!("Failed to fetch OIDC discovery: {e}"))?
            .json()
            .await
            .map_err(|e| format!("Failed to parse OIDC discovery: {e}"))?;

        let jwks_uri = discovery["jwks_uri"]
            .as_str()
            .ok_or("OIDC discovery missing jwks_uri")?;

        // Fetch JWKS from the discovered URI
        let jwks: JwkSet = client
            .get(jwks_uri)
            .send()
            .await
            .map_err(|e| format!("Failed to fetch JWKS: {e}"))?
            .json()
            .await
            .map_err(|e| format!("Failed to parse JWKS: {e}"))?;

        Ok(jwks)
    }

    /// Refresh the JWKS cache if expired.
    pub async fn refresh_jwks_if_needed(&self) -> std::result::Result<(), String> {
        if !self.cache.read().is_expired() {
            return Ok(());
        }
        let jwks = self.fetch_jwks().await?;
        self.cache.write().set(jwks);
        Ok(())
    }
}

/// Without the `embedding-providers` feature, JWKS must be provided manually.
#[cfg(not(feature = "embedding-providers"))]
impl OidcValidator {
    /// Set the JWKS directly (for use when HTTP fetching is not available).
    ///
    /// JWKS fetching requires the `embedding-providers` feature (for `reqwest`).
    /// Without it, call this method to provide the JWKS manually.
    pub fn set_jwks(&self, jwks: JwkSet) {
        self.cache.write().set(jwks);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{encode, EncodingKey, Header};
    use serde_json::json;

    fn rsa_private_key() -> &'static str {
        "\
-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEAwSe3yTr3EOgFxWwOmMdBkpwCLns+sRDAL8bEg72EYisWf0dV
Y1l8H38ddL3F8dYLcOWFwpFjZCJ766Zu4mwH33CZ9A1/ymYcu5MDxDE7e8ZjcdID
6QaiM2fTzG9JmGmn2InHYaKD/irUXKPTUZzs0LC4QGES9GBig4kiCGEkBk/KvlBj
SQVvkJEhppiGAWOBv8iL06NVdMCHzt0O7HnGsqCUAySrLI0T/eiaRStpRC397IJ5
lvaPX+RR2A+wmjCVk0dx/yZi07MyAWT6Ochq4X3uGOvxBvwVGt5S6/No0YYb13/I
fATDs5D5MvTZE9/GBIqFQF1x1pgnonQ2k0vt9wIDAQABAoIBAEjlEWIbI7S4q7zm
29dik2eeAuDB2FYAiVc+f1lsg3J86l+cbygwVDyav2YYXIS5D9ZKeKGGNulKblPv
mrdOp+X2W9OT6J9czAkqIWjAX7+FjnAdHyapPzuBOphTg4XGkfaRgLJjH8cjKMPR
e+W4AFN97fs1525clbEoZrSc3HiYqn6COZAXYlaf2qbYbzF3PRXEM4sxd1k02rvU
ZQBafXgL4vvOt3vT1OzLkX2IpHGDYMB7vOq0VELPR9HGshZWHIT7CDsNYpm4Bz2m
LdBqnwDxbbPHCm8+NQWMI/c0x6LZ3AfjEGSt8Whg+6PXwuFMT/M8QfvBAFBybjqA
6s49TVECgYEA4rabNPn/1hmG5DhdAZ1V83i4k/xtcwy4AzTOAnCYMUjb4xVbbCG7
YCwCH5IKnloTfbm4zPy7ROjyrPDL9/5gTk4jBzjjYAH7Wl1a0/ciZKYLDBhSN7bN
Mu0UqzjcRocZBh0/Mg1ozVgekzf+Ue357KCM5l5L6vsZgQ/U9jhJcO8CgYEA2htY
ew4AFAPt+uknl4pVy/9xqehHRy7C+Smb8S6Jbg4Jwo8aOdSO7wIZAfhMpeuF/Qmo
9l3ANoHg+85TIm6plu1Knmbv7khcIw7k8d69lpM3hrAOIl5IxyAFEn4uV2yEAzaK
Z6ErDB7+4tc6kJ5AgcxyPOpBgt4oT07YX3JpQ3kCgYADpWgtm++vY821kep9AijF
t6VQS/j+pq+27Xx6sZDhCgjvSAKmZIx86XhHRbQCA/TYSspcEZx5aT2t5lmBbYfi
+oK5tQKDIsUGGQZC7nCRKdJ3qVR5LOlz7jgs4Mc6IyYV4RaJGYob81TajUX7z1X7
pkFd2xphdxRb7QNBynnz5QKBgHdSao+30xcgJzwT/lMLnXCjaX240/X/gS9rMiM6
gHkzOOe1/nUQ8rmTfjbzros/VOhgNo3CMHwhhgJ8mELIJAOsAhyy2CSWdcHATkR8
xV/xXnlTLAhlaI931w6M9bFibr6LQiD7rV9OPcfAVAv2Z/ga74yf5ANCou7whbOC
FlCRAoGATY2J0sdCY5EvSbSJ0C3VRdv2stiq0wgbV7uohGYp9OZmV+1o7D8xmfqP
RHIGr59RKxXGDmC9W4qG2BZwBSkiileECQWMZCv7Z8ISkfByid6db7v4UZKP2gdR
4KUqYuGN46DSMpScJyQHIVgXJKO2+UvXfA3JUvFsGW/0u00M5Xs=
-----END RSA PRIVATE KEY-----"
    }

    /// Build a JWKS JSON with the test RSA public key.
    fn rsa_jwks_json(kid: &str) -> String {
        // RSA modulus (n) and exponent (e) extracted from the test public key,
        // base64url-encoded without padding.
        let n = "wSe3yTr3EOgFxWwOmMdBkpwCLns-sRDAL8bEg72EYisWf0dVY1l8H38ddL3F8dYLcOWFwpFjZCJ766Zu4mwH33CZ9A1_ymYcu5MDxDE7e8ZjcdID6QaiM2fTzG9JmGmn2InHYaKD_irUXKPTUZzs0LC4QGES9GBig4kiCGEkBk_KvlBjSQVvkJEhppiGAWOBv8iL06NVdMCHzt0O7HnGsqCUAySrLI0T_eiaRStpRC397IJ5lvaPX-RR2A-wmjCVk0dx_yZi07MyAWT6Ochq4X3uGOvxBvwVGt5S6_No0YYb13_IfATDs5D5MvTZE9_GBIqFQF1x1pgnonQ2k0vt9w";
        let e = "AQAB";
        format!(
            r#"{{"keys":[{{"kty":"RSA","kid":"{kid}","use":"sig","alg":"RS256","n":"{n}","e":"{e}"}}]}}"#,
        )
    }

    fn make_test_config(issuer: &str, audience: &str) -> OidcConfig {
        OidcConfig {
            issuer_url: issuer.into(),
            audience: audience.into(),
            role_mapping: HashMap::new(),
            role_claim: "role".into(),
            jwks_ttl_secs: 3600,
        }
    }

    #[test]
    fn test_oidc_config_defaults() {
        let config: OidcConfig = serde_json::from_str(
            r#"{"issuer_url": "https://example.com", "audience": "my-app"}"#,
        )
        .unwrap();
        assert_eq!(config.role_claim, "role");
        assert_eq!(config.jwks_ttl_secs, DEFAULT_JWKS_TTL_SECS);
        assert!(config.role_mapping.is_empty());
    }

    #[test]
    fn test_jwks_cache_lifecycle() {
        let mut cache = JwksCache::new(std::time::Duration::from_secs(3600));
        assert!(cache.is_expired());
        assert!(cache.get().is_none());

        let jwks: JwkSet = serde_json::from_str(r#"{"keys":[]}"#).unwrap();
        cache.set(jwks);
        assert!(!cache.is_expired());
        assert!(cache.get().is_some());
    }

    #[test]
    fn test_jwks_cache_expiry() {
        let mut cache = JwksCache::new(std::time::Duration::from_secs(0));
        let jwks: JwkSet = serde_json::from_str(r#"{"keys":[]}"#).unwrap();
        cache.set(jwks);
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(cache.is_expired());
        assert!(cache.get().is_none());
    }

    #[test]
    fn test_validator_no_jwks_loaded() {
        let config = make_test_config("https://example.com", "my-app");
        let validator = OidcValidator::new(config);
        let result = validator.validate_token("eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6InRlc3QifQ.eyJzdWIiOiJ0ZXN0In0.fake");
        assert!(result.is_err());
    }

    #[test]
    fn test_validator_unsupported_algorithm() {
        let config = make_test_config("https://example.com", "my-app");
        let validator = OidcValidator::new(config);
        // HS256 token header — not supported for OIDC
        let result = validator.validate_token(
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0IiwiZXhwIjo5OTk5OTk5OTk5fQ.fake",
        );
        assert!(matches!(result, Err(AuthError::InvalidToken(msg)) if msg.contains("Unsupported OIDC algorithm")));
    }

    #[test]
    fn test_role_mapping() {
        let mut role_mapping = HashMap::new();
        role_mapping.insert("admin".into(), "admin".into());
        role_mapping.insert("editor".into(), "writer".into());
        role_mapping.insert("*".into(), "reader".into());

        let config = OidcConfig {
            issuer_url: "https://example.com".into(),
            audience: "my-app".into(),
            role_mapping,
            role_claim: "role".into(),
            jwks_ttl_secs: 3600,
        };
        let validator = OidcValidator::new(config);

        // Test admin mapping
        let mut claims = JwtClaims::new("user1", 3600);
        claims.extra.insert("role".into(), json!("admin"));
        let roles = validator.map_roles(&claims);
        assert_eq!(roles, vec!["admin"]);

        // Test wildcard fallback
        let mut claims2 = JwtClaims::new("user2", 3600);
        claims2.extra.insert("role".into(), json!("unknown-role"));
        let roles2 = validator.map_roles(&claims2);
        assert_eq!(roles2, vec!["reader"]);

        // Test array claim
        let mut claims3 = JwtClaims::new("user3", 3600);
        claims3.extra.insert("role".into(), json!(["admin", "editor"]));
        let roles3 = validator.map_roles(&claims3);
        assert_eq!(roles3, vec!["admin", "writer"]);
    }

    #[test]
    fn test_role_mapping_no_claim() {
        let mut role_mapping = HashMap::new();
        role_mapping.insert("*".into(), "reader".into());

        let config = OidcConfig {
            issuer_url: "https://example.com".into(),
            audience: "my-app".into(),
            role_mapping,
            role_claim: "role".into(),
            jwks_ttl_secs: 3600,
        };
        let validator = OidcValidator::new(config);

        let claims = JwtClaims::new("user1", 3600);
        let roles = validator.map_roles(&claims);
        assert_eq!(roles, vec!["reader"]);
    }

    #[test]
    fn test_load_jwks_from_json() {
        let config = make_test_config("https://example.com", "my-app");
        let validator = OidcValidator::new(config);

        assert!(validator.load_jwks_from_json(r#"{"keys":[]}"#).is_ok());
        assert!(validator.load_jwks_from_json("not-json").is_err());
    }

    #[test]
    fn test_validate_with_rsa_jwks() {
        let kid = "test-key-1";
        let issuer = "https://test-issuer.example.com";
        let audience = "test-audience";

        let config = make_test_config(issuer, audience);
        let validator = OidcValidator::new(config);
        validator
            .load_jwks_from_json(&rsa_jwks_json(kid))
            .unwrap();

        // Create a valid token signed with the matching RSA private key
        let encoding_key =
            EncodingKey::from_rsa_pem(rsa_private_key().as_bytes()).unwrap();
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some(kid.into());

        let mut claims = JwtClaims::new("oidc-user", 3600);
        claims.extra.insert("iss".into(), json!(issuer));
        claims.extra.insert("aud".into(), json!(audience));

        let token = encode(&header, &claims, &encoding_key).unwrap();
        let result = validator.validate_token(&token);
        assert!(result.is_ok(), "Expected Ok, got: {result:?}");
        assert_eq!(result.unwrap().sub, "oidc-user");
    }

    #[test]
    fn test_validate_wrong_kid() {
        let kid = "test-key-1";
        let config = make_test_config("https://example.com", "my-app");
        let validator = OidcValidator::new(config);
        validator
            .load_jwks_from_json(&rsa_jwks_json(kid))
            .unwrap();

        let encoding_key =
            EncodingKey::from_rsa_pem(rsa_private_key().as_bytes()).unwrap();
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some("wrong-kid".into());

        let claims = JwtClaims::new("user", 3600);
        let token = encode(&header, &claims, &encoding_key).unwrap();
        let result = validator.validate_token(&token);
        assert!(matches!(result, Err(AuthError::InvalidToken(msg)) if msg.contains("No key found")));
    }

    #[test]
    fn test_validate_missing_kid() {
        let config = make_test_config("https://example.com", "my-app");
        let validator = OidcValidator::new(config);
        validator
            .load_jwks_from_json(r#"{"keys":[]}"#)
            .unwrap();

        // Build a token with no kid in header
        let encoding_key =
            EncodingKey::from_rsa_pem(rsa_private_key().as_bytes()).unwrap();
        let header = Header::new(Algorithm::RS256); // no kid
        let claims = JwtClaims::new("user", 3600);
        let token = encode(&header, &claims, &encoding_key).unwrap();

        let result = validator.validate_token(&token);
        assert!(matches!(result, Err(AuthError::InvalidToken(msg)) if msg.contains("kid")));
    }

    #[test]
    fn test_validate_wrong_issuer() {
        let kid = "test-key-1";
        let config = make_test_config("https://correct-issuer.com", "my-app");
        let validator = OidcValidator::new(config);
        validator
            .load_jwks_from_json(&rsa_jwks_json(kid))
            .unwrap();

        let encoding_key =
            EncodingKey::from_rsa_pem(rsa_private_key().as_bytes()).unwrap();
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some(kid.into());

        let mut claims = JwtClaims::new("user", 3600);
        claims.extra.insert("iss".into(), json!("https://wrong-issuer.com"));
        claims.extra.insert("aud".into(), json!("my-app"));

        let token = encode(&header, &claims, &encoding_key).unwrap();
        let result = validator.validate_token(&token);
        assert!(matches!(result, Err(AuthError::InvalidToken(msg)) if msg.contains("issuer")));
    }

    #[test]
    fn test_validate_wrong_audience() {
        let kid = "test-key-1";
        let config = make_test_config("https://issuer.com", "correct-audience");
        let validator = OidcValidator::new(config);
        validator
            .load_jwks_from_json(&rsa_jwks_json(kid))
            .unwrap();

        let encoding_key =
            EncodingKey::from_rsa_pem(rsa_private_key().as_bytes()).unwrap();
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some(kid.into());

        let mut claims = JwtClaims::new("user", 3600);
        claims.extra.insert("iss".into(), json!("https://issuer.com"));
        claims.extra.insert("aud".into(), json!("wrong-audience"));

        let token = encode(&header, &claims, &encoding_key).unwrap();
        let result = validator.validate_token(&token);
        assert!(matches!(result, Err(AuthError::InvalidToken(msg)) if msg.contains("audience")));
    }
}
