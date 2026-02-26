//! Middleware functions for the Needle HTTP server.

use super::{AppState, CorsConfig, PerIpRateLimiter, RateLimitConfig};
use super::auth::{AuthConfig, AuthContext, AuthMethod};
use super::types::ApiError;
use axum::{
    body::Body,
    extract::{ConnectInfo, State},
    http::{header, Method, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use governor::{clock::DefaultClock, state::keyed::DashMapStateStore, Quota, RateLimiter};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::num::NonZeroU32;
use std::sync::Arc;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::warn;
use serde_json::json;

static SECURITY_HEADER_XCTO: (header::HeaderName, &str) = (header::X_CONTENT_TYPE_OPTIONS, "nosniff");
static SECURITY_HEADER_XFO: (header::HeaderName, &str) = (header::X_FRAME_OPTIONS, "DENY");

#[cfg(feature = "metrics")]
use crate::metrics::{http_metrics, metrics};

pub(super) fn build_cors_layer(config: &CorsConfig) -> std::result::Result<CorsLayer, String> {
    if !config.enabled {
        return Ok(CorsLayer::new());
    }

    config.validate()?;

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
            let mut parsed = Vec::new();
            for o in origins {
                match o.parse() {
                    Ok(origin) => parsed.push(origin),
                    Err(_) => {
                        tracing::warn!(origin = %o, "Rejected invalid CORS origin at startup");
                    }
                }
            }
            cors.allow_origin(parsed)
        }
    };

    if config.allow_credentials {
        cors = cors.allow_credentials(true);
    }

    Ok(cors)
}


pub(super) fn create_rate_limiter(config: &RateLimitConfig) -> Option<Arc<PerIpRateLimiter>> {
    if !config.enabled {
        return None;
    }
    if config.requests_per_second == 0 {
        warn!("Rate limiting disabled (requests_per_second = 0)");
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

pub(super) fn create_write_rate_limiter(config: &RateLimitConfig) -> Option<Arc<PerIpRateLimiter>> {
    if !config.enabled {
        return None;
    }
    let rps = config.write_requests_per_second.unwrap_or(
        (config.requests_per_second / 5).max(1)
    );
    let burst = config.write_burst_size.unwrap_or(
        (config.burst_size / 5).max(1)
    );

    let quota = Quota::per_second(
        NonZeroU32::new(rps).unwrap_or(NonZeroU32::new(20).expect("20 is non-zero")),
    )
    .allow_burst(
        NonZeroU32::new(burst).unwrap_or(NonZeroU32::new(1).expect("1 is non-zero")),
    );

    Some(Arc::new(RateLimiter::dashmap(quota)))
}

pub(super) fn create_admin_rate_limiter(config: &RateLimitConfig) -> Option<Arc<PerIpRateLimiter>> {
    if !config.enabled {
        return None;
    }
    let rps = config.admin_requests_per_second.unwrap_or(5);
    let burst = config.admin_burst_size.unwrap_or(3);

    let quota = Quota::per_second(
        NonZeroU32::new(rps).unwrap_or(NonZeroU32::new(5).expect("5 is non-zero")),
    )
    .allow_burst(
        NonZeroU32::new(burst).unwrap_or(NonZeroU32::new(1).expect("1 is non-zero")),
    );

    Some(Arc::new(RateLimiter::dashmap(quota)))
}

/// Classify a request into a rate limit tier based on path and method.
fn classify_rate_limit_tier(method: &Method, path: &str) -> RateLimitTier {
    // Admin operations: save, webhooks, aliases mutations, compact
    if path == "/save"
        || path.starts_with("/webhooks")
        || (path.starts_with("/aliases") && !matches!(*method, Method::GET | Method::HEAD))
        || path.ends_with("/compact")
    {
        return RateLimitTier::Admin;
    }

    // Write/search operations: inserts, deletes, search, upserts
    if matches!(*method, Method::POST | Method::PUT | Method::DELETE) {
        return RateLimitTier::Write;
    }

    // Everything else is a read
    RateLimitTier::Read
}

#[derive(Debug, Clone, Copy)]
enum RateLimitTier {
    Read,
    Write,
    Admin,
}


pub(super) fn extract_client_ip(request: &Request<Body>, trusted_proxies: &[IpAddr]) -> IpAddr {
    /// Maximum number of IPs to parse from X-Forwarded-For to prevent DoS.
    const MAX_FORWARDED_IPS: usize = 10;

    let remote_ip = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|info| info.0.ip());
    let trust_headers = remote_ip
        .is_some_and(|ip| trusted_proxies.contains(&ip));

    if trust_headers {
        // Check X-Forwarded-For header first (for proxied requests)
        if let Some(forwarded_for) = request.headers().get("x-forwarded-for") {
            if let Ok(value) = forwarded_for.to_str() {
                // Walk right-to-left to find the first IP not in the trusted proxy list
                let ips: Vec<&str> = value.split(',').collect();
                // Limit parsed entries to prevent DoS via excessively long header
                for raw_ip in ips.iter().rev().take(MAX_FORWARDED_IPS) {
                    if let Ok(ip) = raw_ip.trim().parse::<IpAddr>() {
                        if !trusted_proxies.contains(&ip) {
                            return ip;
                        }
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


pub(super) async fn rate_limit_middleware(
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let tier = classify_rate_limit_tier(request.method(), request.uri().path());
    let limiter = match tier {
        RateLimitTier::Admin => state.admin_rate_limiter.as_ref().or(state.rate_limiter.as_ref()),
        RateLimitTier::Write => state.write_rate_limiter.as_ref().or(state.rate_limiter.as_ref()),
        RateLimitTier::Read => state.rate_limiter.as_ref(),
    };

    if let Some(limiter) = limiter {
        let client_ip = extract_client_ip(&request, &state.trusted_proxies);
        match limiter.check_key(&client_ip) {
            Ok(_) => next.run(request).await,
            Err(_) => {
                warn!(
                    client_ip = %client_ip,
                    tier = ?tier,
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


///
/// Checks for authentication credentials in this order:
/// 1. `X-API-Key` header for API key authentication
/// 2. `Authorization: Bearer <token>` header for JWT authentication
///
/// If authentication is required and no valid credentials are provided,
/// returns 401 Unauthorized. If credentials are invalid, returns 401.
/// Public endpoints (as configured in AuthConfig) bypass authentication.
pub(super) async fn auth_middleware(
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
            if let Some(collection) = extract_collection_from_path(&path) {
                // Collection-level RBAC
                let permission = infer_permission_from_request(&request);
                let resource = crate::security::Resource::Collection(collection);
                if !context.user.has_permission(permission, &resource) {
                    let error = ApiError::new(
                        format!("Insufficient permissions on collection"),
                        "FORBIDDEN".to_string(),
                    );
                    return (StatusCode::FORBIDDEN, Json(error)).into_response();
                }
            } else if requires_admin(&path, request.method()) {
                // Admin-only endpoints (non-collection mutating operations)
                let resource = crate::security::Resource::System;
                if !context.user.has_permission(crate::security::Permission::Admin, &resource) {
                    let error = ApiError::new(
                        "Admin permission required".to_string(),
                        "FORBIDDEN".to_string(),
                    );
                    return (StatusCode::FORBIDDEN, Json(error)).into_response();
                }
            }
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


pub(super) fn extract_collection_from_path(path: &str) -> Option<String> {
    let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    if parts.len() >= 2 && parts[0] == "collections" {
        Some(parts[1].to_string())
    } else {
        None
    }
}


/// Check if a non-collection endpoint requires admin permission.
/// Gates mutating operations on /save, /webhooks, and /aliases.
fn requires_admin(path: &str, method: &Method) -> bool {
    let is_mutating = matches!(*method, Method::POST | Method::PUT | Method::DELETE);
    if !is_mutating {
        return false;
    }
    path == "/save"
        || path.starts_with("/webhooks")
        || path.starts_with("/aliases")
}

pub(super) fn infer_permission_from_request(request: &Request<Body>) -> crate::security::Permission {
    match *request.method() {
        Method::GET | Method::HEAD => crate::security::Permission::Read,
        Method::DELETE => crate::security::Permission::Delete,
        _ => crate::security::Permission::Write,
    }
}


pub(super) fn try_authenticate(request: &Request<Body>, auth_config: &AuthConfig) -> Option<AuthContext> {
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


#[cfg(feature = "metrics")]
pub(super) async fn metrics_middleware(request: Request<Body>, next: Next) -> Response {
    let method = request.method().to_string();
    let path = request.uri().path().to_string();

    let mut timer = http_metrics().start_request(&method, &path);
    let response = next.run(request).await;
    timer.set_status(response.status().as_u16());

    response
}


#[cfg(feature = "metrics")]
pub(super) async fn get_metrics() -> impl IntoResponse {
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


/// Middleware that adds security headers to every response.
///
/// Each `.parse().expect("static header value")` call converts a compile-time
/// string literal into an HTTP `HeaderValue`. These calls cannot fail at
/// runtime because the inputs are hardcoded ASCII strings, so `expect` is
/// acceptable here (tagged with `// allow-expect` for the CI audit). See
/// CONTRIBUTING.md § "Acceptable uses of `expect()`" for the project policy.
pub(super) async fn security_headers_middleware(
    request: Request<Body>,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(SECURITY_HEADER_XCTO.0.clone(), SECURITY_HEADER_XCTO.1.parse().expect("valid static X-Content-Type-Options header")); // allow-expect
    headers.insert(SECURITY_HEADER_XFO.0.clone(), SECURITY_HEADER_XFO.1.parse().expect("valid static X-Frame-Options header")); // allow-expect
    headers.insert(
        header::STRICT_TRANSPORT_SECURITY,
        "max-age=63072000; includeSubDomains".parse().expect("valid static HSTS header"), // allow-expect
    );
    headers.insert(
        header::HeaderName::from_static("x-xss-protection"),
        "1; mode=block".parse().expect("valid static X-XSS-Protection header"), // allow-expect
    );
    headers.insert(
        header::HeaderName::from_static("content-security-policy"),
        "default-src 'none'; frame-ancestors 'none'".parse().expect("valid static Content-Security-Policy header"), // allow-expect
    );
    response
}

// ── API Stability Tiers ──────────────────────────────────────────────────

/// Stability tiers for REST endpoints.
#[derive(Clone, Copy)]
enum StabilityTier {
    Stable,
    Beta,
    Experimental,
}

impl StabilityTier {
    fn as_str(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Beta => "beta",
            Self::Experimental => "experimental",
        }
    }
}

/// Classify a request path into a stability tier.
fn classify_stability(path: &str) -> StabilityTier {
    // Experimental endpoints
    if path.contains("/memory/")
        || path.contains("/search/graph")
        || path.contains("/search/matryoshka")
        || path.contains("/cache/")
        || path.contains("/ingest")
        || path.contains("/search/time-travel")
        || path.contains("/snapshots/diff")
        || path.contains("/search/estimate")
        || path.contains("/diff")
        || path.contains("/changes")
        || path.contains("/benchmark")
        || path.contains("/cluster/")
        || path.contains("/grpc/")
        || path.contains("/tracing/")
        || path.contains("/webhooks")
        || path.contains("/embeddings/router")
        || path.contains("/mcp")
        || path.contains("/playground")
    {
        return StabilityTier::Experimental;
    }

    // Beta endpoints
    if path.contains("/texts")
        || path.contains("/aliases")
        || path.contains("/expire")
        || path.contains("/ttl-stats")
        || path.contains("/index/status")
        || path.contains("/snapshots")
        || path.contains("/dashboard")
        || path.contains("/openapi.json")
    {
        return StabilityTier::Beta;
    }

    // Everything else is stable (health, info, CRUD, search, save, etc.)
    StabilityTier::Stable
}

static STABILITY_HEADER: header::HeaderName = header::HeaderName::from_static("x-needle-api-stability");

/// Middleware that adds an `X-Needle-API-Stability` header to every response.
pub(super) async fn api_stability_middleware(
    request: Request<Body>,
    next: Next,
) -> Response {
    let tier = classify_stability(request.uri().path());
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        STABILITY_HEADER.clone(),
        tier.as_str().parse().expect("stability tier is a valid header value"), // allow-expect
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_stability_stable() {
        assert!(matches!(classify_stability("/health"), StabilityTier::Stable));
        assert!(matches!(classify_stability("/collections"), StabilityTier::Stable));
        assert!(matches!(classify_stability("/collections/docs/vectors"), StabilityTier::Stable));
        assert!(matches!(classify_stability("/collections/docs/search"), StabilityTier::Stable));
        assert!(matches!(classify_stability("/save"), StabilityTier::Stable));
    }

    #[test]
    fn test_classify_stability_beta() {
        assert!(matches!(classify_stability("/collections/docs/texts"), StabilityTier::Beta));
        assert!(matches!(classify_stability("/aliases"), StabilityTier::Beta));
        assert!(matches!(classify_stability("/collections/docs/expire"), StabilityTier::Beta));
        assert!(matches!(classify_stability("/collections/docs/ttl-stats"), StabilityTier::Beta));
        assert!(matches!(classify_stability("/dashboard"), StabilityTier::Beta));
        assert!(matches!(classify_stability("/openapi.json"), StabilityTier::Beta));
    }

    #[test]
    fn test_classify_stability_experimental() {
        assert!(matches!(classify_stability("/collections/docs/memory/recall"), StabilityTier::Experimental));
        assert!(matches!(classify_stability("/collections/docs/search/graph"), StabilityTier::Experimental));
        assert!(matches!(classify_stability("/mcp"), StabilityTier::Experimental));
        assert!(matches!(classify_stability("/playground"), StabilityTier::Experimental));
        assert!(matches!(classify_stability("/webhooks"), StabilityTier::Experimental));
        assert!(matches!(classify_stability("/collections/docs/cache/lookup"), StabilityTier::Experimental));
    }

    // ── extract_collection_from_path ─────────────────────────────────────

    #[test]
    fn test_extract_collection_from_path_valid() {
        assert_eq!(
            extract_collection_from_path("/collections/my_coll/vectors"),
            Some("my_coll".to_string())
        );
    }

    #[test]
    fn test_extract_collection_from_path_root() {
        assert_eq!(
            extract_collection_from_path("/collections/test"),
            Some("test".to_string())
        );
    }

    #[test]
    fn test_extract_collection_from_path_no_collection() {
        assert_eq!(extract_collection_from_path("/health"), None);
        assert_eq!(extract_collection_from_path("/save"), None);
    }

    #[test]
    fn test_extract_collection_from_path_empty() {
        assert_eq!(extract_collection_from_path("/"), None);
    }

    // ── requires_admin ───────────────────────────────────────────────────

    #[test]
    fn test_requires_admin_save_post() {
        assert!(requires_admin("/save", &Method::POST));
    }

    #[test]
    fn test_requires_admin_save_get() {
        assert!(!requires_admin("/save", &Method::GET));
    }

    #[test]
    fn test_requires_admin_webhooks() {
        assert!(requires_admin("/webhooks", &Method::POST));
        assert!(requires_admin("/webhooks/123", &Method::DELETE));
    }

    #[test]
    fn test_requires_admin_regular_path() {
        assert!(!requires_admin("/collections", &Method::POST));
        assert!(!requires_admin("/health", &Method::GET));
    }

    // ── infer_permission_from_request ─────────────────────────────────────

    #[test]
    fn test_infer_permission_read() {
        let req = Request::builder()
            .method(Method::GET)
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        assert_eq!(infer_permission_from_request(&req), crate::security::Permission::Read);
    }

    #[test]
    fn test_infer_permission_write() {
        let req = Request::builder()
            .method(Method::POST)
            .uri("/collections")
            .body(Body::empty())
            .unwrap();
        assert_eq!(infer_permission_from_request(&req), crate::security::Permission::Write);
    }

    #[test]
    fn test_infer_permission_delete() {
        let req = Request::builder()
            .method(Method::DELETE)
            .uri("/collections/test")
            .body(Body::empty())
            .unwrap();
        assert_eq!(infer_permission_from_request(&req), crate::security::Permission::Delete);
    }

    // ── extract_client_ip ────────────────────────────────────────────────

    #[test]
    fn test_extract_client_ip_no_extensions() {
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let ip = extract_client_ip(&req, &[]);
        assert_eq!(ip, IpAddr::V4(Ipv4Addr::LOCALHOST));
    }

    // ── stability tier as_str ────────────────────────────────────────────

    #[test]
    fn test_stability_tier_as_str() {
        assert_eq!(StabilityTier::Stable.as_str(), "stable");
        assert_eq!(StabilityTier::Beta.as_str(), "beta");
        assert_eq!(StabilityTier::Experimental.as_str(), "experimental");
    }

    // ── build_cors_layer ─────────────────────────────────────────────────

    #[test]
    fn test_build_cors_disabled() {
        let config = CorsConfig {
            enabled: false,
            allowed_origins: None,
            allow_credentials: false,
            max_age_secs: 3600,
        };
        let layer = build_cors_layer(&config);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_build_cors_wildcard() {
        let config = CorsConfig {
            enabled: true,
            allowed_origins: None, // None = allow all
            allow_credentials: false,
            max_age_secs: 3600,
        };
        let layer = build_cors_layer(&config);
        assert!(layer.is_ok());
    }

    // ── create_rate_limiter ──────────────────────────────────────────────

    #[test]
    fn test_create_rate_limiter_disabled() {
        let config = RateLimitConfig {
            enabled: false,
            requests_per_second: 100,
            burst_size: 200,
            write_requests_per_second: None,
            write_burst_size: None,
            admin_requests_per_second: None,
            admin_burst_size: None,
        };
        let limiter = create_rate_limiter(&config);
        assert!(limiter.is_none());
    }

    #[test]
    fn test_create_rate_limiter_enabled() {
        let config = RateLimitConfig {
            enabled: true,
            requests_per_second: 100,
            burst_size: 200,
            write_requests_per_second: None,
            write_burst_size: None,
            admin_requests_per_second: None,
            admin_burst_size: None,
        };
        let limiter = create_rate_limiter(&config);
        assert!(limiter.is_some());
    }

    // ── CorsConfig with specific origins ─────────────────────────────────

    #[test]
    fn test_build_cors_specific_origins() {
        let config = CorsConfig {
            enabled: true,
            allowed_origins: Some(vec![
                "https://example.com".to_string(),
                "https://app.example.com".to_string(),
            ]),
            allow_credentials: true,
            max_age_secs: 7200,
        };
        let layer = build_cors_layer(&config);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_build_cors_empty_origins() {
        let config = CorsConfig {
            enabled: true,
            allowed_origins: Some(vec![]),
            allow_credentials: false,
            max_age_secs: 3600,
        };
        let layer = build_cors_layer(&config);
        assert!(layer.is_ok());
    }

    // ── infer_permission HEAD ────────────────────────────────────────────

    #[test]
    fn test_infer_permission_head() {
        let req = Request::builder()
            .method(Method::HEAD)
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        assert_eq!(infer_permission_from_request(&req), crate::security::Permission::Read);
    }

    // ── infer_permission PUT ─────────────────────────────────────────────

    #[test]
    fn test_infer_permission_put() {
        let req = Request::builder()
            .method(Method::PUT)
            .uri("/collections/test/vectors")
            .body(Body::empty())
            .unwrap();
        assert_eq!(infer_permission_from_request(&req), crate::security::Permission::Write);
    }

    // ── requires_admin: aliases ──────────────────────────────────────────

    #[test]
    fn test_requires_admin_aliases() {
        assert!(requires_admin("/aliases", &Method::POST));
        assert!(requires_admin("/aliases/my_alias", &Method::DELETE));
        assert!(!requires_admin("/aliases", &Method::GET));
    }

    // ── extract_collection_from_path: nested paths ───────────────────────

    #[test]
    fn test_extract_collection_from_path_deep() {
        assert_eq!(
            extract_collection_from_path("/collections/my-coll/vectors/v1"),
            Some("my-coll".to_string())
        );
        assert_eq!(
            extract_collection_from_path("/collections/test/search"),
            Some("test".to_string())
        );
    }

    // ── classify_stability: additional paths ─────────────────────────────

    #[test]
    fn test_classify_stability_additional() {
        assert!(matches!(classify_stability("/info"), StabilityTier::Stable));
        assert!(matches!(classify_stability("/metrics"), StabilityTier::Stable));
        assert!(matches!(classify_stability("/collections/x/vectors/y"), StabilityTier::Stable));
        assert!(matches!(classify_stability("/collections/x/snapshots"), StabilityTier::Beta));
        assert!(matches!(classify_stability("/collections/x/ingest"), StabilityTier::Experimental));
        assert!(matches!(classify_stability("/collections/x/benchmark"), StabilityTier::Experimental));
    }

    // ── try_authenticate: no credentials ─────────────────────────────────

    #[test]
    fn test_try_authenticate_no_credentials() {
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let auth_config = AuthConfig::default();
        let result = try_authenticate(&req, &auth_config);
        assert!(result.is_none());
    }

    // ── try_authenticate: invalid api key ────────────────────────────────

    #[test]
    fn test_try_authenticate_invalid_api_key() {
        let mut req = Request::builder()
            .uri("/collections")
            .body(Body::empty())
            .unwrap();
        req.headers_mut().insert("x-api-key", "invalid-key".parse().unwrap());
        let auth_config = AuthConfig::default();
        let result = try_authenticate(&req, &auth_config);
        assert!(result.is_none());
    }

    // ── extract_client_ip: with x-forwarded-for ──────────────────────────

    #[test]
    fn test_extract_client_ip_default_fallback() {
        let req = Request::builder()
            .header("x-forwarded-for", "1.2.3.4")
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        // Without ConnectInfo extension, should fall back to localhost
        // (trusted_proxies check fails without extension)
        let ip = extract_client_ip(&req, &[]);
        assert_eq!(ip, IpAddr::V4(Ipv4Addr::LOCALHOST));
    }

    // ── create_rate_limiter: parameters ──────────────────────────────────

    #[test]
    fn test_rate_limiter_high_burst() {
        let config = RateLimitConfig {
            enabled: true,
            requests_per_second: 1000,
            burst_size: 5000,
            write_requests_per_second: None,
            write_burst_size: None,
            admin_requests_per_second: None,
            admin_burst_size: None,
        };
        let limiter = create_rate_limiter(&config);
        assert!(limiter.is_some());
    }

    // ── tiered rate limiting ─────────────────────────────────────────────

    #[test]
    fn test_classify_rate_limit_tier_reads() {
        assert!(matches!(classify_rate_limit_tier(&Method::GET, "/health"), RateLimitTier::Read));
        assert!(matches!(classify_rate_limit_tier(&Method::GET, "/collections"), RateLimitTier::Read));
        assert!(matches!(classify_rate_limit_tier(&Method::HEAD, "/health"), RateLimitTier::Read));
    }

    #[test]
    fn test_classify_rate_limit_tier_writes() {
        assert!(matches!(classify_rate_limit_tier(&Method::POST, "/collections/test/vectors"), RateLimitTier::Write));
        assert!(matches!(classify_rate_limit_tier(&Method::POST, "/collections/test/search"), RateLimitTier::Write));
        assert!(matches!(classify_rate_limit_tier(&Method::DELETE, "/collections/test/vectors/v1"), RateLimitTier::Write));
    }

    #[test]
    fn test_classify_rate_limit_tier_admin() {
        assert!(matches!(classify_rate_limit_tier(&Method::POST, "/save"), RateLimitTier::Admin));
        assert!(matches!(classify_rate_limit_tier(&Method::POST, "/webhooks"), RateLimitTier::Admin));
        assert!(matches!(classify_rate_limit_tier(&Method::POST, "/collections/test/compact"), RateLimitTier::Admin));
    }

    #[test]
    fn test_create_tiered_rate_limiters() {
        let config = RateLimitConfig::default();
        assert!(create_rate_limiter(&config).is_some());
        assert!(create_write_rate_limiter(&config).is_some());
        assert!(create_admin_rate_limiter(&config).is_some());
    }
}
