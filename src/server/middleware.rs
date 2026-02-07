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

pub(super) fn build_cors_layer(config: &CorsConfig) -> CorsLayer {
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


pub(super) fn create_rate_limiter(config: &RateLimitConfig) -> Option<Arc<PerIpRateLimiter>> {
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


pub(super) fn extract_client_ip(request: &Request<Body>, trusted_proxies: &[IpAddr]) -> IpAddr {
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


pub(super) async fn rate_limit_middleware(
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
            // Auto-enforce collection-level RBAC when auth is required
            if let Some(collection) = extract_collection_from_path(&path) {
                let permission = infer_permission_from_request(&request);
                let resource = crate::security::Resource::Collection(collection);
                if !context.user.has_permission(permission, &resource) {
                    let error = ApiError::new(
                        format!("Insufficient permissions on collection"),
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
pub(super) async fn security_headers_middleware(
    request: Request<Body>,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(SECURITY_HEADER_XCTO.0.clone(), SECURITY_HEADER_XCTO.1.parse().unwrap());
    headers.insert(SECURITY_HEADER_XFO.0.clone(), SECURITY_HEADER_XFO.1.parse().unwrap());
    headers.insert(
        header::STRICT_TRANSPORT_SECURITY,
        "max-age=63072000; includeSubDomains".parse().unwrap(),
    );
    headers.insert(
        header::HeaderName::from_static("x-xss-protection"),
        "1; mode=block".parse().unwrap(),
    );
    response
}
