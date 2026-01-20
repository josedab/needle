# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of Needle seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@needle-db.io**

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours
- **Issue Triage**: Within 7 days
- **Fix Development**: Depends on severity (critical: 7 days, high: 14 days, medium: 30 days)
- **Public Disclosure**: Coordinated after fix is available

## Security Features

Needle includes several security features for production deployments:

### Encryption at Rest

When using the `encryption` feature, Needle encrypts all data using ChaCha20-Poly1305 authenticated encryption:

```rust
use needle::{Database, EncryptionConfig};

let config = EncryptionConfig::new(encryption_key);
let db = Database::open_encrypted("data.needle", config)?;
```

### Authentication & Authorization

The HTTP server (feature: `server`) supports multiple authentication methods:

- **API Key**: Simple token-based authentication
- **JWT**: JSON Web Tokens with HS256 signing
- **RBAC**: Role-based access control (reader/writer/admin)

### Rate Limiting

The server includes per-IP rate limiting to prevent abuse:

- Default: 1000 requests per minute per IP
- Configurable via server options

### Input Validation

- Request body size limits (default: 100MB)
- Batch operation limits (default: 10,000 items)
- Query timeout limits (default: 30 seconds)

## Security Considerations

### Feature Stability Warnings

| Feature | Stability | Security Notes |
|---------|-----------|----------------|
| `server` | Stable | Production-ready with auth/rate limiting |
| `encryption` | Stable | Uses audited ChaCha20-Poly1305 |
| `embeddings` | **Unstable** | Uses pre-release `ort` crate; not recommended for untrusted inputs |
| `wasm` | Stable | Runs in browser sandbox |

### Unsafe Code

Needle contains a limited number of `unsafe` blocks, all of which are:

1. **SIMD operations** (`src/distance.rs`): For optimized distance calculations
2. **Memory-mapped files** (`src/storage.rs`): For efficient large file access

All unsafe code is documented with safety invariants and has been reviewed for memory safety.

### Dependencies

We regularly audit dependencies using:

- `cargo audit` in CI pipeline
- Dependabot for automated updates
- Manual review of security advisories

## Best Practices

When deploying Needle in production:

1. **Enable encryption** for sensitive data at rest
2. **Use authentication** when exposing the HTTP API
3. **Set appropriate rate limits** for your use case
4. **Run with minimal features** enabled (only what you need)
5. **Keep Needle updated** to receive security patches
6. **Use TLS termination** (via reverse proxy) for HTTPS

## Security Audit

Needle has not yet undergone a formal third-party security audit. If you are interested in sponsoring an audit, please contact us.

## Acknowledgments

We would like to thank the following individuals for responsibly disclosing security issues:

*No vulnerabilities have been reported yet.*
