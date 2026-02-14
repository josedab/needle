//! AWS S3 storage backend.

use super::common::MockStorage;
use super::config::{ConnectionPool, RetryPolicy, StorageBackend, StorageConfig};
#[cfg(feature = "cloud-storage-s3")]
use crate::error::NeedleError;
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;

#[cfg(feature = "cloud-storage-s3")]
use aws_sdk_s3::{config::Region, primitives::ByteStream, Client as S3Client};

/// S3-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3Config {
    /// AWS region.
    pub region: String,
    /// S3 bucket name.
    pub bucket: String,
    /// Optional endpoint URL (for S3-compatible services).
    pub endpoint: Option<String>,
    /// Access key ID.
    pub access_key_id: Option<String>,
    /// Secret access key.
    pub secret_access_key: Option<String>,
    /// Use path-style URLs.
    pub path_style: bool,
    /// General storage config.
    pub storage: StorageConfig,
}

impl Default for S3Config {
    fn default() -> Self {
        Self {
            region: "us-east-1".to_string(),
            bucket: "needle-vectors".to_string(),
            endpoint: None,
            access_key_id: None,
            secret_access_key: None,
            path_style: false,
            storage: StorageConfig::default(),
        }
    }
}

/// AWS S3 storage backend with real SDK integration.
///
/// When the `cloud-storage-s3` feature is enabled, this uses the real AWS SDK.
/// Otherwise, it falls back to an in-memory mock for testing.
pub struct S3Backend {
    /// Configuration.
    config: S3Config,
    /// Connection pool.
    pool: ConnectionPool,
    /// Retry policy for transient failures (reserved for future use).
    _retry_policy: RetryPolicy,
    /// Real S3 client (when feature is enabled).
    #[cfg(feature = "cloud-storage-s3")]
    client: Option<S3Client>,
    /// In-memory storage for testing/fallback.
    mock: MockStorage,
}

impl S3Backend {
    /// Create a new S3 backend with default credentials from environment.
    ///
    /// Uses AWS SDK's default credential provider chain:
    /// - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    /// - Shared credentials file (~/.aws/credentials)
    /// - IAM role (when running on AWS)
    #[cfg(feature = "cloud-storage-s3")]
    pub async fn new_with_default_credentials(config: S3Config) -> Result<Self> {
        let region = Region::new(config.region.clone());

        let mut aws_config_builder =
            aws_config::defaults(aws_config::BehaviorVersion::latest()).region(region);

        // Use custom endpoint if provided (for S3-compatible services like MinIO)
        if let Some(ref endpoint) = config.endpoint {
            aws_config_builder = aws_config_builder.endpoint_url(endpoint);
        }

        let aws_config = aws_config_builder.load().await;

        let mut s3_config_builder = aws_sdk_s3::config::Builder::from(&aws_config);

        // Force path-style addressing if configured (required for some S3-compatible services)
        if config.path_style {
            s3_config_builder = s3_config_builder.force_path_style(true);
        }

        let client = S3Client::from_conf(s3_config_builder.build());

        let mock = MockStorage::new("S3 key", format!("{}/", config.bucket));
        Ok(Self {
            pool: ConnectionPool::from_storage_config(&config.storage),
            _retry_policy: RetryPolicy::from_storage_config(&config.storage),
            config,
            client: Some(client),
            mock,
        })
    }

    /// Create a new S3 backend with explicit credentials.
    #[cfg(feature = "cloud-storage-s3")]
    pub async fn new_with_credentials(
        config: S3Config,
        access_key_id: &str,
        secret_access_key: &str,
    ) -> Result<Self> {
        let region = Region::new(config.region.clone());
        let credentials = aws_sdk_s3::config::Credentials::new(
            access_key_id,
            secret_access_key,
            None, // session token
            None, // expiration
            "needle-explicit-credentials",
        );

        let mut s3_config_builder = aws_sdk_s3::config::Builder::new()
            .region(region)
            .credentials_provider(credentials);

        if let Some(ref endpoint) = config.endpoint {
            s3_config_builder = s3_config_builder.endpoint_url(endpoint);
        }

        if config.path_style {
            s3_config_builder = s3_config_builder.force_path_style(true);
        }

        let client = S3Client::from_conf(s3_config_builder.build());

        let mock = MockStorage::new("S3 key", format!("{}/", config.bucket));
        Ok(Self {
            pool: ConnectionPool::from_storage_config(&config.storage),
            _retry_policy: RetryPolicy::from_storage_config(&config.storage),
            config,
            client: Some(client),
            mock,
        })
    }

    /// Create a mock S3 backend for testing (no real S3 connection).
    pub fn new(config: S3Config) -> Self {
        let mock = MockStorage::new("S3 key", format!("{}/", config.bucket));
        Self {
            pool: ConnectionPool::from_storage_config(&config.storage),
            _retry_policy: RetryPolicy::from_storage_config(&config.storage),
            config,
            #[cfg(feature = "cloud-storage-s3")]
            client: None,
            mock,
        }
    }

    /// Get bucket name.
    pub fn bucket(&self) -> &str {
        &self.config.bucket
    }

    /// Get region.
    pub fn region(&self) -> &str {
        &self.config.region
    }

    /// Check if connected to real S3.
    #[cfg(feature = "cloud-storage-s3")]
    pub fn is_connected(&self) -> bool {
        self.client.is_some()
    }

    /// Check if connected to real S3 (always false without feature).
    #[cfg(not(feature = "cloud-storage-s3"))]
    pub fn is_connected(&self) -> bool {
        false
    }
}

impl StorageBackend for S3Backend {
    fn read<'a>(
        &'a self,
        key: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;

            #[cfg(feature = "cloud-storage-s3")]
            if let Some(ref client) = self.client {
                // Real S3 implementation
                let resp = client
                    .get_object()
                    .bucket(&self.config.bucket)
                    .key(key)
                    .send()
                    .await
                    .map_err(|e| {
                        if e.to_string().contains("NoSuchKey")
                            || e.to_string().contains("not found")
                        {
                            NeedleError::NotFound(format!("S3 key '{}' not found", key))
                        } else {
                            NeedleError::Io(std::io::Error::other(format!(
                                "S3 get_object error: {}",
                                e
                            )))
                        }
                    })?;

                let data = resp
                    .body
                    .collect()
                    .await
                    .map_err(|e| {
                        NeedleError::Io(std::io::Error::other(format!("S3 body read error: {}", e)))
                    })?
                    .into_bytes()
                    .to_vec();

                return Ok(data);
            }

            // Fallback to in-memory storage (mock mode)
            self.mock.read(key)
        })
    }

    fn write<'a>(
        &'a self,
        key: &'a str,
        data: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;

            #[cfg(feature = "cloud-storage-s3")]
            if let Some(ref client) = self.client {
                // Real S3 implementation
                let body = ByteStream::from(data.to_vec());

                client
                    .put_object()
                    .bucket(&self.config.bucket)
                    .key(key)
                    .body(body)
                    .send()
                    .await
                    .map_err(|e| {
                        NeedleError::Io(std::io::Error::other(format!(
                            "S3 put_object error: {}",
                            e
                        )))
                    })?;

                return Ok(());
            }

            // Fallback to in-memory storage (mock mode)
            self.mock.write(key, data);
            Ok(())
        })
    }

    fn delete<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;

            #[cfg(feature = "cloud-storage-s3")]
            if let Some(ref client) = self.client {
                // Real S3 implementation - delete is idempotent in S3
                client
                    .delete_object()
                    .bucket(&self.config.bucket)
                    .key(key)
                    .send()
                    .await
                    .map_err(|e| {
                        NeedleError::Io(std::io::Error::other(format!(
                            "S3 delete_object error: {}",
                            e
                        )))
                    })?;

                return Ok(());
            }

            // Fallback to in-memory storage (mock mode)
            self.mock.delete(key);
            Ok(())
        })
    }

    fn list<'a>(
        &'a self,
        prefix: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;

            #[cfg(feature = "cloud-storage-s3")]
            if let Some(ref client) = self.client {
                // Real S3 implementation with pagination
                let mut keys = Vec::new();
                let mut continuation_token: Option<String> = None;

                loop {
                    let mut request = client
                        .list_objects_v2()
                        .bucket(&self.config.bucket)
                        .prefix(prefix);

                    if let Some(token) = continuation_token {
                        request = request.continuation_token(token);
                    }

                    let resp = request.send().await.map_err(|e| {
                        NeedleError::Io(std::io::Error::other(format!(
                            "S3 list_objects_v2 error: {}",
                            e
                        )))
                    })?;

                    if let Some(contents) = resp.contents {
                        for obj in contents {
                            if let Some(key) = obj.key {
                                keys.push(key);
                            }
                        }
                    }

                    if resp.is_truncated.unwrap_or(false) {
                        continuation_token = resp.next_continuation_token;
                    } else {
                        break;
                    }
                }

                return Ok(keys);
            }

            // Fallback to in-memory storage (mock mode)
            Ok(self.mock.list(prefix))
        })
    }

    fn exists<'a>(
        &'a self,
        key: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;

            #[cfg(feature = "cloud-storage-s3")]
            if let Some(ref client) = self.client {
                // Real S3 implementation using HEAD request
                match client
                    .head_object()
                    .bucket(&self.config.bucket)
                    .key(key)
                    .send()
                    .await
                {
                    Ok(_) => return Ok(true),
                    Err(e) => {
                        // Check if it's a "not found" error
                        let err_str = e.to_string();
                        if err_str.contains("NoSuchKey")
                            || err_str.contains("404")
                            || err_str.contains("not found")
                        {
                            return Ok(false);
                        }
                        return Err(NeedleError::Io(std::io::Error::other(format!(
                            "S3 head_object error: {}",
                            e
                        ))));
                    }
                }
            }

            // Fallback to in-memory storage (mock mode)
            Ok(self.mock.exists(key))
        })
    }
}
