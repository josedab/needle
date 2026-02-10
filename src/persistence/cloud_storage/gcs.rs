//! Google Cloud Storage backend.

use crate::error::Result;
#[cfg(feature = "cloud-storage-gcs")]
use crate::error::NeedleError;
use super::common::MockStorage;
use super::config::{ConnectionPool, RetryPolicy, StorageBackend, StorageConfig};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;

#[cfg(feature = "cloud-storage-gcs")]
use google_cloud_storage::{
    client::{Client as GcsClient, ClientConfig as GcsClientConfig},
    http::objects::{
        download::Range as GcsRange,
        get::GetObjectRequest,
        upload::{Media, UploadObjectRequest, UploadType},
        delete::DeleteObjectRequest,
        list::ListObjectsRequest,
    },
};

/// GCS-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCSConfig {
    /// GCP project ID.
    pub project_id: String,
    /// GCS bucket name.
    pub bucket: String,
    /// Path to service account credentials JSON.
    pub credentials_path: Option<String>,
    /// General storage config.
    pub storage: StorageConfig,
}

impl Default for GCSConfig {
    fn default() -> Self {
        Self {
            project_id: "my-project".to_string(),
            bucket: "needle-vectors".to_string(),
            credentials_path: None,
            storage: StorageConfig::default(),
        }
    }
}

/// Google Cloud Storage backend with real SDK integration.
///
/// When the `cloud-storage-gcs` feature is enabled, this uses the real GCS SDK.
/// Otherwise, it falls back to an in-memory mock for testing.
pub struct GCSBackend {
    /// Configuration.
    config: GCSConfig,
    /// Connection pool.
    pool: ConnectionPool,
    /// Retry policy for transient failures (reserved for future use).
    _retry_policy: RetryPolicy,
    /// Real GCS client (when feature is enabled).
    #[cfg(feature = "cloud-storage-gcs")]
    client: Option<GcsClient>,
    /// In-memory storage for testing/fallback.
    mock: MockStorage,
}

impl GCSBackend {
    /// Create a new GCS backend with default credentials.
    ///
    /// Uses Google Cloud's default credential provider:
    /// - GOOGLE_APPLICATION_CREDENTIALS environment variable
    /// - Application default credentials
    /// - GCE metadata service (when running on GCP)
    #[cfg(feature = "cloud-storage-gcs")]
    pub async fn new_with_default_credentials(config: GCSConfig) -> Result<Self> {
        let gcs_config = GcsClientConfig::default()
            .with_auth()
            .await
            .map_err(|e| NeedleError::Io(std::io::Error::other(format!("GCS auth error: {}", e))))?;

        let client = GcsClient::new(gcs_config);

        let mock = MockStorage::new("GCS object", format!("gs://{}/", config.bucket));
        Ok(Self {
            pool: ConnectionPool::from_storage_config(&config.storage),
            _retry_policy: RetryPolicy::from_storage_config(&config.storage),
            config,
            client: Some(client),
            mock,
        })
    }

    /// Create a new GCS backend with service account credentials from file.
    #[cfg(feature = "cloud-storage-gcs")]
    pub async fn new_with_credentials_file(config: GCSConfig, credentials_path: &str) -> Result<Self> {
        // Set the environment variable for the credentials file
        std::env::set_var("GOOGLE_APPLICATION_CREDENTIALS", credentials_path);

        let gcs_config = GcsClientConfig::default()
            .with_auth()
            .await
            .map_err(|e| NeedleError::Io(std::io::Error::other(format!("GCS auth error: {}", e))))?;

        let client = GcsClient::new(gcs_config);

        let mock = MockStorage::new("GCS object", format!("gs://{}/", config.bucket));
        Ok(Self {
            pool: ConnectionPool::from_storage_config(&config.storage),
            _retry_policy: RetryPolicy::from_storage_config(&config.storage),
            config,
            client: Some(client),
            mock,
        })
    }

    /// Create a mock GCS backend for testing (no real GCS connection).
    pub fn new(config: GCSConfig) -> Self {
        let mock = MockStorage::new("GCS object", format!("gs://{}/", config.bucket));
        Self {
            pool: ConnectionPool::from_storage_config(&config.storage),
            _retry_policy: RetryPolicy::from_storage_config(&config.storage),
            config,
            #[cfg(feature = "cloud-storage-gcs")]
            client: None,
            mock,
        }
    }

    /// Get bucket name.
    pub fn bucket(&self) -> &str {
        &self.config.bucket
    }

    /// Get project ID.
    pub fn project_id(&self) -> &str {
        &self.config.project_id
    }

    /// Check if connected to real GCS.
    #[cfg(feature = "cloud-storage-gcs")]
    pub fn is_connected(&self) -> bool {
        self.client.is_some()
    }

    /// Check if connected to real GCS (always false without feature).
    #[cfg(not(feature = "cloud-storage-gcs"))]
    pub fn is_connected(&self) -> bool {
        false
    }
}

impl StorageBackend for GCSBackend {
    fn read<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;

            #[cfg(feature = "cloud-storage-gcs")]
            if let Some(ref client) = self.client {
                // Real GCS implementation
                let data = client
                    .download_object(
                        &GetObjectRequest {
                            bucket: self.config.bucket.clone(),
                            object: key.to_string(),
                            ..Default::default()
                        },
                        &GcsRange::default(),
                    )
                    .await
                    .map_err(|e| {
                        let err_str = e.to_string();
                        if err_str.contains("404") || err_str.contains("not found") || err_str.contains("No such object") {
                            NeedleError::NotFound(format!("GCS object '{}' not found", key))
                        } else {
                            NeedleError::Io(std::io::Error::other(format!("GCS download error: {}", e)))
                        }
                    })?;

                return Ok(data);
            }

            // Fallback to in-memory storage (mock mode)
            self.mock.read(key)
        })
    }

    fn write<'a>(&'a self, key: &'a str, data: &'a [u8]) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;

            #[cfg(feature = "cloud-storage-gcs")]
            if let Some(ref client) = self.client {
                // Real GCS implementation
                let upload_type = UploadType::Simple(Media::new(key.to_string()));

                client
                    .upload_object(
                        &UploadObjectRequest {
                            bucket: self.config.bucket.clone(),
                            ..Default::default()
                        },
                        data.to_vec(),
                        &upload_type,
                    )
                    .await
                    .map_err(|e| NeedleError::Io(std::io::Error::other(format!("GCS upload error: {}", e))))?;

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

            #[cfg(feature = "cloud-storage-gcs")]
            if let Some(ref client) = self.client {
                // Real GCS implementation - ignore "not found" errors for idempotency
                let result = client
                    .delete_object(&DeleteObjectRequest {
                        bucket: self.config.bucket.clone(),
                        object: key.to_string(),
                        ..Default::default()
                    })
                    .await;

                match result {
                    Ok(_) => return Ok(()),
                    Err(e) => {
                        let err_str = e.to_string();
                        if err_str.contains("404") || err_str.contains("not found") {
                            return Ok(()); // Idempotent delete
                        }
                        return Err(NeedleError::Io(std::io::Error::other(format!("GCS delete error: {}", e))));
                    }
                }
            }

            // Fallback to in-memory storage (mock mode)
            self.mock.delete(key);
            Ok(())
        })
    }

    fn list<'a>(&'a self, prefix: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;

            #[cfg(feature = "cloud-storage-gcs")]
            if let Some(ref client) = self.client {
                // Real GCS implementation
                let objects = client
                    .list_objects(&ListObjectsRequest {
                        bucket: self.config.bucket.clone(),
                        prefix: Some(prefix.to_string()),
                        ..Default::default()
                    })
                    .await
                    .map_err(|e| NeedleError::Io(std::io::Error::other(format!("GCS list error: {}", e))))?;

                let keys: Vec<String> = objects
                    .items
                    .unwrap_or_default()
                    .into_iter()
                    .map(|obj| obj.name)
                    .collect();

                return Ok(keys);
            }

            // Fallback to in-memory storage (mock mode)
            Ok(self.mock.list(prefix))
        })
    }

    fn exists<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;

            #[cfg(feature = "cloud-storage-gcs")]
            if let Some(ref client) = self.client {
                // Real GCS implementation - use get metadata to check existence
                match client
                    .get_object(&GetObjectRequest {
                        bucket: self.config.bucket.clone(),
                        object: key.to_string(),
                        ..Default::default()
                    })
                    .await
                {
                    Ok(_) => return Ok(true),
                    Err(e) => {
                        let err_str = e.to_string();
                        if err_str.contains("404") || err_str.contains("not found") {
                            return Ok(false);
                        }
                        return Err(NeedleError::Io(std::io::Error::other(format!("GCS get_object error: {}", e))));
                    }
                }
            }

            // Fallback to in-memory storage (mock mode)
            Ok(self.mock.exists(key))
        })
    }
}
