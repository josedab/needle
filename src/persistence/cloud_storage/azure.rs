//! Azure Blob Storage backend.

use crate::error::Result;
#[cfg(feature = "cloud-storage-azure")]
use crate::error::NeedleError;
use super::common::MockStorage;
use super::config::{ConnectionPool, RetryPolicy, StorageBackend, StorageConfig};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;

#[cfg(feature = "cloud-storage-azure")]
use std::sync::Arc;
#[cfg(feature = "cloud-storage-azure")]
use azure_storage::StorageCredentials;
#[cfg(feature = "cloud-storage-azure")]
use azure_storage_blobs::prelude::*;

/// Azure Blob-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureBlobConfig {
    /// Azure storage account name.
    pub account_name: String,
    /// Container name.
    pub container: String,
    /// Account key (or use managed identity).
    pub account_key: Option<String>,
    /// Connection string (alternative to account name/key).
    pub connection_string: Option<String>,
    /// General storage config.
    pub storage: StorageConfig,
}

impl Default for AzureBlobConfig {
    fn default() -> Self {
        Self {
            account_name: "needlestorage".to_string(),
            container: "vectors".to_string(),
            account_key: None,
            connection_string: None,
            storage: StorageConfig::default(),
        }
    }
}

/// Azure Blob Storage backend with real SDK integration.
///
/// When the `cloud-storage-azure` feature is enabled, this uses the real Azure SDK.
/// Otherwise, it falls back to an in-memory mock for testing.
pub struct AzureBlobBackend {
    /// Configuration.
    config: AzureBlobConfig,
    /// Connection pool.
    pool: ConnectionPool,
    /// Retry policy for transient failures (reserved for future use).
    _retry_policy: RetryPolicy,
    /// Real Azure container client (when feature is enabled).
    #[cfg(feature = "cloud-storage-azure")]
    container_client: Option<ContainerClient>,
    /// In-memory storage for testing/fallback.
    mock: MockStorage,
}

impl AzureBlobBackend {
    /// Create a new Azure Blob backend with account key authentication.
    #[cfg(feature = "cloud-storage-azure")]
    pub fn new_with_account_key(config: AzureBlobConfig, account_key: &str) -> Result<Self> {
        let storage_credentials = StorageCredentials::access_key(
            config.account_name.clone(),
            account_key.to_string(),
        );

        let service_client = BlobServiceClient::new(
            config.account_name.clone(),
            storage_credentials,
        );

        let container_client = service_client.container_client(&config.container);

        let mock = MockStorage::new(
            "Azure blob",
            format!("https://{}.blob.core.windows.net/{}/", config.account_name, config.container),
        );
        Ok(Self {
            pool: ConnectionPool::from_storage_config(&config.storage),
            _retry_policy: RetryPolicy::from_storage_config(&config.storage),
            config,
            container_client: Some(container_client),
            mock,
        })
    }

    /// Create a new Azure Blob backend with access key.
    #[cfg(feature = "cloud-storage-azure")]
    pub fn new_with_access_key(config: AzureBlobConfig, access_key: String) -> Result<Self> {
        let storage_credentials = StorageCredentials::access_key(
            config.account_name.clone(),
            access_key,
        );

        let service_client = BlobServiceClient::new(
            config.account_name.clone(),
            storage_credentials,
        );

        let container_client = service_client.container_client(&config.container);

        let mock = MockStorage::new(
            "Azure blob",
            format!("https://{}.blob.core.windows.net/{}/", config.account_name, config.container),
        );
        Ok(Self {
            pool: ConnectionPool::from_storage_config(&config.storage),
            _retry_policy: RetryPolicy::from_storage_config(&config.storage),
            config,
            container_client: Some(container_client),
            mock,
        })
    }

    /// Create a new Azure Blob backend with default Azure credentials.
    ///
    /// Uses Azure Identity's default credential chain:
    /// - Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
    /// - Azure CLI credentials
    /// - Managed Identity (when running on Azure)
    #[cfg(feature = "cloud-storage-azure")]
    pub async fn new_with_default_credentials(config: AzureBlobConfig) -> Result<Self> {
        use azure_identity::TokenCredentialOptions;

        let credential = azure_identity::DefaultAzureCredential::create(TokenCredentialOptions::default())
            .map_err(|e| NeedleError::Io(std::io::Error::other(format!("Azure credential error: {}", e))))?;
        let storage_credentials = StorageCredentials::token_credential(Arc::new(credential));

        let service_client = BlobServiceClient::new(
            config.account_name.clone(),
            storage_credentials,
        );

        let container_client = service_client.container_client(&config.container);

        let mock = MockStorage::new(
            "Azure blob",
            format!("https://{}.blob.core.windows.net/{}/", config.account_name, config.container),
        );
        Ok(Self {
            pool: ConnectionPool::from_storage_config(&config.storage),
            _retry_policy: RetryPolicy::from_storage_config(&config.storage),
            config,
            container_client: Some(container_client),
            mock,
        })
    }

    /// Create a mock Azure Blob backend for testing (no real Azure connection).
    pub fn new(config: AzureBlobConfig) -> Self {
        let mock = MockStorage::new(
            "Azure blob",
            format!("https://{}.blob.core.windows.net/{}/", config.account_name, config.container),
        );
        Self {
            pool: ConnectionPool::from_storage_config(&config.storage),
            _retry_policy: RetryPolicy::from_storage_config(&config.storage),
            config,
            #[cfg(feature = "cloud-storage-azure")]
            container_client: None,
            mock,
        }
    }

    /// Get container name.
    pub fn container(&self) -> &str {
        &self.config.container
    }

    /// Get account name.
    pub fn account_name(&self) -> &str {
        &self.config.account_name
    }

    /// Check if connected to real Azure.
    #[cfg(feature = "cloud-storage-azure")]
    pub fn is_connected(&self) -> bool {
        self.container_client.is_some()
    }

    /// Check if connected to real Azure (always false without feature).
    #[cfg(not(feature = "cloud-storage-azure"))]
    pub fn is_connected(&self) -> bool {
        false
    }
}

impl StorageBackend for AzureBlobBackend {
    fn read<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;

            #[cfg(feature = "cloud-storage-azure")]
            if let Some(ref container_client) = self.container_client {
                // Real Azure implementation
                let blob_client = container_client.blob_client(key);

                let response = blob_client
                    .get_content()
                    .await
                    .map_err(|e| {
                        let err_str = e.to_string();
                        if err_str.contains("404") || err_str.contains("BlobNotFound") || err_str.contains("not found") {
                            NeedleError::NotFound(format!("Azure blob '{}' not found", key))
                        } else {
                            NeedleError::Io(std::io::Error::other(format!("Azure get_content error: {}", e)))
                        }
                    })?;

                return Ok(response);
            }

            // Fallback to in-memory storage (mock mode)
            self.mock.read(key)
        })
    }

    fn write<'a>(&'a self, key: &'a str, data: &'a [u8]) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;

            #[cfg(feature = "cloud-storage-azure")]
            if let Some(ref container_client) = self.container_client {
                // Real Azure implementation
                let blob_client = container_client.blob_client(key);

                blob_client
                    .put_block_blob(data.to_vec())
                    .await
                    .map_err(|e| NeedleError::Io(std::io::Error::other(format!("Azure put_block_blob error: {}", e))))?;

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

            #[cfg(feature = "cloud-storage-azure")]
            if let Some(ref container_client) = self.container_client {
                // Real Azure implementation - ignore "not found" errors for idempotency
                let blob_client = container_client.blob_client(key);

                match blob_client.delete().await {
                    Ok(_) => return Ok(()),
                    Err(e) => {
                        let err_str = e.to_string();
                        if err_str.contains("404") || err_str.contains("BlobNotFound") {
                            return Ok(()); // Idempotent delete
                        }
                        return Err(NeedleError::Io(std::io::Error::other(format!("Azure delete error: {}", e))));
                    }
                }
            }

            // Fallback to in-memory storage (mock mode)
            self.mock.delete(key);
            Ok(())
        })
    }

    fn list<'a>(&'a self, prefix: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>> {
        #[cfg(feature = "cloud-storage-azure")]
        let container_client = self.container_client.clone();
        let _prefix_owned = prefix.to_string();

        Box::pin(async move {
            let _conn = self.pool.acquire()?;

            #[cfg(feature = "cloud-storage-azure")]
            if let Some(container_client) = container_client {
                // Real Azure implementation with pagination
                use futures::StreamExt;

                let mut keys = Vec::new();
                let mut stream = container_client
                    .list_blobs()
                    .prefix(prefix_owned)
                    .into_stream();

                while let Some(result) = stream.next().await {
                    let response = result
                        .map_err(|e| NeedleError::Io(std::io::Error::other(format!("Azure list_blobs error: {}", e))))?;

                    for blob in response.blobs.blobs() {
                        keys.push(blob.name.clone());
                    }
                }

                return Ok(keys);
            }

            // Fallback to in-memory storage (mock mode)
            Ok(self.mock.list(prefix))
        })
    }

    fn exists<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;

            #[cfg(feature = "cloud-storage-azure")]
            if let Some(ref container_client) = self.container_client {
                // Real Azure implementation - use get_properties to check existence
                let blob_client = container_client.blob_client(key);

                match blob_client.get_properties().await {
                    Ok(_) => return Ok(true),
                    Err(e) => {
                        let err_str = e.to_string();
                        if err_str.contains("404") || err_str.contains("BlobNotFound") {
                            return Ok(false);
                        }
                        return Err(NeedleError::Io(std::io::Error::other(format!("Azure get_properties error: {}", e))));
                    }
                }
            }

            // Fallback to in-memory storage (mock mode)
            Ok(self.mock.exists(key))
        })
    }
}
