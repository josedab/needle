//! One-Click Cloud Deploy Templates
//!
//! Generates deployment configuration files for Fly.io, Railway, and Render
//! with pre-configured monitoring, health checks, and auto-scaling rules.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::cloud_deploy::{
//!     DeployGenerator, Platform, DeployConfig,
//! };
//!
//! let gen = DeployGenerator::new(DeployConfig::default());
//!
//! let fly_config = gen.generate(Platform::FlyIo);
//! println!("{}", fly_config.content);
//!
//! let railway_config = gen.generate(Platform::Railway);
//! println!("{}", railway_config.content);
//! ```

use serde::{Deserialize, Serialize};

// ── Platform ─────────────────────────────────────────────────────────────────

/// Supported deployment platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Platform {
    FlyIo,
    Railway,
    Render,
    Docker,
    AwsEcs,
    GcpCloudRun,
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FlyIo => write!(f, "fly.io"),
            Self::Railway => write!(f, "railway"),
            Self::Render => write!(f, "render"),
            Self::Docker => write!(f, "docker"),
            Self::AwsEcs => write!(f, "aws-ecs"),
            Self::GcpCloudRun => write!(f, "gcp-cloud-run"),
        }
    }
}

// ── Deploy Configuration ─────────────────────────────────────────────────────

/// Deployment configuration.
#[derive(Debug, Clone)]
pub struct DeployConfig {
    /// Application name.
    pub app_name: String,
    /// Region.
    pub region: String,
    /// Memory limit in MB.
    pub memory_mb: usize,
    /// CPU count.
    pub cpus: usize,
    /// Port to listen on.
    pub port: u16,
    /// Database file path.
    pub db_path: String,
    /// Enable metrics endpoint.
    pub metrics: bool,
    /// Auto-scaling min instances.
    pub min_instances: usize,
    /// Auto-scaling max instances.
    pub max_instances: usize,
    /// Health check path.
    pub health_path: String,
}

impl Default for DeployConfig {
    fn default() -> Self {
        Self {
            app_name: "needle-db".into(),
            region: "iad".into(),
            memory_mb: 512,
            cpus: 1,
            port: 8080,
            db_path: "/data/vectors.needle".into(),
            metrics: true,
            min_instances: 1,
            max_instances: 3,
            health_path: "/health".into(),
        }
    }
}

impl DeployConfig {
    /// Set app name.
    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.app_name = name.into();
        self
    }

    /// Set memory.
    #[must_use]
    pub fn with_memory(mut self, mb: usize) -> Self {
        self.memory_mb = mb;
        self
    }

    /// Set region.
    #[must_use]
    pub fn with_region(mut self, region: &str) -> Self {
        self.region = region.into();
        self
    }
}

// ── Generated Config ─────────────────────────────────────────────────────────

/// A generated deployment configuration file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedConfig {
    /// Platform.
    pub platform: String,
    /// File name.
    pub filename: String,
    /// File content.
    pub content: String,
    /// Additional files (e.g., Dockerfile).
    pub additional_files: Vec<(String, String)>,
}

// ── Monitoring Config ────────────────────────────────────────────────────────

/// Generated monitoring configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Prometheus scrape config.
    pub prometheus_config: String,
    /// Grafana dashboard JSON.
    pub grafana_dashboard: String,
}

// ── Deploy Generator ─────────────────────────────────────────────────────────

/// Generates deployment configs for various platforms.
pub struct DeployGenerator {
    config: DeployConfig,
}

impl DeployGenerator {
    /// Create a new generator.
    pub fn new(config: DeployConfig) -> Self {
        Self { config }
    }

    /// Generate config for a platform.
    pub fn generate(&self, platform: Platform) -> GeneratedConfig {
        match platform {
            Platform::FlyIo => self.generate_fly(),
            Platform::Railway => self.generate_railway(),
            Platform::Render => self.generate_render(),
            Platform::Docker => self.generate_docker(),
            Platform::AwsEcs => self.generate_aws_ecs(),
            Platform::GcpCloudRun => self.generate_gcp_cloud_run(),
        }
    }

    /// Generate monitoring config.
    pub fn monitoring(&self) -> MonitoringConfig {
        MonitoringConfig {
            prometheus_config: format!(
                "scrape_configs:\n  - job_name: needle\n    static_configs:\n      - targets: ['localhost:{}']\n    metrics_path: /metrics\n    scrape_interval: 15s\n",
                self.config.port
            ),
            grafana_dashboard: serde_json::json!({
                "dashboard": {
                    "title": "Needle Vector DB",
                    "panels": [
                        {"title": "QPS", "type": "graph", "targets": [{"expr": "rate(needle_operations_total[5m])"}]},
                        {"title": "Latency p99", "type": "graph", "targets": [{"expr": "histogram_quantile(0.99, needle_operation_duration_seconds_bucket)"}]},
                        {"title": "Vector Count", "type": "stat", "targets": [{"expr": "needle_collection_vectors_total"}]},
                    ]
                }
            }).to_string(),
        }
    }

    /// List all supported platforms.
    pub fn platforms() -> Vec<Platform> {
        vec![Platform::FlyIo, Platform::Railway, Platform::Render, Platform::Docker, Platform::AwsEcs, Platform::GcpCloudRun]
    }

    fn generate_fly(&self) -> GeneratedConfig {
        let content = format!(
            r#"# Fly.io deployment config for Needle
app = "{app}"
primary_region = "{region}"

[build]
  dockerfile = "Dockerfile"

[env]
  NEEDLE_DB_PATH = "{db}"
  NEEDLE_HOST = "0.0.0.0"
  NEEDLE_PORT = "{port}"
  RUST_LOG = "info"

[http_service]
  internal_port = {port}
  force_https = true
  auto_stop_machines = false
  auto_start_machines = true
  min_machines_running = {min}

[[vm]]
  memory = "{mem}mb"
  cpus = {cpus}

[checks]
  [checks.health]
    port = {port}
    type = "http"
    interval = "10s"
    timeout = "5s"
    path = "{health}"

[mounts]
  source = "needle_data"
  destination = "/data"
"#,
            app = self.config.app_name,
            region = self.config.region,
            db = self.config.db_path,
            port = self.config.port,
            min = self.config.min_instances,
            mem = self.config.memory_mb,
            cpus = self.config.cpus,
            health = self.config.health_path,
        );

        GeneratedConfig {
            platform: "fly.io".into(),
            filename: "fly.toml".into(),
            content,
            additional_files: Vec::new(),
        }
    }

    fn generate_railway(&self) -> GeneratedConfig {
        let content = format!(
            r#"{{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {{
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  }},
  "deploy": {{
    "startCommand": "needle serve -a 0.0.0.0:{port} -d {db}",
    "healthcheckPath": "{health}",
    "healthcheckTimeout": 5,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }}
}}"#,
            port = self.config.port,
            db = self.config.db_path,
            health = self.config.health_path,
        );

        GeneratedConfig {
            platform: "railway".into(),
            filename: "railway.json".into(),
            content,
            additional_files: Vec::new(),
        }
    }

    fn generate_render(&self) -> GeneratedConfig {
        let content = format!(
            r#"services:
  - type: web
    name: {app}
    runtime: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: NEEDLE_DB_PATH
        value: {db}
      - key: NEEDLE_PORT
        value: "{port}"
    healthCheckPath: {health}
    disk:
      name: needle-data
      mountPath: /data
      sizeGB: 10
    scaling:
      minInstances: {min}
      maxInstances: {max}
      targetMemoryPercent: 80
"#,
            app = self.config.app_name,
            db = self.config.db_path,
            port = self.config.port,
            health = self.config.health_path,
            min = self.config.min_instances,
            max = self.config.max_instances,
        );

        GeneratedConfig {
            platform: "render".into(),
            filename: "render.yaml".into(),
            content,
            additional_files: Vec::new(),
        }
    }

    fn generate_docker(&self) -> GeneratedConfig {
        let content = format!(
            r#"version: "3.8"
services:
  needle:
    image: ghcr.io/anthropics/needle:latest
    ports:
      - "{port}:{port}"
    environment:
      NEEDLE_DB_PATH: {db}
      NEEDLE_HOST: "0.0.0.0"
      NEEDLE_PORT: "{port}"
    volumes:
      - needle_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{port}{health}"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

volumes:
  needle_data:
"#,
            port = self.config.port,
            db = self.config.db_path,
            health = self.config.health_path,
        );

        GeneratedConfig {
            platform: "docker".into(),
            filename: "docker-compose.yml".into(),
            content,
            additional_files: Vec::new(),
        }
    }

    fn generate_aws_ecs(&self) -> GeneratedConfig {
        let content = format!(
            r#"AWSTemplateFormatVersion: '2010-09-09'
Description: Needle Vector DB on AWS ECS Fargate

Resources:
  Cluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: {app}-cluster

  TaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: {app}
      Cpu: '{cpus}024'
      Memory: '{mem}'
      NetworkMode: awsvpc
      RequiresCompatibilities: [FARGATE]
      ContainerDefinitions:
        - Name: needle
          Image: ghcr.io/anthropics/needle:latest
          PortMappings:
            - ContainerPort: {port}
          Environment:
            - Name: NEEDLE_DB_PATH
              Value: {db}
            - Name: NEEDLE_PORT
              Value: '{port}'
          HealthCheck:
            Command: ["CMD-SHELL", "curl -f http://localhost:{port}{health} || exit 1"]
            Interval: 10
            Timeout: 5
            Retries: 3
          LogConfiguration:
            LogDriver: awslogs
            Options:
              awslogs-group: /ecs/{app}
              awslogs-region: {region}
              awslogs-stream-prefix: needle

  Service:
    Type: AWS::ECS::Service
    Properties:
      Cluster: !Ref Cluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: {min}
      LaunchType: FARGATE
"#,
            app = self.config.app_name,
            cpus = self.config.cpus,
            mem = self.config.memory_mb,
            port = self.config.port,
            db = self.config.db_path,
            health = self.config.health_path,
            region = self.config.region,
            min = self.config.min_instances,
        );

        GeneratedConfig {
            platform: "aws-ecs".into(),
            filename: "cloudformation.yaml".into(),
            content,
            additional_files: Vec::new(),
        }
    }

    fn generate_gcp_cloud_run(&self) -> GeneratedConfig {
        let content = format!(
            r#"apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: {app}
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "{min}"
        autoscaling.knative.dev/maxScale: "{max}"
        run.googleapis.com/memory: "{mem}Mi"
        run.googleapis.com/cpu: "{cpus}"
    spec:
      containers:
        - image: ghcr.io/anthropics/needle:latest
          ports:
            - containerPort: {port}
          env:
            - name: NEEDLE_DB_PATH
              value: {db}
            - name: NEEDLE_PORT
              value: "{port}"
          livenessProbe:
            httpGet:
              path: {health}
              port: {port}
            initialDelaySeconds: 5
            periodSeconds: 10
          resources:
            limits:
              memory: {mem}Mi
              cpu: "{cpus}"
"#,
            app = self.config.app_name,
            min = self.config.min_instances,
            max = self.config.max_instances,
            mem = self.config.memory_mb,
            cpus = self.config.cpus,
            port = self.config.port,
            db = self.config.db_path,
            health = self.config.health_path,
        );

        GeneratedConfig {
            platform: "gcp-cloud-run".into(),
            filename: "service.yaml".into(),
            content,
            additional_files: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fly_generation() {
        let gen = DeployGenerator::new(DeployConfig::default());
        let config = gen.generate(Platform::FlyIo);
        assert_eq!(config.filename, "fly.toml");
        assert!(config.content.contains("needle-db"));
        assert!(config.content.contains("8080"));
        assert!(config.content.contains("/health"));
    }

    #[test]
    fn test_railway_generation() {
        let gen = DeployGenerator::new(DeployConfig::default());
        let config = gen.generate(Platform::Railway);
        assert_eq!(config.filename, "railway.json");
        assert!(config.content.contains("healthcheckPath"));
    }

    #[test]
    fn test_render_generation() {
        let gen = DeployGenerator::new(DeployConfig::default());
        let config = gen.generate(Platform::Render);
        assert_eq!(config.filename, "render.yaml");
        assert!(config.content.contains("needle-db"));
    }

    #[test]
    fn test_docker_generation() {
        let gen = DeployGenerator::new(DeployConfig::default());
        let config = gen.generate(Platform::Docker);
        assert!(config.content.contains("docker-compose") || config.content.contains("services:"));
    }

    #[test]
    fn test_monitoring() {
        let gen = DeployGenerator::new(DeployConfig::default());
        let mon = gen.monitoring();
        assert!(mon.prometheus_config.contains("scrape_configs"));
        assert!(mon.grafana_dashboard.contains("Needle"));
    }

    #[test]
    fn test_custom_config() {
        let config = DeployConfig::default().with_name("my-app").with_memory(1024);
        let gen = DeployGenerator::new(config);
        let fly = gen.generate(Platform::FlyIo);
        assert!(fly.content.contains("my-app"));
        assert!(fly.content.contains("1024mb"));
    }

    #[test]
    fn test_all_platforms() {
        let platforms = DeployGenerator::platforms();
        assert_eq!(platforms.len(), 6);
    }

    #[test]
    fn test_aws_ecs_generation() {
        let gen = DeployGenerator::new(DeployConfig::default());
        let config = gen.generate(Platform::AwsEcs);
        assert_eq!(config.filename, "cloudformation.yaml");
        assert!(config.content.contains("ECS"));
        assert!(config.content.contains("Fargate"));
    }

    #[test]
    fn test_gcp_cloud_run_generation() {
        let gen = DeployGenerator::new(DeployConfig::default());
        let config = gen.generate(Platform::GcpCloudRun);
        assert_eq!(config.filename, "service.yaml");
        assert!(config.content.contains("knative"));
        assert!(config.content.contains("needle-db"));
    }
}
