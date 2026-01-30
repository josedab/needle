//! Cluster Bootstrap & Discovery
//!
//! Node discovery, cluster formation, health monitoring, and membership
//! management for distributed Needle deployments.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::cluster_bootstrap::{ClusterManager, NodeInfo, ClusterConfig};
//!
//! let mut mgr = ClusterManager::new(ClusterConfig::new("node-1", "127.0.0.1:8080"));
//! mgr.add_seed("127.0.0.1:8081");
//! mgr.add_seed("127.0.0.1:8082");
//! let health = mgr.health();
//! println!("Cluster: {} nodes, leader: {:?}", health.total_nodes, health.leader);
//! ```

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use crate::error::{NeedleError, Result};

/// Node info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String, pub address: String, pub role: NodeRole,
    pub status: NodeStatus, pub joined_at: u64, pub last_heartbeat: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole { Leader, Follower, Candidate, Observer }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus { Healthy, Suspect, Down, Joining }

/// Cluster configuration.
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub node_id: String, pub address: String,
    pub seeds: Vec<String>, pub heartbeat_interval: Duration,
    pub failure_threshold: Duration,
}

impl ClusterConfig {
    pub fn new(id: &str, addr: &str) -> Self {
        Self { node_id: id.into(), address: addr.into(), seeds: Vec::new(),
            heartbeat_interval: Duration::from_secs(1), failure_threshold: Duration::from_secs(5) }
    }
}

/// Cluster health.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterHealth {
    pub total_nodes: usize, pub healthy_nodes: usize, pub leader: Option<String>,
    pub term: u64, pub cluster_id: String,
}

/// Cluster manager.
pub struct ClusterManager {
    config: ClusterConfig, nodes: HashMap<String, NodeInfo>, term: u64,
}

impl ClusterManager {
    pub fn new(config: ClusterConfig) -> Self {
        let mut nodes = HashMap::new();
        let now = now_secs();
        nodes.insert(config.node_id.clone(), NodeInfo {
            id: config.node_id.clone(), address: config.address.clone(),
            role: NodeRole::Leader, status: NodeStatus::Healthy,
            joined_at: now, last_heartbeat: now,
        });
        Self { config, nodes, term: 1 }
    }

    pub fn add_seed(&mut self, address: &str) {
        let id = format!("node-{}", self.nodes.len());
        let now = now_secs();
        self.nodes.insert(id.clone(), NodeInfo {
            id, address: address.into(), role: NodeRole::Follower,
            status: NodeStatus::Joining, joined_at: now, last_heartbeat: now,
        });
    }

    pub fn heartbeat(&mut self, node_id: &str) -> Result<()> {
        let node = self.nodes.get_mut(node_id).ok_or_else(|| NeedleError::NotFound(format!("Node {node_id}")))?;
        node.last_heartbeat = now_secs();
        node.status = NodeStatus::Healthy;
        Ok(())
    }

    pub fn health(&self) -> ClusterHealth {
        let healthy = self.nodes.values().filter(|n| n.status == NodeStatus::Healthy).count();
        let leader = self.nodes.values().find(|n| n.role == NodeRole::Leader).map(|n| n.id.clone());
        ClusterHealth { total_nodes: self.nodes.len(), healthy_nodes: healthy, leader, term: self.term,
            cluster_id: format!("cluster-{}", self.config.node_id) }
    }

    pub fn nodes(&self) -> Vec<&NodeInfo> { self.nodes.values().collect() }
    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn remove_node(&mut self, id: &str) -> bool { self.nodes.remove(id).is_some() }
}

fn now_secs() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_formation() {
        let mut mgr = ClusterManager::new(ClusterConfig::new("n1", "127.0.0.1:8080"));
        mgr.add_seed("127.0.0.1:8081");
        mgr.add_seed("127.0.0.1:8082");
        assert_eq!(mgr.node_count(), 3);
        let h = mgr.health();
        assert!(h.leader.is_some());
    }

    #[test]
    fn test_heartbeat() {
        let mut mgr = ClusterManager::new(ClusterConfig::new("n1", "addr"));
        mgr.heartbeat("n1").unwrap();
        assert_eq!(mgr.health().healthy_nodes, 1);
    }

    #[test]
    fn test_remove_node() {
        let mut mgr = ClusterManager::new(ClusterConfig::new("n1", "addr"));
        mgr.add_seed("other");
        assert!(mgr.remove_node("node-1"));
        assert_eq!(mgr.node_count(), 1);
    }
}
