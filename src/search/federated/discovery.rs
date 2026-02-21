use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::instance::{HealthStatus, InstanceConfig, InstanceRegistry};

/// Configuration for instance auto-discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// How often instances should send heartbeats.
    pub heartbeat_interval: Duration,
    /// After how many missed heartbeats an instance is marked unhealthy.
    pub missed_heartbeat_threshold: u32,
    /// Whether to automatically remove stale instances.
    pub auto_remove_stale: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(10),
            missed_heartbeat_threshold: 3,
            auto_remove_stale: true,
        }
    }
}

/// Tracks heartbeat state per instance.
#[derive(Debug, Clone)]
struct HeartbeatState {
    last_seen: Instant,
    missed_count: u32,
    metadata: HashMap<String, String>,
}

/// Service that manages instance discovery through heartbeats.
pub struct DiscoveryService {
    registry: Arc<InstanceRegistry>,
    config: DiscoveryConfig,
    heartbeats: RwLock<HashMap<String, HeartbeatState>>,
}

impl DiscoveryService {
    /// Create a new discovery service.
    pub fn new(registry: Arc<InstanceRegistry>, config: DiscoveryConfig) -> Self {
        Self {
            registry,
            config,
            heartbeats: RwLock::new(HashMap::new()),
        }
    }

    /// Record a heartbeat from an instance. Auto-registers unknown instances.
    pub fn heartbeat(&self, instance_id: &str, endpoint: &str, metadata: HashMap<String, String>) {
        // Register if unknown
        if self.registry.get(instance_id).is_none() {
            let mut inst = InstanceConfig::new(instance_id, endpoint);
            if let Some(region) = metadata.get("region") {
                inst = inst.with_region(region);
            }
            self.registry.register(inst);
        }
        self.registry
            .update_health(instance_id, HealthStatus::Healthy);

        let mut hb = self.heartbeats.write();
        hb.insert(
            instance_id.to_string(),
            HeartbeatState {
                last_seen: Instant::now(),
                missed_count: 0,
                metadata,
            },
        );
    }

    /// Check all instances for missed heartbeats.
    pub fn check_heartbeats(&self) -> Vec<String> {
        let threshold = self.config.heartbeat_interval * self.config.missed_heartbeat_threshold;
        let mut stale = Vec::new();
        let mut hb = self.heartbeats.write();

        for (id, state) in hb.iter_mut() {
            if state.last_seen.elapsed() > threshold {
                state.missed_count += 1;
                self.registry.update_health(id, HealthStatus::Unhealthy);
                stale.push(id.clone());
            }
        }

        if self.config.auto_remove_stale {
            for id in &stale {
                if let Some(state) = hb.get(id) {
                    if state.missed_count > self.config.missed_heartbeat_threshold * 2 {
                        self.registry.unregister(id);
                        // Will be cleaned up from hb map below
                    }
                }
            }
            hb.retain(|id, state| {
                state.missed_count <= self.config.missed_heartbeat_threshold * 2
                    || self.registry.get(id).is_some()
            });
        }

        stale
    }

    /// Number of tracked instances.
    pub fn tracked_count(&self) -> usize {
        self.heartbeats.read().len()
    }
}

// ---------------------------------------------------------------------------
// Gossip-Based Peer Discovery Protocol
// ---------------------------------------------------------------------------

/// Gossip protocol configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipConfig {
    /// Number of random peers to contact per gossip round.
    pub fanout: usize,
    /// Gossip interval.
    pub interval: Duration,
    /// How long until a peer is considered suspect.
    pub suspect_timeout: Duration,
    /// How long until a suspect peer is considered dead.
    pub dead_timeout: Duration,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            fanout: 3,
            interval: Duration::from_secs(1),
            suspect_timeout: Duration::from_secs(5),
            dead_timeout: Duration::from_secs(30),
        }
    }
}

/// State of a peer in the gossip protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeerState {
    /// Peer is alive and responsive.
    Alive,
    /// Peer missed a probe — asking others to confirm.
    Suspect,
    /// Peer is confirmed dead.
    Dead,
    /// Peer has voluntarily left.
    Left,
}

/// Information about a peer node in the cluster.
#[derive(Debug, Clone)]
pub struct PeerInfo {
    /// Unique peer identifier.
    pub id: String,
    /// Network endpoint.
    pub endpoint: String,
    /// Current state in the gossip protocol.
    pub state: PeerState,
    /// Logical clock (incarnation number for consistency).
    pub incarnation: u64,
    /// Collections hosted on this peer.
    pub collections: Vec<String>,
    /// Region/zone for locality-aware routing.
    pub region: Option<String>,
    /// Last time this peer's state was updated.
    pub last_updated: Instant,
}

/// A gossip message exchanged between peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GossipMessage {
    /// Periodic ping with sender's view of the cluster.
    Ping {
        sender: String,
        #[serde(skip)]
        members: Vec<GossipMemberState>,
    },
    /// Acknowledgement to a ping.
    Ack {
        sender: String,
        #[serde(skip)]
        members: Vec<GossipMemberState>,
    },
    /// Request another peer to probe a suspect.
    PingReq {
        sender: String,
        target: String,
    },
    /// Voluntary leave notification.
    Leave {
        sender: String,
    },
}

/// Serializable member state for gossip exchange.
#[derive(Debug, Clone)]
pub struct GossipMemberState {
    pub id: String,
    pub endpoint: String,
    pub state: PeerState,
    pub incarnation: u64,
    pub collections: Vec<String>,
    pub region: Option<String>,
}

/// SWIM-inspired gossip protocol for decentralized peer discovery.
///
/// Nodes discover each other transitively: when node A tells node B about
/// node C, all three become aware of each other. This eliminates the need
/// for a central coordinator or static configuration.
pub struct GossipProtocol {
    /// Local node's ID.
    local_id: String,
    /// Local node's endpoint.
    local_endpoint: String,
    /// Current incarnation number.
    incarnation: u64,
    /// Known peers.
    peers: RwLock<HashMap<String, PeerInfo>>,
    /// Configuration.
    config: GossipConfig,
    /// Registry to sync peer state with.
    registry: Arc<InstanceRegistry>,
    /// Collections hosted locally.
    local_collections: Vec<String>,
    /// Local region.
    local_region: Option<String>,
}

impl GossipProtocol {
    /// Create a new gossip protocol instance.
    pub fn new(
        local_id: impl Into<String>,
        local_endpoint: impl Into<String>,
        registry: Arc<InstanceRegistry>,
        config: GossipConfig,
    ) -> Self {
        Self {
            local_id: local_id.into(),
            local_endpoint: local_endpoint.into(),
            incarnation: 0,
            peers: RwLock::new(HashMap::new()),
            config,
            registry,
            local_collections: Vec::new(),
            local_region: None,
        }
    }

    /// Set the collections this node hosts.
    pub fn set_collections(&mut self, collections: Vec<String>) {
        self.local_collections = collections;
    }

    /// Set the region for this node.
    pub fn set_region(&mut self, region: impl Into<String>) {
        self.local_region = Some(region.into());
    }

    /// Add a seed peer to bootstrap the gossip protocol.
    pub fn add_seed(&self, id: impl Into<String>, endpoint: impl Into<String>) {
        let id = id.into();
        let endpoint = endpoint.into();
        let mut peers = self.peers.write();
        peers.insert(
            id.clone(),
            PeerInfo {
                id: id.clone(),
                endpoint: endpoint.clone(),
                state: PeerState::Alive,
                incarnation: 0,
                collections: Vec::new(),
                region: None,
                last_updated: Instant::now(),
            },
        );
        // Also register in the federated search registry
        self.registry
            .register(InstanceConfig::new(&id, &endpoint));
        self.registry.update_health(&id, HealthStatus::Healthy);
    }

    /// Process an incoming gossip message and return the response.
    pub fn handle_message(&self, msg: GossipMessage) -> Option<GossipMessage> {
        match msg {
            GossipMessage::Ping { sender, members } => {
                self.merge_members(&members);
                self.mark_alive(&sender);
                Some(GossipMessage::Ack {
                    sender: self.local_id.clone(),
                    members: self.member_states(),
                })
            }
            GossipMessage::Ack { sender, members } => {
                self.merge_members(&members);
                self.mark_alive(&sender);
                None
            }
            GossipMessage::PingReq { sender: _, target } => {
                // Probe the target on behalf of the sender
                self.mark_alive(&target);
                None
            }
            GossipMessage::Leave { sender } => {
                self.mark_left(&sender);
                None
            }
        }
    }

    /// Generate a ping message to send to a random peer.
    pub fn create_ping(&self) -> GossipMessage {
        GossipMessage::Ping {
            sender: self.local_id.clone(),
            members: self.member_states(),
        }
    }

    /// Generate a leave message for graceful shutdown.
    pub fn create_leave(&self) -> GossipMessage {
        GossipMessage::Leave {
            sender: self.local_id.clone(),
        }
    }

    /// Select random peers for the next gossip round.
    pub fn select_gossip_targets(&self) -> Vec<PeerInfo> {
        let peers = self.peers.read();
        let alive: Vec<PeerInfo> = peers
            .values()
            .filter(|p| p.state == PeerState::Alive)
            .cloned()
            .collect();

        if alive.len() <= self.config.fanout {
            return alive;
        }

        // Deterministic selection based on incarnation for reproducibility
        let mut selected: Vec<PeerInfo> = Vec::new();
        let n = alive.len();
        let step = n / self.config.fanout.max(1);
        for i in 0..self.config.fanout {
            let idx = (i * step + self.incarnation as usize) % n;
            selected.push(alive[idx].clone());
        }
        selected
    }

    /// Run a gossip protocol tick: check for suspect/dead peers.
    pub fn tick(&self) -> GossipTickResult {
        let mut newly_suspect = Vec::new();
        let mut newly_dead = Vec::new();
        let mut peers = self.peers.write();

        for peer in peers.values_mut() {
            let elapsed = peer.last_updated.elapsed();
            match peer.state {
                PeerState::Alive if elapsed > self.config.suspect_timeout => {
                    peer.state = PeerState::Suspect;
                    self.registry.update_health(&peer.id, HealthStatus::Degraded);
                    newly_suspect.push(peer.id.clone());
                }
                PeerState::Suspect if elapsed > self.config.dead_timeout => {
                    peer.state = PeerState::Dead;
                    self.registry
                        .update_health(&peer.id, HealthStatus::Unhealthy);
                    newly_dead.push(peer.id.clone());
                }
                _ => {}
            }
        }

        // Remove dead peers from registry
        for id in &newly_dead {
            self.registry.unregister(id);
        }

        GossipTickResult {
            newly_suspect,
            newly_dead,
            alive_count: peers.values().filter(|p| p.state == PeerState::Alive).count(),
            total_known: peers.len(),
        }
    }

    /// Get the current view of all known peers.
    pub fn peers(&self) -> Vec<PeerInfo> {
        self.peers.read().values().cloned().collect()
    }

    /// Get the number of alive peers.
    pub fn alive_count(&self) -> usize {
        self.peers
            .read()
            .values()
            .filter(|p| p.state == PeerState::Alive)
            .count()
    }

    fn mark_alive(&self, id: &str) {
        let mut peers = self.peers.write();
        if let Some(peer) = peers.get_mut(id) {
            peer.state = PeerState::Alive;
            peer.last_updated = Instant::now();
            self.registry.update_health(id, HealthStatus::Healthy);
        }
    }

    fn mark_left(&self, id: &str) {
        let mut peers = self.peers.write();
        if let Some(peer) = peers.get_mut(id) {
            peer.state = PeerState::Left;
            peer.last_updated = Instant::now();
        }
        self.registry.unregister(id);
    }

    fn merge_members(&self, members: &[GossipMemberState]) {
        let mut peers = self.peers.write();
        for member in members {
            if member.id == self.local_id {
                continue;
            }
            let should_update = match peers.get(&member.id) {
                Some(existing) => member.incarnation > existing.incarnation,
                None => true,
            };
            if should_update {
                let info = PeerInfo {
                    id: member.id.clone(),
                    endpoint: member.endpoint.clone(),
                    state: member.state,
                    incarnation: member.incarnation,
                    collections: member.collections.clone(),
                    region: member.region.clone(),
                    last_updated: Instant::now(),
                };
                peers.insert(member.id.clone(), info);

                // Sync with registry
                if member.state == PeerState::Alive {
                    if self.registry.get(&member.id).is_none() {
                        let mut config = InstanceConfig::new(&member.id, &member.endpoint);
                        if let Some(ref region) = member.region {
                            config = config.with_region(region);
                        }
                        for coll in &member.collections {
                            config = config.with_collection(coll);
                        }
                        self.registry.register(config);
                    }
                    self.registry
                        .update_health(&member.id, HealthStatus::Healthy);
                }
            }
        }
    }

    fn member_states(&self) -> Vec<GossipMemberState> {
        let peers = self.peers.read();
        let mut states: Vec<GossipMemberState> = peers
            .values()
            .map(|p| GossipMemberState {
                id: p.id.clone(),
                endpoint: p.endpoint.clone(),
                state: p.state,
                incarnation: p.incarnation,
                collections: p.collections.clone(),
                region: p.region.clone(),
            })
            .collect();

        // Include self
        states.push(GossipMemberState {
            id: self.local_id.clone(),
            endpoint: self.local_endpoint.clone(),
            state: PeerState::Alive,
            incarnation: self.incarnation,
            collections: self.local_collections.clone(),
            region: self.local_region.clone(),
        });

        states
    }
}

/// Result of a gossip protocol tick.
#[derive(Debug, Clone)]
pub struct GossipTickResult {
    /// Peers that became suspect.
    pub newly_suspect: Vec<String>,
    /// Peers that were declared dead.
    pub newly_dead: Vec<String>,
    /// Number of currently alive peers.
    pub alive_count: usize,
    /// Total known peers (including dead/left).
    pub total_known: usize,
}
