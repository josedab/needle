use needle::Result;

use crate::cli::commands::FederateCommands;

pub fn federate_command(cmd: FederateCommands) -> Result<()> {
    match cmd {
        FederateCommands::Search {
            query,
            collection,
            k,
            instances,
            routing,
            merge,
        } => federate_search(&query, &collection, k, &instances, &routing, &merge),
        FederateCommands::Health { instances } => federate_health(&instances),
        FederateCommands::Stats { instances } => federate_stats(&instances),
    }
}

fn federate_search(
    query_str: &str,
    collection: &str,
    k: usize,
    instances_str: &str,
    routing: &str,
    merge: &str,
) -> Result<()> {
    use needle::federated::{
        Federation, FederationConfig, InstanceConfig, MergeStrategy, RoutingStrategy,
    };

    let query: Vec<f32> = query_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if query.is_empty() {
        eprintln!("Invalid query vector. Use comma-separated floats.");
        return Ok(());
    }

    let instance_urls: Vec<&str> = instances_str.split(',').map(|s| s.trim()).collect();

    if instance_urls.is_empty() {
        eprintln!("No instances specified.");
        return Ok(());
    }

    let routing_strategy = match routing.to_lowercase().as_str() {
        "latency-aware" | "latency" => RoutingStrategy::LatencyAware,
        "round-robin" | "roundrobin" => RoutingStrategy::RoundRobin,
        "geographic" | "geo" => RoutingStrategy::GeographicProximity,
        _ => RoutingStrategy::Broadcast,
    };

    let merge_strategy = match merge.to_lowercase().as_str() {
        "rrf" | "reciprocal" => MergeStrategy::ReciprocalRankFusion,
        "consensus" => MergeStrategy::Consensus,
        "first" => MergeStrategy::FirstResponse,
        _ => MergeStrategy::DistanceBased,
    };

    let config = FederationConfig::default()
        .with_routing(routing_strategy)
        .with_merge(merge_strategy);

    let federation = Federation::new(config);

    for (i, url) in instance_urls.iter().enumerate() {
        let instance_config = InstanceConfig::new(format!("instance-{}", i), *url);
        federation.register_instance(instance_config);
    }

    println!("Federated Search");
    println!("================");
    println!();
    println!("Query: {} dimensions", query.len());
    println!("Collection: {}", collection);
    println!("K: {}", k);
    println!("Instances: {}", instance_urls.len());
    println!("Routing: {:?}", routing_strategy);
    println!("Merge: {:?}", merge_strategy);
    println!();

    println!("Note: Federated search requires the 'server' feature and running instances.");
    println!("      Use 'needle serve' to start instances, then use this command to query them.");
    println!();
    println!("Configured instances:");
    for url in &instance_urls {
        println!("  - {}", url);
    }

    Ok(())
}

fn federate_health(instances_str: &str) -> Result<()> {
    use needle::federated::{Federation, FederationConfig, InstanceConfig};

    let instance_urls: Vec<&str> = instances_str.split(',').map(|s| s.trim()).collect();

    let config = FederationConfig::default();
    let federation = Federation::new(config);

    for (i, url) in instance_urls.iter().enumerate() {
        let instance_config = InstanceConfig::new(format!("instance-{}", i), *url);
        federation.register_instance(instance_config);
    }

    let health = federation.health();

    println!("Federation Health Status");
    println!("========================");
    println!();
    println!("Overall: {:?}", health.status);
    println!(
        "Healthy instances: {}/{}",
        health.healthy_instances, health.total_instances
    );
    println!("Degraded instances: {}", health.degraded_instances);
    println!("Unhealthy instances: {}", health.unhealthy_instances);
    println!("Average latency: {:.2} ms", health.avg_latency_ms);

    Ok(())
}

fn federate_stats(instances_str: &str) -> Result<()> {
    use needle::federated::{Federation, FederationConfig, InstanceConfig};

    let instance_urls: Vec<&str> = instances_str.split(',').map(|s| s.trim()).collect();

    let config = FederationConfig::default();
    let federation = Federation::new(config);

    for (i, url) in instance_urls.iter().enumerate() {
        let instance_config = InstanceConfig::new(format!("instance-{}", i), *url);
        federation.register_instance(instance_config);
    }

    let stats = federation.stats();

    println!("Federation Statistics");
    println!("=====================");
    println!();
    println!("Total queries: {}", stats.total_queries);
    println!("Failed queries: {}", stats.failed_queries);
    println!("Partial results: {}", stats.partial_results);
    println!("Timeouts: {}", stats.timeouts);

    Ok(())
}
