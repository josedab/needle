# Roadmap

This document outlines the planned development direction for Needle. Items are organized by priority and expected milestone, though timelines may shift based on community feedback and priorities.

## Current Status

Needle is currently in **v0.1.x** (pre-1.0), focusing on API stabilization and core feature completion.

## Upcoming: v0.2.0

### Stability & Polish
- [ ] API stabilization and documentation review
- [ ] Performance benchmarking and optimization pass
- [ ] Upgrade ONNX Runtime (`ort`) to stable release
- [ ] Comprehensive error message improvements

### Testing & Quality
- [ ] Increase test coverage to 80%+
- [ ] Add more property-based tests for edge cases
- [ ] Stress testing under high load
- [ ] Memory leak detection and profiling

## Future: v0.3.0

### Performance
- [ ] Improved SIMD support (AVX-512)
- [ ] GPU acceleration improvements
- [ ] Query planning and optimization
- [ ] Connection pooling for server mode

### Features
- [ ] GraphQL API option
- [ ] Improved streaming and pagination
- [ ] Custom distance function plugins
- [ ] Vector compression improvements

## Future: v1.0.0

### Stability Guarantees
- [ ] Stable public API with semver guarantees
- [ ] Stable file format with migration tooling
- [ ] Long-term support commitment
- [ ] Production deployment guides

### Enterprise Readiness
- [ ] SOC 2 compliance documentation
- [ ] High availability deployment patterns
- [ ] Disaster recovery documentation
- [ ] Performance SLA guidelines

## Exploration (No Timeline)

These items are under consideration but not yet planned:

- **Vector Streaming**: Real-time vector ingestion from message queues
- **Federated Search**: Cross-instance search coordination
- **Vector Compression Research**: New quantization techniques
- **Hardware Acceleration**: TPU/specialized hardware support
- **Managed Service**: Cloud-hosted Needle offering

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Priority areas for contributions:
- Documentation improvements
- Test coverage expansion
- Performance optimizations
- Language binding improvements

## Feedback

Have suggestions for the roadmap? Please:
1. Open a [GitHub Issue](https://github.com/anthropics/needle/issues) for feature requests
2. Join discussions in existing roadmap-related issues
3. Submit PRs for items you'd like to work on
