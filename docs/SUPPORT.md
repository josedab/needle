# Support — Needle Vector Database

Thank you for using Needle! Here's how to get help.

## 📖 Documentation

- **[README](../README.md)** — Quick start, features, and examples
- **[QUICKSTART.md](../QUICKSTART.md)** — Get up and running in 2 minutes
- **[API Reference](api-reference.md)** — Complete API documentation
- **[How-To Guides](how-to-guides.md)** — Task-oriented guides
- **[Architecture](../ARCHITECTURE.md)** — System design and internals

## 💬 Community

- **[GitHub Discussions](https://github.com/anthropics/needle/discussions)** — Ask questions, share ideas, show off what you've built
- **[Issue Tracker](https://github.com/anthropics/needle/issues)** — Report bugs or request features

## 🐛 Bug Reports

Found a bug? Please [open an issue](https://github.com/anthropics/needle/issues/new?template=bug_report.md) with:
- Needle version (`cargo run -- --version` or check `Cargo.toml`)
- Rust version (`rustc --version`)
- Operating system and architecture
- Minimal reproduction steps

## 🔒 Security Issues

**Do not report security vulnerabilities via public issues.** Please see [SECURITY.md](../SECURITY.md) for our responsible disclosure process.

## ❓ FAQ

### How do I choose between HNSW, IVF, and brute-force?
See the [Index Selection Guide](index-selection-guide.md) or use `auto_tune()` for automatic recommendations.

### What's the maximum dataset size Needle supports?
Needle is optimized for single-node deployments up to ~50M vectors. For larger datasets, consider sharding or a distributed solution.

### Can I use Needle in production?
Needle is pre-1.0 (currently v0.1.x). Core features are stable, but the API may change between minor versions. See [API Stability](api-stability.md) for tier definitions.

### How do I embed Needle in my application?
Add `needle = "0.1"` to your `Cargo.toml`. For Python, use `pip install needle-db`. For JavaScript, use the WASM package from npm.

## 📋 Before Asking for Help

1. Check the [documentation](.) for your topic
2. Search [existing issues](https://github.com/anthropics/needle/issues) and [discussions](https://github.com/anthropics/needle/discussions)
3. Try the latest version — your issue may already be fixed
4. Prepare a minimal reproduction if reporting a bug
