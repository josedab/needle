# Governance

This document describes the governance model for the Needle project.

## Project Structure

Needle is maintained by [Anthropic](https://anthropic.com) with contributions from the open source community.

### Roles

**Maintainers** have full commit access and are responsible for:
- Reviewing and merging pull requests
- Triaging issues
- Making release decisions
- Setting project direction

**Contributors** are anyone who submits a pull request, opens an issue, or participates in discussions.

### Decision Making

- **Day-to-day decisions** (bug fixes, minor features, documentation) are made by any maintainer.
- **Significant changes** (new features, API changes, architecture decisions) are discussed in GitHub Issues or Discussions before implementation.
- **Breaking changes** require at least two maintainer approvals and must be documented in the CHANGELOG.

### Releases

- Releases follow [Semantic Versioning](https://semver.org/).
- The project is currently pre-1.0; breaking changes may occur in minor versions with advance notice.
- Release candidates are tagged for community testing before stable releases.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting changes.

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.
