---
name: Bug Report
about: Report a bug to help us improve Needle
title: '[BUG] '
labels: bug
assignees: ''
---

## Description

A clear and concise description of the bug.

## Steps to Reproduce

1. Create a database with '...'
2. Insert vectors with '...'
3. Search with '...'
4. See error

## Expected Behavior

A clear and concise description of what you expected to happen.

## Actual Behavior

A clear and concise description of what actually happened.

## Error Messages

```
Paste any error messages or stack traces here
```

## Minimal Reproduction

```rust
// Provide minimal code to reproduce the issue
use needle::Database;

fn main() -> needle::Result<()> {
    // Your reproduction code here
    Ok(())
}
```

## Environment

- **Needle Version**: [e.g., 0.1.0]
- **Rust Version**: [e.g., 1.75.0]
- **OS**: [e.g., Ubuntu 22.04, macOS 14.0, Windows 11]
- **Architecture**: [e.g., x86_64, ARM64]
- **Feature Flags**: [e.g., `server,metrics,hybrid`]

## Additional Context

Add any other context about the problem here, such as:

- Database size (number of vectors, dimensions)
- Memory usage observations
- Performance characteristics
- Whether it's intermittent or consistent

## Checklist

- [ ] I have searched existing issues to ensure this bug hasn't been reported
- [ ] I have tested with the latest version of Needle
- [ ] I have provided a minimal reproduction example
- [ ] I have included all relevant environment information
