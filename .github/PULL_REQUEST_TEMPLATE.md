## Summary

Brief description of the changes in this PR.

## Related Issues

Fixes #(issue number)

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring (no functional changes)
- [ ] CI/CD changes
- [ ] Dependency update

## Changes Made

- Change 1
- Change 2
- Change 3

## Testing

### How has this been tested?

Describe the tests you ran to verify your changes:

- [ ] Unit tests
- [ ] Integration tests
- [ ] Property-based tests (proptest)
- [ ] Manual testing

### Test environment:

- **OS**:
- **Rust version**:
- **Feature flags tested**:

## Checklist

### Code Quality
- [ ] My code follows the project's style guidelines
- [ ] I have run `cargo fmt` and `cargo clippy`
- [ ] I have added tests that prove my fix/feature works
- [ ] All new and existing tests pass (`cargo test --features full`)
- [ ] I have updated documentation as needed

### Security
- [ ] I have not introduced any new `unsafe` code (or have documented safety invariants)
- [ ] I have not introduced any security vulnerabilities (see OWASP top 10)
- [ ] I have not committed any secrets or credentials

### Breaking Changes
- [ ] This PR does not contain breaking changes
- [ ] OR: Breaking changes are documented below and in CHANGELOG.md

## Breaking Changes (if applicable)

Describe any breaking changes and migration steps:

```rust
// Before
old_api_call();

// After
new_api_call();
```

## Performance Impact (if applicable)

- [ ] This change has no performance impact
- [ ] OR: Benchmark results are provided below

```
Benchmark results here
```

## Screenshots (if applicable)

Add screenshots for UI changes or visual improvements.

## Additional Notes

Any additional information that reviewers should know.
