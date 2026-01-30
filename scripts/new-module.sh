#!/usr/bin/env bash
set -euo pipefail

# Scaffold a new Needle service module.
#
# Usage:
#   scripts/new-module.sh <domain> <module_name>
#
# Example:
#   scripts/new-module.sh search my_feature
#
# This creates:
#   1. src/services/<domain>/<module_name>.rs  — module boilerplate
#   2. Adds `pub mod <module_name>;` to src/services/<domain>/mod.rs
#   3. Adds `pub use <domain>::<module_name>;` to src/services/mod.rs
#
# Available domains:
#   ai, client, collection, compute, embedding, governance,
#   infrastructure, observability, pipeline, plugin, search, storage, sync

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICES_DIR="$ROOT_DIR/src/services"

usage() {
    echo "Usage: $0 <domain> <module_name>"
    echo ""
    echo "Scaffold a new service module in src/services/<domain>/<module_name>.rs"
    echo ""
    echo "Available domains:"
    ls -d "$SERVICES_DIR"/*/  2>/dev/null | xargs -I{} basename {} | sort | tr '\n' ', ' | sed 's/,$/\n/'
    echo ""
    echo "Example:"
    echo "  $0 search my_query_optimizer"
    exit 1
}

if [ $# -ne 2 ]; then
    usage
fi

DOMAIN="$1"
MODULE="$2"

# Validate domain exists
if [ ! -d "$SERVICES_DIR/$DOMAIN" ]; then
    echo "Error: Domain '$DOMAIN' does not exist."
    echo "Available domains:"
    ls -d "$SERVICES_DIR"/*/  2>/dev/null | xargs -I{} basename {} | sort
    exit 1
fi

# Validate module name (snake_case)
if ! echo "$MODULE" | grep -qE '^[a-z][a-z0-9_]*$'; then
    echo "Error: Module name must be snake_case (e.g., my_feature)"
    exit 1
fi

MODULE_FILE="$SERVICES_DIR/$DOMAIN/$MODULE.rs"
DOMAIN_MOD="$SERVICES_DIR/$DOMAIN/mod.rs"
SERVICES_MOD="$SERVICES_DIR/mod.rs"

# Check if module already exists
if [ -f "$MODULE_FILE" ]; then
    echo "Error: $MODULE_FILE already exists"
    exit 1
fi

# Convert snake_case to CamelCase for struct name
STRUCT_NAME=$(echo "$MODULE" | sed -E 's/(^|_)([a-z])/\U\2/g')

# 1. Create module file
cat > "$MODULE_FILE" << RUST
//! ${STRUCT_NAME} service.

use crate::error::Result;

/// ${STRUCT_NAME} service configuration.
#[derive(Debug, Clone)]
pub struct ${STRUCT_NAME}Config {
    /// Whether the service is enabled.
    pub enabled: bool,
}

impl Default for ${STRUCT_NAME}Config {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// ${STRUCT_NAME} service.
#[derive(Debug)]
pub struct ${STRUCT_NAME} {
    config: ${STRUCT_NAME}Config,
}

impl ${STRUCT_NAME} {
    /// Create a new ${STRUCT_NAME} with the given configuration.
    pub fn new(config: ${STRUCT_NAME}Config) -> Self {
        Self { config }
    }

    /// Create a new ${STRUCT_NAME} with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(${STRUCT_NAME}Config::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ${STRUCT_NAME}Config::default();
        assert!(config.enabled);
    }

    #[test]
    fn test_new() {
        let service = ${STRUCT_NAME}::with_defaults();
        assert!(service.config.enabled);
    }
}
RUST

echo "Created $MODULE_FILE"

# 2. Add pub mod to domain mod.rs
if ! grep -q "pub mod ${MODULE};" "$DOMAIN_MOD"; then
    # Add with experimental feature gate
    echo "#[cfg(feature = \"experimental\")]" >> "$DOMAIN_MOD"
    echo "pub mod ${MODULE};" >> "$DOMAIN_MOD"
    echo "Added 'pub mod ${MODULE};' to $DOMAIN_MOD"
fi

# 3. Add re-export to services/mod.rs
if ! grep -q "pub use ${DOMAIN}::${MODULE};" "$SERVICES_MOD"; then
    echo "#[cfg(feature = \"experimental\")]" >> "$SERVICES_MOD"
    echo "pub use ${DOMAIN}::${MODULE};" >> "$SERVICES_MOD"
    echo "Added 'pub use ${DOMAIN}::${MODULE};' to $SERVICES_MOD"
fi

echo ""
echo "✓ Module scaffolded successfully!"
echo ""
echo "Next steps:"
echo "  1. Edit $MODULE_FILE to implement your service"
echo "  2. Add 'pub use services::${MODULE};' to src/lib.rs (if public)"
echo "  3. Run 'cargo check --features experimental' to verify"
