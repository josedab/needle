#!/usr/bin/env bash
# Verify that markdown links in documentation files resolve to existing files.
# Usage: ./scripts/verify-docs.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BROKEN_FILE=$(mktemp)
echo 0 > "$BROKEN_FILE"

check_file() {
    local file="$1"
    local dir
    dir="$(dirname "$file")"

    # Extract markdown links: [text](path)
    local links
    links=$(grep -oE '\[[^]]*\]\([^)]+\)' "$file" 2>/dev/null | sed 's/.*](//' | sed 's/)$//' || true)

    while IFS= read -r link; do
        [ -z "$link" ] && continue
        # Skip external URLs and mailto
        [[ "$link" =~ ^https?:// ]] && continue
        [[ "$link" =~ ^mailto: ]] && continue
        # Strip anchor fragment
        local path="${link%%#*}"
        [ -z "$path" ] && continue

        # Resolve relative to file's directory
        local resolved
        if [[ "$path" == /* ]]; then
            resolved="$ROOT/$path"
        else
            resolved="$dir/$path"
        fi

        if [ ! -e "$resolved" ]; then
            echo "BROKEN: $file -> $link"
            echo "        (resolved: $resolved)"
            echo 1 > "$BROKEN_FILE"
        fi
    done <<< "$links"
}

# Files to check
DOC_FILES=(
    "$ROOT/README.md"
    "$ROOT/CONTRIBUTING.md"
    "$ROOT/QUICKSTART.md"
    "$ROOT/ARCHITECTURE.md"
    "$ROOT/CHANGELOG.md"
    "$ROOT/SECURITY.md"
    "$ROOT/ROADMAP.md"
)

# Add docs/*.md if any exist
for f in "$ROOT"/docs/*.md; do
    [ -f "$f" ] && DOC_FILES+=("$f")
done

echo "Checking markdown links..."
echo ""

for file in "${DOC_FILES[@]}"; do
    if [ -f "$file" ]; then
        check_file "$file"
    else
        echo "SKIP: $file (not found)"
    fi
done

echo ""
BROKEN=$(cat "$BROKEN_FILE")
rm -f "$BROKEN_FILE"
if [ "$BROKEN" -ne 0 ]; then
    echo "❌ Found broken links. Fix them or remove stale references."
    exit 1
else
    echo "✅ All markdown links resolve correctly."
fi
