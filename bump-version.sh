#!/usr/bin/env bash
# bump-version.sh — Update version in all 3 locations:
#   mcp_speech.py, gemini-extension.json, CLAUDE.md
#
# Usage: ./bump-version.sh 4.3.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <version>"
    echo "  e.g. $0 4.3.0"
    exit 1
fi

VERSION="$1"

# Validate semver-like X.Y.Z
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: Version must be in X.Y.Z format (e.g. 4.3.0), got: $VERSION"
    exit 1
fi

# Files to update
MCP="$SCRIPT_DIR/mcp_speech.py"
EXT="$SCRIPT_DIR/gemini-extension.json"
CLAUDE="$SCRIPT_DIR/CLAUDE.md"

# Show current versions
echo "Current versions:"
grep -n '"version"' "$MCP"  | head -1
grep -n '"version"' "$EXT"  | head -1
grep -n '^Version ' "$CLAUDE" | head -1

echo ""
echo "Updating to $VERSION ..."

# 1. mcp_speech.py: "version": "X.Y.Z"
sed -i "s/\"version\": \"[0-9]\+\.[0-9]\+\.[0-9]\+\"/\"version\": \"$VERSION\"/" "$MCP"

# 2. gemini-extension.json: "version": "X.Y.Z"
sed -i "s/\"version\": \"[0-9]\+\.[0-9]\+\.[0-9]\+\"/\"version\": \"$VERSION\"/" "$EXT"

# 3. CLAUDE.md: "Version X.Y.Z." on line ~4
sed -i "s/^Version [0-9]\+\.[0-9]\+\.[0-9]\+\./Version $VERSION./" "$CLAUDE"

echo ""
echo "Updated:"
grep -n '"version"' "$MCP"  | head -1
grep -n '"version"' "$EXT"  | head -1
grep -n '^Version ' "$CLAUDE" | head -1

echo ""
echo "Done. Version bumped to $VERSION in all 3 files."
