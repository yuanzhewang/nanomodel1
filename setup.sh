#!/bin/bash
# One-time setup script for a new Lambda instance.
# Creates symlinks from the repo to persistent NFS storage,
# and installs Python dependencies.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NFS_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== NanoModel Setup ==="
echo "Repo:        $SCRIPT_DIR"
echo "NFS root:    $NFS_ROOT"
echo ""

# Create symlinks to persistent NFS dirs
ln -sfn "$NFS_ROOT/data" "$SCRIPT_DIR/data_symlink"
ln -sfn "$NFS_ROOT/checkpoints" "$SCRIPT_DIR/checkpoints_symlink"
echo "Created symlinks:"
echo "  data_symlink        -> $NFS_ROOT/data"
echo "  checkpoints_symlink -> $NFS_ROOT/checkpoints"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "=== Setup complete ==="
