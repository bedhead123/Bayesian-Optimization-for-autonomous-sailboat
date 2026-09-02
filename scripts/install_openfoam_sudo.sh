#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# OpenFOAM v2512 — requires sudo (run separately on your real machine)
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "Installing OpenFOAM v2512 from openfoam.com repo..."
echo "This requires sudo access."

# Add repository
wget -q -O - https://dl.openfoam.com/add-debian-repo.sh | sudo bash

# Update and install
sudo apt-get update
sudo apt-get install -y openfoam2512-dev openfoam2512

# Create symlink at /opt/openfoam2512 (matches config.yaml)
sudo mkdir -p /opt
sudo ln -sf /usr/lib/openfoam/openfoam2512 /opt/openfoam2512

echo ""
echo "Verification:"
echo "  Source:  source /opt/openfoam2512/etc/bashrc"
echo "  Test:    which interFoam"
echo ""
echo "Done. OpenFOAM v2512 installed at /opt/openfoam2512"
