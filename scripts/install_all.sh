#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " Hull-Keel Optimization: Full Install"
echo "=========================================="

# ── 1. System dependencies ───────────────────────────────────────────────
echo ""
echo "[1/5] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libopenmpi-dev \
    openmpi-bin \
    libcgal-dev \
    libboost-all-dev \
    libeigen3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxt-dev \
    libxcursor-dev \
    libxrandr-dev \
    freeglut3-dev \
    libgmp-dev \
    libmpfr-dev \
    zlib1g-dev \
    libncurses-dev \
    flex \
    bison \
    libreadline-dev \
    libfftw3-dev

# ── 2. Python environment ────────────────────────────────────────────────
echo ""
echo "[2/5] Setting up Python environment..."

# Use conda if available, otherwise create a venv
if command -v conda &>/dev/null && [ -f "$(dirname $(which conda))/../envs/boat_env" ]; then
    echo "Found conda environment 'boat_env' — activating and installing deps"
    eval "$(conda shell.bash hook)"
    conda activate boat_env
    pip install -r "$SCRIPT_DIR/requirements.txt"
elif [ -d "/home/anon/miniconda3" ]; then
    echo "Found miniconda at /home/anon/miniconda3 — using it"
    export PATH="/home/anon/miniconda3/bin:$PATH"
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "Creating fresh venv at .venv"
    python3 -m venv "$SCRIPT_DIR/.venv"
    source "$SCRIPT_DIR/.venv/bin/activate"
    pip install --upgrade pip setuptools wheel
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# ── 3. OpenFOAM (latest from openfoam.com repo) ──────────────────────────
echo ""
echo "[3/5] Installing OpenFOAM..."

OF_VERSION="2512"
OF_PACKAGE="openfoam${OF_VERSION}"
OF_DIR="/usr/lib/openfoam/${OF_PACKAGE}"
OF_SYMLINK="/opt/${OF_PACKAGE}"
OF_ENV="${OF_SYMLINK}/etc/bashrc"

if [ -f "$OF_ENV" ]; then
    echo "OpenFOAM ${OF_VERSION} already installed at ${OF_DIR}"
else
    echo "Adding OpenFOAM repository and installing v${OF_VERSION}..."
    wget -q -O - https://dl.openfoam.com/add-debian-repo.sh | sudo bash
    sudo apt-get update
    sudo apt-get install -y "${OF_PACKAGE}-dev" "${OF_PACKAGE}"
    sudo mkdir -p /opt
    sudo ln -sf "$OF_DIR" "$OF_SYMLINK"
    echo "OpenFOAM v${OF_VERSION} installed at ${OF_SYMLINK}"
fi

# ── 4. DualSPHysics v5.4 ─────────────────────────────────────────────────
echo ""
echo "[4/5] Installing DualSPHysics v5.4..."

DS_DIR="/opt/dualsphysics/5.4"
DS_BIN_DIR="${DS_DIR}/bin/linux"
DS_LOCAL="/home/anon/DualSPHysics"

if [ -f "${DS_BIN_DIR}/DualSPHysics5.4_linux64" ]; then
    echo "DualSPHysics already installed at ${DS_DIR}"
elif [ -d "$DS_LOCAL" ]; then
    echo "Copying DualSPHysics from ${DS_LOCAL}..."
    sudo mkdir -p "$DS_BIN_DIR"
    sudo cp -v "${DS_LOCAL}/bin/linux/DualSPHysics5.4_linux64" "${DS_BIN_DIR}/"
    sudo cp -v "${DS_LOCAL}/bin/linux/GenCase_linux64" "${DS_BIN_DIR}/"
    sudo cp -v "${DS_LOCAL}/bin/linux/IsoSurface_linux64" "${DS_BIN_DIR}/"
    sudo cp -v "${DS_LOCAL}/bin/linux/MeasureTool_linux64" "${DS_BIN_DIR}/"
    sudo cp -v "${DS_LOCAL}/bin/linux/PartVTK_linux64" "${DS_BIN_DIR}/"
    sudo cp -v "${DS_LOCAL}/bin/linux/PartVTKOut_linux64" "${DS_BIN_DIR}/"
    sudo cp -v "${DS_LOCAL}/bin/linux/libdsphchrono.so" "${DS_BIN_DIR}/"
    sudo cp -v "${DS_LOCAL}/bin/linux/libChronoEngine.so" "${DS_BIN_DIR}/"
    sudo chmod +x "${DS_BIN_DIR}/"*
    echo "DualSPHysics installed at ${DS_DIR}"
else
    echo "WARNING: DualSPHysics not found at ${DS_LOCAL}"
    echo "  Download from https://dual.sphysics.org/ and extract to ${DS_DIR}"
fi

# Ensure DS XML templates reference the right binary location
DS_CONFIG="${DS_BIN_DIR}/DsphConfig.xml"
if [ -f "$DS_CONFIG" ]; then
    sudo cp -v "${DS_LOCAL}/bin/linux/DsphConfig.xml" "${DS_BIN_DIR}/" 2>/dev/null || true
fi

# ── 5. Verify installation ───────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Installation Summary"
echo "=========================================="

echo ""
echo "Python:"
if command -v conda &>/dev/null; then
    python --version || echo "  NOT FOUND"
else
    python3 --version || echo "  NOT FOUND"
fi

echo ""
echo "Pipeline dependencies:"
python -c "
try:
    import torch; print(f'  PyTorch {torch.__version__}')
except: print('  PyTorch NOT FOUND')
try:
    import botorch; print(f'  BoTorch {botorch.__version__}')
except: print('  BoTorch NOT FOUND')
try:
    import capytaine; print(f'  Capytaine {capytaine.__version__}')
except: print('  Capytaine NOT FOUND')
try:
    import trimesh; print('  trimesh ok')
except: print('  trimesh NOT FOUND')
try:
    import ray; print(f'  Ray {ray.__version__}')
except: print('  Ray NOT FOUND')
" 2>/dev/null || echo "  (check python env)"

echo ""
echo "OpenFOAM:"
if [ -f "$OF_ENV" ]; then
    echo "  OpenFOAM v${OF_VERSION} at ${OF_DIR}"
    echo "  Env file: ${OF_ENV}"
    bash -c "source ${OF_ENV} && which interFoam" 2>/dev/null && echo "  interFoam available" || echo "  interFoam not in PATH (source env first)"
else
    echo "  NOT FOUND"
fi

echo ""
echo "DualSPHysics:"
if [ -f "${DS_BIN_DIR}/DualSPHysics5.4_linux64" ]; then
    echo "  Binary: ${DS_BIN_DIR}/DualSPHysics5.4_linux64"
    file "${DS_BIN_DIR}/DualSPHysics5.4_linux64" 2>/dev/null | grep -q "ELF" && echo "  Valid ELF binary" || echo "  Check binary"
else
    echo "  NOT FOUND"
fi

echo ""
echo "=========================================="
echo " To activate the environment:"
echo "   export PATH=\"/home/anon/miniconda3/bin:\$PATH\""
echo ""
echo " To run the pipeline:"
echo "   python run_optimization.py --config config.yaml"
echo "=========================================="
