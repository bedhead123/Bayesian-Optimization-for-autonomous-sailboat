#!/usr/bin/env bash
set -e
# Hull-Keel Optimization — one-command bootstrap
# Usage: ./run.sh [args]
#   ./run.sh --dry-run      # validate setup (default if no args)
#   ./run.sh --quick-test   # 5 LHS + 2 BO
#   ./run.sh --hyper-test   # 3 LHS smoke
#   ./run.sh                # full optimization (config.yaml)
# Creates ./venv on first run, installs deps, then execs run_optimization.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# --- 1. Create venv if missing ---
if [ ! -x "$PYTHON" ]; then
    echo "[bootstrap] Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "[bootstrap] Upgrading pip ..."
    "$PIP" install --upgrade pip setuptools wheel -q
fi

# --- 2. Install / verify deps ---
NEED_INSTALL=0
if ! "$PYTHON" -c "import torch, botorch, capytaine, trimesh, ray, fast_simplification" 2>/dev/null; then
    NEED_INSTALL=1
fi
# also check requirements.txt newer than venv
if [ "$SCRIPT_DIR/requirements.txt" -nt "$VENV_DIR/.installed" ]; then
    NEED_INSTALL=1
fi

if [ "$NEED_INSTALL" -eq 1 ]; then
    echo "[bootstrap] Installing Python deps (this may take 2-5 min first run) ..."
    # Auto-detect CUDA GPU → install CUDA torch, else CPU torch (saves 2GB on headless boxes)
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo "[bootstrap] CUDA GPU detected ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)) — installing torch CUDA (cu130)"
        "$PIP" install --quiet torch --index-url https://download.pytorch.org/whl/cu130 || \
        "$PIP" install --quiet torch --index-url https://download.pytorch.org/whl/cpu || true
    else
        echo "[bootstrap] No CUDA GPU — installing torch CPU"
        "$PIP" install --quiet torch --index-url https://download.pytorch.org/whl/cpu || true
    fi
    "$PIP" install --quiet -r "$SCRIPT_DIR/requirements.txt"
    touch "$VENV_DIR/.installed"
    echo "[bootstrap] Deps installed."
else
    echo "[bootstrap] Venv ready — deps OK."
    # Self-heal: venv had CPU torch but host now shows GPU → swap to CUDA build
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        if "$PYTHON" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            : # CUDA already working
        else
            if "$PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "+cpu"; then
                echo "[bootstrap] GPU present but venv torch is CPU-only — upgrading to CUDA (cu130) ..."
                "$PIP" install --quiet torch --index-url https://download.pytorch.org/whl/cu130 || true
                touch "$VENV_DIR/.installed"
                echo "[bootstrap] Torch CUDA installed — GPU will be used next run."
            fi
        fi
    fi
fi

# --- 3. Ensure output dir exists ---
mkdir -p "$SCRIPT_DIR/output"

# --- 4. Default to --dry-run if no args ---
if [ $# -eq 0 ]; then
    echo "[bootstrap] No args — defaulting to --dry-run (use --quick-test / --hyper-test / --help for other modes)"
    set -- --dry-run
fi

exec "$PYTHON" "$SCRIPT_DIR/run_optimization.py" "$@"
