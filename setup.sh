#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup.sh — One-shot environment setup for pika
#
# Usage:
#   bash setup.sh            # create env + install package
#   bash setup.sh --update   # update existing env from pika.yml
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="pika"
ENV_FILE="${SCRIPT_DIR}/pika.yml"

# ── Helpers ───────────────────────────────────────────────────────────────────
info()  { echo -e "\033[1;34m[setup]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[setup]\033[0m $*"; }
err()   { echo -e "\033[1;31m[setup]\033[0m $*" >&2; }

# ── Check prerequisites ──────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    err "conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
    err "pika.yml not found at $ENV_FILE"
    exit 1
fi

# ── Create or update conda env ────────────────────────────────────────────────
if conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
    if [[ "${1:-}" == "--update" ]]; then
        info "Updating existing '${ENV_NAME}' env from pika.yml ..."
        conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
    else
        info "Conda env '${ENV_NAME}' already exists. Use --update to refresh."
    fi
else
    info "Creating conda env '${ENV_NAME}' from pika.yml ..."
    conda env create -f "$ENV_FILE"
fi

# ── Install the package in editable mode ──────────────────────────────────────
info "Installing pika in editable mode ..."
conda run -n "$ENV_NAME" --no-capture-output \
    pip install -e "${SCRIPT_DIR}" --no-deps 2>&1 | tail -3

# ── Quick smoke test ──────────────────────────────────────────────────────────
info "Running import smoke test ..."
conda run -n "$ENV_NAME" --no-capture-output \
    python -c "import pika; print('  ✓ pika imported')"

ok "Done. Activate the env with:  conda activate ${ENV_NAME}"

# to update the conda env (removes already installed pika and arrow depndencies.)
# conda env export -n pika --no-builds \
#   | sed '/^  - arrow-cpp=/d; /^  - libabseil=/d; /^  - libgrpc=/d; /^  - libprotobuf=/d; /^  - libre2-11=/d; /^      - pika==/d' \
#   > pika.yml