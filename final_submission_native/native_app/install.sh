#!/bin/sh
set -eu

APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
PYTHON_BIN=${PYTHON:-python3.12}

usage() {
    printf '%s\n' 'Usage: install.sh' \
        'Creates a local .venv, installs pinned CPU inference dependencies,' \
        'verifies the frozen model, and installs the user-wide Git hook.'
}

[ "$#" -eq 1 ] && case "$1" in --help|-h) usage; exit 0 ;; esac
[ "$#" -eq 0 ] || { usage >&2; exit 2; }

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "ERROR: Python 3.12 is required (or set PYTHON to a Python 3.12 executable)." >&2
    exit 1
}
command -v git >/dev/null 2>&1 || { echo "ERROR: Git is required." >&2; exit 1; }

if [ ! -x "$APP_ROOT/.venv/bin/python" ]; then
    "$PYTHON_BIN" -m venv "$APP_ROOT/.venv"
fi
VENV_PYTHON="$APP_ROOT/.venv/bin/python"
"$VENV_PYTHON" -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
"$VENV_PYTHON" -m pip install -r "$APP_ROOT/requirements-model.txt"
unset HF_HUB_CACHE TRANSFORMERS_CACHE
CODEBERT_PYTHON="$VENV_PYTHON" sh "$APP_ROOT/hook/install_global.sh"

echo "Native CodeBERT Git detector is ready at $APP_ROOT"
