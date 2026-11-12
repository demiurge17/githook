#!/bin/sh
set -eu

APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
PYTHON=${CODEBERT_PYTHON:-$APP_ROOT/.venv/bin/python}
HOOK_SCRIPT="$APP_ROOT/hook/pre_commit.py"
CONFIG_ROOT=${XDG_CONFIG_HOME:-$HOME/.config}
INSTALL_ROOT="$CONFIG_ROOT/git/codebert-secret-hook"
HOOKS_DIR="$INSTALL_ROOT/hooks"
TARGET="$HOOKS_DIR/pre-commit"
TEMP_HOOK=

usage() {
    printf '%s\n' 'Usage: install_global.sh' \
        'Installs the detector as the user-wide Git pre-commit hook.' \
        'Refuses to replace an existing global core.hooksPath or pre-commit hook.'
}

case "${1:-}" in
    "") ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

command -v git >/dev/null 2>&1 || { echo "ERROR: Git is required." >&2; exit 1; }
[ -x "$PYTHON" ] || { echo "ERROR: Application Python is unavailable; run install.sh first." >&2; exit 1; }
[ -f "$HOOK_SCRIPT" ] || { echo "ERROR: Native hook implementation is missing." >&2; exit 1; }

existing_global=$(git config --global --get-all core.hooksPath 2>/dev/null || true)
[ -z "$existing_global" ] || {
    echo "ERROR: an existing global core.hooksPath was preserved; remove or migrate it deliberately before installing." >&2
    exit 1
}
existing_system=$(git config --system --get-all core.hooksPath 2>/dev/null || true)
[ -z "$existing_system" ] || {
    echo "ERROR: a system core.hooksPath is configured; it was preserved." >&2
    exit 1
}
[ ! -e "$TARGET" ] && [ ! -L "$TARGET" ] || {
    echo "ERROR: existing global pre-commit hook preserved at $TARGET" >&2
    exit 1
}
if [ -d "$HOOKS_DIR" ]; then
    existing_hook_file=$(find "$HOOKS_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)
    [ -z "$existing_hook_file" ] || {
        echo "ERROR: files already exist in the managed global hooks directory; they were preserved." >&2
        exit 1
    }
fi

# Check the frozen model before making any hook or Git configuration changes.
"$PYTHON" "$HOOK_SCRIPT" --verify-model

mkdir -p "$HOOKS_DIR"
TEMP_HOOK=$(mktemp "$HOOKS_DIR/.pre-commit.codebert.XXXXXX")
cleanup() {
    [ -z "$TEMP_HOOK" ] || rm -f "$TEMP_HOOK"
}
trap cleanup EXIT HUP INT TERM
HOOK_COMMAND=$("$PYTHON" -c 'import shlex,sys; print("exec "+shlex.quote(sys.argv[1])+" "+shlex.quote(sys.argv[2]))' "$PYTHON" "$HOOK_SCRIPT")
{
    printf '%s\n' '#!/bin/sh'
    printf '%s\n' '# Installed by the native CodeBERT secret detector (global).'
    printf '%s\n' '# Chain a repository-default hook first, preserving its decision.'
    cat <<'EOF'
common_git_dir=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || exit 1
local_hook="$common_git_dir/hooks/pre-commit"
if [ -x "$local_hook" ] && [ "$(readlink -f -- "$local_hook" 2>/dev/null || printf '%s' "$local_hook")" != "$(readlink -f -- "$0" 2>/dev/null || printf '%s' "$0")" ]; then
    if "$local_hook" "$@"; then
        :
    else
        status=$?
        exit "$status"
    fi
fi
EOF
    printf '%s\n' "$HOOK_COMMAND"
} > "$TEMP_HOOK"
chmod 755 "$TEMP_HOOK"
if ! ln "$TEMP_HOOK" "$TARGET" 2>/dev/null; then
    echo "ERROR: global hook path became occupied; existing file was preserved." >&2
    exit 1
fi

if ! git config --global core.hooksPath "$HOOKS_DIR"; then
    rm -f "$TARGET"
    echo "ERROR: could not set global core.hooksPath; generated hook was removed." >&2
    exit 1
fi

echo "Installed the native CodeBERT hook globally at $TARGET"
echo "Global core.hooksPath is now $HOOKS_DIR"
