#!/bin/sh
set -eu

APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
CONFIG_ROOT=${XDG_CONFIG_HOME:-$HOME/.config}
INSTALL_ROOT="$CONFIG_ROOT/git/codebert-secret-hook"
HOOKS_DIR="$INSTALL_ROOT/hooks"
TARGET="$HOOKS_DIR/pre-commit"

case "${1:-}" in
    "") ;;
    --help|-h) echo 'Usage: uninstall_global.sh'; exit 0 ;;
    *) echo 'Usage: uninstall_global.sh' >&2; exit 2 ;;
esac

configured=$(git config --global --get-all core.hooksPath 2>/dev/null || true)
[ "$configured" = "$HOOKS_DIR" ] || {
    echo "ERROR: global core.hooksPath changed or has multiple values; no configuration or hook was removed." >&2
    exit 1
}
[ -f "$TARGET" ] && [ ! -L "$TARGET" ] || {
    echo "ERROR: expected regular generated hook missing; configuration was preserved." >&2
    exit 1
}

if ! python3 - "$TARGET" "$APP_ROOT/hook/pre_commit.py" <<'PY'
import pathlib
import shlex
import sys

lines = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
expected_prefix = '''#!/bin/sh
# Installed by the native CodeBERT secret detector (global).
# Chain a repository-default hook first, preserving its decision.
common_git_dir=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || exit 1
local_hook="$common_git_dir/hooks/pre-commit"
if [ -x "$local_hook" ] && [ "$(readlink -f -- "$local_hook" 2>/dev/null || printf '%s' "$local_hook")" != "$(readlink -f -- "$0" 2>/dev/null || printf '%s' "$0")" ]; then
    if "$local_hook" "$@"; then
        :
    else
        status=$?
        exit "$status"
    fi
fi'''.splitlines()
if len(lines) != len(expected_prefix) + 1 or lines[:-1] != expected_prefix:
    raise SystemExit(1)
command = lines[-1]
parts = shlex.split(command)
if len(parts) != 3 or parts[0] != "exec":
    raise SystemExit(1)
if pathlib.Path(parts[2]).resolve() != pathlib.Path(sys.argv[2]).resolve():
    raise SystemExit(1)
PY
then
    echo "ERROR: global hook was changed or does not belong to this application; it was preserved." >&2
    exit 1
fi

other=$(find "$HOOKS_DIR" -mindepth 1 -maxdepth 1 ! -name pre-commit -print -quit 2>/dev/null || true)
[ -z "$other" ] || {
    echo "ERROR: other global hooks exist in the managed directory; move or preserve them before uninstalling." >&2
    exit 1
}

rm "$TARGET"
if ! git config --global --unset-all core.hooksPath; then
    echo "ERROR: hook removed, but global core.hooksPath could not be unset. Inspect Git configuration." >&2
    exit 1
fi
echo "Removed the verified global detector hook and its core.hooksPath setting."
