#!/usr/bin/env bash
# Symlink repo-tracked git hooks into .git/hooks/.
set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_SRC="$REPO_ROOT/scripts/hooks"
HOOKS_DST="$REPO_ROOT/.git/hooks"

for hook in "$HOOKS_SRC"/*; do
    name="$(basename "$hook")"
    target="$HOOKS_DST/$name"
    if [ -e "$target" ] && [ ! -L "$target" ]; then
        echo "Skipping $name — already exists and is not a symlink"
        continue
    fi
    ln -sf "$hook" "$target"
    chmod +x "$hook"
    echo "Installed $name"
done

echo "Done."
