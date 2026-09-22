#!/usr/bin/env bash
# Block commits that add local paths, emails or other private strings.
# Install per clone:
#   ln -sf ../../scripts/fork/leak-check.sh .git/hooks/pre-commit
#   ln -sf ../../scripts/fork/leak-check.sh .git/hooks/commit-msg
# Private patterns (names, hostnames, keys) go in .git/info/leak-patterns,
# one extended regex per line; that file is never committed.
set -euo pipefail

generic='(/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+|/media/[A-Za-z0-9._-]+|/mnt/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})'
allow='(users\.noreply\.github\.com|@ggml\.ai|example\.(com|org)|@anthropic\.com)'

patterns_file="$(git rev-parse --git-dir)/info/leak-patterns"

if [ $# -ge 1 ] && [ -f "$1" ]; then
    # commit-msg hook: check the message
    text="$(grep -v '^#' "$1")"
    what="commit message"
else
    # pre-commit hook: check added lines only
    text="$(git diff --cached -U0 --no-color | grep '^+' | grep -v '^+++' || true)"
    what="staged changes"
fi

hits="$(printf '%s\n' "$text" | grep -nE "$generic" | grep -vE "$allow" || true)"
if [ -f "$patterns_file" ]; then
    while IFS= read -r p; do
        [ -z "$p" ] && continue
        case "$p" in \#*) continue ;; esac
        h="$(printf '%s\n' "$text" | grep -niE -- "$p" || true)"
        [ -n "$h" ] && hits="$hits"$'\n'"$h"
    done < "$patterns_file"
fi

hits="$(printf '%s\n' "$hits" | sed '/^$/d')"
if [ -n "$hits" ]; then
    echo "leak-check: possible private data in $what:" >&2
    printf '%s\n' "$hits" | head -20 >&2
    echo "Fix it, or bypass once with --no-verify if it is a false positive." >&2
    exit 1
fi
