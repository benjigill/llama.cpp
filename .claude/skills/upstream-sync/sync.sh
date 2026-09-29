#!/usr/bin/env bash
# Helper for the upstream-sync skill.
#   sync.sh report [target]            preflight + what an upstream merge would bring in
#   sync.sh verify <old-master> <target>  after merging: did every fork change survive?
#   sync.sh leak [target]              leak-check the staged fork delta (works mid-merge)
# target defaults to upstream/master.
set -euo pipefail
export LC_ALL=C

cd "$(git rev-parse --show-toplevel)"

die() { echo "error: $*" >&2; exit 1; }

# upstream PR numbers listed under "### Carried patches" in AGENTS.md
carried_prs() {
    sed -n '/^### Carried patches/,/^#/p' AGENTS.md | grep -oE '^- #[0-9]+' | grep -oE '[0-9]+' || true
}

report() {
    local target="${1:-upstream/master}"

    git remote get-url upstream >/dev/null 2>&1 || die "no 'upstream' remote (git remote add upstream https://github.com/ggml-org/llama.cpp.git)"
    [ -z "$(git status --porcelain --untracked-files=no)" ] || die "working tree has uncommitted changes"
    [ "$(git rev-parse --abbrev-ref HEAD)" = "master" ] || echo "warning: not on master (on $(git rev-parse --abbrev-ref HEAD))"

    git fetch --quiet upstream
    git fetch --quiet origin
    [ "$(git rev-parse master)" = "$(git rev-parse origin/master)" ] || echo "warning: master != origin/master; pull or push first"

    local base behind ahead
    base="$(git merge-base master "$target")"
    behind="$(git rev-list --count master.."$target")"
    ahead="$(git rev-list --count --no-merges "$target"..master)"

    echo "== divergence"
    echo "base:   $(git log -1 --format='%h %cs %s' "$base")"
    echo "target: $(git log -1 --format='%h %cs %s' "$target")"
    echo "upstream commits to merge: $behind"
    echo "fork-only commits on master: $ahead"
    [ "$behind" -eq 0 ] && { echo "nothing to merge"; return 0; }

    echo
    echo "== carried patches already merged upstream (drop our copy during this sync)"
    local found=0 pr hit
    for pr in $(carried_prs); do
        hit="$(git log --format='%h %s' "$base".."$target" | grep -E "\(#$pr\)" || true)"
        if [ -n "$hit" ]; then echo "#$pr -> $hit"; found=1; fi
    done
    [ "$found" -eq 0 ] && echo "(none)"

    echo
    echo "== files touched by both the fork and incoming upstream commits (likely conflicts)"
    comm -12 \
        <(git diff --name-only "$base" master | sort) \
        <(git diff --name-only "$base" "$target" | sort) \
        | while read -r f; do
            printf '%-60s %s upstream commits\n' "$f" "$(git rev-list --count "$base".."$target" -- "$f")"
        done | { grep . || echo "(none)"; }

    if [ "$behind" -gt 150 ]; then
        echo
        echo "== large gap: suggested intermediate merge targets (every 100 upstream commits)"
        git rev-list --first-parent --reverse "$base".."$target" | awk 'NR % 100 == 0' \
            | xargs -r -n1 git log -1 --format='%h %cs %s'
        echo "then: $target"
    fi
}

verify() {
    local old="${1:?old master sha}" target="${2:-upstream/master}"
    local old_base
    old_base="$(git merge-base "$old" "$target")"
    git merge-base --is-ancestor "$target" HEAD || die "$target is not merged into HEAD"

    echo "== fork delta before (vs $(git rev-parse --short "$old_base")) and after (vs $(git rev-parse --short "$target"))"
    local before after
    before="$(git diff --numstat "$old_base" "$old" | awk '{print $3"\t+"$1" -"$2}' | sort)"
    after="$(git diff --numstat "$target" HEAD | awk '{print $3"\t+"$1" -"$2}' | sort)"

    echo "-- files no longer different from upstream (must be explained: patch merged upstream, or lost in a conflict?)"
    comm -23 <(cut -f1 <<<"$before") <(cut -f1 <<<"$after") | sed 's/^/  /' | { grep . || echo "  (none)"; }
    echo "-- files newly different from upstream (should only be conflict fixups or AGENTS.md)"
    comm -13 <(cut -f1 <<<"$before") <(cut -f1 <<<"$after") | sed 's/^/  /' | { grep . || echo "  (none)"; }
    echo "-- files whose fork diff size changed (inspect: git diff $target HEAD -- <file>)"
    join -t $'\t' <(echo "$before") <(echo "$after") | awk -F'\t' '$2 != $3 {printf "  %-60s before %s   after %s\n", $1, $2, $3}' | { grep . || echo "  (none)"; }

    echo
    leak "$target"
}

# Leak-check only what the fork adds on top of target (index vs target), so
# upstream's own emails/paths don't trip it. Index == HEAD after committing.
leak() {
    local target="${1:-upstream/master}" tmp rc=0
    echo "== leak check on the fork delta only (upstream content is not ours to check)"
    tmp="$(mktemp)"
    git diff --cached -U0 --no-color "$target" | grep '^+' | grep -v '^+++' > "$tmp" || true
    scripts/fork/leak-check.sh "$tmp" && echo "clean" || rc=$?
    rm -f "$tmp"
    return $rc
}

case "${1:-}" in
    report) shift; report "$@" ;;
    verify) shift; verify "$@" ;;
    leak)   shift; leak "$@" ;;
    *) echo "usage: $0 report [target] | verify <old-master> [target] | leak [target]" >&2; exit 2 ;;
esac
