---
name: upstream-sync
description: Merge upstream ggml-org/llama.cpp master into this fork without losing carried patches. Use when asked to sync, update from, catch up with, or pull upstream / ggml-org master into the fork.
---

# Upstream sync

Bring `upstream/master` (ggml-org/llama.cpp) into the fork's `master` (benjigill/llama.cpp) while keeping every carried patch. Read the "Fork Policy" section of `AGENTS.md` first; the "Carried patches" list there is the source of truth for what must survive.

## Strategy (do not deviate without asking)

- **Merge, never rebase `master`.** `master` is published and contains GitHub PR merge commits; rebasing means force-pushing and rewriting history other clones (the Linux GPU box) have pulled.
- Merge on a branch `sync/upstream-<YYYY-MM-DD>` and land it through a PR on **the fork** so the target machine can build and benchmark it before `master` moves.
- The PR must be merged with **"Create a merge commit"**. Squash or rebase merges drop the upstream ancestry, and the next sync would re-conflict everything.
- Small steps: if the report suggests intermediate targets (gap > 150 commits), merge them one at a time on the same branch, building after each.
- When upstream has merged a PR we carry, **upstream's version wins**: resolve to theirs, remove our copy, and delete the line from "Carried patches".

## Steps

### 1. Report

```bash
git config rerere.enabled true      # remember conflict resolutions across chunks and retries
.claude/skills/upstream-sync/sync.sh report
```

It checks for a clean tree and that `master` matches `origin/master`, fetches, and prints:
- how many upstream commits are incoming
- carried patches whose upstream PR has now landed
- files changed on both sides (likely conflicts)
- intermediate targets, if the gap is large

Summarise this for the user before merging. If a carried patch has landed upstream, say which one and check whether any `local` patch depends on it (for example the meta/CUDA gate-up alloc fix depends on #28702).

### 2. Merge

```bash
OLD=$(git rev-parse master)
git switch -c sync/upstream-$(date +%F) master
git merge --no-ff upstream/master -m "Merge upstream/master into fork (ggml-org <short-sha>)"
```

Use an intermediate sha in place of `upstream/master` when chunking. A clean merge runs only the `commit-msg` leak hook, which is fine.

### 3. Resolve conflicts (if any)

For each conflicted file:
1. Find which carried patch owns our side: `git log --oneline $OLD --not upstream/master -- <file>`.
2. Understand what upstream changed: `git log --oneline MERGE_HEAD --not $OLD -- <file>`, then read the diffs.
3. Resolve so that upstream's new code **and** our patch's behaviour both hold. Re-apply our change on top of upstream's version of the code. Do not keep our stale copy of code that upstream refactored.
4. If the patch was merged upstream, take theirs (`git checkout --theirs <file>` only when the whole file is that patch; otherwise resolve by hunk).
5. Never resolve by dropping fork code just to make the conflict go away. If the right resolution is unclear (e.g. upstream rewrote the function we patch), stop and ask the user. Offer to pull the newer revision of the upstream PR instead.

When committing a conflicted merge, the `pre-commit` leak hook scans the **entire upstream diff** and flags upstream's own emails and paths (AUTHORS, docs). Do not trust or silence it blindly. Instead:
```bash
git add <resolved files>
.claude/skills/upstream-sync/sync.sh leak MERGE_HEAD   # leak-checks only the fork delta
scripts/fork/leak-check.sh <msg-file>                  # check the message yourself
git commit --no-verify -F <msg-file>
```
List every conflicted file in the merge message, with one line per file on how it was resolved.

### 4. Verify nothing was lost

```bash
.claude/skills/upstream-sync/sync.sh verify $OLD upstream/master   # or the chunk target
```
This compares the fork's diff against upstream before and after the merge. It also leak-checks the fork's delta.
- **Files no longer different from upstream**: each one must be explained by a patch that landed upstream. Anything else means fork code was lost; fix it before continuing.
- **Files newly different**: should be only AGENTS.md or deliberate conflict fixups.
- **Fork diff size changed**: inspect with `git diff <target> HEAD -- <file>` and confirm the patch is intact.
- For carried patches that landed upstream, `git diff <target> HEAD -- <their files>` should be empty or explained.

Then update the "Carried patches" list in `AGENTS.md` (fork section only) for any dropped patch. Commit that separately: `docs : drop #NNNNN from carried patches (merged upstream)`.

### 5. Build locally (macOS, no CUDA)

```bash
cmake -B build-check -DCMAKE_BUILD_TYPE=Release
cmake --build build-check -j
ctest --test-dir build-check -L main --output-on-failure -j 8
```
This catches API breaks in the shared code (common/, src/, server, and ggml core). It does **not** compile the CUDA patches.

### 6. Push and open the PR on the fork

```bash
git push -u origin HEAD
gh pr create --repo benjigill/llama.cpp --base master \
  --title "Sync upstream master (<date>, ggml-org <short-sha>)" --body-file <file>
```
Always pass `--repo benjigill/llama.cpp`, because `gh` defaults to the parent repo for forks. **Never open anything against ggml-org.** The PR body contains:
- the commit count
- the upstream range
- dropped patches
- conflicted files with their resolutions
- the checklist below

Leak-check the body file before creating the PR.

### 7. Hand off to the target machine

Tell the user that this sync cannot merge until the Linux box checks pass. Give them the **full commands below**, with `<date>`, the PR number and the op list filled in. Never shorten them to "run test-backend-ops" or "A/B with the bench script". Use placeholders only for values you don't know (`<model.gguf>`). Adjust the `-o` list to the ops touched by carried patches and by upstream in this sync.

```sh
# 0. stop any running llama-server (it holds the GPUs and skews the bench)
pkill -f llama-server

# 1. baseline: current master in its own worktree (first time: worktree add; later: checkout)
git fetch origin
git worktree add --detach ../llama-master origin/master 2>/dev/null || git -C ../llama-master checkout --detach origin/master
cmake -S ../llama-master -B ../llama-master/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama-master/build -j

# 2. candidate: the sync branch
git switch sync/upstream-<date>
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. correctness of the CUDA ops our patches touch (all lines must say OK)
build/bin/test-backend-ops -b CUDA0 -j 8 -o MUL_MAT,MUL_MAT_Q_GATE_UP_SWIGLU_DOWN,GATED_DELTA_NET,SSM_CONV

# 4. A/B benchmark (results land in bench-results/<label>)
python3 scripts/fork/bench_qwen38.py run --bin ../llama-master/build/bin --label master --model <model.gguf> --probabilistic
python3 scripts/fork/bench_qwen38.py run --bin build/bin --label sync-<date> --model <model.gguf> --probabilistic
python3 scripts/fork/bench_qwen38.py compare bench-results/master bench-results/sync-<date>

# 5. smoke test: server with MTP + ngram-mod, then two requests from a second terminal
GGML_CUDA_P2P=1 CUDA_VISIBLE_DEVICES=0,1 build/bin/llama-server -m <model.gguf> --port 8080 \
  --split-mode tensor --tensor-split 15,13 --main-gpu 1 --n-gpu-layers 999 \
  -c 180000 --cache-type-k q8_0 --cache-type-v q4_0 --flash-attn on --jinja \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-sampling probabilistic \
  --spec-ngram-mod-n-max 64 --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-match 24
for i in 1 2; do
  curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Write a Python function that parses ISO 8601 dates, with tests."}],"max_tokens":400,"temperature":1.0,"top_p":0.95,"top_k":20}' \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"][:300]); print(r.get("timings"))'
done

# 6. merge with a merge commit (never squash/rebase), then update local master
gh pr merge <PR#> --repo benjigill/llama.cpp --merge --delete-branch
git switch master && git pull
git branch -d sync/upstream-<date>
git branch --list 'feat/*'            # for each one listed:
git switch feat/<name> && git rebase master
```

Pass criteria: test-backend-ops is all OK. `compare` shows pp/tg and spec acceptance within noise. Step 5 returns coherent text, and `timings` shows `draft_n_accepted` > 0.

## Abort

`git merge --abort` (mid-merge), or `git switch master && git branch -D sync/upstream-<date>`. Local `master` is never touched until the PR is merged.
