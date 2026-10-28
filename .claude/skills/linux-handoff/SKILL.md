---
name: linux-handoff
description: Write the full, copy-paste command list for testing a fork branch on the Linux GPU box (build master baseline and candidate, test-backend-ops for the touched ops, bench_qwen38.py A/B with the right suites, KLD for numeric changes, server smoke test, merge steps). Runs in a Haiku subagent. Use at the end of upstream-sync, after adopting an upstream PR, or whenever a branch needs CUDA build and benchmark numbers.
argument-hint: "[branch] [sync | kernel-pp | kernel-tg | spec | numeric]"
context: fork
agent: general-purpose
model: haiku
---

Produce a command list for the Linux target box. Do not run any of the commands. Do not edit files.

Arguments (may be empty): $ARGUMENTS

## 1. Work out what changed

- Branch: from the arguments, otherwise `git rev-parse --abbrev-ref HEAD`.
- `git diff --stat master...<branch>` and `git log --oneline master..<branch> --no-merges`. For a `sync/upstream-*` branch, use the merge commit's message for the conflicts and the upstream range.
- Kind (from the arguments, otherwise from the files):
  - `sync`: branch name starts with `sync/`
  - `kernel-pp`: mmq, fattn mma/tile, chunked gated_delta_net, prefill paths
  - `kernel-tg`: mmvq, fattn vec, recurrent GDN, cpy, sampling, decode paths
  - `spec`: common/speculative*, ngram, MTP, server, prompt cache, checkpoints
  - `numeric`: add this to any of the above when the change alters arithmetic (new kernel math, quantization, accumulation order)
- Ops for `test-backend-ops -o`: map the touched ggml-cuda files to op names (mmq.cu/mmvq.cu -> MUL_MAT; fattn* -> FLASH_ATTN_EXT; gated_delta_net* -> GATED_DELTA_NET; ssm-conv* -> SSM_CONV; cpy* -> CPY; fused gate/up -> MUL_MAT_Q_GATE_UP_SWIGLU_DOWN; ggml-backend-meta -> MUL_MAT,GATED_DELTA_NET). For `sync`, always include `MUL_MAT,MUL_MAT_Q_GATE_UP_SWIGLU_DOWN,GATED_DELTA_NET,SSM_CONV`, plus ops upstream touched in this range (`git diff --stat <old-master> <branch> -- ggml/src/ggml-cuda`).
- Suites: `kernel-pp` -> `micro,batched,long-context`; `kernel-tg` -> `micro,decode,parallel`; `spec` -> `decode,parallel,cache`; `sync` -> the defaults of `bench_qwen38.py run` plus `--probabilistic`. Check the suite names against `grep -n 'suite' scripts/fork/bench_qwen38.py`.

## 2. Write the commands

Use the templates in `.claude/skills/upstream-sync/SKILL.md` step 7 (for `sync`) or `.claude/skills/upstream-scout/SKILL.md` "Adopting a candidate" (for everything else) as the base. Copy them in full and fill in the branch, label, ops and suites. Never shorten a step to "run the tests" or "A/B it". Rules:

- Keep `<model.gguf>`, `<mmproj.gguf>` and `<table.bin>` as placeholders. Never insert real paths.
- Run test-backend-ops on both `CUDA0` and `CUDA1`.
- Include the `test-backend-ops perf` step only for kernel changes. Include the KLD step only for `numeric`.
- Include the server smoke test (two curl requests, `timings` must show `draft_n_accepted` > 0) for `sync` and `spec`.
- For `sync`, end with the merge-commit steps (`gh pr merge <PR#> --repo benjigill/llama.cpp --merge`; never squash or rebase) and the `feat/*` rebase loop.

## 3. Output

First give one line with the branch, its kind, the ops and the suites. Then one `sh` code block with all commands. Last, give the pass criteria in 3 to 5 lines and ask the user to bring back the results for `/bench-review`.
