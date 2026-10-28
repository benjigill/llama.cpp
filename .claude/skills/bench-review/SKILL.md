---
name: bench-review
description: Interpret benchmark or correctness results from the Linux GPU box (bench_qwen38.py compare, llama-bench, test-backend-ops, KLD, nsys CSV, server timings) in a Sonnet subagent and return a keep/drop/rerun verdict plus a numbers block for the commit message and ledger. Use when the user pastes results or names result directories.
argument-hint: "<result dirs or files, or paste output after the command>"
context: fork
agent: bench-analyst
model: sonnet
---

Analyse these results and report in your standard format.

$ARGUMENTS

If the arguments are two directories under `bench-results/`, run `python3 scripts/fork/bench_qwen38.py compare <a> <b>` first and analyse its output, along with any per-request files in the two directories. If they name the branch or patch under test, use `git log -1 --format=%B <branch>` and the matching `IMPROVEMENT_LEDGER.md` item to learn what change is expected, and say whether the numbers confirm it.
