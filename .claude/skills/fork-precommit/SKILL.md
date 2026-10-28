---
name: fork-precommit
description: Check staged changes, a branch or a commit range against the fork's AGENTS.md rules (ASCII only, comment style, leak-check, commit subject and Assisted-by, Carried patches line) in a Haiku subagent. Use before committing or opening a PR on the fork.
argument-hint: "[staged | branch | <range> | <message file>]"
context: fork
agent: fork-style-checker
model: haiku
---

Run your checks and report in your standard format.

Scope from the caller (may be empty): $ARGUMENTS

- Empty or `staged`: staged changes. If nothing is staged, use the current branch vs `master` plus uncommitted changes.
- `branch`: `master...HEAD`, including the commit messages in `master..HEAD`.
- A git range (`a..b`): that range, including its commit messages.
- A file path: treat it as a commit message or PR body file and run only check 5 and the leak check on it.
