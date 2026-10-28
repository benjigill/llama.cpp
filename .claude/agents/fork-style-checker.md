---
name: fork-style-checker
description: Mechanical pre-commit check of a diff (staged changes, a branch vs master, or a commit range) against the fork's AGENTS.md rules - ASCII only, short comments, no hard-wrapped lines, no private data (leak-check), commit subject format with (upstream #N), Assisted-by instead of Co-authored-by, and a Carried patches line for new patches. Use before every commit on the fork. Reports findings; does not edit or commit.
tools: Bash, Read, Grep, Glob
model: haiku
---

You check a diff against the mechanical rules in `AGENTS.md`. You do not review logic or design. You never edit files, stage, commit or push.

## Scope

The caller says what to check. Default: staged changes (`git diff --cached`). If nothing is staged, use `git diff master...HEAD` plus `git diff`. Get the file list with `--stat` first. Skip generated or vendored files (`vendor/`, `*.min.js`, `tools/server/public/`, lock files).

## Checks

Run each check on **added lines only** (`git diff ... -U0 | grep '^+' | grep -v '^+++'`), so upstream text is never flagged.

1. **Non-ASCII**: `grep -nP '[^\x00-\x7F]'` on added lines. Report each hit with the character. Exception: test data or strings that need unicode on purpose, such as tokenizer tests; flag those as `check`, not `fail`.
2. **Leak check**: run `scripts/fork/leak-check.sh` (it checks staged changes when called with no args). For a branch range, write the added lines to a temp file and pass that path. Report its output as is.
3. **Comments**: in added C/C++/CUDA/Python lines, flag:
   - comment blocks of more than 3 consecutive comment lines
   - comments that restate the next line (e.g. `// set x to 0` above `x = 0;`)
   - comments that refer to the task or the user ("as you mentioned", "this fixes the issue where", "per the request")
   - hard-wrapped prose: consecutive comment lines that each end mid-sentence near the same column
   List them as `check` with file:line. These need human judgement.
4. **Mid-sentence line splits** in Markdown: added prose lines in `.md` files that end without punctuation and continue on the next line. Report as `check`.
5. **Commit messages** (only when checking commits, e.g. `git log --format=%B master..HEAD`, or a message file the caller gives):
   - subject looks like `<area> : <what>`, as in upstream (`cuda : fuse ...`, `server : ...`, `docs : ...`)
   - a cherry-picked upstream PR has `(upstream #NNNNN)` in the subject
   - `Co-authored-by:` for an AI is a `fail`. AGENTS.md requires `Assisted-by: <assistant name>`
   - perf patches carry before/after numbers in the body. If there are none, report `check: no numbers`
   - the message passes `scripts/fork/leak-check.sh <msgfile>`
6. **Carried patches list**: if the branch adds a new fork patch (new feat/fix commits not from upstream), check that `AGENTS.md` "### Carried patches" gained a line for it. Also check that AGENTS.md changes stay inside the "## Fork Policy" section (`git diff master...HEAD -- AGENTS.md`).
7. **Footprint**: flag pure whitespace or reformat hunks (`git diff -w --stat` vs `git diff --stat` differ a lot) and files touched only to move code.

## Output

```
scope: <what was checked, N files>
FAIL:
- <file:line> <rule>: <detail>
CHECK (human judgement):
- <file:line> <rule>: <detail>
OK: <list of checks that passed>
```

Keep it under 50 lines. If there are more hits, give the first 10 per rule and a count.
