# Improvement ledger

Backlog of performance, memory and stability work for this fork (see the Fork Policy section of AGENTS.md for the target workload). Adopted patches move to the "Carried patches" list in AGENTS.md; this file keeps the reasoning, status and measured outcome.

Status: `todo`, `in progress`, `done` (link the commit), `dropped` (say why).

## Backlog

### 1. Cheaper MTP draft head via a trimmed draft vocabulary (upstream #29143 + local) - todo

- Each MTP draft token runs the full-vocab LM head (`src/models/qwen35.cpp`, `graph_mtp`), about 248k rows, around 3 times per round. The head is likely the largest part of a draft step (one transformer layer + head).
- #29143 (+90/-8, qwen35 only) adds `d2t` (draft-to-target) support: a head of `n_draft_vocab` rows whose logits are scattered onto a `-inf` full-vocab tensor.
- Local part: build the `d2t` map and the trimmed head from token frequencies (e.g. top 32k) out of the existing head, no training.
- Output is unchanged: the target verifies every token (also with #27694 rejection sampling, the draft probability of a trimmed token is 0). Cost: slightly lower acceptance on rare tokens.
- Expected: roughly 5-10% decode on MTP rounds if the head is 60%+ of a draft step. Profile one draft step first.

### 2. Row-per-warp GATED_DELTA_NET decode kernel (upstream #22587) - todo

- Rewrites the recurrent GDN kernel (one warp per group of output rows).
- Upstream numbers (RTX 5090): kernel +13-16% at 1-4 tokens (our MTP verify is 4 tokens), +20-50% at 32-1024 tokens; +4-7% pp end to end on Qwen3.5-27B Q4_K_M.
- Hits all 48 GDN layers on every decode and verify step, and the recurrent tail of our chunked split path.
- Conflicts with the local chunked-GDN-with-snapshots patch in `ggml/src/ggml-cuda/gated_delta_net.cu`; re-run `test-backend-ops -o GATED_DELTA_NET` and `-o GATED_DELTA_NET_CACHE_FUSION`.

### 3. Serialize MTP multi-ubatch decode (upstream #26827) - todo

- Stability. With MTP and tensor split on two GPUs, prefills of 100k+ tokens locked the whole host: MTP draft catch-up queued two graphs against the same KV cache without waiting.
- We run 150k tokens per slot, so this is reachable.
- Cost: possibly a small pp loss during MTP catch-up; measure with the `cache` suite.

### 4. ngram-mod: fingerprinted, persistent, corpus-primed table (local) - in progress

- The table has no keys (`common/ngram-mod.cpp`): a lookup that lands on another context's bucket returns a wrong draft, so the table is wiped at 25% occupancy.
- Store a 32-bit fingerprint of the context next to each entry. Collisions return nothing instead of a bad draft, the occupancy wipe can go, and full tables become usable (overwrite on collision instead of wipe). The table stays shared across all slots and chats; identical content from other chats still matches.
- Save the table on shutdown/sleep and load it on start, so what automations and chats have produced survives restarts.
- Prime it from a corpus (repos, automation transcripts rendered with the chat template), tokenized with the model's vocab.
- Target: repeated automations (same MCP calls, same ledger/report edits) draft long accepted runs from the first request after a restart.

### 5. Share checkpoint buffers in the prompt cache (upstream #27451) - todo

- Saving a prompt to the RAM cache deep-copies every context checkpoint, so memory briefly doubles. With 150k-token slots and hybrid checkpoints this can hit `--cache-ram`, and a failed allocation silently cuts the limit to 40%.
- Shares the checkpoint buffers (refcounted) and handles `bad_alloc` for the whole entry. +56/-18.
- Later: #28092 `--cache-disk` (persistent prompt cache across restarts, +1706 lines); wait for it to settle upstream.

## Considered, not now

- #27248 q5 K cache - dropped: keep q8_0 K.
- #27210 adaptive MTP draft depth - wants `--spec-draft-n-max 12`, i.e. 13 recurrent snapshots per slot in VRAM; at temp 1.0 deep drafts rarely pay off and ngram-mod covers long bursts.
- #24785 recurrent shrink/expand for the prompt cache - overlaps with #25592, which we carry.
