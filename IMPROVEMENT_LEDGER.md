# Improvement ledger

Backlog of performance, memory and stability work for this fork (see the Fork Policy section of AGENTS.md for the target workload). Adopted patches move to the "Carried patches" list in AGENTS.md; this file keeps the reasoning, status and measured outcome.

Status: `todo`, `in progress`, `done` (link the commit), `deferred` (say what is pending), `dropped` (say why).

## Backlog

### 1. Cheaper MTP draft head via a trimmed draft vocabulary (upstream #29143 + local) - done (opt-in: needs the d2t model copy)

- Each MTP draft token runs the full-vocab LM head (`src/models/qwen35.cpp`, `graph_mtp`), about 248k rows, around 3 times per round. The head is likely the largest part of a draft step (one transformer layer + head).
- #29143 (+90/-8, qwen35 only) adds `d2t` (draft-to-target) support: a head of `n_draft_vocab` rows whose logits are scattered onto a `-inf` full-vocab tensor.
- Local part: build the `d2t` map and the trimmed head from token frequencies (e.g. top 32k) out of the existing head, no training.
- Output is unchanged: the target verifies every token (also with #27694 rejection sampling, the draft probability of a trimmed token is 0). Cost: slightly lower acceptance on rare tokens.
- Expected: roughly 5-10% decode on MTP rounds if the head is 60%+ of a draft step. Profile one draft step first.
- Applied (branch `feat/mtp-ubatch-dvocab-cache`): #29143 only covers MTP sidecars (`mtp_only`), and Qwen3.x GGUFs have no `nextn.shared_head_head` (the draft uses `output.weight`), so a local commit loads `d2t` + a trimmed `nextn.shared_head_head` next to the trunk. `scripts/fork/mtp-d2t.py` writes the model copy (rows copied byte for byte from `output.weight`). Checked on a random fixture: kept logits match the full head exactly.
- Estimate: at n_embd 5120, a Q6_K head of 248k rows is ~1 GB read per draft token vs ~0.3 GB for the MTP block, so the head is ~75% of a draft step; 32k rows cut it to ~0.13 GB. Confirm with the A/B below.
- To measure: build the copy (`--n-draft 32768`, `--table` from the ngram-mod file), then A/B the same build on the original vs the d2t GGUF (decode + parallel suites); watch MTP acceptance, it should barely move.
- Result (same build, original vs d2t copy, `--n-draft 32768`, decode suite): tg +1.8 to +2.6% greedy, +1.9 to +5.2% probabilistic, i.e. about +2-5%. MTP acceptance -1 to -5% relative (e.g. greedy edit 0.44 -> 0.42), within a few hundredths and partly noise.
  | | tg base | tg d2t | accept base | accept d2t |
  | --- | --- | --- | --- | --- |
  | greedy edit / write / prose | 73.45 / 68.68 / 62.32 | 75.08 / 69.91 / 63.92 | 0.44 / 0.38 / 0.31 | 0.42 / 0.36 / 0.30 |
  | probabilistic edit / write / prose | 84.82 / 76.04 / 75.71 | 86.46 / 80.01 / 78.50 | 0.56 / 0.45 / 0.46 | 0.54 / 0.46 / 0.45 |
- Smaller than the 5-10% estimate: the draft steps are a small part of a round next to the 27B verify step, and the slightly lower acceptance gives some of it back.
- `--n-draft 49152` (now the script default), same base run:
  | | tg base | tg d2t | accept base | accept d2t |
  | --- | --- | --- | --- | --- |
  | greedy edit / write / prose | 73.45 / 68.68 / 62.32 | 74.76 / 69.97 / 63.39 | 0.44 / 0.38 / 0.31 | 0.43 / 0.37 / 0.31 |
  | probabilistic edit / write / prose | 84.82 / 76.04 / 75.71 | 87.05 / 82.53 / 79.33 | 0.56 / 0.45 / 0.46 | 0.55 / 0.49 / 0.47 |
- 49152 vs 32768: greedy speed the same (within 0.5 t/s), greedy acceptance about halfway back to base (-2% vs -4% relative); probabilistic +2.6 / +8.5 / +4.8% vs +1.9 / +5.2 / +3.7%. Greedy is the clean signal (a trimmed draft can only lose accepted tokens there); the probabilistic acceptance above base (write 0.45 -> 0.49) cannot come from the trim and is sampling noise at temp 1.0, so read the probabilistic gains as roughly +3-5%.

### 2. Row-per-warp GATED_DELTA_NET decode kernel (upstream #22587) - replaced by upstream #30087

- Rewrites the recurrent GDN kernel (one warp per group of output rows).
- Upstream numbers (RTX 5090): kernel +13-16% at 1-4 tokens (our MTP verify is 4 tokens), +20-50% at 32-1024 tokens; +4-7% pp end to end on Qwen3.5-27B Q4_K_M.
- Hits all 48 GDN layers on every decode and verify step, and the recurrent tail of our chunked split path.
- Conflicts with the local chunked-GDN-with-snapshots patch in `ggml/src/ggml-cuda/gated_delta_net.cu`; re-run `test-backend-ops -o GATED_DELTA_NET` and `-o GATED_DELTA_NET_CACHE_FUSION`.
- Applied (branch `feat/gdn-row-per-warp`) as a port: the PR predates PDL, cache fusion and our snapshots, so the kernel body was moved into our file with those kept. 55/55 + 7/7 tests pass on both GPUs.
- Kernel (`test-backend-ops perf`, 5070 Ti, master -> branch): 1 token 2.93 -> 2.91 us (flat), KDA 1 token 3.05 -> 2.94 us; 64 tokens 32h d128 68.56 -> 43.11 us (1.59x), 4h 22.13 -> 19.77 us, KDA 78.62 -> 46.35 us; 256+ tokens identical (chunked kernel).
- End to end: micro and batched within +-0.4%. Decode moved with acceptance (greedy text differs with the new summation order) and was flat where acceptance was flat, so no decode gain. The gain is on 2-130 token batches that the chunked kernel does not take (short follow-up turns, tool results after a cache hit), which the bench suites barely exercise.
- Dropped in the 2026-10-08 upstream sync: upstream #30087 (4 state columns per warp) rewrote the same recurrent kernel, so the two cannot both be carried. Upstream reports -33% at 64 tokens 32h d128 and -12% at 1 token on a 4090 vs the old kernel; compare against the 5070 Ti numbers above (`test-backend-ops perf -o GATED_DELTA_NET`) and re-port #22587 only if #30087 loses on our GPUs.

### 3. Serialize MTP multi-ubatch decode (upstream #26827) - done

- Stability. With MTP and tensor split on two GPUs, prefills of 100k+ tokens locked the whole host: MTP draft catch-up queued two graphs against the same KV cache without waiting.
- We run 150k tokens per slot, so this is reachable.
- Cost: possibly a small pp loss during MTP catch-up; measure with the `cache` suite.
- Applied (branch `feat/mtp-ubatch-dvocab-cache`). The upstream test uses the abort callback to see the ubatch boundary, which only the CPU backend calls per graph (Metal: capture mode only, CUDA: never); pinned the test model to CPU. Fails without the fix, passes with it.

### 4. ngram-mod: fingerprinted, persistent, corpus-primed table (local) - done, numbers pending

- The table has no keys (`common/ngram-mod.cpp`): a lookup that lands on another context's bucket returns a wrong draft, so the table is wiped at 25% occupancy.
- Store a 32-bit fingerprint of the context next to each entry. Collisions return nothing instead of a bad draft, the occupancy wipe can go, and full tables become usable (overwrite on collision instead of wipe). The table stays shared across all slots and chats; identical content from other chats still matches.
- Save the table on shutdown/sleep and load it on start, so what automations and chats have produced survives restarts.
- Prime it from a corpus (repos, automation transcripts rendered with the chat template), tokenized with the model's vocab.
- Target: repeated automations (same MCP calls, same ledger/report edits) draft long accepted runs from the first request after a restart.

### 5. Share checkpoint buffers in the prompt cache (upstream #27451) - done

- Saving a prompt to the RAM cache deep-copies every context checkpoint, so memory briefly doubles. With 150k-token slots and hybrid checkpoints this can hit `--cache-ram`, and a failed allocation silently cuts the limit to 40%.
- Shares the checkpoint buffers (refcounted) and handles `bad_alloc` for the whole entry. +56/-18.
- Applied (branch `feat/mtp-ubatch-dvocab-cache`), with a local fix: upstream allocates a fresh zeroed buffer for every checkpoint update, including the per-round speculative checkpoint (ngram drafts longer than n_rs_seq); an unshared buffer is now resized in place as before.
- Result (with #26827, `cache` suite, branch vs master build): prompt tokens re-processed identical (warm 18/16/17, multi-turn 26/22); cold TTFT 19.72 -> 19.89 s (+0.9%), multi-turn TTFT 0.52 -> 0.54 / 0.52 -> 0.52 s, single samples, noise. No regression. The suite does not reach the cases these fix (100k+ MTP prefills under -sm tensor; `--cache-ram` near its limit), so there is no gain to show either. Branch side ran on the d2t copy, which only changes draft steps.
- Later: #28092 `--cache-disk` (persistent prompt cache across restarts, +1706 lines); wait for it to settle upstream.

### 6. Cost-based ngram-mod / MTP routing (local) - done, experimental and shelved

- The fixed low-acceptance cooldown chooses based only on ngram acceptance, not the latency of a verification round and its checkpoint replay. With both speculators, ngram-mod is always tried before MTP.
- For the ngram-mod + MTP pair, estimate per-slot time per produced token from complete rounds (draft, verify and replay), using a 1/8-weight moving estimate and a 5% margin before choosing MTP; probe the other source every 32 rounds. A table miss still falls back to MTP. Other speculative configurations keep the old policy.
- Opt-in for the pair with `LLAMA_SPEC_COST_ROUTING=1`; unset or `0` uses the unchanged fixed ngram-first/cooldown policy. The per-request cost estimates and round counts are logged with slot timings when routing is enabled.
- Production baseline supplied before this change: probabilistic edit/write/prose 84.4/77.5/75.3 t/s with aggregate acceptance 0.54/0.46/0.45; `-np 1` 73.2 aggregate t/s, `-np 2` 109.0 aggregate t/s.
- First same-build A/B on the target GPUs (`spec-fixed` vs `spec-cost`): probabilistic edit 84.56 -> 85.27 t/s (+0.8%, acceptance 0.54 -> 0.55); write 77.73 -> 84.87 t/s (+9.2%, acceptance 0.46 -> 0.54); prose 75.57 -> 77.42 t/s (+2.4%, acceptance 0.45 -> 0.47). Parallel `-np 1` aggregate 73.33 -> 75.23 t/s (+2.6%); `-np 2` aggregate 109.09 -> 107.08 t/s (-1.8%) and per-request 59.48 -> 58.95 t/s (-0.9%). Repeat runs and check source-selection logs before treating small differences as reliable; output correctness and distribution still need validation.
- Repeat A/B: edit 85.62 -> 87.08 t/s (+1.7%); write 80.83 -> 80.50 t/s (-0.4%); prose 75.74 -> 77.41 t/s (+2.2%). Parallel `-np 1` 73.23 -> 75.43 t/s (+3.0%), `-np 2` 108.98 -> 106.72 t/s (-2.1%). The first write gain did not reproduce; the two-slot loss did.
- Greedy fixed-vs-cost output hashes matched only 1/3 edit, 1/3 write, 2/3 prose responses; greedy edit/write were ~9% slower under cost routing. A hash difference is not proof of incorrect tokens (the benchmark hashes the entire response, including reasoning), but this is not a passing correctness control. Cost-routing logs show only 0-2 successful ngram rounds versus ~277-325 MTP rounds per request; they do not distinguish skipped ngram attempts from table misses. The policy is opt-in pending diagnosis of output differences and source selection.
- Greedy repeat on the target GPUs: both fixed-vs-fixed and cost-vs-cost matched 3/3 per task (9/9 each), with throughput within 0.4%. Fixed-vs-cost again matched 1/3 edit, 1/3 write and 2/3 prose; only `reasoning_content` hashes differed. All 18 responses had empty visible content at the 768-token limit, so 3/3 matching content hashes do not validate answers. Add a bounded per-request reasoning budget and require a visible completed response before treating a greedy comparison as an answer check. The ~9% greedy edit/write slowdown remains repeatable.
- Greedy with a 128-token reasoning budget and 2048-token response limit: edit 180.63 -> 145.32 t/s (-19.5%), write 138.53 -> 120.77 (-12.8%), prose 86.74 -> 89.03 (+2.6%). Completed visible write answers matched 0/3 and prose answers 2/3 across policies; edit answers matched 1/3 but all six edit responses hit the output limit, so none is a complete answer comparison. The same-policy 9/9 control was on the earlier unbounded-reasoning run, not this bounded run; do not attribute these answer differences solely to routing without a same-policy repeat. Higher draft acceptance with cost routing did not offset the greedy edit/write slowdown. Keep routing opt-in; the result does not support enabling it for the two-slot production workload.
- Shelved: the implementation remains available behind `LLAMA_SPEC_COST_ROUTING=1` for diagnosis, but no further tuning is planned until a credible performance and completed-answer correctness case emerges.

### 7. GDN MMA prefill with rollback snapshots (upstream ggml-org/llama.cpp#29353) - deferred

- The open upstream MMA kernel targets `K == 1`; its author confirmed that MTP prefill with snapshots (`K > 1`) cannot use it as submitted. The fork already runs the chunked kernel on the first `n_tokens - (K - 1)` tokens and the recurrent kernel on the final `K - 1` tokens.
- Wait for upstream support for `K > 1` instead of importing the current `K == 1`-only kernel and maintaining a separate adaptation. If it becomes available, compare its snapshot handling with the fork's prefix/tail path before porting it.
- Upstream reports about 10-13% prompt throughput over its recurrent baseline on Qwen3.8-27B Q4_K_M, but that is not a gain over this fork's existing chunked path. Any later port needs cache-fused snapshot and numeric checks in `test-backend-ops`, followed by server prefill, decode and long-context A/B runs with speculative decoding enabled (`-ub 512`, tensor split). No CUDA code has been imported.

## Considered, not now

- #29208 clamp the draft context to n_ctx_train with --kv-unified - dropped: no effect with our `-c 262144` (n_ctx_seq == n_ctx_train already), and it shrinks the draft KV below the target's in two cases: non-unified `-np N` (draft gets n_ctx/N split N ways) and unified pools above n_ctx_train (slots together can outgrow the clamped draft pool).
- #29187 fused GDN alpha/beta projections - upstream shows +8-16% decode on NVFP4 but 0-4% on Q4_K_M (fused path handles BF16/F16/F32/Q8_0 weights only); touches `gated_delta_net.cu` like our chunked patch. Revisit now that the recurrent kernel is upstream's again (#30087).

- #27248 q5 K cache - dropped: keep q8_0 K.
- #27210 adaptive MTP draft depth - wants `--spec-draft-n-max 12`, i.e. 13 recurrent snapshots per slot in VRAM; at temp 1.0 deep drafts rarely pay off and ngram-mod covers long bursts.
- #24785 recurrent shrink/expand for the prompt cache - overlaps with #25592, which we carry.
