# Fork scripts

Tools for this fork's target setup (Qwen3.8-27B Q4_K_M, 2 GPUs, `llama-server` with MTP + ngram-mod speculation). The Fork Policy section of `AGENTS.md` has the list of carried patches; this file explains how to use the pieces that need a manual step.

Everything these scripts produce from your own data (corpora, ngram tables, trimmed model copies, bench results) is private: keep it out of the repo.

| Script | What it's for |
| --- | --- |
| `bench_qwen38.py` | A/B benchmark of two builds or two models (`run`, then `compare`) |
| `server-risk-checks.py` | Probe long-context recall and per-slot MTP acceptance against a running server |
| `ngram-corpus.py` | Collect repos, text and chat transcripts into a corpus for an ngram-mod table |
| `mtp-d2t.py` | Make a model copy whose MTP draft step uses a trimmed vocabulary (faster drafts) |
| `mmproj-q8_0.py` | Requantize an F16 mmproj (vision projector) GGUF to Q8_0 |
| `leak-check.sh` | Commit hook that blocks local paths, emails and other private strings |

Each script's header comment has the full usage.

## Mixed FP8/NVFP4 conversion

The HF converter supports compressed-tensors checkpoints with mixed per-channel FP8 and packed NVFP4 groups. NVFP4 weights are repacked directly into GGUF without re-quantization; their global scales are preserved as separate tensors. FP8 weights are dequantized and written using the selected `--outtype`.

```sh
python3 convert_hf_to_gguf.py <checkpoint-dir> --outfile <model.gguf> --outtype bf16
```

Add `--fp8-as-q8` to store only the FP8-origin weights as Q8_0 instead of BF16. This adds re-quantization error but reduces their memory footprint; the NVFP4 weights stay unchanged. There is no `--outtype nvfp4`: the converter detects packed NVFP4 weights automatically, and `--outtype` controls the other weights. Only NVFP4 plus per-channel FP8 has been tested; other mixed-precision groups are not checked by the converter.

## Production-like latency and long-context benchmark

`bench_qwen38.py run --production` uses the server startup's 262144-token shared KV pool, two slots capped at 160000 tokens each, probabilistic MTP drafting, draft KV types and CUDA checkpoint settings. It leaves the existing micro and batched suites unchanged. Pass `--mmproj <mmproj.gguf>` to match the multimodal load, and `--ngram-table <table.bin>` to match the primed ngram-mod table. The script copies that table for each server start, so benchmarking never changes the original.

`--suite long-context` measures streamed time to the first generated content or reasoning chunk (not the initial role event), prompt throughput and cache reuse at approximately 8192, 32768, 128000 and 155000 tokens. The script measures actual chat-template token counts before sending each prompt, records processed and cached tokens, and exercises two simultaneous slots through 128000 tokens when `--production` is set. It does not try two 155000-token slots: they cannot both fit in the 262144-token pool. Each depth gets one cold request and one warm request; the context suite is intentionally costly.

```sh
python3 scripts/fork/bench_qwen38.py run --bin <base-build/bin> --label baseline --model <model.gguf> --mmproj <mmproj.gguf> --production --ngram-table <table.bin> --suite long-context,decode,parallel
python3 scripts/fork/bench_qwen38.py run --bin <candidate-build/bin> --label candidate --model <model.gguf> --mmproj <mmproj.gguf> --production --ngram-table <table.bin> --suite long-context,decode,parallel
python3 scripts/fork/bench_qwen38.py compare bench-results/baseline bench-results/candidate
```

Use `--suite decode,parallel` for a shorter decode/concurrency run, or `--server-extra "--spec-type draft-mtp"` on both builds to isolate MTP from ngram-mod. Stop other servers using the GPUs first. Results and server logs go under `bench-results/` and stay private; compare the same suite and flags on both builds.

## Server stability probes

`server-risk-checks.py` exercises upstream ggml-org/llama.cpp#27756 (silent EOS or lost recall at scattered long-context depths) and ggml-org/llama.cpp#27572 (zero MTP acceptance in a parallel slot). These are reproduction checks, not fixes. Use an idle server with `--slots` enabled and no other clients. Use the original model first, not the d2t copy. The script sends greedy raw completions, pins slot IDs and disables prompt reuse; it does not change or restart the server.

```sh
bash scripts/get-wikitext-2.sh
python3 scripts/fork/server-risk-checks.py recall --server http://127.0.0.1:<port> --text wikitext-2-raw/wiki.test.raw
python3 scripts/fork/server-risk-checks.py parallel --server http://127.0.0.1:<port> --text wikitext-2-raw/wiki.test.raw --rounds 3
```

The recall probe repeats tokenized corpus text to reach exact prompt depths of 96000, 96519, 100000, 120000 and 140000 tokens, with the needle at 10% and 256 output tokens reserved. Run it with a slot capacity of at least 140256 tokens (e.g. the production 160000 cap), then repeat with speculation disabled to isolate the long-context bug. A missing code is a failed probe, not proof of an engine defect: reasoning can exhaust the response budget, and prompt dependence requires more depths and prompts. `--depths` overrides the depths.

The parallel probe needs exactly two slots (`-np 2`), `--spec-type draft-mtp` without ngram-mod, and `--spec-draft-n-max 3`. Both requests have 4096 actual prompt tokens, so they span multiple ubatches with `-ub 512`; change `--parallel-depth` if needed. It reports each slot's draft and accepted counts over three concurrent rounds. Zero acceptance with at least eight generated and eight drafted tokens flags the reported failure signature; restart with `-np 1` and compare acceptance separately before diagnosing a race.

JSON summaries go to stdout, without response text. Exit codes: 0 = no failure signature observed, 1 = missing needle or zero acceptance, 2 = invalid setup, insufficient drafting, token-count mismatch or request failure. Passing probes do not rule out either intermittent bug.

## Pending CUDA adoption checks

The checkpoint change from ggml-org/llama.cpp#29509 and graph recapture change from ggml-org/llama.cpp#29768 need target-GPU A/B before retention. Build the unchanged baseline and candidate separately with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_FA_QUANTS=q8_0-q4_0`; use the same model, mmproj, ngram table and environment.

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_FA_QUANTS=q8_0-q4_0 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python3 scripts/fork/bench_qwen38.py run --bin <baseline-build/bin> --label scout-base --model <model.gguf> --mmproj <mmproj.gguf> --production --ngram-table <table.bin> --suite micro,decode,parallel,cache
python3 scripts/fork/bench_qwen38.py run --bin build/bin --label scout-candidate --model <model.gguf> --mmproj <mmproj.gguf> --production --ngram-table <table.bin> --suite micro,decode,parallel,cache
python3 scripts/fork/bench_qwen38.py compare bench-results/scout-base bench-results/scout-candidate
```

Before the graph A/B, confirm both GPU startup lines report VMM and count `CUDA graph warmup reset` lines in baseline logs from a production-like debug-verbosity run (`-lv 4` or higher). A Release build still logs these lines at debug verbosity. Check again with the candidate; rare baseline resets mean little potential gain. The benchmark sets `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` in production mode, but `new_pool_for_device` still selects the VMM scratch pool when compiled with `GGML_USE_VMM` and the device reports VMM support.

For checkpoint memory, use a long prompt with MTP and inspect `created context checkpoint ... size = ... MiB` logs: size should no longer grow with draft KV length. Exercise warm prefix reuse, a divergent continuation that restores an earlier checkpoint, RAM-cache eviction/reload and slot save/restore; compare processed-token counts and completed greedy answers with baseline. Repeat same-build greedy checks first because the separately tracked MTP hidden-state restore bug can confound cross-build hashes. The updated d2t loader also needs original-vs-trimmed decode/parallel checks, with MTP enabled and disabled.

## Ngram-mod / MTP cost routing

When both `ngram-mod` and `draft-mtp` are the only configured speculators, the server chooses per slot using measured time from drafting through verification and any checkpoint replay, divided by the number of tokens produced. Each source has a 1/8-weight moving estimate; MTP must be at least 5% faster to displace ngram. It tries ngram first, then MTP to establish estimates, and probes the other choice every 32 rounds. An ngram miss still falls through to MTP. The estimates reset with each request and appear in the slot timing log. Other speculative configurations keep their existing ordering and cooldown.

The fixed ngram-first policy and low-acceptance cooldown remain the default (unset or `LLAMA_SPEC_COST_ROUTING=0`); `LLAMA_SPEC_COST_ROUTING=1` opts into experimental cost routing. Set the variable on the `bench_qwen38.py run` command so its child server inherits it. Compare both decode and parallel suites with the same model, table and seeds; the policy affects decode, not cold prefill.

Repeat the A/B runs in alternating order before interpreting small changes, especially for `-np 2`. The `--reps` flag repeats decode requests but not the six-request parallel suite. To check greedy correctness, run `--suite decode --greedy-check` with each policy and compare the results. This sends temperature 0 requests and stores hashes of complete responses and separate content, reasoning, tool-call and finish-reason hashes, not response text. `compare` reports exact and per-field match counts and fails if any complete responses differ. With the production reasoning budget and 768-token response limit, all nine fixed-policy responses and all nine cost-policy responses ended before visible content; matching empty content hashes do not validate answers. Use `--greedy-reasoning-budget 128` to limit reasoning for this check only. A 2048-token limit was insufficient to finish the long edit task in either policy; use a larger `--max-tokens` for a completed-answer check. The comparison also requires visible, completed answers and refuses to compare different budgets or token limits. This shorter reasoning workload is a control, not a correctness check of the full production reasoning trace or a distribution test at temperature 1.

```sh
LLAMA_SPEC_COST_ROUTING=0 python3 scripts/fork/bench_qwen38.py run --bin build/bin --label spec-fixed --model "$M" --mmproj "$MP" --production --ngram-table "$NG" --suite decode,parallel
LLAMA_SPEC_COST_ROUTING=1 python3 scripts/fork/bench_qwen38.py run --bin build/bin --label spec-cost --model "$M" --mmproj "$MP" --production --ngram-table "$NG" --suite decode,parallel
python3 scripts/fork/bench_qwen38.py compare bench-results/spec-fixed bench-results/spec-cost
LLAMA_SPEC_COST_ROUTING=0 python3 scripts/fork/bench_qwen38.py run --bin build/bin --label spec-fixed-complete --model "$M" --mmproj "$MP" --production --ngram-table "$NG" --suite decode --greedy-check --greedy-reasoning-budget 128 --max-tokens 8192
LLAMA_SPEC_COST_ROUTING=1 python3 scripts/fork/bench_qwen38.py run --bin build/bin --label spec-cost-complete --model "$M" --mmproj "$MP" --production --ngram-table "$NG" --suite decode --greedy-check --greedy-reasoning-budget 128 --max-tokens 8192
python3 scripts/fork/bench_qwen38.py compare bench-results/spec-fixed-complete bench-results/spec-cost-complete
```

If greedy output differs, rerun each policy with a new label and compare it against itself before interpreting the cross-policy difference. Increase `--max-tokens` on both runs if the comparison reports unfinished answers. Leave cost routing disabled for production until it is validated.

## Persistent, primed ngram-mod table

`--spec-ngram-mod-file <table.bin>` loads the ngram-mod table at startup and saves it at clean shutdown, so what the server learned from your automations survives restarts. The file is binary (use `.bin`) and always the full table size; the log line `ngram_mod table loaded from ...: <used>/<size> cells used` shows how full it is.

- First start (no file yet): the table size comes from `--spec-ngram-mod-size` (MiB, 8 bytes per n-gram). After that the file decides the size.
- To prime it from your own material, build a corpus and then the table:

```sh
python3 scripts/fork/ngram-corpus.py -o corpus.txt --repo <repo> --chat <runs.jsonl> --server http://127.0.0.1:8080
build/bin/llama-ngram-mod-build -m <model.gguf> -o <table.bin> --size 4096 corpus.txt
```

`--append` extends an existing table. Plan the table at 3-4x the corpus size in MiB so it stays under half full.

## Trimmed MTP draft vocabulary (`mtp-d2t.py`)

**What the MTP draft does.** Qwen3.x GGUFs carry one extra MTP layer (`blk.<N>.nextn.*`) that predicts the next token cheaply. A draft step has two parts:

1. the MTP layer itself (one attention + FFN block), which turns the last hidden state into a new one, and
2. the output projection, which turns that hidden state into a score for every token in the vocabulary.

The model has no projection of its own for step 2: the MTP layer reuses the main model's `output.weight`, one row per token, about 248k rows. That makes step 2 about 3x as expensive as step 1 (roughly 1 GB of weights read per draft token), although almost all of those tokens never come up.

**What the script does.** It writes a copy of the model with two extra tensors:

- `blk.<N>.nextn.shared_head_head.weight`: a projection used only for drafting, with the rows of `output.weight` for ~49k tokens (`--n-draft`, default 49152), copied exactly (same quantization);
- `d2t`: the real token id of each of those rows.

When the file has `d2t`, the draft step uses the small projection and the scores of all other tokens are set to -inf. The target model keeps the full `output.weight`. Output quality does not change: the target still checks every drafted token, and a token outside the trimmed set can't be drafted but is still generated normally by the target. With the original GGUF nothing changes; the trim only exists in the copy.

**Which tokens are kept**, until `--n-draft` is reached: all special tokens (tool-call and think tags, etc.) and single characters, then the most frequent tokens of your traffic (from your ngram-mod table and/or a corpus), then the lowest token ids.

```sh
python3 scripts/fork/mtp-d2t.py <model.gguf> <model-d2t.gguf> --n-draft 49152 --table <table.bin>
# optional, count tokens from a corpus through a running llama-server with the same model:
#   --corpus corpus.txt --server http://127.0.0.1:8080
```

Then start `llama-server` with `-m <model-d2t.gguf>` (everything else unchanged; the ngram-mod table still works, the vocabulary size is the same). The load log shows `QWEN35 MTP using d2t draft-vocab trim of the embedded head (n_vocab_mtp = ...)`.

- Rebuild the copy when your traffic changes a lot (run it again from the original GGUF with a fresh table), or when you update the base model.
- It needs disk space for a full model copy. Under `-sm tensor` the small projection is kept whole on each GPU (~200 MB each at 49152).
- Measured on the target machine: 49152 kept greedy acceptance closer to the original than 32768 at the same speed; with probabilistic drafting decode was +2.6 to +8.5% faster. Smaller sets draft faster but miss more tokens.
