# Fork scripts

Tools for this fork's target setup (Qwen3.8-27B Q4_K_M, 2 GPUs, `llama-server` with MTP + ngram-mod speculation). The Fork Policy section of `AGENTS.md` has the list of carried patches; this file explains how to use the pieces that need a manual step.

Everything these scripts produce from your own data (corpora, ngram tables, trimmed model copies, bench results) is private: keep it out of the repo.

| Script | What it's for |
| --- | --- |
| `bench_qwen38.py` | A/B benchmark of two builds or two models (`run`, then `compare`) |
| `ngram-corpus.py` | Collect repos, text and chat transcripts into a corpus for an ngram-mod table |
| `mtp-d2t.py` | Make a model copy whose MTP draft step uses a trimmed vocabulary (faster drafts) |
| `mmproj-q8_0.py` | Requantize an F16 mmproj (vision projector) GGUF to Q8_0 |
| `leak-check.sh` | Commit hook that blocks local paths, emails and other private strings |

Each script's header comment has the full usage.

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

- `blk.<N>.nextn.shared_head_head.weight`: a projection used only for drafting, with the rows of `output.weight` for ~32k tokens, copied exactly (same quantization);
- `d2t`: the real token id of each of those rows.

When the file has `d2t`, the draft step uses the small projection and the scores of all other tokens are set to -inf. The target model keeps the full `output.weight`. Output quality does not change: the target still checks every drafted token, and a token outside the 32k can't be drafted but is still generated normally by the target. With the original GGUF nothing changes; the trim only exists in the copy.

**Which tokens are kept**, until `--n-draft` is reached: all special tokens (tool-call and think tags, etc.) and single characters, then the most frequent tokens of your traffic (from your ngram-mod table and/or a corpus), then the lowest token ids.

```sh
python3 scripts/fork/mtp-d2t.py <model.gguf> <model-d2t.gguf> --n-draft 32768 --table <table.bin>
# optional, count tokens from a corpus through a running llama-server with the same model:
#   --corpus corpus.txt --server http://127.0.0.1:8080
```

Then start `llama-server` with `-m <model-d2t.gguf>` (everything else unchanged; the ngram-mod table still works, the vocabulary size is the same). The load log shows `QWEN35 MTP using d2t draft-vocab trim of the embedded head (n_vocab_mtp = ...)`.

- Rebuild the copy when your traffic changes a lot (run it again from the original GGUF with a fresh table), or when you update the base model.
- It needs disk space for a full model copy. Under `-sm tensor` the small projection is kept whole on each GPU (~130 MB each).
