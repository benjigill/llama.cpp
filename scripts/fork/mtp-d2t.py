#!/usr/bin/env python3
# Make a copy of a qwen35 GGUF whose MTP draft step uses a trimmed vocabulary (d2t).
# Background and usage: scripts/fork/README.md
#
# The MTP layer has no output projection of its own: every draft token multiplies against the main
# model's full output.weight (~248k rows). This writes a copy of the model with two extra tensors:
#   d2t                                   I64 [n_draft]       target token id of each draft row
#   blk.<N>.nextn.shared_head_head.weight [n_embd, n_draft]   those rows of output.weight, copied byte
#                                                             for byte (same quant type, no requant)
# for every MTP layer N. The target still uses the full output.weight; only the draft head shrinks.
# Output is unchanged: the target verifies every drafted token, a token outside the draft vocab is
# just never drafted (its draft logit is -inf).
#
# Which tokens are kept, in this order until --n-draft is reached:
#   1. every non-normal token (control, user-defined, byte) and every single-character token
#   2. the most frequent tokens of your traffic, from --table (an ngram-mod table file, whose cells
#      hold the tokens that followed each context) and/or --corpus (text tokenized by a running
#      llama-server with this model, --server)
#   3. the lowest token ids (BPE merge order, a rough frequency order)
#
# usage:
#   python3 scripts/fork/mtp-d2t.py <model.gguf> <model-d2t.gguf> [--n-draft 49152] [--table <table.bin>]
#                                   [--corpus <corpus.txt> --server http://127.0.0.1:8080]
#
# note: with --table/--corpus the output file is derived from your data; keep it out of the repo

import argparse
import json
import re
import struct
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))

import gguf  # noqa: E402

NGRAM_MOD_MAGIC = b"NGRAMMOD"
NGRAM_MOD_HDR   = struct.Struct("<8sIIIIQQ")  # magic, version, n, n_vocab, cell_size, size, used


def counts_from_table(path: str, n_vocab: int) -> Counter:
    with open(path, "rb") as f:
        magic, version, n, table_n_vocab, cell_size, size, used = NGRAM_MOD_HDR.unpack(f.read(NGRAM_MOD_HDR.size))
        if magic != NGRAM_MOD_MAGIC or version != 1 or cell_size != 8:
            sys.exit(f"{path}: not an ngram-mod table (version 1)")
        if table_n_vocab != n_vocab:
            sys.exit(f"{path}: table was built for n_vocab = {table_n_vocab}, the model has {n_vocab}")
        cells = np.fromfile(f, dtype=np.dtype([("key", "<u4"), ("token", "<i4")]), count=size)
    tokens = cells["token"]
    tokens = tokens[(tokens >= 0) & (tokens < n_vocab)]
    print(f"{path}: {len(tokens)} tokens from {size} cells ({used} used)", file=sys.stderr)
    return Counter(dict(zip(*np.unique(tokens, return_counts=True))))


def counts_from_corpus(path: str, server: str) -> Counter:
    counts = Counter()
    url = server.rstrip("/") + "/tokenize"
    data = Path(path).read_bytes().decode("utf-8", "replace")
    n_tokens = 0
    for doc in data.split("\0"):
        # the tokenizer is linear in the input, chunk long documents to keep requests small
        for i in range(0, len(doc), 1 << 20):
            chunk = doc[i:i + (1 << 20)]
            if not chunk:
                continue
            body = json.dumps({"content": chunk, "add_special": False, "parse_special": True}).encode()
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                tokens = json.load(r)["tokens"]
            counts.update(tokens)
            n_tokens += len(tokens)
    print(f"{path}: {n_tokens} tokens", file=sys.stderr)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="add a trimmed MTP draft head (d2t) to a qwen35 GGUF (see the header of this file)")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--n-draft", type=int, default=49152, help="draft vocabulary size (default: 49152)")
    ap.add_argument("--table", action="append", default=[], help="ngram-mod table file to count tokens from")
    ap.add_argument("--corpus", action="append", default=[], help="text file (NUL-separated documents) to count tokens from")
    ap.add_argument("--server", help="llama-server URL with this model, for --corpus")
    args = ap.parse_args()

    if args.corpus and not args.server:
        ap.error("--corpus needs --server")

    reader = gguf.GGUFReader(args.input)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    if arch != "qwen35":
        sys.exit(f"architecture {arch} is not supported (qwen35 only)")

    tensors = {t.name: t for t in reader.tensors}
    if "d2t" in tensors or any(".nextn.shared_head_head." in name for name in tensors):
        sys.exit("the model already has a d2t or nextn.shared_head_head tensor")

    mtp_layers = sorted(int(m.group(1)) for name in tensors if (m := re.fullmatch(r"blk\.(\d+)\.nextn\.eh_proj\.weight", name)))
    if not mtp_layers:
        sys.exit("the model has no embedded MTP layer")

    head = tensors.get("output.weight") or tensors["token_embd.weight"]
    n_vocab = int(head.shape[1])

    tok_texts = reader.fields[gguf.Keys.Tokenizer.LIST].contents()
    tok_types = reader.fields[gguf.Keys.Tokenizer.TOKEN_TYPE].contents() if gguf.Keys.Tokenizer.TOKEN_TYPE in reader.fields \
        else [gguf.TokenType.NORMAL] * len(tok_texts)
    if len(tok_types) != n_vocab:
        print(f"note: {len(tok_types)} tokenizer tokens, {n_vocab} head rows", file=sys.stderr)

    n_draft = min(args.n_draft, n_vocab)
    keep = set()
    for i, (ttype, text) in enumerate(zip(tok_types, tok_texts)):
        if i < n_vocab and (ttype != gguf.TokenType.NORMAL or len(text) == 1):
            keep.add(i)
    n_base = len(keep)

    counts = Counter()
    for path in args.table:
        counts.update(counts_from_table(path, n_vocab))
    for path in args.corpus:
        counts.update(counts_from_corpus(path, args.server))

    for tok, _ in counts.most_common():
        if len(keep) >= n_draft:
            break
        keep.add(int(tok))
    n_counted = len(keep) - n_base

    for tok in range(n_vocab):
        if len(keep) >= n_draft:
            break
        keep.add(tok)

    d2t = np.array(sorted(keep), dtype=np.int64)

    covered = sum(c for t, c in counts.items() if int(t) in keep)
    total   = sum(counts.values())
    print(f"draft vocab: {len(d2t)} of {n_vocab} tokens ({n_base} special/single-char, {n_counted} by frequency, "
          f"{len(d2t) - n_base - n_counted} by id)" + (f", covers {100.0 * covered / total:.2f}% of the counted tokens" if total else ""),
          file=sys.stderr)

    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        if val_type == gguf.GGUFValueType.ARRAY and not field.data:
            print(f"note: dropping empty array {field.name} (gguf-py cannot write it)", file=sys.stderr)
            continue
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    # rows are contiguous in every type (quantized blocks never span rows), so gathering whole rows of
    # the raw data keeps the exact weights
    head_rows = np.ascontiguousarray(head.data[d2t])

    out = [(t.name, t.data, t.tensor_type) for t in reader.tensors]
    for il in mtp_layers:
        out.append((f"blk.{il}.nextn.shared_head_head.weight", head_rows, head.tensor_type))
    out.append(("d2t", d2t, gguf.GGMLQuantizationType.I64))

    for name, data, qtype in out:
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, data, _ in out:
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()

    print(f"wrote {args.output}: head {head.name} {head.tensor_type.name} -> {len(mtp_layers)} x "
          f"[{int(head.shape[0])}, {len(d2t)}] ({head_rows.nbytes / 1024 / 1024:.1f} MiB each)", file=sys.stderr)


if __name__ == "__main__":
    main()
