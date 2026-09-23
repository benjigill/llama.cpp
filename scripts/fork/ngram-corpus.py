#!/usr/bin/env python3
# Build a text corpus for llama-ngram-mod-build (a table for --spec-ngram-mod-file).
#
# Sources are processed in command line order, and later documents win a shared table bucket,
# so list the most relevant material last:
#   --repo DIR     tracked text files of a git repo (git ls-files), one document per file
#   --text PATH    a text file, or every text file under a directory, one document per file
#   --chat FILE    JSONL of OpenAI chat requests, one per line: {"messages": [...], "tools": [...]};
#                  an optional "response" (a chat completion) is appended as the last assistant turn.
#                  Rendered with the chat template of a running llama-server (--server), so the
#                  special tokens and tool-call format match what the server sees.
#
# Output: documents separated by NUL bytes.
#
# usage:
#   python3 scripts/fork/ngram-corpus.py -o corpus.txt --repo <repo> --chat <runs.jsonl> --server http://127.0.0.1:8080
#   build/bin/llama-ngram-mod-build -m <model.gguf> -o <table.bin> --size 4096 corpus.txt
#
# note: the corpus and the table contain your data; keep them out of the repo

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

SKIP_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock", "uv.lock", "go.sum"}
SKIP_SUFFIXES = {".min.js", ".min.css", ".map", ".svg", ".gguf", ".bin", ".pdf"}


def read_text(path: Path, max_bytes: int) -> str | None:
    if path.name in SKIP_NAMES or any(path.name.endswith(s) for s in SKIP_SUFFIXES):
        return None
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def docs_repo(root: str, max_bytes: int):
    files = subprocess.run(["git", "-C", root, "ls-files", "-z"], check=True, capture_output=True).stdout
    for name in files.split(b"\0"):
        if name:
            text = read_text(Path(root) / name.decode("utf-8", "replace"), max_bytes)
            if text:
                yield text


def docs_text(path: str, max_bytes: int):
    p = Path(path)
    for f in sorted(p.rglob("*")) if p.is_dir() else [p]:
        text = read_text(f, max_bytes)
        if text:
            yield text


def apply_template(server: str, body: dict) -> str:
    req = urllib.request.Request(server.rstrip("/") + "/apply-template", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["prompt"]


def docs_chat(path: str, server: str | None):
    if not server:
        sys.exit("--chat needs --server (a running llama-server with the target model, for its chat template)")
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                messages = list(rec["messages"])
                resp = rec.get("response")
                if resp and resp.get("choices"):
                    messages.append(resp["choices"][0]["message"])
                body = {"messages": messages}
                if rec.get("tools"):
                    body["tools"] = rec["tools"]
                yield apply_template(server, body)
            except Exception as e:  # keep going on a bad line
                print(f"{path}:{i}: skipped ({e})", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="build a NUL-separated corpus for llama-ngram-mod-build (see the header of this file)")
    ap.add_argument("-o", "--output", required=True, help="corpus file to write ('-' for stdout)")
    ap.add_argument("--repo", dest="sources", action="append", type=lambda p: ("repo", p), default=[])
    ap.add_argument("--text", dest="sources", action="append", type=lambda p: ("text", p))
    ap.add_argument("--chat", dest="sources", action="append", type=lambda p: ("chat", p))
    ap.add_argument("--server", help="llama-server URL for --chat")
    ap.add_argument("--max-file-kb", type=int, default=512, help="skip larger files (default: 512)")
    args = ap.parse_args()

    if not args.sources:
        ap.error("no sources, use --repo, --text or --chat")

    max_bytes = args.max_file_kb * 1024
    out = sys.stdout.buffer if args.output == "-" else open(args.output, "wb")

    seen = set()
    n_docs, n_dup, n_bytes = 0, 0, 0
    for kind, path in args.sources:
        gen = docs_repo(path, max_bytes) if kind == "repo" else docs_text(path, max_bytes) if kind == "text" else docs_chat(path, args.server)
        n_src = 0
        for text in gen:
            h = hashlib.blake2b(text.encode(), digest_size=16).digest()
            if h in seen:
                n_dup += 1
                continue
            seen.add(h)
            data = text.replace("\0", "").encode()
            out.write(data + b"\0")
            n_docs, n_src, n_bytes = n_docs + 1, n_src + 1, n_bytes + len(data)
        print(f"{kind} {path}: {n_src} documents", file=sys.stderr)

    if out is not sys.stdout.buffer:
        out.close()
    print(f"{n_docs} documents ({n_dup} duplicates skipped), {n_bytes / 1024 / 1024:.1f} MiB", file=sys.stderr)


if __name__ == "__main__":
    main()
