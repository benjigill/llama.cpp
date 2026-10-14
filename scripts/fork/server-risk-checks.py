#!/usr/bin/env python3
# Reproduce upstream #27756 and #27572 against an idle server. Stdlib only.

import argparse
import concurrent.futures
import json
import sys
import threading
import urllib.request
from pathlib import Path


def request(server, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(server.rstrip("/") + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as response:
        return json.load(response)


def tokenize(server, text):
    tokens = request(server, "/tokenize", {"content": text, "add_special": False})["tokens"]
    if not tokens:
        raise ValueError("input text produced no tokens")
    return tokens


def fill_tokens(tokens, count, offset=0):
    return [tokens[(i + offset) % len(tokens)] for i in range(count)]


def complete(server, prompt, n_predict, slot=0):
    result = request(server, "/completion", {
        "prompt": prompt, "n_predict": n_predict, "id_slot": slot,
        "temperature": 0, "top_k": 1, "cache_prompt": False,
    })
    if result["tokens_evaluated"] != len(prompt):
        raise ValueError(f"prompt depth mismatch: sent {len(prompt)}, evaluated {result['tokens_evaluated']}")
    return result


def check_recall(server, filler, depths):
    code = "BLUE-HARBOR-7342"
    needle = tokenize(server, f"\nThe secret launch code is {code}.\n")
    question = tokenize(server, "\n\nQuestion: what is the secret launch code? Answer:")
    failed = False
    for depth in depths:
        count = depth - len(needle) - len(question)
        if count < 1:
            raise ValueError("depth must exceed the needle and question token count")
        body = fill_tokens(filler, count)
        cut = count // 10
        prompt = body[:cut] + needle + body[cut:] + question
        result = complete(server, prompt, 256)
        recalled = code in result["content"]
        failed |= not recalled
        print(json.dumps({
            "issue": 27756, "tokens_evaluated": result["tokens_evaluated"],
            "tokens_predicted": result["tokens_predicted"], "stop_type": result["stop_type"],
            "recalled": recalled,
            "silent_eos": result["stop_type"] == "eos" and not result["content"].strip(),
        }), flush=True)
    return failed


def check_parallel(server, filler, rounds, depth, slot_ids):
    question = tokenize(server, "\n\nSummarize the text above in detail:")
    if depth <= len(question):
        raise ValueError("depth must exceed the question token count")
    failed = False
    for round_id in range(rounds):
        barrier = threading.Barrier(2)

        def run(slot):
            prompt = fill_tokens(filler, depth - len(question), (round_id * 2 + slot) * 2000) + question
            barrier.wait(timeout=30)
            result = complete(server, prompt, 256, slot)
            timings = result["timings"]
            draft = timings["draft_n"]
            accepted = timings["draft_n_accepted"]
            if result["tokens_predicted"] < 8 or draft < 8:
                raise ValueError(f"slot {slot}: insufficient generation/drafting to check acceptance; use draft-mtp only")
            if not 0 <= accepted <= draft:
                raise ValueError(f"slot {slot}: invalid draft acceptance counters")
            return {
                "issue": 27572, "round": round_id + 1, "slot": slot,
                "tokens_evaluated": result["tokens_evaluated"],
                "tokens_predicted": result["tokens_predicted"],
                "draft_n": draft, "accepted": accepted, "acceptance": accepted / draft,
                "zero_acceptance": accepted == 0,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, slot_ids))
        for result in results:
            failed |= result["zero_acceptance"]
            print(json.dumps(result), flush=True)
    return failed


def main():
    parser = argparse.ArgumentParser(description="Probe long-context recall or two-slot MTP acceptance on an idle server.")
    parser.add_argument("check", choices=["recall", "parallel"])
    parser.add_argument("--server", default="http://127.0.0.1:8090")
    parser.add_argument("--text", type=Path, required=True, help="filler corpus, e.g. wiki.test.raw")
    parser.add_argument("--depths", type=int, nargs="+", default=[96000, 96519, 100000, 120000, 140000])
    parser.add_argument("--parallel-depth", type=int, default=4096, help="must exceed the server ubatch size")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1 or args.parallel_depth < 1 or any(depth < 1 for depth in args.depths):
        parser.error("depths and rounds must be positive")
    slots = request(args.server, "/slots")
    if not slots or any(slot["is_processing"] for slot in slots):
        raise ValueError("server must have idle slots and no competing clients")
    slot_ids = sorted(slot["id"] for slot in slots)
    if args.check == "parallel" and slot_ids != [0, 1]:
        raise ValueError("parallel check requires exactly two slots (-np 2) and --slots")
    if slot_ids[0] != 0:
        raise ValueError("recall check requires slot 0")
    filler = tokenize(args.server, args.text.read_text(encoding="utf-8"))
    if args.check == "recall":
        failed = check_recall(args.server, filler, args.depths)
    else:
        failed = check_parallel(args.server, filler, args.rounds, args.parallel_depth, slot_ids)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, threading.BrokenBarrierError) as error:
        print(f"check failed or inconclusive: {error}", file=sys.stderr)
        sys.exit(2)
