#!/usr/bin/env python3
# A/B benchmark for the Qwen3.8-27B fork. Stdlib only.
#
#   run:     bench_qwen38.py run --bin <build/bin> --label <name> --model <gguf> [--mmproj <gguf>] [--suite ...]
#   compare: bench_qwen38.py compare <results/a> <results/b>
#
# Stop any other llama-server using the GPUs first, or the numbers are meaningless.

import argparse
import concurrent.futures
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# mirrors the production launch; model/mmproj/port/np/spec flags are added per test
SERVER_ARGS = [
    "--cache-type-k", "q8_0", "--cache-type-v", "q4_0",
    "-c", "180000", "--split-mode", "tensor", "--tensor-split", "15,13", "--main-gpu", "1",
    "-b", "2048", "-ub", "512",
    "--spec-type", "draft-mtp,ngram-mod", "--spec-draft-n-max", "3",
    "--spec-ngram-mod-n-max", "64", "--spec-ngram-mod-n-min", "8", "--spec-ngram-mod-n-match", "24",
    "-t", "8", "--threads-batch", "12", "--flash-attn", "on", "--jinja", "--no-slots",
    "--n-gpu-layers", "999", "--cache-ram", "80000",
]

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repeat_penalty": 1.0}

ENV_DEFAULTS = {
    "GGML_CUDA_GRAPH_OPT": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "CUDA_SCALE_LAUNCH_QUEUES": "4x",
    "GGML_CUDA_P2P": "1",
    "CUDA_VISIBLE_DEVICES": "0,1",
}

PORT = 11555


def env():
    e = dict(os.environ)
    for k, v in ENV_DEFAULTS.items():
        e.setdefault(k, v)
    return e


def log(msg):
    print(f"[bench] {msg}", flush=True)


# ---------------------------------------------------------------------------
# kernel-level: llama-bench and llama-batched-bench


def run_micro(a, out):
    cmd = [str(Path(a.bin) / "llama-bench"), "-m", a.model,
           "-sm", "tensor", "-ts", "15/13", "-mg", "1", "-ngl", "999", "-fa", "1",
           "-ctk", "q8_0", "-ctv", "q4_0", "-b", "2048", "-ub", "512,1024,2048",
           "-p", "4096", "-n", "128", "-d", "0,32768", "-r", str(a.reps), "-o", "json"]
    log(" ".join(cmd))
    res = subprocess.run(cmd, env=env(), capture_output=True, text=True)
    if res.returncode != 0:
        log(res.stderr[-3000:])
        raise SystemExit("llama-bench failed")
    rows = []
    for r in json.loads(res.stdout):
        test = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
        rows.append({"test": f"{test} d{r['n_depth']} ub{r['n_ubatch']}", "tps": r["avg_ts"], "std": r["stddev_ts"]})
    (out / "micro.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        log(f"  {r['test']:<24} {r['tps']:9.1f} +- {r['std']:.1f}")


def run_batched(a, out):
    cmd = [str(Path(a.bin) / "llama-batched-bench"), "-m", a.model,
           "-sm", "tensor", "-ts", "15,13", "-mg", "1", "-ngl", "999", "-fa", "on",
           "-ctk", "q8_0", "-ctv", "q4_0", "-c", "32768", "-b", "2048", "-ub", "512", "--kv-unified",
           "-npp", "2048", "-ntg", "256", "-npl", "1,2,4", "--output-format", "jsonl"]
    log(" ".join(cmd))
    res = subprocess.run(cmd, env=env(), capture_output=True, text=True)
    if res.returncode != 0:
        log(res.stderr[-3000:])
        raise SystemExit("llama-batched-bench failed")
    rows = []
    # the jsonl rows go through the logger, which may write to either stream
    for line in (res.stdout + "\n" + res.stderr).splitlines():
        line = line[line.find("{"):] if "{\"n_kv_max\"" in line or "\"speed_tg\"" in line else ""
        if line:
            r = json.loads(line)
            rows.append({"test": f"npl{r['pl']} tg total", "tps": r["speed_tg"]})
            rows.append({"test": f"npl{r['pl']} pp total", "tps": r["speed_pp"]})
    (out / "batched.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        log(f"  {r['test']:<24} {r['tps']:9.1f}")


# ---------------------------------------------------------------------------
# server-level


class Server:
    def __init__(self, a, out, name, extra):
        self.cmd = [str(Path(a.bin) / "llama-server"), "--model", a.model, "--host", "127.0.0.1", "--port", str(PORT)]
        if a.mmproj:
            self.cmd += ["--mmproj", a.mmproj]
        self.cmd += SERVER_ARGS + extra
        self.logf = open(out / f"server-{name}.log", "w")

    def __enter__(self):
        log(" ".join(self.cmd))
        self.p = subprocess.Popen(self.cmd, env=env(), stdout=self.logf, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise SystemExit(f"llama-server exited early, see {self.logf.name}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(2)
        raise SystemExit("llama-server did not become healthy")

    def __exit__(self, *exc):
        os.killpg(self.p.pid, signal.SIGTERM)
        try:
            self.p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(self.p.pid, signal.SIGKILL)
        self.logf.close()
        time.sleep(3)


def chat(messages, max_tokens, seed):
    body = {"messages": messages, "max_tokens": max_tokens, "seed": seed, "stream": False, **SAMPLING}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.loads(r.read())
    wall = time.time() - t0
    t = res.get("timings", {})
    msg = res["choices"][0]["message"]
    return {
        "wall_s": wall,
        "prompt_n": t.get("prompt_n", 0), "cache_n": t.get("cache_n", 0),
        "prompt_ms": t.get("prompt_ms", 0.0), "prompt_tps": t.get("prompt_per_second", 0.0),
        "gen_n": t.get("predicted_n", 0), "gen_tps": t.get("predicted_per_second", 0.0),
        "draft_n": t.get("draft_n", 0), "draft_acc": t.get("draft_n_accepted", 0),
        "content": msg.get("content") or "",
    }


def system_prompt(n_chars):
    # deterministic, code-heavy stand-in for a long shared system prompt
    src = ""
    for f in ["src/llama-context.cpp", "src/llama-model.cpp", "tools/server/server-context.cpp"]:
        src += (REPO / f).read_text(errors="ignore")
        if len(src) >= n_chars:
            break
    return "You are a senior C++ engineer. Reference code follows.\n\n```cpp\n" + src[:n_chars] + "\n```"


FUNC = None


def some_function():
    global FUNC
    if FUNC is None:
        text = (REPO / "src/llama-batch.cpp").read_text()
        m = re.search(r"\nbool llama_batch_allocr::init\(.*?\n}\n", text, re.S)
        FUNC = m.group(0) if m else text[:6000]
    return FUNC


DECODE_TASKS = [
    # high ngram overlap: output mostly repeats the input
    ("edit", lambda: "Return this function unchanged except rename every local variable to snake_case with a `v_` prefix. "
                     "Output only the full function in one code block.\n\n```cpp\n" + some_function() + "\n```"),
    # low overlap: new code
    ("write", lambda: "Write a C++17 thread-safe LRU cache class template with get/put/erase, a capacity limit and unit tests using assert. Output only code."),
    ("prose", lambda: "Explain in about 400 words how speculative decoding with a draft head keeps the output distribution unchanged."),
]


def summarize(rows):
    gen_n = sum(r["gen_n"] for r in rows)
    gen_s = sum(r["gen_n"] / r["gen_tps"] for r in rows if r["gen_tps"] > 0)
    dn = sum(r["draft_n"] for r in rows)
    da = sum(r["draft_acc"] for r in rows)
    return {"gen_tps": gen_n / gen_s if gen_s else 0.0, "accept": da / dn if dn else 0.0, "gen_n": gen_n}


def test_decode(a, out, name, extra):
    res = {}
    with Server(a, out, name, extra):
        chat([{"role": "user", "content": "hi"}], 16, 1)  # warmup
        for task, prompt in DECODE_TASKS:
            rows = [chat([{"role": "user", "content": prompt()}], a.max_tokens, 1000 + i) for i in range(a.reps)]
            res[task] = summarize(rows)
            log(f"  [{name}] {task:<6} tg {res[task]['gen_tps']:6.1f} t/s  accept {res[task]['accept']:.2f}")
    return res


def test_cache(a, out):
    # automations: same long system prompt, different user turns; then an agent-style follow-up
    # that drops the reasoning of the previous reply (the case that used to force re-processing)
    sp = system_prompt(a.sys_chars)
    res = {}
    with Server(a, out, "cache", []):
        cold = chat([{"role": "system", "content": sp}, {"role": "user", "content": "Summarize llama_context::decode in 3 bullets."}], 256, 1)
        res["cold"] = cold
        warm = []
        for i, q in enumerate(["List the public methods of llama_context.", "What does n_outputs control?", "Name 3 server slot states."]):
            warm.append(chat([{"role": "system", "content": sp}, {"role": "user", "content": q}], 256, 10 + i))
        res["warm"] = warm
        hist = [{"role": "system", "content": sp}, {"role": "user", "content": "List the public methods of llama_context."}]
        turn1 = chat(hist, 256, 20)
        hist += [{"role": "assistant", "content": turn1["content"]}, {"role": "user", "content": "Now describe the first one in detail."}]
        turn2 = chat(hist, 256, 21)
        hist += [{"role": "assistant", "content": turn2["content"]}, {"role": "user", "content": "And the second."}]
        turn3 = chat(hist, 256, 22)
        res["multiturn"] = [turn2, turn3]
    for r in [res["cold"]] + res["warm"] + res["multiturn"]:
        r.pop("content", None)
    log(f"  cold      prompt_n {cold['prompt_n']:6d}  {cold['prompt_tps']:8.1f} pp t/s  ttft {cold['prompt_ms']/1000:6.2f}s")
    for r in res["warm"]:
        log(f"  warm      prompt_n {r['prompt_n']:6d}  cache_n {r['cache_n']:6d}  ttft {r['prompt_ms']/1000:6.2f}s")
    for r in res["multiturn"]:
        log(f"  multiturn prompt_n {r['prompt_n']:6d}  cache_n {r['cache_n']:6d}  ttft {r['prompt_ms']/1000:6.2f}s")
    return res


def test_parallel(a, out, extra):
    res = {}
    for np_ in [1, 3]:
        with Server(a, out, f"np{np_}", ["-np", str(np_), "--kv-unified"] + extra):
            chat([{"role": "user", "content": "hi"}], 16, 1)
            prompts = [DECODE_TASKS[i % len(DECODE_TASKS)][1]() for i in range(6)]
            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(np_) as ex:
                rows = list(ex.map(lambda ip: chat([{"role": "user", "content": ip[1]}], a.max_tokens, 2000 + ip[0]), enumerate(prompts)))
            wall = time.time() - t0
            gen = sum(r["gen_n"] for r in rows)
            res[f"np{np_}"] = {"agg_tps": gen / wall, "per_req_tps": summarize(rows)["gen_tps"]}
            log(f"  np{np_}: aggregate {gen / wall:6.1f} t/s, per request {res[f'np{np_}']['per_req_tps']:6.1f} t/s")
    return res


def cmd_run(a):
    out = Path(a.out) / a.label
    out.mkdir(parents=True, exist_ok=True)
    extra = a.server_extra.split() if a.server_extra else []
    suites = a.suite.split(",")
    meta = {"label": a.label, "bin": a.bin, "model": a.model, "suites": suites, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        meta["version"] = subprocess.run([str(Path(a.bin) / "llama-server"), "--version"], capture_output=True, text=True).stderr.strip()
    except OSError:
        pass
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    if "micro" in suites:
        log("== micro (llama-bench)")
        run_micro(a, out)
    if "batched" in suites:
        log("== batched (llama-batched-bench)")
        run_batched(a, out)
    if "cache" in suites:
        log("== prompt cache")
        (out / "cache.json").write_text(json.dumps(test_cache(a, out), indent=1))
    if "decode" in suites:
        log("== decode (spec greedy)")
        res = {"greedy": test_decode(a, out, "greedy", extra)}
        if a.probabilistic:
            log("== decode (spec probabilistic)")
            res["probabilistic"] = test_decode(a, out, "prob", extra + ["--spec-draft-sampling", "probabilistic"])
        (out / "decode.json").write_text(json.dumps(res, indent=1))
    if "parallel" in suites:
        log("== parallel slots")
        (out / "parallel.json").write_text(json.dumps(test_parallel(a, out, extra), indent=1))
    log(f"results in {out}")


# ---------------------------------------------------------------------------


def load(d, name):
    p = Path(d) / name
    return json.loads(p.read_text()) if p.exists() else None


def row(name, x, y, unit="", higher=True):
    if x is None or y is None:
        return
    d = (y - x) / x * 100 if x else 0.0
    better = (d > 0) == higher
    mark = "" if abs(d) < 2 else ("  <-- better" if better else "  <-- WORSE")
    print(f"  {name:<34} {x:10.2f} {y:10.2f} {d:+7.1f}%{unit}{mark}")


def cmd_compare(a):
    A, B = a.a, a.b
    print(f"{'':36} {Path(A).name:>10} {Path(B).name:>10}  delta")
    for f in ["micro.json", "batched.json"]:
        ra, rb = load(A, f), load(B, f)
        if ra and rb:
            print(f"\n{f}")
            mb = {r["test"]: r["tps"] for r in rb}
            for r in ra:
                row(r["test"] + " t/s", r["tps"], mb.get(r["test"]))
    ca, cb = load(A, "cache.json"), load(B, "cache.json")
    if ca and cb:
        print("\ncache.json (prompt tokens re-processed, lower is better)")
        row("cold ttft s", ca["cold"]["prompt_ms"] / 1000, cb["cold"]["prompt_ms"] / 1000, higher=False)
        for i, (x, y) in enumerate(zip(ca["warm"], cb["warm"])):
            row(f"warm{i} prompt_n", x["prompt_n"], y["prompt_n"], higher=False)
        for i, (x, y) in enumerate(zip(ca["multiturn"], cb["multiturn"])):
            row(f"multiturn{i} prompt_n", x["prompt_n"], y["prompt_n"], higher=False)
            row(f"multiturn{i} ttft s", x["prompt_ms"] / 1000, y["prompt_ms"] / 1000, higher=False)
    da, db = load(A, "decode.json"), load(B, "decode.json")
    if da and db:
        print("\ndecode.json")
        for mode in ["greedy", "probabilistic"]:
            base = da.get(mode) or da.get("greedy")
            if mode not in db:
                continue
            for task, r in db[mode].items():
                row(f"{mode} {task} tg t/s", base[task]["gen_tps"], r["gen_tps"])
                row(f"{mode} {task} accept", base[task]["accept"], r["accept"])
    pa, pb = load(A, "parallel.json"), load(B, "parallel.json")
    if pa and pb:
        print("\nparallel.json")
        for k in pb:
            row(f"{k} aggregate t/s", pa[k]["agg_tps"], pb[k]["agg_tps"])
            row(f"{k} per-request t/s", pa[k]["per_req_tps"], pb[k]["per_req_tps"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bin", required=True, help="build/bin directory")
    r.add_argument("--label", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--mmproj", default=None)
    r.add_argument("--out", default="bench-results")
    r.add_argument("--suite", default="micro,batched,cache,decode,parallel")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--max-tokens", type=int, default=768)
    r.add_argument("--sys-chars", type=int, default=90000, help="size of the shared system prompt (~3.5 chars/token)")
    r.add_argument("--probabilistic", action="store_true", help="also run decode with --spec-draft-sampling probabilistic")
    r.add_argument("--server-extra", default="", help="extra llama-server args, space separated")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
