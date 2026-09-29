#!/usr/bin/env python3
# A/B benchmark for the Qwen3.8-27B fork. Stdlib only.
#
#   run:     bench_qwen38.py run --bin <build/bin> --label <name> --model <gguf> [--mmproj <gguf>] [--production --ngram-table <table.bin>] [--suite ...]
#   compare: bench_qwen38.py compare <results/a> <results/b>
#
# Stop any other llama-server using the GPUs first, or the numbers are meaningless.

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
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

PRODUCTION_ARGS = [
    "-c", "262144", "-np", "2", "--kv-unified", "--kv-unified-per-slot", "160000",
    "-ctkd", "q8_0", "-ctvd", "q4_0", "--spec-draft-sampling", "probabilistic",
    "--reasoning-budget", "16384", "--reasoning-budget-message", "Thinking budget exceeded. Proceed to final answer.",
    "--metrics",
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
LONG_DEPTHS = [8192, 32768, 128000, 155000]


def env(production=False):
    e = dict(os.environ)
    for k, v in ENV_DEFAULTS.items():
        e.setdefault(k, v)
    if production:
        e.setdefault("LLAMA_SPEC_CKPT_ON_DEVICE", "1")
        e.setdefault("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "1")
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


def merge_args(base, extra):
    # a flag in extra replaces the same flag (and its value) in base, so the command line shows what runs
    # note: aliases (e.g. -c vs --ctx-size) are not matched
    flags = {t for t in extra if t.startswith("-")}
    out, i = [], 0
    while i < len(base):
        has_val = i + 1 < len(base) and not base[i + 1].startswith("-")
        if base[i] in flags:
            i += 2 if has_val else 1
            continue
        out.append(base[i])
        i += 1
    return out + extra


class Server:
    def __init__(self, a, out, name, extra):
        self.cmd = [str(Path(a.bin) / "llama-server"), "--model", a.model, "--host", "127.0.0.1", "--port", str(PORT)]
        if a.mmproj:
            self.cmd += ["--mmproj", a.mmproj]
        self.cmd += merge_args(SERVER_ARGS, (PRODUCTION_ARGS if a.production else []) + extra)
        self.production = a.production
        self.ngram_table = a.ngram_table
        self.out = out
        self.table_dir = None
        self.logf = open(out / f"server-{name}.log", "w")

    def __enter__(self):
        try:
            if self.ngram_table:
                self.table_dir = tempfile.TemporaryDirectory(prefix="ngram-", dir=self.out)
                table_copy = Path(self.table_dir.name) / "table.bin"
                shutil.copyfile(self.ngram_table, table_copy)
                self.cmd += ["--spec-ngram-mod-file", str(table_copy)]
            log(" ".join(self.cmd))
            self.p = subprocess.Popen(self.cmd, env=env(self.production), stdout=self.logf, stderr=subprocess.STDOUT, start_new_session=True)
            t0 = time.time()
            while time.time() - t0 < 900:
                if self.p.poll() is not None:
                    raise RuntimeError(f"llama-server exited early, see {self.logf.name}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                        if r.status == 200:
                            return self
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    pass
                time.sleep(2)
            raise RuntimeError(f"llama-server did not become healthy, see {self.logf.name}")
        except BaseException:
            if hasattr(self, "p") and self.p.poll() is None:
                os.killpg(self.p.pid, signal.SIGTERM)
                self.p.wait()
            self.logf.close()
            if self.table_dir:
                self.table_dir.cleanup()
            raise

    def __exit__(self, *exc):
        try:
            if self.p.poll() is None:
                os.killpg(self.p.pid, signal.SIGTERM)
            try:
                self.p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)
                self.p.wait()
            time.sleep(3)
        finally:
            self.logf.close()
            if self.table_dir:
                self.table_dir.cleanup()


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


def post_json(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def chat_stream(messages, max_tokens, seed):
    body = {"messages": messages, "max_tokens": max_tokens, "seed": seed, "stream": True,
            "stream_options": {"include_usage": True}, **SAMPLING}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    timings = None
    done = False
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if "timings" in chunk:
                timings = chunk["timings"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if ttft is None and any(delta.get(k) for k in ("content", "reasoning_content", "tool_calls")):
                    ttft = time.perf_counter() - t0
    if not done or ttft is None or timings is None:
        raise RuntimeError("stream ended without a generated token, final timings, or [DONE]")
    return {
        "ttft_s": ttft, "wall_s": time.perf_counter() - t0,
        "prompt_n": timings["prompt_n"], "cache_n": timings["cache_n"],
        "prompt_tps": timings["prompt_per_second"],
        "gen_n": timings["predicted_n"], "gen_tps": timings["predicted_per_second"],
        "draft_n": timings.get("draft_n", 0), "draft_acc": timings.get("draft_n_accepted", 0),
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


def test_cache(a, out, extra):
    # automations: same long system prompt, different user turns; then an agent-style follow-up
    # that drops the reasoning of the previous reply (the case that used to force re-processing)
    sp = system_prompt(a.sys_chars)
    res = {}
    with Server(a, out, "cache", extra):
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
    for np_ in ([1, 2] if a.production else [1, 3]):
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


def context_messages(depth, sequence):
    unit = system_prompt(25000)
    prefix = f"Depth {depth}, sequence {sequence}. Reference code:\n"
    user = {"role": "user", "content": "Describe this code in one sentence."}
    chars = depth * 4
    for _ in range(8):
        text = (unit * (chars // len(unit) + 1))[:chars]
        messages = [{"role": "system", "content": prefix + text}, user]
        prompt = post_json("/apply-template", {"messages": messages})["prompt"]
        count = len(post_json("/tokenize", {"content": prompt, "add_special": True})["tokens"])
        if abs(count - depth) <= max(128, depth // 100):
            return messages
        chars = max(1, int(chars * depth / count))
    raise RuntimeError(f"could not build a {depth}-token chat prompt (last count: {count})")


def test_long_context(a, out, extra):
    res = {}
    with Server(a, out, "long-context", extra):
        chat([{"role": "user", "content": "hi"}], 16, 1)
        for depth in LONG_DEPTHS:
            messages = context_messages(depth, "single")
            cold = chat_stream(messages, 128, 100 + depth)
            warm = chat_stream(messages[:-1] + [{"role": "user", "content": "Name one function from this code."}], 128, 100 + depth)
            for name, sample in (("cold", cold), ("warm", warm)):
                if abs(sample["prompt_n"] + sample["cache_n"] - depth) > max(256, depth // 50):
                    raise RuntimeError(f"{depth}-token {name} request used {sample['prompt_n'] + sample['cache_n']} tokens")
            res[str(depth)] = {"cold": cold, "warm": warm}
            log(f"  depth {depth:6d}: cold {cold['prompt_n']:6d} processed, {cold['ttft_s']:7.2f}s TTFT, "
                f"{cold['prompt_tps']:7.1f} pp t/s; warm {warm['cache_n']:6d} cached, {warm['ttft_s']:7.2f}s TTFT")

            if depth <= 128000 and a.production:
                prompts = [context_messages(depth, f"parallel-{i}") for i in range(2)]
                t0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(2) as ex:
                    rows = list(ex.map(lambda ip: chat_stream(ip[1], 128, 200 + depth + ip[0]), enumerate(prompts)))
                wall = time.perf_counter() - t0
                for sample in rows:
                    if abs(sample["prompt_n"] + sample["cache_n"] - depth) > max(256, depth // 50):
                        raise RuntimeError(f"{depth}-token two-slot request used {sample['prompt_n'] + sample['cache_n']} tokens")
                res[str(depth)]["parallel"] = {
                    "requests": rows, "end_to_end_tps": sum(r["gen_n"] for r in rows) / wall,
                }
                log(f"             two slots: TTFT {rows[0]['ttft_s']:.2f}/{rows[1]['ttft_s']:.2f}s, "
                    f"end-to-end {res[str(depth)]['parallel']['end_to_end_tps']:.1f} t/s")
    return res


def cmd_run(a):
    out = Path(a.out) / a.label
    extra = a.server_extra.split() if a.server_extra else []
    suites = a.suite.split(",")
    valid = {"micro", "batched", "cache", "decode", "parallel", "long-context"}
    if set(suites) - valid:
        raise ValueError(f"unknown suites: {', '.join(sorted(set(suites) - valid))}")
    if a.ngram_table and "--spec-ngram-mod-file" in extra:
        raise ValueError("use either --ngram-table or --spec-ngram-mod-file in --server-extra")
    if a.ngram_table and not Path(a.ngram_table).is_file():
        raise FileNotFoundError(a.ngram_table)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"label": a.label, "bin": a.bin, "model": a.model, "suites": suites, "production": a.production,
            "ngram_table": a.ngram_table, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
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
        (out / "cache.json").write_text(json.dumps(test_cache(a, out, extra), indent=1))
    if "decode" in suites:
        res = {}
        if not a.production:
            log("== decode (spec greedy)")
            res["greedy"] = test_decode(a, out, "greedy", extra)
        if a.production or a.probabilistic:
            log("== decode (spec probabilistic)")
            prob_extra = extra if a.production else extra + ["--spec-draft-sampling", "probabilistic"]
            res["probabilistic"] = test_decode(a, out, "prob", prob_extra)
        (out / "decode.json").write_text(json.dumps(res, indent=1))
    if "parallel" in suites:
        log("== parallel slots")
        (out / "parallel.json").write_text(json.dumps(test_parallel(a, out, extra), indent=1))
    if "long-context" in suites:
        log("== long context (streamed TTFT, cold/warm, two slots)")
        (out / "long-context.json").write_text(json.dumps(test_long_context(a, out, extra), indent=1))
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
            base = da.get(mode)
            if base is None or mode not in db:
                continue
            for task, r in db[mode].items():
                row(f"{mode} {task} tg t/s", base[task]["gen_tps"], r["gen_tps"])
                row(f"{mode} {task} accept", base[task]["accept"], r["accept"])
    pa, pb = load(A, "parallel.json"), load(B, "parallel.json")
    if pa and pb:
        print("\nparallel.json")
        for k in pb:
            if k in pa:
                row(f"{k} aggregate t/s", pa[k]["agg_tps"], pb[k]["agg_tps"])
                row(f"{k} per-request t/s", pa[k]["per_req_tps"], pb[k]["per_req_tps"])
    la, lb = load(A, "long-context.json"), load(B, "long-context.json")
    if la and lb:
        print("\nlong-context.json")
        for depth in lb:
            if depth not in la:
                continue
            for mode in ("cold", "warm"):
                if mode not in la[depth] or mode not in lb[depth]:
                    continue
                x, y = la[depth][mode], lb[depth][mode]
                row(f"{depth} {mode} TTFT s", x["ttft_s"], y["ttft_s"], higher=False)
                if mode == "cold":
                    row(f"{depth} cold prompt t/s", x["prompt_tps"], y["prompt_tps"])
                else:
                    row(f"{depth} warm cached tokens", x["cache_n"], y["cache_n"])
            if "parallel" in la[depth] and "parallel" in lb[depth]:
                x, y = la[depth]["parallel"], lb[depth]["parallel"]
                row(f"{depth} two-slot max TTFT s", max(r["ttft_s"] for r in x["requests"]),
                    max(r["ttft_s"] for r in y["requests"]), higher=False)
                row(f"{depth} two-slot end-to-end t/s", x["end_to_end_tps"], y["end_to_end_tps"])


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
    r.add_argument("--production", action="store_true", help="use startup's context, two-slot, draft, and CUDA settings for server suites")
    r.add_argument("--ngram-table", help="copy this table for each server start; the original is never modified")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
