#!/usr/bin/env python3
# Helper for the upstream-scout skill. Stdlib only; needs an authenticated gh and git >= 2.38. Read-only towards GitHub.
#
#   scout.py known                              carried, watched and deferred upstream items: merged? closed? new revision?
#   scout.py scan [--since DATE | --full] [--top N] [--all]
#                                               rank open upstream PRs, issues and discussions that are not tracked yet
#   scout.py check <N>                          size, fork overlap and merge conflicts of upstream PR N against master
#   scout.py options [--since DATE]             env vars, CLI options and CMake options that upstream added since DATE
#   scout.py mark <N> <verdict> <note> [--head SHA]
#                                               record a triage verdict in triaged.tsv (carried|adopt|test|watch|skip)
#   scout.py finish                             record today and the upstream/master sha as the last scan
#
# DATE is YYYY-MM-DD and defaults to the last scan recorded in triaged.tsv (30 days ago if none).

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

REPO = "ggml-org/llama.cpp"
HERE = Path(__file__).resolve().parent
TSV = HERE / "triaged.tsv"
VERDICTS = ("carried", "adopt", "test", "watch", "skip")
TSV_HEADER = [
    "# upstream-scout triage state (see SKILL.md). One row per upstream PR, issue or discussion triaged.",
    "# verdict: carried | adopt | test | watch | skip ; head: PR head sha when triaged, '-' for issues and discussions",
]
TSV_COLS = ["number", "kind", "verdict", "head", "date", "note"]


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd, check=True):
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if check and p.returncode != 0:
        die(f"{' '.join(cmd[:4])} ...: {p.stderr.strip()}")
    return p


def git(*args, check=True):
    return run(["git", *args], check=check).stdout.strip()


ROOT = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, cwd=HERE).stdout.strip())


def gh(args, retries=4):
    """Run gh and parse its JSON output. Retries on GitHub 5xx and timeouts."""
    for i in range(retries):
        p = subprocess.run(["gh", *args], capture_output=True, text=True, cwd=ROOT)
        if p.returncode == 0:
            return json.loads(p.stdout) if p.stdout.strip() else None
        err = p.stderr.strip()
        if i + 1 < retries and re.search(r"HTTP 5\d\d|timeout|timed out|EOF|connection reset|stream error", err, re.I):
            time.sleep(4 * (i + 1))
            continue
        raise RuntimeError(f"gh {' '.join(args[:3])}: {err}")


def graphql(query, **variables):
    args = ["api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        args += ["-F" if isinstance(v, int) else "-f", f"{k}={v}"]
    for i in range(4):
        p = subprocess.run(["gh", *args], capture_output=True, text=True, cwd=ROOT)
        # gh exits 1 on partial errors (e.g. a number that is a discussion), but stdout still has the data
        try:
            out = json.loads(p.stdout) if p.stdout.strip() else None
        except json.JSONDecodeError:
            out = None
        if out and out.get("data") is not None:
            return out
        if i < 3 and re.search(r"HTTP 5\d\d|timeout|timed out|EOF|connection reset", p.stderr, re.I):
            time.sleep(4 * (i + 1))
            continue
        raise RuntimeError(f"graphql: {p.stderr.strip() or p.stdout.strip()}")


# ---------------------------------------------------------------------------
# relevance rules for the target workload (see the Fork Policy section of AGENTS.md)

# first match wins; weight 0 means "not our path"
PATH_RULES = [
    (r"^tools/server/(webui|public)/|^tools/ui/", 0, None),
    (r"^ggml/src/ggml-cuda/(mmq|mmvq|mmvf|mmf|fattn|gated_delta_net|chunk_gated_delta_net|ssm-conv|norm|rope|cpy|getrows|softmax|unary|glu|quantize|convert|argsort|top-k|topk|cumsum|ggml-cuda\.cu)", 6, "cuda-hot"),
    (r"^ggml/src/ggml-cuda/", 5, "cuda"),
    (r"^ggml/src/ggml-backend-meta\.cpp", 6, "meta(-sm tensor)"),
    (r"^src/models/(qwen35|delta-net-base)", 6, "qwen35/gdn graph"),
    (r"^common/(speculative|ngram-mod)", 6, "speculative"),
    (r"^src/llama-memory-(recurrent|hybrid)", 5, "recurrent mem"),
    (r"^tools/server/server-(context|task|common)", 4, "server"),
    (r"^src/llama-(context|graph|kv-cache|batch)", 4, "llama core"),
    (r"^ggml/src/ggml-(backend|alloc)", 4, "ggml sched"),
    (r"^common/sampling|^src/llama-sampl", 3, "sampling"),
    (r"^src/models/models\.h", 2, "models.h"),
    (r"^ggml/src/ggml\.c|^ggml/include/ggml\.h|^ggml/src/ggml-impl\.h", 2, "ggml core"),
    (r"^src/llama-(model|arch|hparams|cparams)", 2, "llama model"),
    (r"^common/(arg|common)\.", 1, "args"),
    (r"^tools/server/", 1, "server other"),
    (r"^tools/mtmd/", 1, "mtmd"),
    (r"^ggml/CMakeLists\.txt|^ggml/cmake/", 1, "build"),
]
PATH_RULES = [(re.compile(r), w, lbl) for r, w, lbl in PATH_RULES]

# (pattern, weight, label, counts in body). Bodies mention everything (templates, bench tables), so only the specific terms count there.
KEYWORDS = [
    (r"qwen ?3\.?[5-8]\b|qwen35|qwen3[ ._-]?next", 3, "qwen3.x", True),
    (r"gated[ _-]?delta|delta[ _-]?net|\bgdn\b", 3, "gdn", True),
    (r"\bmtp\b|nextn|multi[ -]token pred", 3, "mtp", True),
    (r"split[ -]mode[ =]?tensor|-sm tensor|tensor[ -]parallel|meta backend|all-?reduce|\bnccl\b|\bp2p\b", 3, "tensor-split", True),
    (r"blackwell|sm_?120|rtx ?50[6-9]0|\b50[6-9]0 ?ti\b|compute capability 12", 3, "blackwell", True),
    (r"speculative|\bdraft(ing|s)?\b|ngram", 2, "spec", False),
    (r"\bq4_k\b|\bq6_k\b|q4_k_m", 2, "q4_k", False),
    (r"\bmmq\b|\bmmvq\b|mul_mat_vec|mat-?vec", 2, "matmul", False),
    (r"flash[ _-]?attn|flash attention|\bfattn\b", 2, "fa", False),
    (r"kv[ -]cache|prompt[ -]cache|cache-ram|checkpoint", 1, "cache", False),
    (r"recurrent|hybrid|ssm[_ ]conv", 1, "recurrent", False),
    (r"cuda graph|\bpdl\b|programmatic dependent|\bfus(e|ed|ion)\b", 1, "fusion", False),
    (r"\bperf\b|performance|speed ?up|faster|throughput|t/s|tok/s|latency", 1, "perf", False),
]
KEYWORDS = [(re.compile(r, re.I), w, lbl, b) for r, w, lbl, b in KEYWORDS]
BUG = re.compile(r"regression|since b\d{4,}|slower|performance drop|degrad|\bhang|deadlock|crash", re.I)
OTHER_BACKEND_TITLE = re.compile(
    r"^\s*\[?(vulkan|metal|sycl|opencl|hip|rocm|musa|hexagon|cann|webgpu|openvino|zdnn|zendnn|virtgpu|hrx|cpu|arm|x86|riscv|ppc|s390|loongarch|"
    r"webui|ui|docs?|ci|android|ios|swift|kleidiai|amd|intel|apple)\b", re.I)
AMD = re.compile(r"\b(?:hip|rocm|amd|radeon|rx ?\d{4}|gfx\d+|rdna\d?|cdna\d?|gcn|wave64|mi\d{3}|strix|musa)\b", re.I)
NVIDIA = re.compile(r"\b(?:cuda|nvidia|blackwell|rtx|geforce|sm_?\d{2,3})\b", re.I)
OTHER_HW = re.compile(r"\b(?:vulkan|sycl|metal|opencl|intel|arc ?[ab]\d+|apple|m[1-5] (?:max|pro|ultra)|snapdragon|qualcomm|adreno|hexagon|cpu only)\b", re.I)
OTHER_MODEL_FILE = re.compile(r"^src/models/(?!qwen35|delta-net-base|models\.h)[^/]+\.cpp$")
GGML_BACKENDS_FIELD = re.compile(r"###\s*GGML backends\s*\n+\s*(.+)", re.I)


def score_paths(paths):
    best, labels = 0, {}
    for p in paths:
        for rx, w, lbl in PATH_RULES:
            if rx.search(p):
                if w > 0:
                    best = max(best, w)
                    labels.setdefault(lbl, []).append(p.rsplit("/", 1)[-1])
                break
    return best, labels


def score_text(title, body):
    """Title hits (upper case in the output) count double; body hits count 1 each, at most 4."""
    st, sb, hits = 0, 0, []
    body = (body or "")[:4000]
    for rx, w, lbl, in_body in KEYWORDS:
        if rx.search(title):
            st += 2 * w
            hits.append(lbl.upper())
        elif in_body and rx.search(body):
            sb += 1
            hits.append(lbl)
    return min(st, 12) + min(sb, 4), hits


def other_hw_penalty(title, cuda_path=False):
    """'cuda, hip: X' still concerns us; 'ggml-cuda: HIP: wave64 for GCN' or 'RX 7800 XT Vulkan ...' does not."""
    if OTHER_BACKEND_TITLE.search(title) and not cuda_path:
        return -6
    amd = {m.lower() for m in AMD.findall(title)}
    nv = NVIDIA.search(title)
    if len(amd) >= 2 or (amd and not nv):
        return -6
    if amd:
        return -3
    if OTHER_HW.search(title) and not nv:
        return -6
    return 0


# ---------------------------------------------------------------------------
# state: tracked items in AGENTS.md / IMPROVEMENT_LEDGER.md and triaged.tsv


def read_tsv():
    meta, rows = {}, {}
    if TSV.exists():
        for line in TSV.read_text().splitlines():
            if line.startswith("# last-scan:"):
                parts = line.split()
                meta["last_scan"] = parts[2] if len(parts) > 2 else None
                meta["last_sha"] = parts[3] if len(parts) > 3 else None
            elif line.startswith("#") or not line.strip() or line.startswith("number\t"):
                continue
            else:
                f = line.split("\t")
                f += ["-"] * (len(TSV_COLS) - len(f))
                rows[int(f[0])] = dict(zip(TSV_COLS, f))
    return meta, rows


def write_tsv(meta, rows):
    lines = list(TSV_HEADER)
    if meta.get("last_scan"):
        lines.append(f"# last-scan: {meta['last_scan']} {meta.get('last_sha') or '-'}")
    lines.append("\t".join(TSV_COLS))
    for n in sorted(rows):
        lines.append("\t".join(str(rows[n][c]) for c in TSV_COLS))
    TSV.write_text("\n".join(lines) + "\n")


def section(text, heading):
    i = text.find(heading)
    if i < 0:
        return ""
    level = len(heading) - len(heading.lstrip("#"))
    m = re.compile(r"^#{1,%d} " % level, re.M).search(text, i + len(heading))
    return text[i:m.start() if m else len(text)]


def carried_prs():
    agents = (ROOT / "AGENTS.md").read_text()
    sec = section(agents, "### Carried patches")
    return [int(n) for n in re.findall(r"^- #(\d+)", sec, re.M)]


def tracked():
    """number -> short description of where it is tracked."""
    refs = {}
    agents = (ROOT / "AGENTS.md").read_text()
    fork = agents[agents.find("## Fork Policy"):] if "## Fork Policy" in agents else ""
    for n in carried_prs():
        refs[n] = "carried"
    for m in re.finditer(r"#(\d{4,6})\b", fork):
        refs.setdefault(int(m.group(1)), "AGENTS.md")
    ledger_path = ROOT / "IMPROVEMENT_LEDGER.md"
    if ledger_path.exists():
        where = "ledger"
        for line in ledger_path.read_text().splitlines():
            h = re.match(r"^###\s+(\d+)\..*?(?: - (.+))?$", line)
            if h:
                where = f"ledger #{h.group(1)}: {(h.group(2) or '').strip()[:40]}"
            elif line.startswith("## "):
                where = f"ledger: {line[3:].strip()[:30]}"
            for m in re.finditer(r"#(\d{4,6})\b", line):
                refs.setdefault(int(m.group(1)), where)
    _, rows = read_tsv()
    for n, r in rows.items():
        refs.setdefault(n, f"tsv: {r['verdict']}")
    return refs


def default_since(meta):
    if meta.get("last_scan"):
        return meta["last_scan"]
    return (date.today() - timedelta(days=30)).isoformat()


# ---------------------------------------------------------------------------
# fetching


def fetch_window(kind, field, lo, hi, fields, extra=""):
    """All open PRs/issues with <field> in [lo, hi]; splits the range when the search cap or a timeout is hit."""
    q = f"{field}:{lo.isoformat()}..{hi.isoformat()} {extra}".strip()
    try:
        items = gh([kind, "list", "-R", REPO, "--state", "open", "--search", q, "--limit", "1000", "--json", fields])
    except RuntimeError as e:
        if lo == hi:
            raise
        print(f"[scout] {kind} {lo}..{hi}: {str(e)[:80]}; splitting", file=sys.stderr)
        items = None
    if items is not None and (len(items) < 1000 or lo == hi):
        return items
    mid = lo + (hi - lo) // 2
    return fetch_window(kind, field, lo, mid, fields, extra) + fetch_window(kind, field, mid + timedelta(days=1), hi, fields, extra)


def fetch_open(kind, since, full, fields):
    today = date.today()
    if full:
        field, lo, step = "created", date(2023, 3, 1), 120
    else:
        field, lo, step = "updated", date.fromisoformat(since), 3650
    out, d = {}, lo
    while d <= today:
        e = min(d + timedelta(days=step - 1), today)
        for it in fetch_window(kind, field, d, e, fields):
            out[it["number"]] = it
        d = e + timedelta(days=1)
    return list(out.values())


def fetch_discussions(since, full):
    lo = (date.today() - timedelta(days=180)).isoformat() if full else since
    groups = ["qwen", "MTP OR speculative", "\"tensor parallel\" OR \"split-mode\"", "blackwell OR 5070 OR 5090",
              "\"gated delta\" OR GDN", "CUDA performance"]
    query = """query($q: String!) { search(type: DISCUSSION, query: $q, first: 50) { nodes { ... on Discussion {
      number title url updatedAt category { name } upvoteCount comments { totalCount } bodyText } } } }"""
    out = {}
    for g in groups:
        try:
            res = graphql(query, q=f"repo:{REPO} updated:>={lo} {g}")
        except RuntimeError as e:
            print(f"[scout] discussions '{g}': {str(e)[:80]}", file=sys.stderr)
            continue
        for d in res["data"]["search"]["nodes"]:
            if d:
                out[d["number"]] = d
    return list(out.values())


def pr_files(n):
    p = subprocess.run(["gh", "api", f"repos/{REPO}/pulls/{n}/files", "--paginate", "--jq", ".[].filename"], capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        raise RuntimeError(f"files of #{n}: {p.stderr.strip()}")
    return p.stdout.split()


def cached(name, max_age_h, refresh, fn):
    """Keep raw fetches for a few hours in the git dir (untracked), so re-running scan with other flags is cheap."""
    d = Path(git("rev-parse", "--absolute-git-dir")) / "upstream-scout"
    f = d / f"{name}.json"
    if not refresh and f.exists() and time.time() - f.stat().st_mtime < max_age_h * 3600:
        print(f"[scout] using cached {f.name} (--refresh to fetch again)", file=sys.stderr)
        return json.loads(f.read_text())
    data = fn()
    d.mkdir(exist_ok=True)
    f.write_text(json.dumps(data))
    return data


def fork_files():
    base = git("merge-base", "master", "upstream/master")
    return set(git("diff", "--name-only", base, "master").splitlines())


# ---------------------------------------------------------------------------
# commands


def cmd_scan(a):
    meta, _ = read_tsv()
    since = a.since or default_since(meta)
    known = tracked()
    ff = fork_files()
    scope = "all open" if a.full else f"updated since {since}"

    key = "full" if a.full else f"since-{since}"

    def get_prs():
        print(f"[scout] fetching PRs ({scope}), takes a few minutes ...", file=sys.stderr)
        prs = fetch_open("pr", since, a.full, "number,title,author,updatedAt,isDraft,additions,deletions,changedFiles,labels,files,reviewDecision,body")
        for p in prs:
            p["body"] = (p.get("body") or "")[:4000]
            p["files"] = [f["path"] for f in p["files"]]
            if p.get("changedFiles", 0) > len(p["files"]):
                try:
                    p["files"] = pr_files(p["number"])
                except RuntimeError as e:
                    print(f"[scout] {e}", file=sys.stderr)
        return prs

    def get_issues():
        print(f"[scout] fetching issues ({scope}) ...", file=sys.stderr)
        return fetch_open("issue", since, a.full, "number,title,labels,updatedAt,body")

    prs = cached(f"prs-{key}", 6, a.refresh, get_prs)
    issues = cached(f"issues-{key}", 6, a.refresh, get_issues)
    discs = cached(f"discussions-{key}", 6, a.refresh, lambda: fetch_discussions(since, a.full))

    # ---- PRs
    rows, n_known, n_none = [], 0, 0
    for p in prs:
        if p["number"] in known and not a.all:
            n_known += 1
            continue
        paths = p["files"]
        ps, plabels = score_paths(paths)
        ts, hits = score_text(p["title"], p.get("body"))
        s = ps + ts + other_hw_penalty(p["title"], any(l in plabels for l in ("cuda-hot", "cuda", "meta(-sm tensor)")))
        if any(OTHER_MODEL_FILE.match(f) for f in paths) and "QWEN3.X" not in hits:
            s -= 4  # support for another model; shared-code changes in it still score through their paths
        if ps == 0 and ts < 6:
            n_none += 1
            continue
        if s <= 0:
            n_none += 1
            continue
        state = "draft" if p["isDraft"] else {"APPROVED": "approved", "CHANGES_REQUESTED": "changes"}.get(p.get("reviewDecision") or "", "open")
        overlap = len(ff & set(paths))
        rows.append((s, p, state, overlap, plabels, hits))
    rows.sort(key=lambda r: (-r[0], r[1]["number"]))

    print(f"== open PRs not yet tracked ({scope}; {len(prs)} fetched, {n_known} already tracked, {n_none} not on our path)")
    print("score  number  state     +add/-del     files fork  updated     author           title")
    for s, p, state, overlap, plabels, hits in rows[:a.top]:
        tag = f"  [{known[p['number']]}]" if p["number"] in known else ""
        print(f"{s:5d}  #{p['number']:<6} {state:9} +{p['additions']}/-{p['deletions']:<8} {p.get('changedFiles', len(p['files'])):5d} {overlap:4d}  "
              f"{p['updatedAt'][:10]}  {p['author']['login'][:16]:16} {p['title'][:90]}{tag}")
        paths_s = "; ".join(f"{k}: {', '.join(sorted(set(v))[:4])}" for k, v in sorted(plabels.items(), key=lambda kv: -len(kv[1]))[:3])
        print(f"{'':15}paths: {paths_s or '-'} | kw: {', '.join(hits) or '-'}")
    if len(rows) > a.top:
        print(f"... {len(rows) - a.top} more with score > 0 (use --top)")

    # ---- issues
    irows, i_known = [], 0
    for it in issues:
        if it["number"] in known and not a.all:
            i_known += 1
            continue
        body = it.get("body") or ""
        ts, hits = score_text(it["title"], body)
        if BUG.search(it["title"]):
            ts += 2
            hits.append("BUG/REGRESSION")
        m = GGML_BACKENDS_FIELD.search(body)
        if m:
            ts += 1 if "cuda" in m.group(1).lower() else -6
        ts += other_hw_penalty(it["title"])
        if ts >= 6:
            irows.append((ts, it, hits))
    irows.sort(key=lambda r: (-r[0], -r[1]["number"]))
    n_issues = min(len(irows), max(10, a.top // 2))
    print(f"\n== open issues not yet tracked ({scope}; {len(issues)} fetched, {i_known} already tracked, {len(irows)} relevant)")
    for s, it, hits in irows[:n_issues]:
        labels = ",".join(l["name"] for l in it["labels"] if l["name"] not in ("bug-unconfirmed",))
        print(f"{s:5d}  #{it['number']:<6} {it['updatedAt'][:10]}  {it['title'][:100]}  [{labels}] kw: {', '.join(hits)}")
    if len(irows) > n_issues:
        print(f"... {len(irows) - n_issues} more (use --top)")

    # ---- discussions
    drows = []
    for d in discs:
        if d["number"] in known and not a.all:
            continue
        ts, hits = score_text(d["title"], d.get("bodyText"))
        ts += other_hw_penalty(d["title"]) // 2  # a tuned fork on other hardware can still carry ideas that transfer
        if ts >= 6:
            drows.append((ts, d, hits))
    drows.sort(key=lambda r: (-r[0], -r[1]["number"]))
    print(f"\n== discussions not yet tracked ({'last 180 days' if a.full else 'updated since ' + since}; {len(discs)} fetched, {len(drows)} relevant)")
    for s, d, hits in drows[:max(8, a.top // 3)]:
        print(f"{s:5d}  #{d['number']:<6} {d['updatedAt'][:10]}  [{d['category']['name']}] {d['title'][:100]}  "
              f"(+{d['upvoteCount']}, {d['comments']['totalCount']} comments) kw: {', '.join(hits)}")


def our_copy(n):
    """Latest fork commit that carries upstream PR n: (sha, committer date ISO, subject) or None."""
    out = git("log", "master", "--not", "upstream/master", "--no-merges", "-E", f"--grep=upstream #{n}([^0-9]|$)",
              "--format=%H%x09%cI%x09%s", check=False)
    line = out.splitlines()[0] if out else ""
    return tuple(line.split("\t", 2)) if line else None


def fetch_states(numbers):
    frag = """__typename ... on PullRequest { title state isDraft mergedAt closedAt updatedAt headRefOid
      commits(last: 60) { totalCount nodes { commit { oid committedDate } } } }
      ... on Issue { title state closedAt updatedAt }"""
    out = {}
    numbers = sorted(numbers)
    for i in range(0, len(numbers), 30):
        chunk = numbers[i:i + 30]
        q = "query { repository(owner: \"ggml-org\", name: \"llama.cpp\") { " + " ".join(
            f"n{n}: issueOrPullRequest(number: {n}) {{ {frag} }}" for n in chunk) + " } }"
        res = graphql(q)
        repo = res["data"].get("repository") or {}
        for n in chunk:
            out[n] = repo.get(f"n{n}")
    missing = [n for n, v in out.items() if v is None]
    for n in missing:
        try:
            res = graphql("query($n: Int!) { repository(owner: \"ggml-org\", name: \"llama.cpp\") { discussion(number: $n) { title updatedAt closed } } }", n=n)
            d = res["data"]["repository"]["discussion"]
            if d:
                out[n] = {"__typename": "Discussion", "title": d["title"], "updatedAt": d["updatedAt"], "state": "CLOSED" if d["closed"] else "OPEN"}
        except RuntimeError:
            pass
    return out


def cmd_known(a):
    meta, rows = read_tsv()
    last = meta.get("last_scan") or "0000-00-00"
    refs = tracked()
    for n, r in rows.items():
        if r["verdict"] == "skip":
            refs.pop(n, None)
    carried = set(carried_prs()) | {n for n, r in rows.items() if r["verdict"] == "carried"}
    states = fetch_states(refs.keys())

    groups = {k: [] for k in ("c_merged", "c_closed", "c_newer", "c_ok", "t_changed", "t_closed", "t_quiet", "t_merged")}
    for n in sorted(refs):
        st = states.get(n)
        if not st:
            groups["t_quiet"].append(f"#{n}  (not found upstream)  [{refs[n]}]")
            continue
        title = st.get("title", "")[:70]
        kind = st["__typename"]
        rec = rows.get(n)
        if n in carried:
            if st["state"] == "MERGED":
                groups["c_merged"].append(f"#{n}  merged {st['mergedAt'][:10]}: next upstream-sync drops our copy  {title}")
                continue
            if st["state"] == "CLOSED":
                groups["c_closed"].append(f"#{n}  closed unmerged {st['closedAt'][:10]}: our carry is now permanent (keep or drop?)  {title}")
                continue
            commits = [(c["commit"]["oid"], c["commit"]["committedDate"]) for c in st["commits"]["nodes"]]
            head = st["headRefOid"]
            if rec and rec["head"] not in ("-", ""):
                idx = next((i for i, (oid, _) in enumerate(commits) if oid.startswith(rec["head"])), None)
                if head.startswith(rec["head"]):
                    groups["c_ok"].append(f"#{n}  head {head[:12]} = recorded  {title}")
                elif idx is not None:
                    groups["c_newer"].append(f"#{n}  {len(commits) - 1 - idx} PR commits after recorded {rec['head']} (head {head[:12]}, updated {st['updatedAt'][:10]})  {title}")
                else:
                    groups["c_newer"].append(f"#{n}  head rewritten since recorded {rec['head']} (force-push?); now {head[:12]}: compare gh pr diff with our commit  {title}")
                continue
            copy = our_copy(n)
            if not copy:
                groups["c_newer"].append(f"#{n}  no fork commit with 'upstream #{n}' in its subject; cannot compare (head {head[:12]})  {title}")
                continue
            newer = [c for c in commits if c[1] > copy[1]]
            older = [c for c in commits if c[1] <= copy[1]]
            if newer and not older:
                groups["c_newer"].append(f"#{n}  all {len(newer)} listed PR commits are dated after our copy {copy[0][:9]} ({copy[1][:10]}): "
                                         f"rebased or rewritten; compare gh pr diff {n} with git show {copy[0][:9]} (head {head[:12]})  {title}")
            elif newer:
                groups["c_newer"].append(f"#{n}  {len(newer)} PR commit(s) dated after our copy {copy[0][:9]} ({copy[1][:10]}); "
                                         f"head {head[:12]}, likely copied head {older[-1][0][:12]}  {title}")
            else:
                # committer dates move when feat branches are rebased, so this can miss a newer revision
                groups["c_ok"].append(f"#{n}  no PR commit dated after our copy {copy[0][:9]} ({copy[1][:10]}); unrecorded: verify with the "
                                      f"changed-lines diff (SKILL.md step 1), then scout.py mark {n} carried \"<name>\" --head {head[:12]}  {title}")
            continue
        where = refs[n]
        if st["state"] == "MERGED":
            groups["t_merged"].append(f"#{n}")
            continue
        if st["state"] == "CLOSED":
            closed = (st.get("closedAt") or st.get("updatedAt") or "")[:10]
            if closed > last:
                groups["t_closed"].append(f"#{n}  {kind} closed {closed}  {title}  [{where}]")
            else:
                groups["t_merged"].append(f"#{n}")
            continue
        moved = ""
        if kind == "PullRequest" and rec and rec["head"] not in ("-", "") and not st["headRefOid"].startswith(rec["head"]):
            moved = f"head moved since triage ({rec['head']} -> {st['headRefOid'][:12]}), "
        draft = "draft, " if st.get("isDraft") else ""
        line = f"#{n}  {kind} {draft}{moved}updated {st['updatedAt'][:10]}  {title}  [{where}]"
        (groups["t_changed"] if moved or st["updatedAt"][:10] > last else groups["t_quiet"]).append(line)

    def show(key, header):
        if groups[key]:
            print(f"== {header}")
            for line in groups[key]:
                print(f"  {line}")
    print(f"(last scan: {meta.get('last_scan') or 'never'})")
    show("c_merged", "carried, merged upstream")
    show("c_closed", "carried, closed upstream without merging")
    show("c_newer", "carried, upstream PR has newer commits than our copy (check for fixes)")
    show("c_ok", "carried, no newer upstream revision")
    show("t_changed", "tracked (ledger/watch/test), changed since last scan: re-check the revisit condition")
    show("t_closed", "tracked, closed since last scan")
    show("t_quiet", "tracked, open and unchanged")
    if groups["t_merged"]:
        print(f"== tracked, merged or closed before the last scan (no action): {' '.join(groups['t_merged'])}")


def cmd_check(a):
    n = a.number
    info = gh(["api", f"repos/{REPO}/pulls/{n}"])
    files = pr_files(n)
    print(f"#{n} {info['title']}")
    print(f"  {info['state']}{' (draft)' if info['draft'] else ''}, merged={info['merged']}, +{info['additions']}/-{info['deletions']}, "
          f"{info['changed_files']} files, {info['commits']} commits, head {info['head']['sha'][:12]}, updated {info['updated_at'][:10]}")
    ff = fork_files()
    ov = sorted(ff & set(files))
    print(f"\n== files also changed by the fork ({len(ov)} of {len(files)})")
    for f in ov:
        commits = git("log", "--format=%h %s", "master", "--not", "upstream/master", "--no-merges", "--", f).splitlines()
        print(f"  {f}")
        for c in commits[:5]:
            print(f"      {c}")
        if len(commits) > 5:
            print(f"      ... {len(commits) - 5} more")
    ref = f"refs/scout/pr-{n}"
    run(["git", "fetch", "-q", "upstream", f"+pull/{n}/head:{ref}"])
    behind = int(git("rev-list", "--count", f"master..{ref}"))
    if behind > info["commits"]:
        print(f"\nnote: the PR branch has {behind} commits not in master but the PR has {info['commits']}: "
              f"it is based on newer upstream code; run upstream-sync first or expect extra conflicts")
    def conflicts(onto):
        p = run(["git", "merge-tree", "--write-tree", "--name-only", "--no-messages", onto, ref], check=False)
        if p.returncode not in (0, 1):
            die(f"merge-tree {onto}: {p.stderr.strip()}")
        return [l for l in p.stdout.strip().splitlines()[1:] if l.strip()]

    vs_master, vs_upstream = conflicts("master"), set(conflicts("upstream/master"))
    print(f"\n== merge into master: {'clean' if not vs_master else f'{len(vs_master)} conflicted file(s)'}")
    for f in vs_master:
        why = "PR is stale against upstream too" if f in vs_upstream else "conflicts with fork patches" if f in ff else "conflicts with fork history"
        print(f"  {f}  ({why})")
    print(f"\nlocal ref: {ref}  (diff: git diff master...{ref})")


def cmd_options(a):
    meta, _ = read_tsv()
    since = a.since or default_since(meta)
    base = git("rev-list", "-1", f"--before={since}T00:00:00", "upstream/master")
    if not base:
        die(f"no upstream/master commit before {since}")
    paths = ["ggml/src", "ggml/CMakeLists.txt", "ggml/cmake", "src", "common", "tools/server"]
    diff = git("diff", "-U0", "--no-color", base, "upstream/master", "--", *paths)
    pats = [
        ("env", re.compile(r"getenv\(\s*\"([A-Z][A-Z0-9_]+)\"")),
        ("env", re.compile(r"set_env\(\s*\"([A-Z][A-Z0-9_]+)\"")),
        ("cli", re.compile(r"\{\s*\"(-{1,2}[a-z0-9][a-z0-9-]*)\"")),
        ("cmake", re.compile(r"option\s*\(\s*(GGML_[A-Z0-9_]+)")),
        ("cmake", re.compile(r"set\s*\(\s*(GGML_[A-Z0-9_]+)\s[^)]*CACHE")),
    ]
    other_file = re.compile(r"^ggml/src/ggml-(?!cuda/)[a-z0-9]+/")
    # other backends' knobs, and LLAMA_ARG_* env aliases of CLI options (a new option shows up as cli)
    other_name = re.compile(r"^(GGML_(HEXAGON|OPENCL|VK|VULKAN|SYCL|METAL|CANN|CPU|RPC|WEBGPU|ZDNN|ZENDNN|OPENVINO|MUSA|HIP|HRX|BLAS|VIRTGPU)_|LLAMA_ARG_)")
    found, cur = {}, None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            cur = line[6:]
            if other_file.match(cur):
                cur = None
        elif line.startswith("+") and cur:
            for kind, rx in pats:
                if kind == "cli" and not cur.startswith("common/arg"):
                    continue
                for m in rx.finditer(line):
                    if not other_name.match(m.group(1)):
                        found.setdefault((kind, m.group(1)), cur)
    bench = ""
    for f in ("scripts/fork/bench_qwen38.py", ".claude/skills/upstream-sync/SKILL.md"):
        if (ROOT / f).exists():
            bench += (ROOT / f).read_text()
    in_fork = run(["git", "merge-base", "--is-ancestor", "upstream/master", "master"], check=False).returncode == 0
    rows = []
    for (kind, name), path in sorted(found.items()):
        if run(["git", "grep", "-q", "-F", "-e", name, base, "--"] + paths, check=False).returncode == 0:
            continue  # existed before, only moved or edited
        intro = git("log", "--reverse", "--format=%h %s", f"-S{name}", f"{base}..upstream/master", "--", *paths, check=False).splitlines()
        used = "used by bench/sync" if name in bench else "not used"
        rows.append(f"  {kind:5} {name:42} {used:18} {intro[0][:90] if intro else path}")
    print(f"== options added upstream since {since} (base {base[:9]}){'' if in_fork else '; upstream/master is not fully merged into master yet'}")
    print("\n".join(rows) if rows else "  (none)")


def cmd_mark(a):
    if a.verdict not in VERDICTS:
        die(f"verdict must be one of {', '.join(VERDICTS)}")
    note = re.sub(r"[\t\r\n]+", " ", a.note).strip()
    head, kind = "-", "pr"
    try:
        it = gh(["api", f"repos/{REPO}/issues/{a.number}"])
        kind = "pr" if it.get("pull_request") else "issue"
    except RuntimeError:
        kind = "disc"
    if kind == "pr":
        head = (a.head or gh(["api", f"repos/{REPO}/pulls/{a.number}", "--jq", "{sha: .head.sha}"])["sha"])[:12]
    meta, rows = read_tsv()
    rows[a.number] = dict(number=a.number, kind=kind, verdict=a.verdict, head=head, date=date.today().isoformat(), note=note)
    write_tsv(meta, rows)
    print(f"#{a.number} {kind} {a.verdict} {head} {note}")


def cmd_finish(a):
    meta, rows = read_tsv()
    meta["last_scan"] = date.today().isoformat()
    meta["last_sha"] = git("rev-parse", "--short=9", "upstream/master")
    write_tsv(meta, rows)
    print(f"last-scan: {meta['last_scan']} {meta['last_sha']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--since")
    s.add_argument("--full", action="store_true", help="all open PRs/issues, not only recently updated ones")
    s.add_argument("--top", type=int, default=30)
    s.add_argument("--all", action="store_true", help="also list items that are already tracked")
    s.add_argument("--refresh", action="store_true", help="ignore the fetch cache (kept 6 h in the git dir)")
    sub.add_parser("known")
    c = sub.add_parser("check")
    c.add_argument("number", type=int)
    o = sub.add_parser("options")
    o.add_argument("--since")
    m = sub.add_parser("mark")
    m.add_argument("number", type=int)
    m.add_argument("verdict")
    m.add_argument("note")
    m.add_argument("--head")
    sub.add_parser("finish")
    a = ap.parse_args()
    if run(["git", "remote", "get-url", "upstream"], check=False).returncode != 0:
        die("no 'upstream' remote (git remote add upstream https://github.com/ggml-org/llama.cpp.git)")
    {"scan": cmd_scan, "known": cmd_known, "check": cmd_check, "options": cmd_options, "mark": cmd_mark, "finish": cmd_finish}[a.cmd](a)


if __name__ == "__main__":
    main()
