#!/usr/bin/env python3
"""DFlash2 / spec-decode A-B harness for the GB10. Stdlib only.

Measures, for one running vLLM engine:
  * single-stream decode tok/s and TTFT on a fixed prompt set (code / prose / math)
  * concurrent aggregate throughput at a given batch size
  * vLLM's own spec-decode acceptance counters, deltaed across the run
  * host memory footprint
"""
import argparse, json, os, statistics, sys, threading, time, urllib.request, urllib.error

PROMPTS = {
 "code": "Write a complete Go function `MergeIntervals(in [][]int) [][]int` that merges overlapping intervals. Include the sort, the merge loop, and a short doc comment. Then write a table-driven test for it with four cases.",
 "prose": "Explain, in about 300 words and in plain language, why memory bandwidth rather than raw FLOPs is usually the limiting factor when a single user runs a large dense language model on one machine.",
 "math": "A train leaves station A at 9:00 travelling 60 km/h. A second train leaves station B, 300 km away, at 10:30 travelling 90 km/h toward A. At what time do they meet, and how far from A? Show every step of the arithmetic.",
 "refactor": "Here is a Python function:\n\ndef p(d):\n    r=[]\n    for k in d:\n        if d[k]!=None and d[k]!='':\n            r.append((k,str(d[k]).strip().lower()))\n    r.sort()\n    return dict(r)\n\nRewrite it idiomatically with type hints, a docstring, and explain each change you made.",
}

def post_stream(base, model, prompt, max_tokens, temperature, think=False, timeout=1800):
    """Returns (ttft, total_wall, completion_tokens, text)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": bool(think)},
    }
    if temperature == 0:
        body["top_p"] = 1.0
    req = urllib.request.Request(base + "/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; ntok = 0; chunks = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            try: d = json.loads(payload)
            except Exception: continue
            if d.get("usage"): ntok = d["usage"].get("completion_tokens", ntok)
            for ch in d.get("choices", []):
                delta = ch.get("delta", {}) or {}
                piece = delta.get("content") or delta.get("reasoning") or ""
                if piece:
                    if ttft is None: ttft = time.perf_counter() - t0
                    chunks.append(piece)
    total = time.perf_counter() - t0
    if ttft is None: ttft = total
    text = "".join(chunks)
    if not ntok: ntok = max(1, len(text) // 4)
    return ttft, total, ntok, text

def scrape(url):
    out = {}
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("#") or " " not in line: continue
                k, _, v = line.rpartition(" ")
                try: out[k] = float(v)
                except ValueError: pass
    except Exception as e:
        out["_error"] = str(e)
    return out

def spec_stats(before, after):
    """vLLM spec-decode counters -> acceptance length / rate."""
    def d(sub):
        tot = 0.0
        for k, v in after.items():
            if sub in k and not k.endswith("_created"):
                tot += v - before.get(k, 0.0)
        return tot
    drafts   = d("spec_decode_num_drafts")
    dtok     = d("spec_decode_num_draft_tokens")
    atok     = d("spec_decode_num_accepted_tokens_total")
    if drafts <= 0: return {"spec_active": False}
    return {
        "spec_active": True,
        "num_drafts": drafts,
        "draft_tokens": dtok,
        "accepted_tokens": atok,
        "draft_len": round(dtok / drafts, 3),
        "accept_rate": round(atok / dtok, 4) if dtok else None,
        # acceptance length = accepted drafts + the always-free verifier token
        "acceptance_length": round(atok / drafts + 1, 3),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--metrics", default="http://127.0.0.1:8002/metrics")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/home/bakeer/dflash2-bench/results")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--skip-batch", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {"tag": a.tag, "model": a.model, "temp": a.temp, "max_tokens": a.max_tokens,
           "think": a.think, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "single": {}, }

    print(f"[{a.tag}] warmup", flush=True)
    post_stream(a.base, a.model, "Say OK.", 16, 0.0)

    m0 = scrape(a.metrics)
    for name, prompt in PROMPTS.items():
        runs = []
        for i in range(a.reps):
            ttft, total, ntok, text = post_stream(a.base, a.model, prompt, a.max_tokens, a.temp, a.think)
            dec = ntok / max(total - ttft, 1e-6)
            runs.append({"ttft": round(ttft,4), "wall": round(total,3), "tokens": ntok,
                         "decode_tps": round(dec,2)})
            print(f"  {name}[{i}] {ntok:4d} tok  ttft {ttft:.3f}s  {dec:6.2f} tok/s", flush=True)
            if i == 0: res.setdefault("samples", {})[name] = text[:4000]
        res["single"][name] = {
            "runs": runs,
            "decode_tps_median": round(statistics.median(r["decode_tps"] for r in runs), 2),
            "ttft_median": round(statistics.median(r["ttft"] for r in runs), 4),
        }
    m1 = scrape(a.metrics)
    res["spec"] = spec_stats(m0, m1)
    tps = [v["decode_tps_median"] for v in res["single"].values()]
    res["decode_tps_mean_of_medians"] = round(statistics.mean(tps), 2)

    if not a.skip_batch:
        print(f"[{a.tag}] batch x{a.batch}", flush=True)
        m2 = scrape(a.metrics)
        outs = [None]*a.batch; t0 = time.perf_counter()
        def work(i):
            p = list(PROMPTS.values())[i % len(PROMPTS)] + f"\n\n(variant {i})"
            outs[i] = post_stream(a.base, a.model, p, a.max_tokens, max(a.temp, 0.0), a.think)
        ths = [threading.Thread(target=work, args=(i,)) for i in range(a.batch)]
        [t.start() for t in ths]; [t.join() for t in ths]
        wall = time.perf_counter() - t0
        m3 = scrape(a.metrics)
        tot_tok = sum(o[2] for o in outs if o)
        res["batch"] = {"n": a.batch, "wall": round(wall,2), "total_tokens": tot_tok,
                        "aggregate_tps": round(tot_tok/wall, 2),
                        "per_stream_tps": round(tot_tok/wall/a.batch, 2),
                        "spec": spec_stats(m2, m3)}
        print(f"  aggregate {res['batch']['aggregate_tps']} tok/s over {a.batch} streams", flush=True)

    # host memory
    try:
        with open("/proc/meminfo") as f:
            mi = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
        res["mem_gib"] = {"total": round(mi["MemTotal"]/1048576,1),
                          "available": round(mi["MemAvailable"]/1048576,1),
                          "used": round((mi["MemTotal"]-mi["MemAvailable"])/1048576,1)}
    except Exception: pass
    kv = {k: v for k, v in scrape(a.metrics).items() if "cache_usage" in k or "kv_cache" in k}
    res["kv"] = kv

    path = os.path.join(a.out, f"{a.tag}.json")
    json.dump(res, open(path, "w"), indent=2)
    print(f"\n[{a.tag}] decode {res['decode_tps_mean_of_medians']} tok/s (mean of per-prompt medians)")
    print(f"[{a.tag}] spec: {json.dumps(res['spec'])}")
    print(f"[{a.tag}] -> {path}", flush=True)

if __name__ == "__main__":
    main()
