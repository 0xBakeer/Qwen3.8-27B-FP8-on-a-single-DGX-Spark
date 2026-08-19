#!/usr/bin/env python3
"""Concurrency sweep 1..16. Aggregate + per-stream tok/s, TTFT, acceptance per point."""
import argparse, json, os, statistics, threading, time
import bench_dflash as B

def run_point(base, model, n, max_tokens, temp, think, metrics):
    prompts = list(B.PROMPTS.values())
    out = [None] * n
    m0 = B.scrape(metrics)
    def work(i):
        p = prompts[i % len(prompts)] + (f"\n\n(variant {i})" if i >= len(prompts) else "")
        try: out[i] = B.post_stream(base, model, p, max_tokens, temp, think)
        except Exception as e: out[i] = None
    t0 = time.perf_counter()
    ths = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.perf_counter() - t0
    m1 = B.scrape(metrics)
    ok = [o for o in out if o]
    tot = sum(o[2] for o in ok)
    ttfts = sorted(o[0] for o in ok)
    per = [o[2] / max(o[1] - o[0], 1e-6) for o in ok]
    return {
        "concurrency": n, "completed": len(ok), "wall_s": round(wall, 2),
        "total_tokens": tot,
        "aggregate_tps": round(tot / wall, 2),
        "per_stream_tps_mean": round(statistics.mean(per), 2) if per else 0,
        "ttft_p50": round(statistics.median(ttfts), 3) if ttfts else None,
        "ttft_p95": round(ttfts[int(len(ttfts) * 0.95) - 1], 3) if len(ttfts) > 1 else (ttfts[0] if ttfts else None),
        "spec": B.spec_stats(m0, m1),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--metrics", default="http://127.0.0.1:8002/metrics")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/home/bakeer/dflash2-bench/results")
    ap.add_argument("--levels", default="1,2,4,8,12,16")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--temp", type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(f"[{a.tag}] warmup", flush=True)
    B.post_stream(a.base, a.model, "Say OK.", 16, 0.0)
    res = {"tag": a.tag, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "max_tokens": a.max_tokens, "points": []}
    for n in [int(x) for x in a.levels.split(",")]:
        p = run_point(a.base, a.model, n, a.max_tokens, a.temp, False, a.metrics)
        res["points"].append(p)
        acc = p["spec"].get("acceptance_length", "-")
        print(f"  c={n:<3d} agg {p['aggregate_tps']:8.2f}  per-stream {p['per_stream_tps_mean']:6.2f}"
              f"  ttft p50 {p['ttft_p50']}  acc {acc}  ({p['completed']}/{n} ok)", flush=True)
    base = res["points"][0]["aggregate_tps"]
    for p in res["points"]:
        p["scaling_vs_c1"] = round(p["aggregate_tps"] / base, 3)
    path = os.path.join(a.out, f"sweep-{a.tag}.json")
    json.dump(res, open(path, "w"), indent=2)
    print(f"[{a.tag}] -> {path}", flush=True)

if __name__ == "__main__":
    main()
