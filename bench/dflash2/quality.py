#!/usr/bin/env python3
"""Quality suite: GSM8K (generative, exact-match) + MMLU (single-letter).
Greedy, thinking off, concurrency-parallel. Stdlib only."""
import argparse, json, os, re, threading, queue, time, urllib.request

def chat(base, model, prompt, max_tokens, timeout=900):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "top_p": 1.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode())
            m = d["choices"][0]["message"]
            return (m.get("content") or m.get("reasoning") or ""), d.get("usage", {}).get("completion_tokens", 0)
        except Exception as e:
            if attempt == 2: return f"__ERROR__ {e}", 0
            time.sleep(2)

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
def gsm_pred(t):
    if "####" in t: t = t.split("####")[-1]
    n = NUM.findall(t.replace("$", ""))
    if not n: return None
    v = n[-1].replace(",", "").rstrip(".")
    try: return str(int(float(v)))
    except Exception: return v

def run_pool(items, fn, workers):
    q = queue.Queue(); [q.put(i) for i in range(len(items))]
    out = [None]*len(items); lock = threading.Lock(); done = [0]
    def w():
        while True:
            try: i = q.get_nowait()
            except queue.Empty: return
            out[i] = fn(items[i])
            with lock:
                done[0] += 1
                if done[0] % 25 == 0: print(f"    {done[0]}/{len(items)}", flush=True)
    ts = [threading.Thread(target=w) for _ in range(workers)]
    [t.start() for t in ts]; [t.join() for t in ts]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="/home/bakeer/dflash2-bench/data")
    ap.add_argument("--out", default="/home/bakeer/dflash2-bench/results")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gsm", type=int, default=200)
    ap.add_argument("--mmlu", type=int, default=400)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {"tag": a.tag, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

    if a.gsm:
        gs = json.load(open(f"{a.data}/gsm8k.json"))[:a.gsm]
        print(f"[{a.tag}] GSM8K n={len(gs)}", flush=True)
        t0 = time.perf_counter()
        def one(r):
            p = r["q"] + "\n\nSolve this step by step. End your reply with the final numeric answer on its own line in the form: #### <number>"
            txt, ntok = chat(a.base, a.model, p, 640)
            return {"pred": gsm_pred(txt), "gold": r["a"], "tok": ntok, "err": txt.startswith("__ERROR__")}
        rr = run_pool(gs, one, a.workers)
        corr = sum(1 for r in rr if r["pred"] is not None and r["pred"] == r["gold"])
        res["gsm8k"] = {"n": len(rr), "correct": corr, "acc": round(100*corr/len(rr), 2),
                        "errors": sum(1 for r in rr if r["err"]),
                        "mean_tokens": round(sum(r["tok"] for r in rr)/len(rr), 1),
                        "wall_s": round(time.perf_counter()-t0, 1),
                        "preds": [r["pred"] for r in rr]}
        print(f"[{a.tag}] GSM8K acc {res['gsm8k']['acc']}%  ({res['gsm8k']['wall_s']}s)", flush=True)

    if a.mmlu:
        mm = json.load(open(f"{a.data}/mmlu.json"))[:a.mmlu]
        print(f"[{a.tag}] MMLU n={len(mm)}", flush=True)
        t0 = time.perf_counter()
        def one(r):
            ch = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", r["ch"]))
            p = f"{r['q']}\n{ch}\n\nAnswer with a single letter (A, B, C, or D) and nothing else."
            txt, ntok = chat(a.base, a.model, p, 8)
            m = re.search(r"\b([ABCD])\b", txt.strip().upper())
            return {"pred": m.group(1) if m else None, "gold": r["a"], "err": txt.startswith("__ERROR__")}
        rr = run_pool(mm, one, a.workers)
        corr = sum(1 for r in rr if r["pred"] == r["gold"])
        res["mmlu"] = {"n": len(rr), "correct": corr, "acc": round(100*corr/len(rr), 2),
                       "unparsed": sum(1 for r in rr if r["pred"] is None),
                       "wall_s": round(time.perf_counter()-t0, 1),
                       "preds": [r["pred"] for r in rr]}
        print(f"[{a.tag}] MMLU acc {res['mmlu']['acc']}%  ({res['mmlu']['wall_s']}s)", flush=True)

    p = os.path.join(a.out, f"quality-{a.tag}.json")
    json.dump(res, open(p, "w"), indent=2)
    print(f"[{a.tag}] -> {p}", flush=True)

if __name__ == "__main__":
    main()
