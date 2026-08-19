#!/usr/bin/env python3
"""Aggregate dflash2 bench results into comparison tables + losslessness diff."""
import glob, json, os, sys, hashlib

RES = sys.argv[1] if len(sys.argv) > 1 else "/home/bakeer/dflash2-bench/results"
ORDER = ["fp8-nospec","fp8-mtp3","fp8-dspark7","fp8-dflash2",
         "nvfp4-nospec","nvfp4-mtp3","nvfp4-dspark7","nvfp4-dflash2"]
perf = {}
for f in glob.glob(os.path.join(RES, "*.json")):
    b = os.path.basename(f)[:-5]
    if b.startswith("quality-"): continue
    try: perf[b] = json.load(open(f))
    except Exception as e: print("skip", b, e)

tags = [t for t in ORDER if t in perf] + sorted(t for t in perf if t not in ORDER)
if not tags: print("no results yet"); sys.exit()

print("\n=== SINGLE-STREAM DECODE (tok/s, greedy, thinking off, 512 max tokens) ===")
hdr = f"{'config':<16}" + "".join(f"{k:>10}" for k in ["code","prose","math","refactor","MEAN"]) + f"{'TTFT_s':>9}"
print(hdr); print("-"*len(hdr))
base = None
for t in tags:
    d = perf[t]; s = d.get("single", {})
    row = f"{t:<16}"
    for k in ["code","prose","math","refactor"]:
        row += f"{s.get(k,{}).get('decode_tps_median','-'):>10}"
    mean = d.get("decode_tps_mean_of_medians","-")
    row += f"{mean:>10}"
    ttfts = [v.get("ttft_median") for v in s.values() if v.get("ttft_median")]
    row += f"{round(sum(ttfts)/len(ttfts),3) if ttfts else '-':>9}"
    print(row)

print("\n=== SPECULATIVE DECODE BEHAVIOUR ===")
hdr = f"{'config':<16}{'active':>8}{'draft_len':>11}{'accept_rate':>13}{'accept_len':>12}{'speedup_vs_nospec':>19}"
print(hdr); print("-"*len(hdr))
for prec in ("fp8","nvfp4"):
    b = perf.get(f"{prec}-nospec", {}).get("decode_tps_mean_of_medians")
    for t in tags:
        if not t.startswith(prec+"-"): continue
        d = perf[t]; sp = d.get("spec", {})
        m = d.get("decode_tps_mean_of_medians")
        su = f"{m/b:.2f}x" if (b and m) else "-"
        print(f"{t:<16}{str(sp.get('spec_active')):>8}{str(sp.get('draft_len','-')):>11}"
              f"{str(sp.get('accept_rate','-')):>13}{str(sp.get('acceptance_length','-')):>12}{su:>19}")

print("\n=== BATCH x8 (aggregate tok/s) ===")
hdr = f"{'config':<16}{'aggregate':>11}{'per_stream':>12}{'accept_len':>12}{'wall_s':>9}"
print(hdr); print("-"*len(hdr))
for t in tags:
    b = perf[t].get("batch")
    if not b: continue
    print(f"{t:<16}{b.get('aggregate_tps','-'):>11}{b.get('per_stream_tps','-'):>12}"
          f"{str(b.get('spec',{}).get('acceptance_length','-')):>12}{b.get('wall','-'):>9}")

print("\n=== MEMORY (host, GiB) ===")
for t in tags:
    m = perf[t].get("mem_gib", {})
    if m: print(f"  {t:<16} used {m.get('used')}  avail {m.get('available')}  / {m.get('total')}")

print("\n=== LOSSLESSNESS: greedy output identity within a precision ===")
for prec in ("fp8","nvfp4"):
    ref = f"{prec}-nospec"
    if ref not in perf: continue
    rs = perf[ref].get("samples", {})
    for t in tags:
        if not t.startswith(prec+"-") or t == ref: continue
        ss = perf[t].get("samples", {})
        marks = []
        for k in rs:
            a, b = rs.get(k,""), ss.get(k,"")
            if a and b:
                if a == b: marks.append(f"{k}=IDENTICAL")
                else:
                    # first divergence point
                    i = next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
                    marks.append(f"{k}=DIVERGES@{i}ch")
        print(f"  {t:<16} vs {ref:<14} " + "  ".join(marks))

print("\n=== QUALITY ===")
hdr = f"{'config':<16}{'GSM8K':>9}{'MMLU':>9}{'gsm_tok':>9}{'unparsed':>10}"
print(hdr); print("-"*len(hdr))
qual = {}
for f in sorted(glob.glob(os.path.join(RES, "quality-*.json"))):
    d = json.load(open(f)); qual[d["tag"]] = d
    print(f"{d['tag']:<16}{d.get('gsm8k',{}).get('acc','-'):>9}{d.get('mmlu',{}).get('acc','-'):>9}"
          f"{d.get('gsm8k',{}).get('mean_tokens','-'):>9}{d.get('mmlu',{}).get('unparsed','-'):>10}")
if len(qual) > 1:
    print("\n  per-item answer agreement (share of identical predictions):")
    ks = list(qual)
    for i in range(len(ks)):
        for j in range(i+1, len(ks)):
            a, b = qual[ks[i]], qual[ks[j]]
            for suite in ("gsm8k","mmlu"):
                pa, pb = a.get(suite,{}).get("preds"), b.get(suite,{}).get("preds")
                if pa and pb and len(pa)==len(pb):
                    agr = sum(1 for x,y in zip(pa,pb) if x==y)/len(pa)*100
                    print(f"    {ks[i]:<16} vs {ks[j]:<16} {suite:<6} {agr:5.1f}%")
