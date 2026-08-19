import json, os
R = "/home/bakeer/dflash2-bench/results"
def g(t):
    p = os.path.join(R, t + ".json")
    return json.load(open(p)) if os.path.exists(p) else None

ROWS = [("FP8 (vLLM, 27.6 GB)", "fp8"),
        ("NVFP4 (vLLM, ~16 GB)", "nvfp4"),
        ("int4-AutoRound (vLLM)", "int4"),
        ("Q4_K_M (llama.cpp, 19 GB)", "gguf-q4km")]
COLS = [("no spec", "nospec"), ("MTP k=3", "mtp3"),
        ("DSpark k=7", "dspark7"), ("DFlash2 k=7", "dflash2")]

def table(title, pick):
    print(title)
    print(f"{'weights':27s}" + "".join(f"{c[0]:>13s}" for c in COLS))
    print("-" * 79)
    for label, pfx in ROWS:
        line = f"{label:27s}"
        for _, cs in COLS:
            d = g(f"{pfx}-{cs}")
            if d is None:
                v = "BLOCKED" if cs == "dflash2" and pfx in ("nvfp4", "int4") else "--"
                line += f"{v:>13s}"
            else:
                line += f"{pick(d):>13s}"
        print(line)
    print()

table("SINGLE-STREAM GENERATIVE DECODE (tok/s, greedy, thinking off)",
      lambda d: f"{d['decode_tps_mean_of_medians']:.2f}")
table("ACCEPTANCE LENGTH (tokens emitted per verification pass)",
      lambda d: str(d.get("spec", {}).get("acceptance_length", "--")))
table("BATCH x8 AGGREGATE (tok/s)",
      lambda d: f"{d.get('batch', {}).get('aggregate_tps', 0):.2f}")

print("SPEEDUP vs SAME-PRECISION no-spec BASELINE")
print(f"{'weights':27s}" + "".join(f"{c[0]:>13s}" for c in COLS))
print("-" * 79)
for label, pfx in ROWS:
    b = g(f"{pfx}-nospec")
    base = b["decode_tps_mean_of_medians"] if b else None
    line = f"{label:27s}"
    for _, cs in COLS:
        d = g(f"{pfx}-{cs}")
        if d is None:
            line += f"{('BLOCKED' if cs=='dflash2' and pfx in ('nvfp4','int4') else '--'):>13s}"
        elif base:
            line += f"{d['decode_tps_mean_of_medians']/base:>12.2f}x"
        else:
            line += f"{'--':>13s}"
    print(line)

print("\nEDIT REGIME (high copy fraction) — single stream")
print(f"{'config':32s}{'tok/s':>10s}{'accept_len':>12s}{'accept_rate':>13s}{'batch x8':>11s}")
print("-" * 78)
for t in ["edit-nvfp4-nospec", "edit-nvfp4-dspark7", "edit-nvfp4-dspark14",
          "edit-fp8-dflash2", "gguf-q4km-nospec-ed", "gguf-q4km-dflash2ed"]:
    d = g(t)
    if not d: continue
    s = d.get("spec", {})
    print(f"{t:32s}{d['decode_tps_mean_of_medians']:>10.2f}"
          f"{str(s.get('acceptance_length','--')):>12s}{str(s.get('accept_rate','--')):>13s}"
          f"{d.get('batch',{}).get('aggregate_tps',0):>11.2f}")

print("\nDRAFTER PRECISION (llama.cpp Q4_K_M target)")
for t in ["gguf-q4km-dflash2", "gguf-q4km-dflash2q8"]:
    d = g(t)
    if d: print(f"  {t:24s} {d['decode_tps_mean_of_medians']:6.2f} tok/s   "
                f"accept_len {d['spec']['acceptance_length']}")
