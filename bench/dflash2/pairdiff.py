import json, itertools, sys, glob, os
R = "/home/bakeer/dflash2-bench/results/"
prec = sys.argv[1] if len(sys.argv) > 1 else "fp8"
tags = [t for t in [f"{prec}-nospec", f"{prec}-mtp3", f"{prec}-dspark7", f"{prec}-dflash2"]
        if os.path.exists(R + t + ".json")]
S = {t: json.load(open(R + t + ".json")).get("samples", {}) for t in tags}
KS = ["code", "prose", "math", "refactor"]
print("Pairwise greedy-output identity (T=0):")
print(f"{'pair':38s}" + "".join(f"{k:>13s}" for k in KS))
print("-" * 90)
for a, b in itertools.combinations(tags, 2):
    row = f"{a + ' vs ' + b:38s}"
    for k in KS:
        x, y = S[a].get(k, ""), S[b].get(k, "")
        if not x or not y:
            row += f"{'-':>13s}"; continue
        if x == y:
            row += f"{'IDENTICAL':>13s}"
        else:
            i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            row += f"{'div@' + str(i):>13s}"
    print(row)
# show the actual divergence context for one case
print("\nDivergence context (first differing region), nospec vs dflash2, prompt=code:")
x = S.get(f"{prec}-nospec", {}).get("code", ""); y = S.get(f"{prec}-dflash2", {}).get("code", "")
if x and y and x != y:
    i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), 0)
    lo = max(0, i - 90)
    print(f"  nospec : ...{x[lo:i]}[[{x[i:i+45]}]]")
    print(f"  dflash2: ...{y[lo:i]}[[{y[i:i+45]}]]")
