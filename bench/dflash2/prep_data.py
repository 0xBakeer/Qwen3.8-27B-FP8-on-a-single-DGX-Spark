import glob, json, random, os
import pyarrow.parquet as pq
out = "/home/bakeer/dflash2-bench/data"; os.makedirs(out, exist_ok=True)
random.seed(1234)

g = glob.glob("/root/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/*/main/test-*.parquet")[0]
t = pq.read_table(g).to_pylist()
rows = [{"q": r["question"], "a": r["answer"].split("####")[-1].strip().replace(",", "")} for r in t]
random.shuffle(rows); rows = rows[:200]
json.dump(rows, open(f"{out}/gsm8k.json", "w")); print("gsm8k", len(rows))

mm = []
for f in sorted(glob.glob("/root/.cache/huggingface/hub/datasets--cais--mmlu/snapshots/*/*/test-*.parquet")):
    subj = f.split("/")[-2]
    if subj in ("all", "auxiliary_train"): continue
    for r in pq.read_table(f).to_pylist():
        mm.append({"subj": subj, "q": r["question"], "ch": r["choices"], "a": "ABCD"[r["answer"]]})
random.shuffle(mm); mm = mm[:400]
json.dump(mm, open(f"{out}/mmlu.json", "w")); print("mmlu", len(mm), "subjects", len({r["subj"] for r in mm}))
