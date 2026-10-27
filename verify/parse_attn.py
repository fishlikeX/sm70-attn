import os
#!/usr/bin/env python3
import csv, re
from collections import defaultdict

CSV_PATH = os.environ.get("SM70_ATTN_CSV", "ncu176k_attn.csv")
# dump one with: ncu --csv --metrics ... > ncu176k_attn.csv, or set SM70_ATTN_CSV
rows = []
with open(CSV_PATH, newline="") as f:
    lines = [ln for ln in f if ln.startswith('"')]
reader = csv.DictReader(lines)
for r in reader:
    try:
        inv = int(r["Invocations"])
        avg_ns = float(r["Average"].replace(",", ""))
    except (KeyError, ValueError):
        continue
    name = r["Kernel Name"]
    grid = r["Grid Size"]
    rows.append((name, grid, inv, avg_ns))

def clean(n):
    n = re.sub(r"<[^<>]*>", "", n)
    n = re.sub(r"\(.*$", "", n)
    n = n.replace("void ", "").replace("flash::", "")
    return n.strip()

agg = defaultdict(lambda: [0, 0.0])
for name, grid, inv, avg in rows:
    c = clean(name)
    agg[c][0] += inv
    agg[c][1] += avg * inv

total = sum(v[1] for v in agg.values())
print(f"attention-path total: {total/1e6:.0f} ms over {sum(v[0] for v in agg.values())} launches\n")
for n, (inv, tot) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print(f"  {tot/total*100:5.1f}%  {tot/1e6:8.0f} ms  x{inv:<6d} {n}")
