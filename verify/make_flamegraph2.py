import os
#!/usr/bin/env python3
# Social preview flamegraph from REAL Nsight Compute data:
# 176k-context prefill, attention subsystem, V100.
import csv, re, sys
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

CSV_PATH = "/home/fishlikexie/ncu176k_attn.csv"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SM70_FLAMEGRAPH_PNG", "/tmp/flamegraph.png")
W, H = 1280, 640

rows = []
with open(CSV_PATH, newline="") as f:
    lines = [ln for ln in f if ln.startswith('"')]
for r in csv.DictReader(lines):
    try:
        inv = int(r["Invocations"]); avg = float(r["Average"].replace(",", ""))
    except (KeyError, ValueError):
        continue
    rows.append((r["Kernel Name"], inv, avg))

def clean(n):
    n = re.sub(r"<[^<>]*>", "", n); n = re.sub(r"\(.*$", "", n)
    return n.replace("void ", "").replace("flash::", "").strip()

agg = defaultdict(lambda: [0, 0.0])
for name, inv, avg in rows:
    c = clean(name); agg[c][0] += inv; agg[c][1] += avg * inv
total = sum(v[1] for v in agg.values())
kernels = sorted(agg.items(), key=lambda kv: -kv[1][1])

BG, TITLE, SUB, TXT = (24, 26, 32), (240, 243, 250), (150, 158, 170), (18, 18, 22)
FLAME = [(255, 84, 37), (255, 140, 50), (255, 190, 60), (255, 220, 110), (255, 240, 160)]

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

def font(sz, bold=False):
    p = f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"
    try: return ImageFont.truetype(p, sz)
    except OSError: return ImageFont.load_default()

f_title, f_sub, f_box, f_small = font(34, True), font(16), font(14), font(12)

d.text((44, 30), "sm70-attn", font=f_title, fill=TITLE)
d.text((50, 76), "Tesla V100 · Qwen3.8-27B · prefill 176,000 tokens · attention path, real kernel times (Nsight Compute)",
       font=f_sub, fill=SUB)

X0, X1 = 44, W - 44
BH, GAP = 30, 5

# root bar
y = 132
d.rectangle([X0, y, X1, y + BH], fill=(64, 68, 80))
d.text((X0 + 10, y + 7), f"attention path — {total/1e6:.1f} s measured GPU time", font=f_box, fill=TITLE)

# main kernel level (log-ish visual: linear shares, min width for slivers)
y += BH + GAP
x = X0
x = X0
positions = []
for i, (name, (inv, tot)) in enumerate(kernels):
    share = tot / total
    w = int((X1 - X0) * share)
    positions.append((name, inv, tot, share, x, w))
    x += w
x = X0
for i, (name, inv, tot, share, _, _) in enumerate(positions):
    w = max(int((X1 - X0) * share), 10 if share > 0.001 else 4)
    col = FLAME[min(i, len(FLAME) - 1)]
    d.rectangle([x, y, min(x + w, X1), y + BH], fill=col)
    if w > 150:
        d.text((x + 12, y + 7), f"{name}  {share*100:.1f}%", font=f_box, fill=TXT)
    elif w > 40:
        d.text((x + 8, y + 9), f"{share*100:.1f}%", font=f_small, fill=TXT)
    x += w
    if x >= X1: break

# zoom bar: the non-dominant kernels magnified
rest = kernels[1:]
rest_tot = sum(v[1] for _, v in rest) or 1
y += BH + 14
d.text((X0, y - 4), f"×32 zoom — the other {rest_tot/1e6:.1f} s:", font=f_small, fill=SUB)
y += 18
x = X0
for i, (name, (inv, tot)) in enumerate(rest):
    share = tot / rest_tot
    w = int((X1 - X0) * share)
    if w < 4: continue
    col = FLAME[min(i + 1, len(FLAME) - 1)]
    d.rectangle([x, y, min(x + w, X1), y + BH - 8], fill=col)
    if w > 130:
        d.text((x + 10, y + 3), f"{name}  {share*100:.1f}%", font=f_small, fill=TXT)
    x += w
    if x >= X1: break

# headline stat block
ky = y + BH + 26
main_name, (main_inv, main_tot) = kernels[0]
d.text((44, ky), f"{main_tot/1e6:.0f} s of HMMA in one kernel — {main_tot/total*100:.0f}% of the attention path, {main_inv} launches",
       font=f_sub, fill=TITLE)

d.text((44, H - 58), "Split-D N32 D256 · SplitKV3 · q4_0 KV · DFlash2 speculative decoding · 23/23 regression harness",
       font=f_small, fill=SUB)
d.text((44, H - 36), "github.com/fishlikeX/sm70-attn  ·  +42.9% 176k prefill vs stock llama.cpp flash-attn",
       font=f_small, fill=SUB)

img.save(OUT)
print("saved", OUT)
