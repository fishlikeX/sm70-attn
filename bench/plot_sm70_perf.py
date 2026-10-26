#!/usr/bin/env python3
"""Render the sm70-attn perf chart (README style) with zero third-party deps.

Input : bench/perf-2026-10-26-tp2.csv   (series,ctx_tokens,prefill_tps,decode_tps)
Output: bench/sm70-attn-perf.html        (standalone dark-theme SVG page)
        then rasterise with a headless browser, e.g.
        msedge --headless --screenshot=sm70-attn-perf.png --window-size=1400,940 \
               --hide-scrollbars file:///.../bench/sm70-attn-perf.html

Design mirrors the existing sm70-attn.png: dark panel, green tuned curve with a
fade fill, grey dashed stock curve, endpoint value labels, a gain arrow and a
row of KPI cards.
"""
import csv
import io
import os

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "perf-2026-10-26-tp2.csv")
OUT = os.path.join(HERE, "sm70-attn-perf.html")
TITLE2 = "Throughput"
SUB = ("2×V100-SXM2 TP2 · Qwen3.8-27B GSQ-RCO-IQ3_S-mtp · "
       "-fa on, q4_0 KV · tok/s vs prompt length")
NOTE = ("实测 2026-10-26 · 2×V100 TCC · n_predict=96 · cache_prompt=false · "
        "decode 受 MTP 接受率影响（±15%），仅同 prompt 对比有效 · "
        "数据 bench/perf-2026-10-26-tp2.csv")


def parse_args():
    global CSV, OUT, SUB, NOTE, TITLE2, YMAX, YSTEP, TICKS_Y
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        v = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
        if a == "--csv":
            CSV = v
        elif a == "--out":
            OUT = v
        elif a == "--sub":
            SUB = v
        elif a == "--note":
            NOTE = v
        elif a == "--title2":
            TITLE2 = v
        elif a == "--ymax":
            YMAX = int(v)
        elif a == "--ystep":
            YSTEP = int(v)
        TICKS_Y = list(range(0, YMAX + 1, YSTEP))
        i += 2

W, H = 1400, 940
X0, X1, Y0, Y1 = 124, 1236, 176, 556          # plot rect
XMIN, XMAX = 18000, 180000
YMIN, YMAX = 0, 1700        # override with --ymax / --ystep

BG = "#0b0f14"
CARD = "#0f141b"
BORDER = "#1f262e"
GRID = "#1b222b"
TXT = "#e6edf3"
DIM = "#8b949e"
GREEN = "#2ecc71"
GREEN_DIM = "#3fb950"
GREY = "#8b949e"

YSTEP = 400
TICKS_Y = [0, 400, 800, 1200, 1600]
TICKS_X = [20000, 60000, 100000, 140000, 180000]


def xs(v):
    return X0 + (v - XMIN) / (XMAX - XMIN) * (X1 - X0)


def ys(v):
    return Y1 - (v - YMIN) / (YMAX - YMIN) * (Y1 - Y0)


def fmt(v):
    return "%.0f" % v


def load():
    rows = {"tuned": [], "stock": []}
    with io.open(CSV) as f:
        for r in csv.DictReader(f):
            rows[r["series"]].append((int(r["ctx_tokens"]),
                                      float(r["prefill_tps"]),
                                      float(r["decode_tps"])))
    for k in rows:
        rows[k].sort()
    return rows


def svg(data):
    t, s = data["tuned"], data["stock"]
    out = []
    A = out.append

    A('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" '
      'font-family="Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif">' % (W, H, W, H))
    A('<defs>')
    A('<linearGradient id="fade" x1="0" y1="0" x2="0" y2="1">'
      '<stop offset="0%%" stop-color="%s" stop-opacity="0.26"/>'
      '<stop offset="100%%" stop-color="%s" stop-opacity="0.0"/></linearGradient>' % (GREEN, GREEN))
    A('</defs>')
    A('<rect width="%d" height="%d" fill="%s"/>' % (W, H, BG))

    # header
    A('<text x="44" y="62" font-size="34" font-weight="700" fill="%s">Prefill</text>' % TXT)
    A('<text x="168" y="62" font-size="34" font-weight="700" fill="%s">%s</text>' % (GREEN, TITLE2))
    A('<text x="44" y="94" font-size="15" fill="%s">%s</text>' % (DIM, SUB))

    # legend
    A('<line x1="1160" y1="50" x2="1196" y2="50" stroke="%s" stroke-width="3"/>' % GREEN)
    A('<text x="1204" y="55" font-size="15" fill="%s">sm70-attn</text>' % TXT)
    A('<line x1="1160" y1="82" x2="1196" y2="82" stroke="%s" stroke-width="2" '
      'stroke-dasharray="5 4"/>' % GREY)
    A('<text x="1204" y="87" font-size="15" fill="%s">stock</text>' % GREY)

    # panel + grid
    A('<rect x="24" y="118" width="1352" height="500" rx="14" fill="%s" stroke="%s"/>' % (CARD, BORDER))
    for v in TICKS_Y:
        y = ys(v)
        A('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s"/>' % (X0, y, X1, y, GRID))
        A('<text x="%d" y="%.1f" font-size="13" fill="%s" text-anchor="end">%s</text>'
          % (X0 - 14, y + 4, DIM, fmt(v)))
    for v in TICKS_X:
        x = xs(v)
        A('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s"/>' % (x, Y0, x, Y1, GRID))
        lab = "%dk" % (v // 1000)
        A('<text x="%.1f" y="%d" font-size="13" fill="%s" text-anchor="middle">%s</text>'
          % (x, Y1 + 26, DIM, lab))

    # area + tuned line
    pts = " ".join("%.1f,%.1f" % (xs(c), ys(p)) for c, p, _ in t)
    area = "%.1f,%.1f " % (xs(t[0][0]), Y1) + pts + " %.1f,%.1f" % (xs(t[-1][0]), Y1)
    A('<polygon points="%s" fill="url(#fade)"/>' % area)
    A('<polyline points="%s" fill="none" stroke="%s" stroke-width="3.5" '
      'stroke-linejoin="round" stroke-linecap="round"/>' % (pts, GREEN))
    for c, p, _ in t:
        A('<circle cx="%.1f" cy="%.1f" r="4.5" fill="%s"/>' % (xs(c), ys(p), GREEN))

    # stock line
    pts2 = " ".join("%.1f,%.1f" % (xs(c), ys(p)) for c, p, _ in s)
    A('<polyline points="%s" fill="none" stroke="%s" stroke-width="2" '
      'stroke-dasharray="6 5" stroke-linejoin="round"/>' % (pts2, GREY))
    for c, p, _ in s:
        A('<circle cx="%.1f" cy="%.1f" r="3.6" fill="%s"/>' % (xs(c), ys(p), GREY))

    # vertex values (tuned; the last one is covered by the endpoint callout)
    for c, p, _ in t[:-1]:
        A('<text x="%.1f" y="%.1f" font-size="13" fill="%s" text-anchor="middle">%s</text>'
          % (xs(c), ys(p) - 12, GREEN_DIM, fmt(p)))

    # endpoint callout + gain arrow
    ct, pt, _ = t[-1]
    cs, ps, _ = s[-1]
    xg = xs(ct) + 24
    A('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2"/>'
      % (xg, ys(pt), xg, ys(ps), GREEN))
    A('<path d="M %.1f %.1f l -5 -8 l 10 0 z" fill="%s"/>' % (xg, ys(pt) - 2, GREEN))
    A('<path d="M %.1f %.1f l -5 8 l 10 0 z" fill="%s"/>' % (xg, ys(ps) + 2, GREEN))
    A('<text x="%.1f" y="%.1f" font-size="15" font-weight="700" fill="%s" '
      'text-anchor="end">%s tok/s</text>' % (xs(ct) - 12, ys(pt) - 14, GREEN, fmt(pt)))
    A('<text x="%.1f" y="%.1f" font-size="13" fill="%s" text-anchor="end">stock %s</text>'
      % (xs(cs) - 10, ys(ps) + 22, GREY, fmt(ps)))
    gain = (pt / ps - 1.0) * 100.0
    A('<text x="%.1f" y="%.1f" font-size="22" font-weight="700" fill="%s">+%.1f%%</text>'
      % (xg + 12, (ys(pt) + ys(ps)) / 2 + 7, GREEN, gain))
    A('<text x="%.1f" y="%.1f" font-size="12" fill="%s">prefill @174k</text>'
      % (xg + 12, (ys(pt) + ys(ps)) / 2 + 28, DIM))
    A('<text x="%.1f" y="%.1f" font-size="12" fill="%s">peak</text>'
      % (xs(t[0][0]) + 12, ys(t[0][1]) + 22, GREEN_DIM))

    # KPI cards
    cards = [
        ("%s tok/s" % fmt(t[0][1]), "prefill @22k（峰值）"),
        ("+%.1f%%" % gain, "prefill vs stock @174k"),
        ("%s tok/s" % fmt(pt), "prefill @174k"),
        ("%.1f tok/s" % t[-1][2], "decode @174k（stock %.1f，+%.0f%%）"
         % (s[-1][2], (t[-1][2] / s[-1][2] - 1) * 100)),
    ]
    cw, ch, gap = 320, 150, 24
    for i, (big, small) in enumerate(cards):
        x = 44 + i * (cw + gap)
        A('<rect x="%d" y="650" width="%d" height="%d" rx="14" fill="%s" stroke="%s"/>'
          % (x, cw, ch, CARD, BORDER))
        A('<text x="%d" y="716" font-size="40" font-weight="700" fill="%s">%s</text>'
          % (x + 26, GREEN if i != 2 else TXT, big))
        A('<text x="%d" y="750" font-size="14" fill="%s">%s</text>' % (x + 26, DIM, small))

    A('<text x="44" y="888" font-size="13" fill="%s">%s</text>' % (DIM, NOTE))
    A('</svg>')
    return "\n".join(out)


def main():
    parse_args()
    data = load()
    body = svg(data)
    html = ('<!doctype html><html><head><meta charset="utf-8">'
            '<title>sm70-attn prefill throughput</title>'
            '<style>html,body{margin:0;padding:0;background:%s}</style></head>'
            '<body>%s</body></html>' % (BG, body))
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
