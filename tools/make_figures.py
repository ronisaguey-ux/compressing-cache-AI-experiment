#!/usr/bin/env python3
"""Render the results figure as a standalone SVG -- no plotting library, no dependencies.

★ WHY SVG AND NOT PNG. A paper is rendered to PDF, where a raster figure is either blurry or
enormous; a vector figure stays sharp at any size and is a few KB. It also removes matplotlib
from the critical path, which matters because this has to be regenerable on whatever machine
holds the results, and "the plotting library is missing" must not be the reason a figure is
absent from a submission.

★ THE FIGURE IS GENERATED FROM THE SAME `cache_rows` THE TABLES USE, so it cannot disagree with
them. The paper's central claim is a *shape*: one policy's context grows to the ceiling and then
stops, the other stays flat. A table states the two endpoints; only the figure shows the shape,
and the shape is what a reviewer can check in one glance.

Two panels: (a) prompt tokens per turn, (b) prefix reuse per turn -- the mechanism behind (a).

Usage: python3 tools/make_figures.py [results_dir] [out.svg]
"""
import glob
import json
import os
import sys

OUT_DIR = os.environ.get("RESULTS_DIR",
                         os.path.expanduser("~/.local/share/ccai-results/fixmode-v3"))
OUT_SVG = os.environ.get("FIG_PATH",
                         os.path.expanduser("~/.local/share/ccai-repo/paper/figs/trajectory.svg"))
ORDER = ["runtime", "linear", "prune"]
COLOUR = {"runtime": "#1b6ca8", "linear": "#c0392b", "prune": "#2e7d32"}

W, H = 1120, 420
PAD_L, PAD_T = 74, 34
PANEL_W, PANEL_H = 470, 300
GAP = 56


def load(d):
    runs = {}
    for p in sorted(glob.glob(os.path.join(d, "incremental_*.json"))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if isinstance(r, dict) and r.get("arm") and r.get("cache_rows"):
            runs[r["arm"]] = r
    return runs


def _nice(v):
    """A round-ish upper bound so the axis label is readable."""
    if v <= 0:
        return 1
    step = 10 ** (len(str(int(v))) - 1)
    for m in (1, 2, 2.5, 5, 10):
        if v <= step * m:
            return step * m
    return v


def panel(series, x0, y0, title, ylab, ymax_override=None, ymin=0):
    """series: list of (label, xs, ys). Returns SVG fragments."""
    xs_all = [x for _, xs, _ in series for x in xs]
    ys_all = [y for _, _, ys in series for y in ys]
    xmax = max(xs_all) if xs_all else 1
    ymax = ymax_override or _nice(max(ys_all) if ys_all else 1)
    ymin = min(ymin, min(ys_all) if ys_all else 0)

    def sx(x):
        return x0 + (x / xmax) * PANEL_W

    def sy(y):
        span = (ymax - ymin) or 1
        return y0 + PANEL_H - ((y - ymin) / span) * PANEL_H

    out = []
    # axes
    out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#333" stroke-width="1"/>'
               % (x0, y0 + PANEL_H, x0 + PANEL_W, y0 + PANEL_H))
    out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#333" stroke-width="1"/>'
               % (x0, y0, x0, y0 + PANEL_H))
    # gridlines + y labels
    for i in range(5):
        yv = ymin + (ymax - ymin) * i / 4.0
        yy = sy(yv)
        out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#e6e6e6" stroke-width="1"/>'
                   % (x0, yy, x0 + PANEL_W, yy))
        out.append('<text x="%.1f" y="%.1f" font-size="11" fill="#555" text-anchor="end">%d</text>'
                   % (x0 - 8, yy + 4, int(yv)))
    # x ticks
    for i in range(5):
        xv = xmax * i / 4.0
        out.append('<text x="%.1f" y="%.1f" font-size="11" fill="#555" text-anchor="middle">%d</text>'
                   % (sx(xv), y0 + PANEL_H + 18, int(xv)))
    out.append('<text x="%.1f" y="%.1f" font-size="12" fill="#333" text-anchor="middle">turn</text>'
               % (x0 + PANEL_W / 2, y0 + PANEL_H + 38))
    out.append('<text x="%.1f" y="%.1f" font-size="12" fill="#333" text-anchor="middle" '
               'transform="rotate(-90 %.1f %.1f)">%s</text>'
               % (x0 - 50, y0 + PANEL_H / 2, x0 - 50, y0 + PANEL_H / 2, ylab))
    out.append('<text x="%.1f" y="%.1f" font-size="13" font-weight="600" fill="#111" '
               'text-anchor="middle">%s</text>' % (x0 + PANEL_W / 2, y0 - 12, title))
    # the lines
    for label, xs, ys in series:
        pts = " ".join("%.1f,%.1f" % (sx(x), sy(y)) for x, y in zip(xs, ys))
        out.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>'
                   % (pts, COLOUR.get(label, "#555")))
    return out, sy, (x0, y0)


def main(argv):
    d = argv[1] if len(argv) > 1 else OUT_DIR
    out = argv[2] if len(argv) > 2 else OUT_SVG
    runs = load(d)
    if not runs:
        print("no arms with cache_rows in %s -- nothing to plot" % d)
        return 1

    order = [a for a in ORDER if a in runs] + [a for a in sorted(runs) if a not in ORDER]

    growth, reuse = [], []
    for a in order:
        rows = sorted(runs[a]["cache_rows"], key=lambda x: x.get("turn", 0))
        xs = [r["turn"] for r in rows]
        growth.append((a, xs, [r["prompt"] for r in rows]))
        reuse.append((a, xs, [100.0 * (r["hit"] / r["prompt"] if r.get("prompt") else 0)
                              for r in rows]))

    frag = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">'
            % (W, H, W, H),
            '<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H)]

    frag += panel(growth, PAD_L, PAD_T, "(a) context growth", "prompt tokens (tokens)")[0]
    x2 = PAD_L + PANEL_W + GAP
    frag += panel(reuse, x2, PAD_T, "(b) prefix-cache reuse", "reused (%)",
                  ymax_override=100)[0]

    # legend, top-right of the first panel's header row
    lx = x2 + PANEL_W - 150
    ly = 16
    for i, a in enumerate(order):
        yy = ly + i * 15
        frag.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="3"/>'
                    % (lx, yy, lx + 22, yy, COLOUR.get(a, "#555")))
        frag.append('<text x="%d" y="%d" font-size="12" fill="#333">%s</text>' % (lx + 28, yy + 4, a))

    frag.append("</svg>")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(frag))
    print("wrote %s  (%d arms: %s)" % (out, len(order), ", ".join(order)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
