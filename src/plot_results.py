"""Plot batch_size vs throughput from mar16.tsv."""

import csv
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TSV_PATH = "outputs/mar16.tsv"
OUT_PATH = "outputs/mar16_throughput.png"
MAX_LABEL_LEN = 32

COLOR = {"keep": "#2ecc71", "discard": "#aaaaaa", "crash": "#e74c3c"}

def parse_batch(desc):
    m = re.search(r"batch=(\d+)", desc)
    return int(m.group(1)) if m else None

def shorten(desc):
    # Strip leading "batch=NNN — " or "batch=NNN "
    desc = re.sub(r"^batch=\d+[\s\u2014-]*", "", desc).strip()
    if len(desc) > MAX_LABEL_LEN:
        desc = desc[:MAX_LABEL_LEN - 1] + "\u2026"
    return desc

rows = []
with open(TSV_PATH, newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        batch = parse_batch(row["description"])
        if batch is None:
            continue
        rows.append({
            "batch": batch,
            "tk_s": float(row["tk_s"]),
            "status": row["status"].strip(),
            "label": shorten(row["description"]),
        })

fig, ax = plt.subplots(figsize=(13, 7))

# Track how many points share each (batch, tk_s) to jitter x slightly
x_counts = defaultdict(int)
for r in rows:
    key = (r["batch"], r["tk_s"])
    r["_jitter_idx"] = x_counts[key]
    x_counts[key] += 1

# Small x-jitter per overlapping point at same position
X_JITTER = 6

# Collect annotation positions to alternate offsets for same-x clusters
batch_point_count = defaultdict(int)

for r in rows:
    x = r["batch"] + r["_jitter_idx"] * X_JITTER
    y = r["tk_s"]
    color = COLOR[r["status"]]
    ax.scatter(x, y, color=color, s=80, zorder=3, edgecolors="white", linewidths=0.5)

    # Alternate label above/below for each successive point at this batch size
    idx = batch_point_count[r["batch"]]
    batch_point_count[r["batch"]] += 1

    # For the rightmost batch size, put labels to the left to avoid clipping
    if r["batch"] >= 400:
        x_offset, ha = -8, "right"
    else:
        x_offset, ha = 8, "left"

    # Larger spread for dense cluster on the right
    if r["batch"] >= 256:
        y_offsets = [22, -28, 52, -58, 82, -88]
    else:
        y_offsets = [18, -24, 44, -50]
    y_offset = y_offsets[idx % len(y_offsets)]
    va = "bottom" if y_offset > 0 else "top"

    ax.annotate(
        r["label"],
        xy=(x, y),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        fontsize=7,
        color=color,
        va=va,
        ha=ha,
    )

# Draw the "keep" line connecting kept points in order of batch size
kept = sorted([r for r in rows if r["status"] == "keep"], key=lambda r: r["batch"])
if kept:
    ax.plot(
        [r["batch"] for r in kept],
        [r["tk_s"] for r in kept],
        color="#2ecc71",
        linewidth=1.2,
        linestyle="--",
        alpha=0.5,
        zorder=2,
    )

ax.set_xlabel("Batch size", fontsize=12)
ax.set_ylabel("Throughput (tokens/s)", fontsize=12)
ax.set_title("Batch size vs. throughput — mar16 experiment run", fontsize=13)
ax.set_xscale("log", base=2)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: str(int(v))))
ax.set_ylim(bottom=-200)   # room for crash labels below y=0
ax.grid(True, which="both", linestyle=":", alpha=0.4)

legend_handles = [
    mpatches.Patch(color=COLOR["keep"],    label="keep"),
    mpatches.Patch(color=COLOR["discard"], label="discard"),
    mpatches.Patch(color=COLOR["crash"],   label="crash"),
]
ax.legend(handles=legend_handles, fontsize=10)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"Saved to {OUT_PATH}")
