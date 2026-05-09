"""render the 'scene = graph, video = graph evolving' figure.

shows:
  - 5 graph snapshots laid out as a timeline
  - each snapshot is the same N nodes with shifting edges
  - the rightmost snapshot has one node-at-time marked as masked (predict-this)
  - arrow from context snapshots to the masked slot
  - matches v-jepa 2 fig 2 (right panel) for masked prediction, but on graphs

style: light blue palette, sharp corners, serif (Times-like via STIX), no italics.
saves to figures/scene_graph.png.
"""
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch


mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "stix"


# colors
border = "#5B9BD5"
node_fill = "#EAF3FB"
masked_fill = "#FFE5B4"   # warm tan to mark the predict-this node
edge_color = "#A0A0A0"
mask_edge = "#D17A00"     # warm orange for the masked node outline
text_color = "#000000"
arrow_color = "#5B9BD5"
font = "serif"


# layout: 5 snapshots side by side, each a copy of the same graph with edge variation
n_snapshots = 5
n_nodes = 7
snapshot_w = 2.4    # width of each panel
snapshot_h = 2.4    # height of each panel
gap = 0.4
fig_w = n_snapshots * snapshot_w + (n_snapshots - 1) * gap + 1.0
fig_h = snapshot_h + 2.2

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(-0.2, fig_h)
ax.axis("off")
ax.set_aspect("equal")


# fixed node positions on a circle (same across all snapshots — same N nodes, different edges)
def node_positions(cx, cy, r, n):
    pos = []
    for i in range(n):
        theta = 2 * math.pi * i / n - math.pi / 2  # start at top
        pos.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return pos


# generate edge sets per snapshot — same N nodes, edges shift over time
rng = np.random.default_rng(seed=0)


def random_edges(n, density=0.35, seed=0):
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                w = 0.3 + rng.random() * 1.2
                edges.append((i, j, w))
    return edges


snapshots = [random_edges(n_nodes, density=0.45, seed=s + 1) for s in range(n_snapshots)]


# render each snapshot
y_center = snapshot_h / 2 + 0.6
node_radius = 0.13

for s in range(n_snapshots):
    cx = 0.5 + s * (snapshot_w + gap) + snapshot_w / 2
    cy = y_center
    pos = node_positions(cx, cy, snapshot_h / 3.0, n_nodes)

    # panel border
    ax.add_patch(plt.Rectangle(
        (cx - snapshot_w / 2, cy - snapshot_h / 2),
        snapshot_w, snapshot_h,
        linewidth=0.8, edgecolor=border, facecolor="white",
    ))

    # is this the target snapshot (t+1)?
    is_target = (s == n_snapshots - 1)
    masked_node_idx = 2 if is_target else None

    # draw edges
    for i, j, w in snapshots[s]:
        if is_target and (i == masked_node_idx or j == masked_node_idx):
            continue  # skip edges incident to the masked node — they're hidden from the predictor
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        ax.plot([x0, x1], [y0, y1], color=edge_color, linewidth=0.4 + w * 0.6, alpha=0.6, zorder=1)

    # draw nodes
    for i, (x, y) in enumerate(pos):
        if i == masked_node_idx:
            # masked target — orange dashed circle, ? inside
            ax.add_patch(Circle((x, y), node_radius, facecolor=masked_fill,
                                edgecolor=mask_edge, linewidth=1.2, linestyle=(0, (3, 2)), zorder=3))
            ax.text(x, y, "?", ha="center", va="center", fontsize=11,
                    color=mask_edge, family=font, zorder=4)
        else:
            ax.add_patch(Circle((x, y), node_radius, facecolor=node_fill,
                                edgecolor=border, linewidth=0.8, zorder=2))

    # snapshot label below the panel
    if is_target:
        label = r"$t+1$ (target)"
    else:
        offset = n_snapshots - 2 - s   # K, K-1, ..., 1, 0
        label = f"$t-{offset}$" if offset > 0 else "$t$"
    ax.text(cx, cy - snapshot_h / 2 - 0.18, label,
            ha="center", va="top", fontsize=10, color=text_color, family=font)


# context label spanning snapshots 0..K-1
ctx_left = 0.5
ctx_right = ctx_left + (n_snapshots - 1) * (snapshot_w + gap) - gap + snapshot_w / 2
target_left = ctx_right + gap + snapshot_w / 2 - snapshot_w  # rough
target_center = 0.5 + (n_snapshots - 1) * (snapshot_w + gap) + snapshot_w / 2

# bracket / span line above context
ctx_y = y_center + snapshot_h / 2 + 0.25
ctx_x_left = 0.5
ctx_x_right = 0.5 + (n_snapshots - 2) * (snapshot_w + gap) + snapshot_w
ax.plot([ctx_x_left, ctx_x_right], [ctx_y, ctx_y], color=text_color, linewidth=0.7)
ax.plot([ctx_x_left, ctx_x_left], [ctx_y - 0.06, ctx_y + 0.06], color=text_color, linewidth=0.7)
ax.plot([ctx_x_right, ctx_x_right], [ctx_y - 0.06, ctx_y + 0.06], color=text_color, linewidth=0.7)
ax.text((ctx_x_left + ctx_x_right) / 2, ctx_y + 0.18,
        "context: K history snapshots (online encoder reads each)",
        ha="center", va="bottom", fontsize=9, color=text_color, family=font)

# arrow from context to target (predictor)
arr = FancyArrowPatch(
    (ctx_x_right + 0.05, y_center),
    (target_center - snapshot_w / 2 - 0.05, y_center),
    arrowstyle="-|>", mutation_scale=14,
    linewidth=1.0, color=arrow_color, zorder=5,
)
ax.add_patch(arr)
ax.text((ctx_x_right + target_center - snapshot_w / 2) / 2, y_center + 0.28,
        "predictor", ha="center", va="bottom", fontsize=6, color=text_color, family=font)
ax.text((ctx_x_right + target_center - snapshot_w / 2) / 2, y_center - 0.28,
        "(latent space)", ha="center", va="top", fontsize=5, color=text_color, family=font)

# label above the target panel
ax.text(target_center, ctx_y + 0.18,
        "target: predict the masked node-at-time",
        ha="center", va="bottom", fontsize=9, color=text_color, family=font)

# title / caption
ax.text(fig_w / 2, fig_h - 0.05,
        "Scene = graph snapshot. Video = graph evolving over time.",
        ha="center", va="top", fontsize=10.5, color=text_color, family=font)

plt.tight_layout()
out = "figures/scene_graph.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
