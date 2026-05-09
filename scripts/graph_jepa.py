"""render the graph-jepa architecture flow diagram.

equivalent of V-JEPA 2 Figure 2 left panel and I-JEPA's ijepa.png:
context view + masked target  -> online encoder -> predictor (+ mask token)
unmasked target               -> EMA encoder    -> stop-grad
predictor and EMA target outputs meet at L2 loss; EMA update from online to target.

style: light blue palette, sharp corners, serif (Times-like via STIX), no italics.
saves to figures/graph_jepa.png.
"""
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "stix"


# colors
border = "#5B9BD5"
fill = "#EAF3FB"
fill_focal = "#C9E0F2"
fill_loss = "#FCE7B0"
edge_color = "#A0A0A0"
node_fill = "#EAF3FB"
masked_fill = "#FFE5B4"
mask_edge = "#D17A00"
text_color = "#000000"
font = "serif"


fig_w = 14.0
fig_h = 9.0

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")
ax.set_aspect("equal")


def box(x, y, w, h, label, fc=fill, ec=border, fs=11, sub=None, sub_fs=8.5):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=0.9, edgecolor=ec, facecolor=fc))
    if sub is None:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=text_color, family=font)
    else:
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                fontsize=fs, color=text_color, family=font)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=sub_fs, color=text_color, family=font)


def arrow(x1, y1, x2, y2, color=border, lw=1.0, style="-|>", linestyle="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=12,
                        linewidth=lw, color=color, zorder=2,
                        linestyle=linestyle)
    ax.add_patch(a)


def graph_icon(cx, cy, r=0.40, n=6, density=0.5, seed=0, masked_idx=None):
    rng = np.random.default_rng(seed)
    pos = []
    for i in range(n):
        theta = 2 * math.pi * i / n - math.pi / 2
        pos.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                edges.append((i, j))
    for i, j in edges:
        if masked_idx is not None and (i == masked_idx or j == masked_idx):
            continue
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        ax.plot([x0, x1], [y0, y1], color=edge_color, linewidth=0.5, alpha=0.7, zorder=1)
    nr = 0.07
    for i, (x, y) in enumerate(pos):
        if i == masked_idx:
            ax.add_patch(Circle((x, y), nr, facecolor=masked_fill, edgecolor=mask_edge,
                                linewidth=1.2, linestyle=(0, (3, 2)), zorder=3))
            ax.text(x, y, "?", ha="center", va="center", fontsize=6,
                    color=mask_edge, family=font, zorder=4)
        else:
            ax.add_patch(Circle((x, y), nr, facecolor=node_fill, edgecolor=border,
                                linewidth=0.7, zorder=2))


# row 1 (bottom): inputs --- context view + masked target | unmasked target ---
n_ctx = 4
n_nodes_icon = 6
gap_icon = 1.10
masked_node = 2
input_y = 1.0

# left input panel: K context graphs + masked target snapshot
left_panel_left = 0.6
for s in range(n_ctx):
    cx_g = 1.2 + s * gap_icon
    graph_icon(cx_g, input_y, r=0.42, n=n_nodes_icon, density=0.55, seed=s + 7)
    offset = n_ctx - s
    label = rf"$t{{-}}{offset}$"
    ax.text(cx_g, input_y - 0.62, label, ha="center", va="top",
            fontsize=8.5, color=text_color, family=font)

masked_cx = 1.2 + n_ctx * gap_icon
graph_icon(masked_cx, input_y, r=0.42, n=n_nodes_icon, density=0.55,
           seed=99, masked_idx=masked_node)
ax.text(masked_cx, input_y - 0.62, r"$t$ (masked)",
        ha="center", va="top", fontsize=8.5, color=text_color, family=font)

left_panel_right = masked_cx + 0.7
ax.add_patch(Rectangle((left_panel_left, input_y - 0.50),
                       left_panel_right - left_panel_left, 1.0,
                       linewidth=0.8, edgecolor=border, facecolor="white", zorder=0))
ax.text((left_panel_left + left_panel_right) / 2, input_y - 1.05,
        r"context view $x$: K history snapshots $\{G_{\tau-K}, \dots, G_{\tau-1}\}$"
        r" + visible portion of $G_\tau$",
        ha="center", va="top", fontsize=9, color=text_color, family=font)

# right input panel: unmasked target
right_cx = 11.6
graph_icon(right_cx, input_y, r=0.42, n=n_nodes_icon, density=0.55, seed=99)
ax.text(right_cx, input_y - 0.62, r"$t$",
        ha="center", va="top", fontsize=8.5, color=text_color, family=font)
ax.add_patch(Rectangle((right_cx - 0.75, input_y - 0.50), 1.5, 1.0,
                       linewidth=0.8, edgecolor=border, facecolor="white", zorder=0))
ax.text(right_cx, input_y - 1.05,
        r"target $y$: unmasked $G_\tau$",
        ha="center", va="top", fontsize=9, color=text_color, family=font)

# row 2: encoders
enc_y = 3.6
enc_h = 0.85

online_x = 2.6
online_w = 2.2
box(online_x, enc_y, online_w, enc_h, r"online encoder $E_\theta$",
    fc=fill_focal, fs=10,
    sub="GATv2, 3 layers, 4 heads", sub_fs=7.5)

ema_x = 10.7
ema_w = 1.7
box(ema_x, enc_y, ema_w, enc_h, r"EMA encoder $E_{\bar\theta}$",
    fc=fill, fs=10,
    sub=r"EMA of $\theta$", sub_fs=7.5)

# arrows: inputs --> encoders
arrow((left_panel_left + left_panel_right) / 2, input_y + 0.55,
      online_x + online_w / 2, enc_y - 0.05)
arrow(right_cx, input_y + 0.55, ema_x + ema_w / 2, enc_y - 0.05)

# EMA update arrow (online -> EMA, dashed)
ema_arr = FancyArrowPatch((online_x + online_w, enc_y + enc_h / 2),
                          (ema_x, enc_y + enc_h / 2),
                          arrowstyle="->", mutation_scale=10,
                          linewidth=0.9, color=border,
                          linestyle=(0, (3, 2)),
                          connectionstyle="arc3,rad=-0.30", zorder=2)
ax.add_patch(ema_arr)
ax.text((online_x + online_w + ema_x) / 2, enc_y + enc_h + 0.55,
        "EMA update",
        ha="center", va="bottom", fontsize=8.5, color=text_color, family=font)

# row 3: predictor + mask token
pred_y = 5.5
pred_h = 0.85
pred_x = 2.6
pred_w = 2.2
box(pred_x, pred_y, pred_w, pred_h, r"predictor $P_\phi$",
    fc=fill_focal, fs=10,
    sub="bidirectional transformer", sub_fs=7.5)

arrow(online_x + online_w / 2, enc_y + enc_h, pred_x + pred_w / 2, pred_y - 0.05)

# mask token Δ_(v, τ) — small box feeding into predictor
mt_x = 5.3
mt_w = 1.3
mt_y = pred_y + pred_h / 2
ax.add_patch(Rectangle((mt_x, mt_y - 0.28), mt_w, 0.56,
                       linewidth=1.0, edgecolor=mask_edge, facecolor=masked_fill))
ax.text(mt_x + mt_w / 2, mt_y + 0.06, r"$\Delta_{(v,\tau)}$",
        ha="center", va="center", fontsize=10, color=mask_edge, family=font)
ax.text(mt_x + mt_w / 2, mt_y - 0.16, "mask token",
        ha="center", va="center", fontsize=7, color=mask_edge, family=font)
arrow(mt_x, mt_y, pred_x + pred_w + 0.02, mt_y, color=mask_edge)
ax.text(mt_x + mt_w / 2, mt_y - 0.46,
        r"posemb$_T(\tau)$ + nodeemb$(v)$",
        ha="center", va="top", fontsize=7, color=text_color, family=font)

# row 4: loss
loss_y = 7.1
loss_h = 0.65
loss_x = 5.4
loss_w = 3.2
box(loss_x, loss_y, loss_w, loss_h,
    r"$\| P_\phi(\Delta_{(v,\tau)}, E_\theta(x)) - \mathrm{sg}(E_{\bar\theta}(y)) \|_2^2$",
    fc=fill_loss, fs=9)

# predictor -> loss
arrow(pred_x + pred_w / 2, pred_y + pred_h,
      loss_x + loss_w * 0.30, loss_y - 0.05)
# EMA -> loss (with sg)
arrow(ema_x + ema_w / 2, enc_y + enc_h,
      loss_x + loss_w * 0.75, loss_y - 0.05)

# sg(·) label on EMA -> loss arrow
sg_mid_x = (ema_x + ema_w / 2 + loss_x + loss_w * 0.75) / 2
sg_mid_y = (enc_y + enc_h + loss_y) / 2
ax.text(sg_mid_x + 0.20, sg_mid_y, r"sg$(\cdot)$",
        ha="left", va="center", fontsize=9.5, color=text_color, family=font)

# title / caption
ax.text(fig_w / 2, fig_h - 0.30,
        "Graph-JEPA: predict the masked node-at-time embedding in latent space.",
        ha="center", va="top", fontsize=10.5, color=text_color, family=font)


plt.tight_layout()
out = "figures/graph_jepa.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
