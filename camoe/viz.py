"""Visualization helpers for the CAMoE demo.

Produces three figures:

1. ``accuracy_vs_channel.png`` — task accuracy as channel quality varies.
2. ``cost_vs_channel.png``     — communication cost as channel quality varies.
3. ``routing_map.png``         — which route the gate picks across the input
   plane, at a poor channel vs a good channel.

All figures use a transparent-friendly white background and a colour-blind-safe
palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; safe on servers and CI
import matplotlib.pyplot as plt
import numpy as np
import torch

from camoe.model import ChannelAwareMoE

# Colour-blind-safe pair.
C_CA = "#1d9e75"    # channel-aware (teal)
C_BASE = "#d85a30"  # baseline (coral)


def plot_accuracy(
    res_ca: dict,
    res_base: dict,
    out: str | Path,
) -> None:
    """Accuracy vs channel quality, channel-aware vs baseline."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res_ca["channel"], res_ca["accuracy"], "o-", color=C_CA, lw=2,
            label="Channel-aware gate")
    ax.plot(res_base["channel"], res_base["accuracy"], "s--", color=C_BASE, lw=2,
            label="Baseline gate (channel-blind)")
    ax.set_xlabel("Channel quality  (0 = poor, 1 = good)")
    ax.set_ylabel("Task accuracy")
    ax.set_title("Accuracy adapts to channel conditions")
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_cost(
    res_ca: dict,
    res_base: dict,
    out: str | Path,
) -> None:
    """Communication cost vs channel quality, channel-aware vs baseline."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res_ca["channel"], res_ca["comm_cost"], "o-", color=C_CA, lw=2,
            label="Channel-aware gate")
    ax.plot(res_base["channel"], res_base["comm_cost"], "s--", color=C_BASE, lw=2,
            label="Baseline gate (channel-blind)")
    ax.set_xlabel("Channel quality  (0 = poor, 1 = good)")
    ax.set_ylabel("Communication cost  (per query)")
    ax.set_title("Channel-aware routing avoids expensive transmissions")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


@torch.no_grad()
def plot_routing_map(
    model: ChannelAwareMoE,
    X: torch.Tensor,
    out: str | Path,
    grid: int = 220,
) -> None:
    """Show the chosen route across the input plane at poor vs good channel.

    Specialist routes are coloured; the local fallback is grey. The shift from
    mostly-grey (poor channel, falling back) to mostly-coloured (good channel,
    using specialists) is the visual signature of channel-aware routing.
    """
    xmin, ymin = X.min(0).values.tolist()
    xmax, ymax = X.max(0).values.tolist()
    pad = 0.5
    xs = torch.linspace(xmin - pad, xmax + pad, grid)
    ys = torch.linspace(ymin - pad, ymax + pad, grid)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    cmap = plt.get_cmap("tab10")

    for ax, c in zip(axes, [0.05, 0.95]):
        cb = torch.full((len(pts), 1), float(c))
        _, route_idx, _ = model.hard_route(pts, cb)
        route_idx = route_idx.reshape(grid, grid).numpy()

        # Colour specialists with distinct hues; local fallback (last idx) grey.
        n_routes = model.n_routes
        colours = np.zeros((*route_idx.shape, 3))
        for r in range(n_routes):
            mask = route_idx == r
            if r == model.local_idx:
                colours[mask] = (0.72, 0.72, 0.72)
            else:
                colours[mask] = cmap(r % 10)[:3]
        ax.imshow(
            colours,
            origin="lower",
            extent=(xs[0], xs[-1], ys[0], ys[-1]),
            aspect="auto",
        )
        # Overlay a sample of the data for context.
        sub = X[torch.randint(0, len(X), (400,))].numpy()
        ax.scatter(sub[:, 0], sub[:, 1], s=4, c="black", alpha=0.35)
        label = "poor" if c < 0.5 else "good"
        ax.set_title(f"Routing at {label} channel (c = {c:.2f})")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(
        0.5, 0.03,
        "Grey = cheap local fallback    ·    colour = remote specialist",
        ha="center", fontsize=11,
    )
    fig.savefig(out, dpi=130)
    plt.close(fig)
