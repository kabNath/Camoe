"""Animated routing map: watch the gate's decisions morph as the channel changes.

Sweeps channel quality from poor to good and back, rendering the routing map at
each step and stitching the frames into a GIF. As the channel improves, the
coloured specialist regions grow (the gate pays for remote experts); as it
degrades, the grey local-fallback region takes over.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from camoe.model import ChannelAwareMoE


@torch.no_grad()
def animate_routing(
    model: ChannelAwareMoE,
    X: torch.Tensor,
    out: str | Path,
    grid: int = 160,
    n_frames: int = 40,
    fps: int = 12,
) -> None:
    """Render a GIF of the routing map as channel quality sweeps 0 -> 1 -> 0."""
    xmin, ymin = X.min(0).values.tolist()
    xmax, ymax = X.max(0).values.tolist()
    pad = 0.5
    xs = torch.linspace(xmin - pad, xmax + pad, grid)
    ys = torch.linspace(ymin - pad, ymax + pad, grid)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    sample = X[torch.randint(0, len(X), (350,))].numpy()
    cmap = plt.get_cmap("tab10")

    # Channel sweep: 0 -> 1 -> 0 for a smooth there-and-back loop.
    half = np.linspace(0.0, 1.0, n_frames // 2)
    channels = np.concatenate([half, half[::-1]])

    def routing_rgb(c: float) -> np.ndarray:
        cb = torch.full((len(pts), 1), float(c))
        _, route_idx, _ = model.hard_route(pts, cb)
        route_idx = route_idx.reshape(grid, grid).numpy()
        rgb = np.zeros((*route_idx.shape, 3))
        for r in range(model.n_routes):
            mask = route_idx == r
            rgb[mask] = (0.72, 0.72, 0.72) if r == model.local_idx else cmap(r % 10)[:3]
        return rgb

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(
        routing_rgb(channels[0]), origin="lower",
        extent=(xs[0], xs[-1], ys[0], ys[-1]), aspect="auto",
    )
    ax.scatter(sample[:, 0], sample[:, 1], s=4, c="black", alpha=0.3)
    title = ax.set_title("")

    def update(frame_idx: int):
        c = float(channels[frame_idx])
        im.set_data(routing_rgb(c))
        label = "poor" if c < 0.5 else "good"
        title.set_text(f"Channel quality c = {c:.2f}  ({label})")
        return (im, title)

    anim = FuncAnimation(fig, update, frames=len(channels), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
