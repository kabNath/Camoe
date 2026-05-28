"""Mobility scenario: a device moving through varying radio coverage.

Where the per-channel plots show steady-state behaviour, this scenario shows the
gate reacting to a channel that changes over time, as a device moves along a
trajectory (think a UAV or a phone passing in and out of good coverage).

At each time step the whole test set is served under the current channel value.
We track, for the channel-aware gate and the channel-blind baseline:

- accuracy over time,
- communication cost over time,
- the fraction of queries routed to a remote specialist.

The channel-aware gate tracks the channel — spending on specialists when
coverage is good and falling back when it is poor — while the baseline, blind to
the channel, behaves the same throughout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from camoe.model import ChannelAwareMoE
from camoe.viz import C_BASE, C_CA


def coverage_profile(n_steps: int = 120, seed: int = 0) -> np.ndarray:
    """A smooth-ish channel-quality trajectory in [0, 1] with two coverage dips."""
    t = np.linspace(0, 1, n_steps)
    base = 0.5 + 0.42 * np.sin(2 * np.pi * (t - 0.1)) * np.cos(np.pi * t)
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=0.03, size=n_steps)
    return np.clip(base + noise, 0.0, 1.0)


@torch.no_grad()
def _serve(model: ChannelAwareMoE, X, y, c: float):
    cb = torch.full((len(X), 1), float(c))
    logits, route_idx, comm_cost = model.hard_route(X, cb)
    acc = (logits.argmax(-1) == y).float().mean().item()
    remote_frac = (model.route_is_remote[route_idx]).mean().item()
    return acc, comm_cost.mean().item(), remote_frac


def run_mobility(
    ca: ChannelAwareMoE,
    base: ChannelAwareMoE,
    X: torch.Tensor,
    y: torch.Tensor,
    n_steps: int = 120,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Serve the test set under a time-varying channel; collect per-step metrics."""
    channel = coverage_profile(n_steps, seed)
    rows = {k: np.zeros(n_steps) for k in
            ["channel", "ca_acc", "ca_cost", "ca_remote", "base_acc", "base_cost", "base_remote"]}
    rows["channel"] = channel
    for i, c in enumerate(channel):
        a, co, rf = _serve(ca, X, y, c)
        rows["ca_acc"][i], rows["ca_cost"][i], rows["ca_remote"][i] = a, co, rf
        a, co, rf = _serve(base, X, y, c)
        rows["base_acc"][i], rows["base_cost"][i], rows["base_remote"][i] = a, co, rf
    return rows


def plot_mobility(rows: dict[str, np.ndarray], out: str | Path) -> None:
    """Three stacked time-series panels: channel, accuracy, cost."""
    t = np.arange(len(rows["channel"]))
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.2), sharex=True)

    axes[0].fill_between(t, rows["channel"], color="#378add", alpha=0.25)
    axes[0].plot(t, rows["channel"], color="#185fa5", lw=1.8)
    axes[0].set_ylabel("Channel\nquality")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("A device moving through varying coverage")

    axes[1].plot(t, rows["ca_acc"], color=C_CA, lw=2, label="Channel-aware")
    axes[1].plot(t, rows["base_acc"], color=C_BASE, lw=2, ls="--", label="Baseline")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend(frameon=False, loc="lower right")

    axes[2].plot(t, rows["ca_cost"], color=C_CA, lw=2, label="Channel-aware")
    axes[2].plot(t, rows["base_cost"], color=C_BASE, lw=2, ls="--", label="Baseline")
    axes[2].set_ylabel("Comm.\ncost")
    axes[2].set_xlabel("Time step (device moving along its trajectory)")
    axes[2].legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
