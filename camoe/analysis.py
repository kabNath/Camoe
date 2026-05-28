"""Pareto-frontier analysis: how the cost weight beta trades accuracy for cost.

This is supplementary and not featured in the README: averaged over a uniform
channel the two frontiers cross, because uniform averaging discards exactly the
channel signal the gate exploits (the conditional advantage shows up in the
mobility scenario instead). Kept here for completeness; run via
``python extras.py --pareto``.

Training minimises ``task_loss + beta * comm_cost``. Sweeping ``beta`` traces an
operating curve in the (communication cost, accuracy) plane. This module trains
a channel-aware model and a channel-blind baseline at several ``beta`` values
and plots both frontiers, so you can see the channel-aware curve dominate
(higher accuracy at equal cost, lower cost at equal accuracy).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from camoe.data import make_clusters, train_test_split
from camoe.model import ChannelAwareMoE
from camoe.train import TrainConfig, evaluate_by_channel, train
from camoe.viz import C_BASE, C_CA


def sweep_beta(
    betas: list[float],
    steps: int = 1600,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Train CA and baseline at each beta; return mean accuracy and cost for both.

    Returns a dict with keys ``beta``, ``ca_acc``, ``ca_cost``, ``base_acc``,
    ``base_cost`` (each a list aligned with ``betas``).
    """
    X, y = make_clusters(seed=seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=seed)
    levels = torch.linspace(0, 1, 6)

    out: dict[str, list[float]] = {
        "beta": [], "ca_acc": [], "ca_cost": [], "base_acc": [], "base_cost": []
    }
    for beta in betas:
        cfg = TrainConfig(steps=steps, beta=beta, seed=seed)

        torch.manual_seed(seed)
        ca = ChannelAwareMoE(channel_aware=True)
        train(ca, Xtr, ytr, cfg)
        rca = evaluate_by_channel(ca, Xte, yte, levels)

        torch.manual_seed(seed)
        base = ChannelAwareMoE(channel_aware=False)
        train(base, Xtr, ytr, cfg)
        rb = evaluate_by_channel(base, Xte, yte, levels)

        out["beta"].append(beta)
        out["ca_acc"].append(float(np.mean(rca["accuracy"])))
        out["ca_cost"].append(float(np.mean(rca["comm_cost"])))
        out["base_acc"].append(float(np.mean(rb["accuracy"])))
        out["base_cost"].append(float(np.mean(rb["comm_cost"])))
    return out


def plot_pareto(res: dict[str, list[float]], out: str | Path) -> None:
    """Plot accuracy vs communication cost for both gates across the beta sweep."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    # Sort each curve by cost for a clean line.
    for cost_k, acc_k, colour, marker, label in [
        ("ca_cost", "ca_acc", C_CA, "o", "Channel-aware gate"),
        ("base_cost", "base_acc", C_BASE, "s", "Baseline gate (channel-blind)"),
    ]:
        pts = sorted(zip(res[cost_k], res[acc_k]))
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marker, color=colour, lw=2, label=label)

    ax.set_xlabel("Mean communication cost  (lower is better)")
    ax.set_ylabel("Mean task accuracy  (higher is better)")
    ax.set_title("Accuracy / cost frontier across the cost weight beta")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    # Annotate the direction of "better".
    ax.annotate(
        "better", xy=(0.04, 0.95), xytext=(0.22, 0.86),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", color="#444441",
        arrowprops=dict(arrowstyle="->", color="#444441"),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
