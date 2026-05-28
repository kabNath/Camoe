"""End-to-end CAMoE demo.

Trains a channel-aware MoE and a channel-blind baseline on the synthetic
multi-regime task, evaluates both across the full range of channel conditions,
prints a summary table, and writes three figures to ``assets/``.

Run:

    python demo.py

Takes a few seconds on CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from camoe.data import make_clusters, train_test_split
from camoe.model import ChannelAwareMoE
from camoe.train import TrainConfig, evaluate_by_channel, train
from camoe.viz import plot_accuracy, plot_cost, plot_routing_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CAMoE demo.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    args = parser.parse_args()

    args.assets.mkdir(parents=True, exist_ok=True)

    X, y = make_clusters(seed=args.seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=args.seed)
    cfg = TrainConfig(steps=args.steps, beta=args.beta, seed=args.seed)

    print("Training channel-aware MoE...")
    torch.manual_seed(args.seed)
    ca = ChannelAwareMoE(channel_aware=True)
    train(ca, Xtr, ytr, cfg)

    print("Training channel-blind baseline...")
    torch.manual_seed(args.seed)
    base = ChannelAwareMoE(channel_aware=False)
    train(base, Xtr, ytr, cfg)

    levels = torch.linspace(0, 1, 6)
    rca = evaluate_by_channel(ca, Xte, yte, levels)
    rb = evaluate_by_channel(base, Xte, yte, levels)

    print("\nchannel | channel-aware acc/cost | baseline acc/cost")
    print("-" * 56)
    for i, c in enumerate(levels):
        print(
            f"  {float(c):.1f}   |      {rca['accuracy'][i]:.2f} / {rca['comm_cost'][i]:.2f}      "
            f"|    {rb['accuracy'][i]:.2f} / {rb['comm_cost'][i]:.2f}"
        )

    ca_acc, ca_cost = np.mean(rca["accuracy"]), np.mean(rca["comm_cost"])
    b_acc, b_cost = np.mean(rb["accuracy"]), np.mean(rb["comm_cost"])
    cost_red = 100 * (1 - ca_cost / b_cost) if b_cost > 0 else 0.0

    print("-" * 56)
    print(f"  mean  |      {ca_acc:.2f} / {ca_cost:.2f}      |    {b_acc:.2f} / {b_cost:.2f}")
    print()
    print(f"Channel-aware vs baseline (averaged over channel conditions):")
    print(f"  accuracy:            {b_acc:.3f} -> {ca_acc:.3f}  ({ca_acc - b_acc:+.3f})")
    print(f"  communication cost:  {b_cost:.3f} -> {ca_cost:.3f}  ({cost_red:+.0f}%)")

    print("\nWriting figures to", args.assets)
    plot_accuracy(rca, rb, args.assets / "accuracy_vs_channel.png")
    plot_cost(rca, rb, args.assets / "cost_vs_channel.png")
    plot_routing_map(ca, X, args.assets / "routing_map.png")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
