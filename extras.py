"""Generate the supplementary figures: the mobility scenario and the routing
animation GIF (and, with --pareto, the optional accuracy/cost frontier).

These are slower than ``demo.py`` (they train several models / render many
frames). Run:

    python extras.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from camoe.analysis import plot_pareto, sweep_beta
from camoe.animate import animate_routing
from camoe.data import make_clusters, train_test_split
from camoe.mobility import plot_mobility, run_mobility
from camoe.model import ChannelAwareMoE
from camoe.train import TrainConfig, train


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate supplementary figures.")
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-gif", action="store_true", help="skip the (slower) GIF")
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="also generate the supplementary beta-sweep frontier (not featured in the README)",
    )
    args = parser.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)

    # Train the reference channel-aware + baseline pair once, reuse for mobility/gif.
    X, y = make_clusters(seed=args.seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=args.seed)
    cfg = TrainConfig(steps=2000, beta=5.0, seed=args.seed)
    torch.manual_seed(args.seed)
    ca = ChannelAwareMoE(channel_aware=True)
    train(ca, Xtr, ytr, cfg)
    torch.manual_seed(args.seed)
    base = ChannelAwareMoE(channel_aware=False)
    train(base, Xtr, ytr, cfg)

    if args.pareto:
        print("Pareto sweep across beta (supplementary)...")
        res = sweep_beta(betas=[0.0, 1.0, 2.0, 4.0, 6.0, 10.0], steps=1600, seed=args.seed)
        plot_pareto(res, args.assets / "pareto.png")

    print("Mobility scenario (time-varying channel)...")
    rows = run_mobility(ca, base, Xte, yte, n_steps=120, seed=args.seed)
    plot_mobility(rows, args.assets / "mobility.png")
    corr = np.corrcoef(rows["channel"], rows["ca_remote"])[0, 1]
    print(f"  channel-aware accuracy: {rows['ca_acc'].mean():.3f}  "
          f"(baseline {rows['base_acc'].mean():.3f})")
    print(f"  corr(channel, remote routing): channel-aware = {corr:.3f}, "
          f"baseline = 0 (constant by construction)")

    if not args.no_gif:
        print("Rendering routing animation GIF...")
        animate_routing(ca, X, args.assets / "routing_animation.gif", n_frames=40, fps=12)

    print("Done. Figures in", args.assets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
