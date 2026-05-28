"""Synthetic multi-regime classification task with intra-cluster XOR structure.

We generate 2D points in ``n_clusters`` Gaussian clusters. Within each cluster
the points carry a *sub-class* determined by an XOR of their local coordinates,
so each cluster contains two interleaved classes that are NOT linearly
separable. Total classes = ``n_clusters * 2``.

Why XOR? It guarantees a genuine accuracy gap:

- A linear local fallback can identify the cluster but cannot separate the
  interleaved sub-classes, so it tops out near 50% within each cluster.
- A nonlinear specialist expert dedicated to one cluster learns the local XOR
  and reaches high accuracy.

This sets up the trade-off CAMoE demonstrates: pay to reach a remote specialist
(high accuracy) or fall back to the cheap local head (lower accuracy, no cost).
"""

from __future__ import annotations

import numpy as np
import torch


def make_clusters(
    n_per_cluster: int = 800,
    n_clusters: int = 4,
    spread: float = 0.55,
    radius: float = 3.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(X, y)`` for a 2D XOR-in-cluster classification task.

    Labels run from ``0`` to ``2 * n_clusters - 1`` (two sub-classes per cluster).
    """
    rng = np.random.default_rng(seed)
    centres = np.stack(
        [
            radius * np.array([np.cos(2 * np.pi * k / n_clusters),
                               np.sin(2 * np.pi * k / n_clusters)])
            for k in range(n_clusters)
        ]
    )
    xs, ys = [], []
    for k in range(n_clusters):
        local = rng.normal(scale=spread, size=(n_per_cluster, 2))
        pts = local + centres[k]
        # XOR sub-class from the sign of the local coordinates.
        sub = ((local[:, 0] > 0) ^ (local[:, 1] > 0)).astype(np.int64)
        xs.append(pts)
        ys.append(k * 2 + sub)
    X = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)

    perm = rng.permutation(len(X))
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])


def sample_channel(n: int, seed: int | None = None) -> torch.Tensor:
    """Sample channel-quality values in ``[0, 1]`` (1 = great, 0 = poor)."""
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return torch.rand(n, 1, generator=g)


def train_test_split(
    X: torch.Tensor,
    y: torch.Tensor,
    test_frac: float = 0.2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split tensors into train / test partitions."""
    n = len(X)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = int(n * test_frac)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
