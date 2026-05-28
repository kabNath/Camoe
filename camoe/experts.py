"""Expert and gating network building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class Expert(nn.Module):
    """A small nonlinear MLP expert mapping 2D input to class logits.

    With ReLU hidden layers it can learn the local XOR structure of its
    assigned cluster.
    """

    def __init__(self, in_dim: int = 2, hidden: int = 32, n_classes: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalFallback(nn.Module):
    """A cheap, shared generalist head (small MLP).

    Routing here costs nothing (no transmission). It is deliberately smaller
    than the specialists, so it handles the task decently but not as well —
    creating the accuracy/cost tension the demo illustrates. Falling back
    sacrifices some accuracy to save communication cost.
    """

    def __init__(self, in_dim: int = 2, hidden: int = 6, n_classes: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Gate(nn.Module):
    """Router producing a distribution over experts (+ local fallback).

    If ``channel_aware`` is True, the gate additionally consumes a scalar
    channel-quality feature, letting it modulate routing by radio conditions.
    """

    def __init__(
        self,
        in_dim: int = 2,
        n_routes: int = 5,
        hidden: int = 32,
        channel_aware: bool = True,
    ) -> None:
        super().__init__()
        self.channel_aware = channel_aware
        gate_in = in_dim + (1 if channel_aware else 0)
        self.net = nn.Sequential(
            nn.Linear(gate_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_routes),
        )

    def forward(self, x: torch.Tensor, channel: torch.Tensor) -> torch.Tensor:
        """Return routing logits of shape ``(batch, n_routes)``.

        ``channel`` has shape ``(batch, 1)`` and is ignored when the gate is
        not channel-aware (kept in the signature so the two gates are
        drop-in interchangeable).
        """
        inp = torch.cat([x, channel], dim=-1) if self.channel_aware else x
        return self.net(inp)
