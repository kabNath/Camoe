"""Training and evaluation routines."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from camoe.data import sample_channel
from camoe.model import ChannelAwareMoE


@dataclass
class TrainConfig:
    steps: int = 1500
    lr: float = 5e-3
    batch_size: int = 256
    beta: float = 5.0           # target weight on communication cost
    warmup_frac: float = 0.4    # fraction of steps at beta=0 (learn task first)
    load_balance: float = 0.05  # weight on the expert load-balancing term
    seed: int = 0


def _load_balance_loss(soft: torch.Tensor, n_specialists: int) -> torch.Tensor:
    """Encourage uniform usage of the specialist experts (standard MoE aux loss)."""
    usage = soft[:, :n_specialists].mean(dim=0)
    return (usage * usage).sum() * n_specialists


def train(
    model: ChannelAwareMoE,
    X: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainConfig,
) -> list[dict[str, float]]:
    """Train to minimise task loss + beta * comm cost + load balancing.

    Channel quality is resampled each step so the model sees the full range of
    radio conditions. Returns a per-step history of scalar metrics.
    """
    torch.manual_seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = len(X)
    history: list[dict[str, float]] = []

    for step in range(cfg.steps):
        # Beta warmup: spend the first warmup_frac of steps at beta=0 so the
        # specialists and the gate learn the task before the cost pressure kicks in,
        # then linearly ramp beta to its target. Without this the gate collapses to
        # always-local and the specialists never receive gradient.
        warmup_steps = int(cfg.warmup_frac * cfg.steps)
        if step < warmup_steps:
            beta = 0.0
        else:
            ramp = (step - warmup_steps) / max(1, cfg.steps - warmup_steps)
            beta = cfg.beta * ramp

        idx = torch.randint(0, n, (cfg.batch_size,))
        xb, yb = X[idx], y[idx]
        cb = sample_channel(cfg.batch_size)

        logits, soft, comm_cost = model(xb, cb)
        task_loss = F.cross_entropy(logits, yb)
        cost_loss = comm_cost.mean()
        lb = _load_balance_loss(soft, model.n_clusters)
        loss = task_loss + beta * cost_loss + cfg.load_balance * lb

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == cfg.steps - 1:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "task_loss": float(task_loss.detach()),
                    "comm_cost": float(cost_loss.detach()),
                }
            )
    return history


@torch.no_grad()
def evaluate_by_channel(
    model: ChannelAwareMoE,
    X: torch.Tensor,
    y: torch.Tensor,
    channel_levels: torch.Tensor,
) -> dict[str, list[float]]:
    """Evaluate hard-routing accuracy and comm cost at each channel level."""
    accs, costs = [], []
    for c in channel_levels:
        cb = torch.full((len(X), 1), float(c))
        logits, _, comm_cost = model.hard_route(X, cb)
        accs.append((logits.argmax(dim=-1) == y).float().mean().item())
        costs.append(comm_cost.mean().item())
    return {
        "channel": [float(c) for c in channel_levels],
        "accuracy": accs,
        "comm_cost": costs,
    }
