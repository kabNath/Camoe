"""Channel-aware Mixture-of-Experts model.

Routing options are: ``n_clusters`` remote specialist experts plus one local
fallback (the last route index).

Routing
-------
We use **straight-through top-1 routing**: the forward pass uses the hard
argmax route (so each expert specialises on the points actually sent to it),
while gradients flow through the soft softmax weights. This gives clean expert
specialisation while keeping the gate trainable.

Communication cost model
-------------------------
Reaching a remote specialist over the wireless link costs

    cost = (1 - channel) * TX_COST

so transmission is expensive when the channel is poor (``channel`` near 0) and
free when it is good (``channel`` near 1). The local fallback costs nothing.
Training minimises ``task_loss + beta * comm_cost + lb * load_balance``, so a
channel-aware gate learns to fall back locally exactly when the channel is poor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from camoe.experts import Expert, Gate, LocalFallback

TX_COST = 1.0  # nominal transmission cost for reaching a remote specialist


class ChannelAwareMoE(nn.Module):
    """Mixture of specialist experts + a local fallback, with a (channel-aware) gate."""

    def __init__(
        self,
        n_clusters: int = 4,
        n_classes: int = 8,
        in_dim: int = 2,
        channel_aware: bool = True,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.n_routes = n_clusters + 1
        self.local_idx = self.n_routes - 1

        self.experts = nn.ModuleList(
            [Expert(in_dim=in_dim, n_classes=n_classes) for _ in range(n_clusters)]
        )
        self.local = LocalFallback(in_dim=in_dim, n_classes=n_classes)
        self.gate = Gate(in_dim=in_dim, n_routes=self.n_routes, channel_aware=channel_aware)

        route_is_remote = torch.ones(self.n_routes)
        route_is_remote[self.local_idx] = 0.0
        self.register_buffer("route_is_remote", route_is_remote)

    def route_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Stack every route's class logits: shape (batch, n_routes, n_classes)."""
        specialists = torch.stack([e(x) for e in self.experts], dim=1)
        local = self.local(x).unsqueeze(1)
        return torch.cat([specialists, local], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        channel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Straight-through top-1 routed forward pass.

        Returns ``(logits, soft_weights, comm_cost)``.
        """
        gate_logits = self.gate(x, channel)
        soft = F.softmax(gate_logits, dim=-1)
        hard = F.one_hot(gate_logits.argmax(dim=-1), self.n_routes).float()
        routing = hard + soft - soft.detach()  # straight-through

        route_logits = self.route_outputs(x)
        logits = torch.einsum("br,brc->bc", routing, route_logits)

        remote_weight = (routing * self.route_is_remote).sum(dim=-1)
        comm_cost = remote_weight * (1.0 - channel.squeeze(-1)) * TX_COST
        return logits, soft, comm_cost

    @torch.no_grad()
    def hard_route(
        self,
        x: torch.Tensor,
        channel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-1 routing for evaluation. Returns ``(logits, route_idx, comm_cost)``."""
        gate_logits = self.gate(x, channel)
        route_idx = gate_logits.argmax(dim=-1)
        route_logits = self.route_outputs(x)
        logits = route_logits[torch.arange(len(x)), route_idx]
        is_remote = self.route_is_remote[route_idx]
        comm_cost = is_remote * (1.0 - channel.squeeze(-1)) * TX_COST
        return logits, route_idx, comm_cost
