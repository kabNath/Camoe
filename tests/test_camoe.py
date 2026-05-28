"""Tests for CAMoE: shapes, cost model, and the core channel-aware claim."""

from __future__ import annotations

import torch

from camoe.data import make_clusters, sample_channel, train_test_split
from camoe.model import ChannelAwareMoE
from camoe.train import TrainConfig, evaluate_by_channel, train


def test_data_shapes_and_labels() -> None:
    X, y = make_clusters(n_per_cluster=100, n_clusters=4, seed=0)
    assert X.shape == (400, 2)
    assert y.shape == (400,)
    assert int(y.min()) == 0
    assert int(y.max()) == 7  # 4 clusters x 2 sub-classes


def test_channel_sampling_range() -> None:
    c = sample_channel(1000, seed=0)
    assert c.shape == (1000, 1)
    assert float(c.min()) >= 0.0
    assert float(c.max()) <= 1.0


def test_forward_shapes() -> None:
    model = ChannelAwareMoE(channel_aware=True)
    x = torch.randn(16, 2)
    c = torch.rand(16, 1)
    logits, soft, cost = model(x, c)
    assert logits.shape == (16, 8)
    assert soft.shape == (16, model.n_routes)
    assert cost.shape == (16,)


def test_cost_is_zero_on_perfect_channel() -> None:
    """With channel == 1, transmission is free regardless of routing."""
    model = ChannelAwareMoE(channel_aware=True)
    x = torch.randn(32, 2)
    c = torch.ones(32, 1)
    _, _, cost = model(x, c)
    assert torch.allclose(cost, torch.zeros_like(cost))


def test_local_fallback_costs_nothing() -> None:
    """Routing to the local fallback never incurs communication cost."""
    model = ChannelAwareMoE(channel_aware=True)
    assert float(model.route_is_remote[model.local_idx]) == 0.0
    assert float(model.route_is_remote[:model.local_idx].min()) == 1.0


def test_channel_aware_adapts_but_baseline_does_not() -> None:
    """The core claim, tested via the robust qualitative signature.

    A channel-aware gate adapts its routing to the channel: its accuracy at a
    good channel is meaningfully higher than at a poor channel (it pays for
    specialists when transmission is cheap, falls back when it is expensive).

    A channel-blind baseline cannot adapt: because its gate never sees the
    channel, its routing — and therefore its accuracy — is identical at every
    channel level. This difference is the whole point of the project.
    """
    X, y = make_clusters(seed=0)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=0)
    cfg = TrainConfig(steps=1500, beta=5.0, seed=0)

    torch.manual_seed(0)
    ca = ChannelAwareMoE(channel_aware=True)
    train(ca, Xtr, ytr, cfg)

    torch.manual_seed(0)
    base = ChannelAwareMoE(channel_aware=False)
    train(base, Xtr, ytr, cfg)

    ends = torch.tensor([0.0, 1.0])
    rca = evaluate_by_channel(ca, Xte, yte, ends)
    rb = evaluate_by_channel(base, Xte, yte, ends)

    ca_adaptation = rca["accuracy"][1] - rca["accuracy"][0]
    base_adaptation = rb["accuracy"][1] - rb["accuracy"][0]

    # Channel-aware accuracy climbs by a clear margin as the channel improves.
    assert ca_adaptation > 0.1
    # The channel-blind baseline is, by construction, flat across the channel.
    assert abs(base_adaptation) < 1e-6
