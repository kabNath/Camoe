"""CAMoE — Channel-Aware Mixture-of-Experts.

A minimal, runnable proof-of-concept: a Mixture-of-Experts router that takes
wireless channel quality into account when deciding whether to route to a
remote specialist expert or a cheap local fallback.
"""

from camoe.model import ChannelAwareMoE
from camoe.train import TrainConfig, evaluate_by_channel, train

__version__ = "0.1.0"
__all__ = ["ChannelAwareMoE", "TrainConfig", "train", "evaluate_by_channel"]
