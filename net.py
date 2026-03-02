"""
Policy network for Sixty-Six.

MLP that maps a 248-dim state tensor to 27 action logits,
with built-in valid-action masking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from features import FEATURE_DIM, ACTION_DIM, VALID_ACTIONS_OFFSET


class PolicyNet(nn.Module):
    """
    MLP policy network.

    forward(x)          → raw logits          (batch, 27)
    masked_logits(x)    → masked logits       (batch, 27)  invalid → -inf
    masked_log_probs(x) → masked log-probs    (batch, 27)
    greedy(x)           → argmax action        (batch,)
    sample(x)           → sampled action       (batch,)
    """

    def __init__(self, hidden_dims: tuple[int, ...] = (256, 128),
                 dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = FEATURE_DIM
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, ACTION_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits, shape (batch, 27)."""
        return self.net(x)

    # ------------------------------------------------------------------
    # Masked variants
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_mask(x: torch.Tensor) -> torch.Tensor:
        """Extract the 27-bit valid-actions mask from the state tensor."""
        return x[..., VALID_ACTIONS_OFFSET:VALID_ACTIONS_OFFSET + ACTION_DIM]

    def masked_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Logits with invalid actions set to -inf."""
        logits = self.forward(x)
        mask = self._valid_mask(x)
        return logits.masked_fill(mask == 0, float('-inf'))

    def masked_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Log-probabilities over valid actions only."""
        return F.log_softmax(self.masked_logits(x), dim=-1)

    def masked_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilities over valid actions (invalid → 0)."""
        return F.softmax(self.masked_logits(x), dim=-1)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def greedy(self, x: torch.Tensor) -> torch.Tensor:
        """Argmax action per batch element. Shape: (batch,)."""
        return self.masked_logits(x).argmax(dim=-1)

    @torch.no_grad()
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Sample one action per batch element. Shape: (batch,)."""
        probs = self.masked_probs(x)
        return torch.multinomial(probs, 1).squeeze(-1)
