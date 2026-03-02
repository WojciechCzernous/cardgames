"""
Policy and actor-critic networks for Sixty-Six.

PolicyNet:      248-dim state → 27 action logits (with valid-action masking).
ActorCriticNet: 248-dim state → 27 action logits + scalar value V(s).
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
        return logits.masked_fill(mask == 0, -1e9)

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


# ---------------------------------------------------------------------------
# Actor-Critic (shared trunk + policy head + value head)
# ---------------------------------------------------------------------------

class ActorCriticNet(nn.Module):
    """
    Shared-trunk actor-critic network.

    Trunk:  248 → hidden_dims[:-1]  (shared feature extraction)
    Policy: → hidden_dims[-1] → 27  (action logits, masked)
    Value:  → hidden_dims[-1] → 1   (state value)

    With default (256, 128): trunk = 248→256, policy = 256→128→27, value = 256→128→1.
    """

    def __init__(self, hidden_dims: tuple[int, ...] = (256, 128),
                 dropout: float = 0.0):
        super().__init__()
        assert len(hidden_dims) >= 1

        # Shared trunk: all layers except the last hidden
        trunk_layers: list[nn.Module] = []
        prev = FEATURE_DIM
        for h in hidden_dims[:-1]:
            trunk_layers.append(nn.Linear(prev, h))
            trunk_layers.append(nn.ReLU())
            if dropout > 0:
                trunk_layers.append(nn.Dropout(dropout))
            prev = h
        self.trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()

        last_h = hidden_dims[-1]

        # Policy head
        policy_layers: list[nn.Module] = [nn.Linear(prev, last_h), nn.ReLU()]
        if dropout > 0:
            policy_layers.append(nn.Dropout(dropout))
        policy_layers.append(nn.Linear(last_h, ACTION_DIM))
        self.policy_head = nn.Sequential(*policy_layers)

        # Value head
        value_layers: list[nn.Module] = [nn.Linear(prev, last_h), nn.ReLU()]
        if dropout > 0:
            value_layers.append(nn.Dropout(dropout))
        value_layers.append(nn.Linear(last_h, 1))
        self.value_head = nn.Sequential(*value_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (raw_logits (batch, 27), value (batch,))."""
        h = self.trunk(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    # ------------------------------------------------------------------
    # Masking helpers (same interface as PolicyNet)
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_mask(x: torch.Tensor) -> torch.Tensor:
        return x[..., VALID_ACTIONS_OFFSET:VALID_ACTIONS_OFFSET + ACTION_DIM]

    def masked_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        mask = self._valid_mask(x)
        return logits.masked_fill(mask == 0, -1e9)

    def masked_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.masked_logits(x), dim=-1)

    def masked_probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.masked_logits(x), dim=-1)

    def policy_and_value(self, x: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (masked_logits, value)."""
        logits, value = self.forward(x)
        mask = self._valid_mask(x)
        masked = logits.masked_fill(mask == 0, -1e9)
        return masked, value

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def greedy(self, x: torch.Tensor) -> torch.Tensor:
        return self.masked_logits(x).argmax(dim=-1)

    @torch.no_grad()
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.masked_probs(x)
        return torch.multinomial(probs, 1).squeeze(-1)

    # ------------------------------------------------------------------
    # Load SL policy weights into the policy head (for warm-starting)
    # ------------------------------------------------------------------

    def load_sl_policy(self, sl_path: str, strict: bool = False):
        """
        Load a PolicyNet checkpoint and transfer weights into this
        actor-critic's trunk + policy head. Value head is left random.
        """
        ckpt = torch.load(sl_path, weights_only=True)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # PolicyNet stores everything in net.0, net.1, ...
        # Map to trunk + policy_head
        new_sd: dict[str, torch.Tensor] = {}
        # Count how many weight layers (Linear) exist in the SL net
        weight_keys = sorted(k for k in sd if k.startswith("net.") and "weight" in k)
        # All except the last Linear → trunk; last Linear → policy_head's last Linear
        n_layers = len(weight_keys)

        # Identify trunk vs policy head boundary
        # trunk has (n_layers - 2) Linears if we have ≥2, else 0
        # With default (256, 128): net.0=Linear(248,256), net.2=Linear(256,128), net.4=Linear(128,27)
        # trunk = [net.0, net.1], policy_head = [net.2, net.3, net.4]
        # i.e. trunk holds hidden_dims[:-1] layers, policy_head holds last hidden + output

        # Simpler: just iterate and map by position
        trunk_param_names = list(self.trunk.state_dict().keys())
        policy_param_names = list(self.policy_head.state_dict().keys())

        sl_keys = sorted(sd.keys(), key=lambda k: int(k.split(".")[1]))

        idx = 0
        for name in trunk_param_names:
            new_sd[f"trunk.{name}"] = sd[sl_keys[idx]]
            idx += 1
        for name in policy_param_names:
            new_sd[f"policy_head.{name}"] = sd[sl_keys[idx]]
            idx += 1

        self.load_state_dict(new_sd, strict=False)
