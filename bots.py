import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RandomBot:
    def act(self, obs: np.ndarray) -> int:
        hand = [c for c in obs[:10] if c > 0]
        idxs = [i for i, c in enumerate(obs[:10]) if c > 0]
        return random.choice(idxs)


class RuleBot:
    def act(self, obs: np.ndarray) -> int:
        hand = [c for c in obs[:10] if c > 0]
        row_tops = obs[10:14]
        best_idx = None
        best_diff = 105
        for i, card in enumerate(hand):
            diffs = [card - top for top in row_tops if card > top]
            if diffs:
                d = min(diffs)
                if d < best_diff:
                    best_diff = d
                    best_idx = i
        if best_idx is None:
            # no card fits, play smallest
            return int(np.argmin(obs[:10]))
        # need to convert best_idx from hand order to position in obs
        card_value = hand[best_idx]
        for i, c in enumerate(obs[:10]):
            if c == card_value:
                return i
        return 0


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256, n_actions: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RLAgent:
    """Simple actor-critic agent."""

    def __init__(self, obs_dim: int, lr: float = 3e-4):
        self.policy = PolicyNet(obs_dim)
        self.value = ValueNet(obs_dim)
        params = list(self.policy.parameters()) + list(self.value.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

    def act(self, obs: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select an action and return logging info.

        Returns the sampled action index, log-probability, entropy and value
        estimate so the training loop can apply entropy regularisation and
        advantage-based updates.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32)
        logits = self.policy(obs_t)
        mask = torch.tensor([1.0 if c > 0 else 0.0 for c in obs[:10]], dtype=torch.float32)
        masked = logits - (1 - mask) * 1e9
        probs = torch.softmax(masked, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        ent = dist.entropy()
        value = self.value(obs_t)
        return int(action.item()), logp, ent, value

    def update(
        self,
        log_probs: List[torch.Tensor],
        values: List[torch.Tensor],
        rewards: List[float],
        entropies: List[torch.Tensor],
        gamma: float = 0.99,
    ) -> None:
        """Update policy and value networks from episode logs."""
        returns: List[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)
        advantages = returns_t - values_t.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_loss = -(log_probs_t * advantages).mean()
        value_loss = advantages.pow(2).mean()
        entropy_loss = entropies_t.mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Utility methods for persistence
    def save(self, path: str) -> None:
        torch.save({
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(state["policy"])
        self.value.load_state_dict(state["value"])
        self.policy.eval()
        self.value.eval()
