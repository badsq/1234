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
    def __init__(self, obs_dim: int, hidden: int = 512, n_actions: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 512):
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

    def __init__(self, obs_dim: int, lr: float = 3e-4, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = PolicyNet(obs_dim).to(self.device)
        self.value = ValueNet(obs_dim).to(self.device)
        params = list(self.policy.parameters()) + list(self.value.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

    def act(self, obs: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select an action for a single observation."""
        acts, logp, ent, val = self.act_batch(obs[None, :])
        return int(acts[0]), logp[0], ent[0], val[0]

    def act_batch(
        self, obs_batch: np.ndarray
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorised action selection for a batch of observations."""
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        logits = self.policy(obs_t)
        mask = obs_t[:, :10].gt(0).float()
        masked = logits - (1 - mask) * 1e9
        dist = torch.distributions.Categorical(logits=masked)
        actions = dist.sample()
        logp = dist.log_prob(actions)
        ent = dist.entropy()
        values = self.value(obs_t)
        return actions.cpu().numpy(), logp, ent, values

    def update(
        self,
        log_probs: List[torch.Tensor],
        values: List[torch.Tensor],
        rewards: List[float],
        entropies: List[torch.Tensor],
        gamma: float = 0.97,
    ) -> None:
        """Update policy and value networks from episode logs."""
        returns: List[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values_t = torch.stack(values).to(self.device)
        log_probs_t = torch.stack(log_probs).to(self.device)
        entropies_t = torch.stack(entropies).to(self.device)
        advantages = returns_t - values_t
        policy_adv = advantages.detach()
        policy_adv = (policy_adv - policy_adv.mean()) / (policy_adv.std() + 1e-8)
        policy_loss = -(log_probs_t * policy_adv).mean()
        value_loss = (returns_t - values_t).pow(2).mean()
        entropy_loss = entropies_t.mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_batch(
        self,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        entropies: torch.Tensor,
        gamma: float = 0.97,
    ) -> None:
        """Update from batched trajectories.

        All tensors are expected to have shape ``(batch, steps)`` where batch
        is the number of parallel episodes."""
        returns = torch.zeros_like(rewards)
        G = torch.zeros(rewards.size(0), device=self.device)
        for t in reversed(range(rewards.size(1))):
            G = rewards[:, t] + gamma * G
            returns[:, t] = G
        log_probs_t = log_probs.reshape(-1)
        values_t = values.reshape(-1)
        returns_t = returns.reshape(-1)
        entropies_t = entropies.reshape(-1)
        advantages = returns_t - values_t
        policy_adv = advantages.detach()
        policy_adv = (policy_adv - policy_adv.mean()) / (policy_adv.std() + 1e-8)
        policy_loss = -(log_probs_t * policy_adv).mean()
        value_loss = (returns_t - values_t).pow(2).mean()
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
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state["policy"])
        self.value.load_state_dict(state["value"])
        self.policy.to(self.device).eval()
        self.value.to(self.device).eval()
