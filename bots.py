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
        """Select an action and return logging info.

        Returns the sampled action index, log-probability, entropy and value
        estimate so the training loop can apply entropy regularisation and
        advantage-based updates.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        logits = self.policy(obs_t)
        mask = torch.tensor([1.0 if c > 0 else 0.0 for c in obs[:10]], dtype=torch.float32, device=self.device)
        masked = logits - (1 - mask) * 1e9
        probs = torch.softmax(masked, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        ent = dist.entropy()
        value = self.value(obs_t)
        return int(action.item()), logp, ent, value

    def act_batch(self, obs_batch: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorised action selection for a batch of observations."""
        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        logits = self.policy(obs_t)
        masks = torch.tensor([[1.0 if c > 0 else 0.0 for c in obs[:10]] for obs in obs_batch], dtype=torch.float32, device=self.device)
        masked = logits - (1 - masks) * 1e9
        probs = torch.softmax(masked, dim=-1)
        dist = torch.distributions.Categorical(probs)
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
        gamma: float = 0.99,
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
        batch_log_probs: List[List[torch.Tensor]],
        batch_values: List[List[torch.Tensor]],
        batch_rewards: List[List[float]],
        batch_entropies: List[List[torch.Tensor]],
        gamma: float = 0.99,
    ) -> None:
        policy_losses = []
        value_losses = []
        entropy_terms = []
        for log_probs, values, rewards, entropies in zip(
            batch_log_probs, batch_values, batch_rewards, batch_entropies
        ):
            returns = []
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
            policy_losses.append(-(log_probs_t * policy_adv).mean())
            value_losses.append((returns_t - values_t).pow(2).mean())
            entropy_terms.append(entropies_t.mean())
        loss = (
            torch.stack(policy_losses).mean()
            + 0.5 * torch.stack(value_losses).mean()
            - 0.01 * torch.stack(entropy_terms).mean()
        )
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
