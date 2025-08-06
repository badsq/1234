import random
from typing import List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

np.seterr(over='ignore')


def bull_value(card: int) -> int:
    """Return penalty (bull heads) for a card."""
    if card == 55:
        return 7
    if card % 11 == 0:
        return 5
    if card % 10 == 0:
        return 3
    if card % 5 == 0:
        return 2
    return 1


class SixNimmtEnv(gym.Env):
    """Multi-agent environment for 6 nimmt!.

    The environment operates with four players simultaneously. Observations and
    actions are arrays with shape (n_players, ...). Each action is an index of a
    card in player's hand (0-9). Invalid indices are mapped to the smallest
    available card.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, n_players: int = 4):
        super().__init__()
        self.n_players = n_players
        self.action_space = spaces.MultiDiscrete([10] * n_players)
        # Observation: 10 cards + 4 row tops + 4 row lengths
        self.obs_dim = 18
        low = np.zeros((n_players, self.obs_dim), dtype=np.int32)
        high = np.full((n_players, self.obs_dim), 104, dtype=np.int32)
        self.observation_space = spaces.Box(low, high, dtype=np.int32)
        self.rows: List[List[int]] = []
        self.hands: List[List[int]] = []
        self.scores: List[int] = []
        self.current_step: int = 0

    # ------------------------------------------------------------------ utils
    def _deal(self) -> None:
        deck = list(range(1, 105))
        random.shuffle(deck)
        self.rows = [[deck.pop()] for _ in range(4)]
        self.hands = [sorted(deck[i * 10:(i + 1) * 10]) for i in range(self.n_players)]
        self.scores = [0 for _ in range(self.n_players)]
        self.current_step = 0

    def _row_penalty(self, row: List[int]) -> int:
        return sum(bull_value(c) for c in row)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.n_players, self.obs_dim), dtype=np.int32)
        row_tops = [r[-1] for r in self.rows]
        row_lens = [len(r) for r in self.rows]
        for p in range(self.n_players):
            hand = self.hands[p]
            padded = hand + [0] * (10 - len(hand))
            obs[p, :10] = padded
            obs[p, 10:14] = row_tops
            obs[p, 14:18] = row_lens
        return obs

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._deal()
        return self._get_obs(), {}

    # ------------------------------------------------------------------ step
    def step(self, actions: List[int]):
        assert len(actions) == self.n_players
        # Retrieve selected cards; replace invalid indices with smallest card
        plays: List[Tuple[int, int]] = []  # (player, card)
        for p, act in enumerate(actions):
            hand = self.hands[p]
            valid = [c for c in hand if c > 0]
            if not valid:
                # should not happen, but guard
                card = 0
            else:
                idx = act
                if idx >= len(hand) or hand[idx] == 0:
                    # map to smallest available card
                    idx = next(i for i, c in enumerate(hand) if c > 0)
                card = hand.pop(idx)
            plays.append((p, card))
        # resolve plays in ascending order of card
        plays.sort(key=lambda x: x[1])
        rewards = np.zeros(self.n_players, dtype=np.float32)
        for p, card in plays:
            # choose row
            diffs = [(card - row[-1] if card > row[-1] else 105) for row in self.rows]
            min_diff = min(diffs)
            if min_diff == 105:  # card smaller than all row tops
                # take row with least penalty
                row_idx = min(range(4), key=lambda i: self._row_penalty(self.rows[i]))
                penalty = self._row_penalty(self.rows[row_idx])
                rewards[p] -= penalty
                self.rows[row_idx] = [card]
                self.scores[p] += penalty
                continue
            row_idx = diffs.index(min_diff)
            if len(self.rows[row_idx]) >= 5:
                # take row
                penalty = self._row_penalty(self.rows[row_idx])
                rewards[p] -= penalty
                self.rows[row_idx] = [card]
                self.scores[p] += penalty
            else:
                self.rows[row_idx].append(card)
        self.current_step += 1
        terminated = self.current_step >= 10
        return self._get_obs(), rewards, terminated, False, {}

    # ---------------------------------------------------------------- render
    def render(self):
        row_str = " | ".join(f"{i}:{row}" for i, row in enumerate(self.rows))
        score_str = ", ".join(f"P{i}:{s}" for i, s in enumerate(self.scores))
        print(f"Rows: {row_str}\nScores: {score_str}\n")
