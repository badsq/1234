import os
import json
import numpy as np
from typing import List

import torch

from six_nimmt_env import SixNimmtEnv, bull_value
from bots import RLAgent, RuleBot


def load_opponents(env: SixNimmtEnv, device: str) -> List:
    scores_path = "agent_scores_last.json"
    worst = 3
    if os.path.exists(scores_path):
        with open(scores_path, "r") as f:
            scores = json.load(f)
        worst = int(np.argmax(scores))
    opponents: List = []
    for i in range(4):
        if i == worst:
            continue
        path = f"agent{i}_last.pth"
        if os.path.exists(path):
            print(f"Loading agent {i} from {path}")
            agent = RLAgent(env.obs_dim, device=device)
            agent.load(path)
            opponents.append(agent)
    return opponents


def print_board(env: SixNimmtEnv) -> None:
    print("\n" + "=" * 50)
    for idx, row in enumerate(env.rows):
        penalty = sum(bull_value(c) for c in row)
        cards = " ".join(f"{c:02d}" for c in row)
        print(f"Row {idx + 1}: {cards:<25} | penalty {penalty:2d}")
    score_line = " ".join(f"P{i}:{s}" for i, s in enumerate(env.scores))
    print(f"Scores -> {score_line}")
    print("=" * 50)


def choose_card(hand: List[int]) -> int:
    valid = [c for c in hand if c > 0]
    mapping = [i for i, c in enumerate(hand) if c > 0]
    while True:
        choices = "  ".join(f"[{j}] {valid[j]}" for j in range(len(valid)))
        print(f"Your hand: {choices}")
        raw = input("Select card index: ")
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(mapping):
                return mapping[idx]
        print("Invalid choice, try again.")


def main(device: str = "cpu") -> None:
    env = SixNimmtEnv()
    opponents = load_opponents(env, device)
    obs, _ = env.reset()
    done = False
    while not done:
        print_board(env)
        action = choose_card(list(obs[0, :10]))
        actions = [action]
        for i, bot in enumerate(opponents, start=1):
            if isinstance(bot, RLAgent):
                act, _, _, _ = bot.act(obs[i])
            else:
                act = bot.act(obs[i])
            actions.append(act)
        obs, rewards, done, _, _ = env.step(actions)
    print_board(env)
    print("Game over!")
    for i, score in enumerate(env.scores):
        tag = "(You)" if i == 0 else ""
        print(f"Player {i} {tag}: {score} penalty")
    winner = min(range(4), key=lambda i: env.scores[i])
    if winner == 0:
        print("You win!")
    else:
        print(f"Player {winner} wins.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    main(device=args.device)
