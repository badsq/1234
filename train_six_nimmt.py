import os
import random
from typing import List, Sequence

import numpy as np
import torch

from six_nimmt_env import SixNimmtEnv
from bots import RLAgent


# ---------------------------------------------------------------------------
# Training and evaluation utilities

def run_episode(env: SixNimmtEnv, players: Sequence, collect_logs: bool = True):
    obs, _ = env.reset()
    log_probs = [[] for _ in players]
    entropies = [[] for _ in players]
    values = [[] for _ in players]
    rewards = [[] for _ in players]
    done = False
    while not done:
        actions = []
        for i, p in enumerate(players):
            if isinstance(p, RLAgent):
                act, logp, ent, val = p.act(obs[i])
                actions.append(act)
                if collect_logs:
                    log_probs[i].append(logp)
                    entropies[i].append(ent)
                    values[i].append(val)
            else:
                actions.append(p.act(obs[i]))
        obs, step_rewards, done, _, _ = env.step(actions)
        for i in range(len(players)):
            rewards[i].append(step_rewards[i])
    return log_probs, values, rewards, entropies, env.scores


def evaluate_agents(env: SixNimmtEnv, agents: Sequence, games: int = 150):
    total = np.zeros(len(agents))
    for _ in range(games):
        _, _, _, _, scores = run_episode(env, agents, collect_logs=False)
        total += np.array(scores)
    return total / games


def train_selfplay(
    cycles: int = 30,
    episodes_per_cycle: int = 1200,
) -> tuple[List[RLAgent], List[float]]:
    """Train six diverse agents in self-play."""
    env = SixNimmtEnv(n_players=6)
    base_lr = 3e-4
    lrs = [base_lr * (1 + 0.1 * i) for i in range(env.n_players)]
    agents = [RLAgent(env.obs_dim, lr=lr) for lr in lrs]
    best_scores = [float("inf")] * env.n_players
    for cycle in range(cycles):
        for _ in range(episodes_per_cycle):
            logps, vals, rews, ents, _ = run_episode(env, agents)
            for i, ag in enumerate(agents):
                ag.update(logps[i], vals[i], rews[i], ents[i])
        avg = evaluate_agents(env, agents, games=200)
        for i in range(env.n_players):
            if avg[i] < best_scores[i]:
                best_scores[i] = avg[i]
                agents[i].save(f"agent{i}_best.pth")
        print(f"Cycle {cycle}: avg penalties {avg}")
    return agents, best_scores


def render_games(env: SixNimmtEnv, agents: Sequence, n: int = 3):
    for g in range(n):
        print(f"=== Game {g+1} ===")
        obs, _ = env.reset()
        done = False
        while not done:
            actions = []
            for i, ag in enumerate(agents):
                act, _, _, _ = ag.act(obs[i])
                actions.append(act)
            obs, _, done, _, _ = env.step(actions)
            env.render()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="skip training and load saved agents")
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=1200)
    args = parser.parse_args()

    env = SixNimmtEnv(n_players=6)
    if args.load:
        agents = [RLAgent(env.obs_dim) for _ in range(env.n_players)]
        for i, ag in enumerate(agents):
            ag.load(f"agent{i}_best.pth")
        best_scores = evaluate_agents(env, agents, games=300)
    else:
        agents, best_scores = train_selfplay(args.cycles, args.episodes)

    best_idx = int(np.argmin(best_scores))
    print(f"Best agent: {best_idx} with avg penalty {best_scores[best_idx]:.2f}")
    render_games(env, agents, n=3)
