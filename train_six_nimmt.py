import os
import json
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


def run_parallel(envs: List[SixNimmtEnv], players: Sequence, collect_logs: bool = True):
    """Run multiple environments in lockstep and collect trajectories."""
    n_envs = len(envs)
    obs = np.stack([e.reset()[0] for e in envs])
    log_probs = [[[] for _ in range(n_envs)] for _ in players]
    entropies = [[[] for _ in range(n_envs)] for _ in players]
    values = [[[] for _ in range(n_envs)] for _ in players]
    rewards = [[[] for _ in range(n_envs)] for _ in players]
    for _ in range(10):
        step_actions = [[] for _ in range(n_envs)]
        for i, p in enumerate(players):
            if isinstance(p, RLAgent):
                acts, logp, ent, val = p.act_batch(obs[:, i, :])
                for e in range(n_envs):
                    step_actions[e].append(int(acts[e]))
                    if collect_logs:
                        log_probs[i][e].append(logp[e])
                        entropies[i][e].append(ent[e])
                        values[i][e].append(val[e])
            else:
                for e in range(n_envs):
                    step_actions[e].append(p.act(obs[e, i]))
        for e, env in enumerate(envs):
            obs_e, step_rew, _, _, _ = env.step(step_actions[e])
            obs[e] = obs_e
            for i in range(len(players)):
                rewards[i][e].append(step_rew[i])
    scores = [env.scores for env in envs]
    return log_probs, values, rewards, entropies, scores


def evaluate_agents(env: SixNimmtEnv, agents: Sequence, games: int = 150):
    total = np.zeros(len(agents))
    for _ in range(games):
        _, _, _, _, scores = run_episode(env, agents, collect_logs=False)
        total += np.array(scores)
    return total / games


def train_selfplay(
    cycles: int = 30,
    episodes_per_cycle: int = 1200,
    device: str | None = None,
    num_envs: int = 32,
) -> tuple[List[RLAgent], List[float]]:
    """Train four diverse agents in self-play using batched environments."""
    base_env = SixNimmtEnv(n_players=4)
    base_lr = 3e-4
    lrs = [base_lr * (1 + 0.1 * i) for i in range(base_env.n_players)]
    agents = [RLAgent(base_env.obs_dim, lr=lr, device=device) for lr in lrs]
    envs = [SixNimmtEnv(n_players=base_env.n_players) for _ in range(num_envs)]
    best_scores = [float("inf")] * base_env.n_players
    for cycle in range(cycles):
        remaining = episodes_per_cycle
        while remaining > 0:
            batch = min(remaining, num_envs)
            batch_envs = envs[:batch]
            logps, vals, rews, ents, _ = run_parallel(batch_envs, agents)
            for i, ag in enumerate(agents):
                ag.update_batch(logps[i], vals[i], rews[i], ents[i])
            remaining -= batch
        avg = evaluate_agents(base_env, agents, games=200)
        for i in range(base_env.n_players):
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-envs", type=int, default=32, help="parallel environments for self-play")
    args = parser.parse_args()

    env = SixNimmtEnv(n_players=4)
    if args.load:
        agents = [RLAgent(env.obs_dim, device=args.device) for _ in range(env.n_players)]
        for i, ag in enumerate(agents):
            ag.load(f"agent{i}_best.pth")
        if os.path.exists("agent_scores.json"):
            with open("agent_scores.json", "r") as f:
                best_scores = json.load(f)
        else:
            best_scores = evaluate_agents(env, agents, games=300).tolist()
            with open("agent_scores.json", "w") as f:
                json.dump(best_scores, f)
    else:
        agents, best_scores = train_selfplay(args.cycles, args.episodes, device=args.device, num_envs=args.num_envs)
        with open("agent_scores.json", "w") as f:
            json.dump(best_scores, f)

    best_idx = int(np.argmin(best_scores))
    print(f"Best agent: {best_idx} with avg penalty {best_scores[best_idx]:.2f}")
    render_games(env, agents, n=3)
