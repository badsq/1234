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


def run_episode_batch(envs: List[SixNimmtEnv], players: Sequence):
    """Run one episode in each environment and collect batched logs."""
    n_envs = len(envs)
    obs = [env.reset()[0] for env in envs]
    log_probs = [[] for _ in players]
    entropies = [[] for _ in players]
    values = [[] for _ in players]
    rewards = [[] for _ in players]
    for _ in range(10):
        actions = np.zeros((n_envs, len(players)), dtype=np.int64)
        for i, p in enumerate(players):
            batch_obs = np.stack([o[i] for o in obs])
            if isinstance(p, RLAgent):
                act, logp, ent, val = p.act(batch_obs)
                actions[:, i] = act
                log_probs[i].append(logp)
                entropies[i].append(ent)
                values[i].append(val)
            else:
                acts = np.array([p.act(o[i]) for o in obs])
                actions[:, i] = acts
        step_rewards = []
        for e, env in enumerate(envs):
            obs_e, rew, _, _, _ = env.step(actions[e].tolist())
            obs[e] = obs_e
            step_rewards.append(rew)
        step_rewards = np.array(step_rewards)
        for i in range(len(players)):
            rewards[i].append(torch.as_tensor(step_rewards[:, i], dtype=torch.float32, device=players[i].device if isinstance(players[i], RLAgent) else "cpu"))
    return log_probs, values, rewards, entropies


def evaluate_agents(env: SixNimmtEnv, agents: Sequence, games: int = 150):
    total = np.zeros(len(agents))
    for _ in range(games):
        _, _, _, _, scores = run_episode(env, agents, collect_logs=False)
        total += np.array(scores)
    return total / games


def train_selfplay(
    cycles: int = 30,
    episodes_per_cycle: int = 1200,
    batch_size: int = 8,
    device: str | None = None,
    num_envs: int = 1,
) -> tuple[List[RLAgent], List[float]]:
    """Train four diverse agents in self-play.

    Experiences from ``batch_size``\*``num_envs`` episodes are accumulated before
    every optimisation step so that each update uses a larger tensor batch. This
    improves GPU utilisation and reduces per-episode overhead."""
    train_envs = [SixNimmtEnv(n_players=4) for _ in range(num_envs)]
    eval_env = SixNimmtEnv(n_players=4)
    base_lr = 3e-4
    lrs = [base_lr * (1 + 0.1 * i) for i in range(eval_env.n_players)]
    agents = [RLAgent(eval_env.obs_dim, lr=lr, device=device) for lr in lrs]
    best_scores = [float("inf")] * eval_env.n_players
    for cycle in range(cycles):
        batch_logs = [
            {"log_probs": [], "values": [], "rewards": [], "entropies": []}
            for _ in agents
        ]
        for ep in range(episodes_per_cycle // num_envs):
            logps, vals, rews, ents = run_episode_batch(train_envs, agents)
            for i in range(eval_env.n_players):
                batch_logs[i]["log_probs"].extend(logps[i])
                batch_logs[i]["values"].extend(vals[i])
                batch_logs[i]["rewards"].extend(rews[i])
                batch_logs[i]["entropies"].extend(ents[i])
            if (ep + 1) % batch_size == 0:
                for i, ag in enumerate(agents):
                    ag.update(
                        batch_logs[i]["log_probs"],
                        batch_logs[i]["values"],
                        batch_logs[i]["rewards"],
                        batch_logs[i]["entropies"],
                    )
                    batch_logs[i] = {"log_probs": [], "values": [], "rewards": [], "entropies": []}
        # flush remaining trajectories
        for i, ag in enumerate(agents):
            if batch_logs[i]["log_probs"]:
                ag.update(
                    batch_logs[i]["log_probs"],
                    batch_logs[i]["values"],
                    batch_logs[i]["rewards"],
                    batch_logs[i]["entropies"],
                )
        avg = evaluate_agents(eval_env, agents, games=200)
        for i in range(eval_env.n_players):
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-envs", type=int, default=1, help="number of parallel environments")
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
        agents, best_scores = train_selfplay(
            args.cycles, args.episodes, args.batch_size, device=args.device, num_envs=args.num_envs
        )
        with open("agent_scores.json", "w") as f:
            json.dump(best_scores, f)

    best_idx = int(np.argmin(best_scores))
    print(f"Best agent: {best_idx} with avg penalty {best_scores[best_idx]:.2f}")
    render_games(env, agents, n=3)
