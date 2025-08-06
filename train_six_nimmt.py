import os
import json
import random
from typing import List, Sequence

import numpy as np
import torch

from six_nimmt_env import SixNimmtEnv
from bots import RLAgent, RuleBot


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


def run_batch(envs: List[SixNimmtEnv], players: Sequence):
    """Run one episode concurrently in each environment.

    ``players`` may contain fixed bots; logs are collected only for RL agents."""

    num_envs = len(envs)
    n_players = len(players)
    obs = np.stack([env.reset()[0] for env in envs])
    logs = [
        {"log_probs": [], "values": [], "rewards": [], "entropies": []}
        if isinstance(p, RLAgent)
        else None
        for p in players
    ]
    rl_device = next((p.device for p in players if isinstance(p, RLAgent)), "cpu")
    for _ in range(10):
        actions = np.zeros((num_envs, n_players), dtype=int)
        for i, p in enumerate(players):
            if isinstance(p, RLAgent):
                acts, logp, ent, val = p.act_batch(obs[:, i, :])
                actions[:, i] = acts
                logs[i]["log_probs"].append(logp)
                logs[i]["entropies"].append(ent)
                logs[i]["values"].append(val)
            else:
                acts = np.array([p.act(obs[e, i, :]) for e in range(num_envs)], dtype=int)
                actions[:, i] = acts
        step_rew = torch.zeros(num_envs, n_players, device=rl_device)
        for e, env in enumerate(envs):
            obs_e, rew, _, _, _ = env.step(actions[e].tolist())
            obs[e] = obs_e
            step_rew[e] = torch.tensor(rew, device=rl_device)
        for i, p in enumerate(players):
            if isinstance(p, RLAgent):
                logs[i]["rewards"].append(step_rew[:, i])
    out = []
    for i, p in enumerate(players):
        if isinstance(p, RLAgent):
            logp = torch.stack(logs[i]["log_probs"], dim=1)
            vals = torch.stack(logs[i]["values"], dim=1)
            rews = torch.stack(logs[i]["rewards"], dim=1)
            ents = torch.stack(logs[i]["entropies"], dim=1)
            out.append((logp, vals, rews, ents))
        else:
            out.append(None)
    return out


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
) -> tuple[List, List[float]]:
    """Train three learning agents against a fixed opponent."""

    envs = [SixNimmtEnv(n_players=4) for _ in range(num_envs)]
    eval_env = SixNimmtEnv(n_players=4)
    base_lr = 3e-4
    lrs = [base_lr * (1 + 0.1 * i) for i in range(3)]
    agents: List = [RuleBot()] + [RLAgent(eval_env.obs_dim, lr=lr, device=device) for lr in lrs]
    best_scores = [float("inf")] * len(agents)
    for cycle in range(cycles):
        batch_logs = [
            {"log_probs": [], "values": [], "rewards": [], "entropies": []}
            if isinstance(p, RLAgent)
            else None
            for p in agents
        ]
        batches = int(np.ceil(episodes_per_cycle / num_envs))
        for b in range(batches):
            logs = run_batch(envs, agents)
            for i, p in enumerate(agents):
                if isinstance(p, RLAgent):
                    logp, val, rew, ent = logs[i]
                    batch_logs[i]["log_probs"].append(logp)
                    batch_logs[i]["values"].append(val)
                    batch_logs[i]["rewards"].append(rew)
                    batch_logs[i]["entropies"].append(ent)
            if (b + 1) % batch_size == 0:
                for i, p in enumerate(agents):
                    if isinstance(p, RLAgent) and batch_logs[i]["log_probs"]:
                        log_probs = torch.cat(batch_logs[i]["log_probs"], dim=0)
                        values = torch.cat(batch_logs[i]["values"], dim=0)
                        rewards = torch.cat(batch_logs[i]["rewards"], dim=0)
                        entropies = torch.cat(batch_logs[i]["entropies"], dim=0)
                        p.update_batch(log_probs, values, rewards, entropies)
                        batch_logs[i] = {"log_probs": [], "values": [], "rewards": [], "entropies": []}
        for i, p in enumerate(agents):
            if isinstance(p, RLAgent) and batch_logs[i]["log_probs"]:
                log_probs = torch.cat(batch_logs[i]["log_probs"], dim=0)
                values = torch.cat(batch_logs[i]["values"], dim=0)
                rewards = torch.cat(batch_logs[i]["rewards"], dim=0)
                entropies = torch.cat(batch_logs[i]["entropies"], dim=0)
                p.update_batch(log_probs, values, rewards, entropies)
        avg = evaluate_agents(eval_env, agents, games=200)
        for i, p in enumerate(agents):
            if isinstance(p, RLAgent) and avg[i] < best_scores[i]:
                best_scores[i] = avg[i]
                p.save(f"agent{i}_best.pth")
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
                if isinstance(ag, RLAgent):
                    act, _, _, _ = ag.act(obs[i])
                else:
                    act = ag.act(obs[i])
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
        agents = [RuleBot()] + [RLAgent(env.obs_dim, device=args.device) for _ in range(env.n_players - 1)]
        for i in range(1, env.n_players):
            agents[i].load(f"agent{i}_best.pth")
        if os.path.exists("agent_scores.json"):
            with open("agent_scores.json", "r") as f:
                best_scores = json.load(f)
        else:
            best_scores = evaluate_agents(env, agents, games=300).tolist()
            with open("agent_scores.json", "w") as f:
                json.dump(best_scores, f)
    else:
        agents, best_scores = train_selfplay(
            args.cycles,
            args.episodes,
            args.batch_size,
            device=args.device,
            num_envs=args.num_envs,
        )
        with open("agent_scores.json", "w") as f:
            json.dump(best_scores, f)

    rl_scores = best_scores[1:]
    best_idx = int(np.argmin(rl_scores)) + 1
    print(f"Best agent: {best_idx} with avg penalty {best_scores[best_idx]:.2f}")
    render_games(env, agents, n=3)
