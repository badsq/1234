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

def run_episode(
    env: SixNimmtEnv,
    players: Sequence,
    collect_logs: bool = True,
    target: int | None = None,
):
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
        if target is not None:
            penalty = -step_rewards[target]
            for j in range(len(players)):
                if j != target:
                    step_rewards[j] += penalty
        for i in range(len(players)):
            rewards[i].append(step_rewards[i])
    return log_probs, values, rewards, entropies, env.scores


def run_batch(
    envs: List[SixNimmtEnv],
    agents: Sequence[RLAgent],
    target: int | None = None,
):
    """Run one episode in each environment in ``envs`` concurrently."""
    num_envs = len(envs)
    n_players = len(agents)
    obs = np.stack([env.reset()[0] for env in envs])  # (num_envs, n_players, obs_dim)
    logs = [
        {"log_probs": [], "values": [], "rewards": [], "entropies": []}
        for _ in agents
    ]
    for _ in range(10):
        actions = np.zeros((num_envs, n_players), dtype=int)
        for i, ag in enumerate(agents):
            acts, logp, ent, val = ag.act_batch(obs[:, i, :])
            actions[:, i] = acts
            logs[i]["log_probs"].append(logp)
            logs[i]["entropies"].append(ent)
            logs[i]["values"].append(val)
        step_rew = torch.zeros(num_envs, n_players, device=agents[0].device)
        for e, env in enumerate(envs):
            obs_e, rew, _, _, _ = env.step(actions[e].tolist())
            obs[e] = obs_e
            step_rew[e] = torch.tensor(rew, device=agents[0].device)
        if target is not None:
            penalty = -step_rew[:, target].unsqueeze(1)
            mask = torch.ones(n_players, device=agents[0].device)
            mask[target] = 0
            step_rew += penalty * mask
        for i in range(n_players):
            logs[i]["rewards"].append(step_rew[:, i])
    out = []
    for i in range(n_players):
        logp = torch.stack(logs[i]["log_probs"], dim=1)
        vals = torch.stack(logs[i]["values"], dim=1)
        rews = torch.stack(logs[i]["rewards"], dim=1)
        ents = torch.stack(logs[i]["entropies"], dim=1)
        out.append((logp, vals, rews, ents))
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
) -> tuple[List[RLAgent], List[float]]:
    """Train four diverse agents in self-play.

    ``num_envs`` parallel environments are rolled out concurrently so that
    policy/value evaluation runs on large batches, improving GPU utilisation.

    Players other than index ``0`` receive additional reward equal to any
    penalty incurred by player ``0`` so that they learn to focus attacks on the
    human-controlled seat."""
    envs = [SixNimmtEnv(n_players=4) for _ in range(num_envs)]
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
        batches = int(np.ceil(episodes_per_cycle / num_envs))
        for b in range(batches):
            logs = run_batch(envs, agents, target=0)
            for i in range(eval_env.n_players):
                logp, val, rew, ent = logs[i]
                batch_logs[i]["log_probs"].append(logp)
                batch_logs[i]["values"].append(val)
                batch_logs[i]["rewards"].append(rew)
                batch_logs[i]["entropies"].append(ent)
            if (b + 1) % batch_size == 0:
                for i, ag in enumerate(agents):
                    log_probs = torch.cat(batch_logs[i]["log_probs"], dim=0)
                    values = torch.cat(batch_logs[i]["values"], dim=0)
                    rewards = torch.cat(batch_logs[i]["rewards"], dim=0)
                    entropies = torch.cat(batch_logs[i]["entropies"], dim=0)
                    ag.update_batch(log_probs, values, rewards, entropies)
                    batch_logs[i] = {"log_probs": [], "values": [], "rewards": [], "entropies": []}
        # flush remaining trajectories
        for i, ag in enumerate(agents):
            if batch_logs[i]["log_probs"]:
                log_probs = torch.cat(batch_logs[i]["log_probs"], dim=0)
                values = torch.cat(batch_logs[i]["values"], dim=0)
                rewards = torch.cat(batch_logs[i]["rewards"], dim=0)
                entropies = torch.cat(batch_logs[i]["entropies"], dim=0)
                ag.update_batch(log_probs, values, rewards, entropies)
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
            args.cycles,
            args.episodes,
            args.batch_size,
            device=args.device,
            num_envs=args.num_envs,
        )
        with open("agent_scores.json", "w") as f:
            json.dump(best_scores, f)

    best_idx = int(np.argmin(best_scores))
    print(f"Best agent: {best_idx} with avg penalty {best_scores[best_idx]:.2f}")
    render_games(env, agents, n=3)
