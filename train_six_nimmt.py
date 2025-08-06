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
    target_weight: float = 1.0,
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
            step_rewards[target] -= penalty * (target_weight - 1)
            for j in range(len(players)):
                if j != target:
                    step_rewards[j] += penalty * target_weight
        for i in range(len(players)):
            rewards[i].append(step_rewards[i])
    return log_probs, values, rewards, entropies, env.scores


def run_batch(
    envs: List[SixNimmtEnv],
    agents: Sequence[RLAgent],
    target: int | None = None,
    target_weight: float = 1.0,
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
            step_rew[:, target] -= penalty.squeeze(1) * (target_weight - 1)
            mask = torch.ones(n_players, device=agents[0].device)
            mask[target] = 0
            step_rew += penalty * target_weight * mask
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


def evaluate_agents(
    env: SixNimmtEnv,
    agents: Sequence,
    games: int = 150,
    target: int | None = None,
    target_weight: float = 1.0,
):
    """Average penalties over ``games`` episodes.

    When ``target`` is provided the episode uses the same reward shaping as
    training so that all agents focus on the specified seat. The returned array
    always contains the raw penalty scores from the environment."""
    total = np.zeros(len(agents))
    for _ in range(games):
        _, _, _, _, scores = run_episode(
            env,
            agents,
            collect_logs=False,
            target=target,
            target_weight=target_weight,
        )
        total += np.array(scores, dtype=np.float64)
    return total / games


def train_selfplay(
    cycles: int = 100,
    episodes_per_cycle: int = 3000,
    batch_size: int = 16,
    device: str | None = None,
    num_envs: int = 256,
    target_weight: float = 2.0,
) -> tuple[List[RLAgent], float]:
    """Train four diverse agents in self-play.

    ``num_envs`` parallel environments are rolled out concurrently so that
    policy/value evaluation runs on large batches, improving GPU utilisation.

    Players other than index ``0`` receive additional reward equal to any
    penalty incurred by player ``0`` so that they learn to focus attacks on the
    human-controlled seat.  Best checkpoints are selected using the average
    penalty accumulated by player ``0`` so that the team is optimised for
    harassing the human opponent."""
    envs = [SixNimmtEnv(n_players=4) for _ in range(num_envs)]
    eval_env = SixNimmtEnv(n_players=4)
    base_lr = 3e-4
    lrs = [base_lr * (1 + 0.1 * i) for i in range(eval_env.n_players)]
    agents = [RLAgent(eval_env.obs_dim, lr=lr, device=device) for lr in lrs]
    best_target = -float("inf")
    for cycle in range(cycles):
        batch_logs = [
            {"log_probs": [], "values": [], "rewards": [], "entropies": []}
            for _ in agents
        ]
        batches = int(np.ceil(episodes_per_cycle / num_envs))
        for b in range(batches):
            logs = run_batch(envs, agents, target=0, target_weight=target_weight)
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
        avg = evaluate_agents(
            eval_env, agents, games=200, target=0, target_weight=target_weight
        )
        team_score = avg[0]
        for i in range(eval_env.n_players):
            agents[i].save(f"agent{i}_last.pth")
        if team_score > best_target:
            best_target = team_score
            for i in range(eval_env.n_players):
                agents[i].save(f"agent{i}_best.pth")
        print(f"Cycle {cycle}: avg penalties {avg}")
    return agents, best_target


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
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-envs", type=int, default=256, help="number of parallel environments")
    parser.add_argument(
        "--attack-weight",
        type=float,
        default=2.0,
        help="reward bonus applied when seat 0 takes a penalty",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
        help="which saved models to load when using --load",
    )
    args = parser.parse_args()

    env = SixNimmtEnv(n_players=4)
    if args.load:
        agents = [RLAgent(env.obs_dim, device=args.device) for _ in range(env.n_players)]
        for i, ag in enumerate(agents):
            ag.load(f"agent{i}_{args.checkpoint}.pth")
        if os.path.exists("team_score.json"):
            with open("team_score.json", "r") as f:
                best_target = json.load(f)["best_player0_penalty"]
        else:
            avg = evaluate_agents(
                env,
                agents,
                games=300,
                target=0,
                target_weight=args.attack_weight,
            )
            best_target = float(avg[0])
            with open("team_score.json", "w") as f:
                json.dump({"best_player0_penalty": best_target}, f)
    else:
        agents, best_target = train_selfplay(
            args.cycles,
            args.episodes,
            args.batch_size,
            device=args.device,
            num_envs=args.num_envs,
            target_weight=args.attack_weight,
        )
        with open("team_score.json", "w") as f:
            json.dump({"best_player0_penalty": best_target}, f)

    print(f"Best recorded player0 penalty: {best_target:.2f}")
    render_games(env, agents, n=3)
