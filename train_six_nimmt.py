import os
import random
from typing import List, Sequence

import numpy as np
import torch

from six_nimmt_env import SixNimmtEnv
from bots import RLAgent, RandomBot, RuleBot


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
    bot_episodes: int = 200,
    target_winrate: float = 0.75,
) -> tuple[List[RLAgent], List[float]]:
    """Train four agents in self-play with extra fine-tuning against RuleBot.

    Training stops early once any agent achieves the ``target_winrate`` against
    ``RuleBot`` to focus computation on producing a policy strong enough to
    beat human opponents.
    """
    env = SixNimmtEnv()
    agents = [RLAgent(env.obs_dim) for _ in range(4)]
    best_scores = [float("inf")] * 4
    for cycle in range(cycles):
        for _ in range(episodes_per_cycle):
            logps, vals, rews, ents, _ = run_episode(env, agents)
            for i, ag in enumerate(agents):
                ag.update(logps[i], vals[i], rews[i], ents[i])
        # additional training vs heuristic bots to boost win-rate
        for i, ag in enumerate(agents):
            opponents = [ag if j == i else RuleBot() for j in range(4)]
            for _ in range(bot_episodes):
                logps, vals, rews, ents, _ = run_episode(env, opponents)
                ag.update(logps[i], vals[i], rews[i], ents[i])
        # evaluation and saving
        avg = evaluate_agents(env, agents, games=200)
        best_win = 0.0
        for i in range(4):
            if avg[i] < best_scores[i]:
                best_scores[i] = avg[i]
                agents[i].save(f"agent{i}_best.pth")
            # quick duel check vs RuleBot to gauge human-level strength
            _, _, win = duel(agents[i], RuleBot, games=150)
            best_win = max(best_win, win)
        print(
            f"Cycle {cycle}: avg penalties {avg}, best win-rate vs RuleBot {best_win:.2f}"
        )
        if best_win >= target_winrate:
            print(
                f"Target win-rate {target_winrate:.2f} reached; stopping training early."
            )
            break

    return agents, best_scores


def duel(agent: RLAgent, opponent_factory, games: int = 300):
    env = SixNimmtEnv(n_players=2)
    wins = 0
    sum_agent = 0
    sum_opponent = 0
    for _ in range(games):
        players = [agent, opponent_factory()]
        _, _, _, _, scores = run_episode(env, players, collect_logs=False)
        sum_agent += scores[0]
        sum_opponent += scores[1]
        if scores[0] < scores[1]:
            wins += 1
    return sum_agent / games, sum_opponent / games, wins / games


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
    parser.add_argument(
        "--bot-episodes", type=int, default=200, help="fine-tune vs RuleBot per cycle"
    )
    parser.add_argument(
        "--target-winrate",
        type=float,
        default=0.75,
        help="early-stop once any agent surpasses this win-rate vs RuleBot",
    )
    args = parser.parse_args()

    if args.load:
        env = SixNimmtEnv()
        agents = [RLAgent(env.obs_dim) for _ in range(4)]
        for i, ag in enumerate(agents):
            ag.load(f"agent{i}_best.pth")
    else:
        agents, _ = train_selfplay(
            args.cycles, args.episodes, args.bot_episodes, args.target_winrate
        )

        env = SixNimmtEnv()

    # find best agent based on saved scores
    results = []
    for i in range(4):
        ag = RLAgent(env.obs_dim)
        ag.load(f"agent{i}_best.pth")
        res = evaluate_agents(env, [ag] + [RandomBot(), RandomBot(), RandomBot()], games=300)
        results.append(res[0])
    best_idx = int(np.argmin(results))
    best_agent = RLAgent(env.obs_dim)
    best_agent.load(f"agent{best_idx}_best.pth")
    # duels
    avg_a, avg_o, win = duel(best_agent, RandomBot)
    print(f"Vs RandomBot: agent {avg_a:.2f}, opp {avg_o:.2f}, win-rate {win:.2f}")
    avg_a, avg_o, win = duel(best_agent, RuleBot)
    print(f"Vs RuleBot: agent {avg_a:.2f}, opp {avg_o:.2f}, win-rate {win:.2f}")
    # render
    agents = [RLAgent(env.obs_dim) for _ in range(4)]
    for i, ag in enumerate(agents):
        ag.load(f"agent{i}_best.pth")
    render_games(env, agents, n=3)
