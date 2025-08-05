# Six Nimmt! RL

This repository implements a reinforcement learning setup for the card game *6 nimmt!*.

## Files
- `six_nimmt_env.py` – Gymnasium environment of the game.
- `bots.py` – helper agents (`RandomBot`, `RuleBot`, and `RLAgent`).

- `train_six_nimmt.py` – training script performing long self-play training with an actor-critic agent, extra fine-tuning against the rule-based bot, evaluation, duels and rendering. Training can stop early once a target win-rate versus `RuleBot` is reached.
- `play_six_nimmt.py` – console interface to play against trained agents.

## Usage
Run extended self-play training (will overwrite previous models):

```bash
python train_six_nimmt.py --cycles 30 --episodes 1200 --bot-episodes 200 --target-winrate 0.75
```

Use `--load` to skip training and reuse existing `agent*_best.pth`. After training the script prints duel statistics against baseline bots and renders three sample games in text form. Extended training is required to push the learned agents toward the 75% win-rate target versus the heuristic RuleBot; once that threshold is exceeded training stops automatically.

To challenge the trained policies interactively, run:

```bash
python play_six_nimmt.py
```

The script loads `agent1_best.pth`, `agent2_best.pth` and `agent3_best.pth` if present (otherwise it falls back to rule-based bots) and lets you play as the first player with a simple text interface.
