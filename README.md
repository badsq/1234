# Six Nimmt! RL

This repository implements a reinforcement learning setup for the card game *6 nimmt!*.

## Files
- `six_nimmt_env.py` – Gymnasium environment of the game.
- `bots.py` – helper agents (`RandomBot`, `RuleBot`, and `RLAgent`).
- `train_six_nimmt.py` – self-play training of six diverse actor-critic agents and text rendering of sample games.
- `play_six_nimmt.py` – console interface to play against trained agents.

## Usage
Run extended self-play training (will overwrite previous models):

```bash
python train_six_nimmt.py --cycles 30 --episodes 1200
```

Use `--load` to skip training and reuse existing `agent*_best.pth`. After training the script evaluates the agents against each other and renders three sample games in text form.

To challenge the trained policies interactively, run:

```bash
python play_six_nimmt.py
```

The script loads `agent1_best.pth`, `agent2_best.pth` and `agent3_best.pth` if present (otherwise it falls back to heuristic bots) and lets you play as the first player with a simple text interface.
