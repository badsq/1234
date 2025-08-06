# Six Nimmt! RL

This repository implements a reinforcement learning setup for the card game *6 nimmt!*.

## Files
- `six_nimmt_env.py` – Gymnasium environment of the game.
- `bots.py` – helper agents (`RandomBot`, `RuleBot`, and `RLAgent`).
- `train_six_nimmt.py` – self-play training of four actor-critic agents with optional GPU acceleration and text rendering of sample games.
- `play_six_nimmt.py` – console interface to play against trained agents (the human replaces the weakest agent).

## Usage
Run extended self-play training (will overwrite previous models):

```bash
python train_six_nimmt.py --cycles 30 --episodes 1200 --batch-size 8 --num-envs 256 --device cuda
```

`--batch-size` controls how many *batches* of parallel episodes contribute to
each optimisation step. Combine it with `--num-envs` to roll out many games in
lockstep and feed massive tensors to the GPU.

Use `--load` to skip training and reuse existing `agent*_best.pth`. After training the script evaluates the agents against each other, stores their average penalties in `agent_scores.json`, and renders three sample games in text form.

To challenge the trained policies interactively, run:

```bash
python play_six_nimmt.py --device cuda
```

The script loads all trained models and drops the weakest one so the human fills that seat. Missing models fall back to the heuristic `RuleBot`.
