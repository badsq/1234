# Six Nimmt! RL

This repository implements a reinforcement learning setup for the card game *6 nimmt!*.

## Files
- `six_nimmt_env.py` – Gymnasium environment of the game.
- `bots.py` – helper agents (`RandomBot`, `RuleBot`, and `RLAgent`).
- `train_six_nimmt.py` – self-play training of four actor-critic agents with optional GPU acceleration.
  Players 1–3 receive extra reward when player 0 takes penalty so they learn to focus on the human seat.
  The strength of this incentive is controlled by `--attack-weight`.
- `play_six_nimmt.py` – console interface to play as player 0 against three trained agents.

## Usage
Run extended self-play training (will overwrite previous models):

```bash
python train_six_nimmt.py --cycles 30 --episodes 1200 --batch-size 8 --num-envs 256 --device cuda --attack-weight 2.0
```

`--batch-size` controls how many *batches* of parallel episodes contribute to
each optimisation step. Combine it with `--num-envs` to roll out many games in
lockstep and feed massive tensors to the GPU.

Use `--load` to skip training and reuse existing checkpoints. Pass `--checkpoint best` (default) to load the strongest group of models saved for maximising the penalty of player 0, or `--checkpoint last` to inspect the most recent training snapshot. After loading or training the script evaluates the agents as a team and renders three sample games in text form.

To challenge the trained policies interactively, run:

```bash
python play_six_nimmt.py --device cuda --checkpoint best
```

The script loads models for players 1–3; the human always takes seat 0. Use `--checkpoint last` to play against the latest training snapshot. Missing models fall back to the heuristic `RuleBot`.
