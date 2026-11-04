# OSRS-RL

Humanlike bots for a private Old School RuneScape server (RSMod), trained with
reinforcement learning on top of behaviour cloning from recorded human play, and
scored by a bot detector trained on the same server's sessions.

| Part | Where | Status |
|---|---|---|
| Tick simulator (gem crab, Draynor willows) + Gymnasium env | `osrs_rl/sim`, `osrs_rl/env.py` | working |
| Scripted bot and synthetic-human stand-ins | `osrs_rl/agents` | working |
| Behaviour cloning + PPO with KL to the BC prior | `osrs_rl/policy.py`, `osrs_rl/train.py` | working |
| Detectors (rules floor, gradient-boosted features, GRU) | `osrs_rl/detector.py`, `osrs_rl/metrics.py` | working |
| RuneLite recording plugin (passive) | `recorder/` | ready to try |
| Recording reader / converter | `osrs_rl/recordings.py` | first pass |
| RSMod gem crab + agent plugin (bots over a socket, lockstep) | `rsmod-content/`, `osrs_rl/rsmod_bridge.py`, `run_server.py` | runs headless; watching from a client untested |

Scope: your own private server only. Recordings of your own play may come from
the live game; bots never connect to it.

## Quick start

```
pip install -r requirements.txt
python -m pytest -q tests
python run_experiment.py                 # both activities, ~5M ticks each, CPU
python run_experiment.py --activity gemcrab --ticks 500000   # a few minutes
```

Results land in `results/<activity>/results.json`. See `docs/first-experiment.md`.
