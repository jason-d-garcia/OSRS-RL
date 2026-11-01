# First experiment: gem crab and Draynor willows in the simulator

Run with `python run_experiment.py` (5M game ticks of PPO per run, CPU only, about 7 minutes
for both activities on 4 cores). Raw numbers: `results/<activity>/results.json`.

## Read this first

There are no human recordings yet, so the "humans" here are **synthetic players**
(`osrs_rl/agents/controllers.py: SyntheticHuman`): 12 made-up players with their own
reaction speed, attention lapses, fatigue, camera and tab habits, redundant clicks and
imperfect target choice. Behaviour cloning learns from 7 of them; the detectors are
tested on 4 held-out ones. Every humanness number below measures closeness to that
model, not to people. The pipeline is what this run tests. Once Jason's gem crab
recordings exist, `osrs_rl/recordings.py` turns them into the same session format and
the same scripts run on real play.

## Setup

* **Simulator**: tick-based (600 ms). Woodcutting rolls, axe rates, willow despawn (50
  ticks of cutting) and respawn (15 ticks) are read from RSMod v2's content. Gem crab: 10
  minute timer, 90 s shell, top-16 damage rule, 3 gems, three mines linked by a cave.
  Layout and a few timings are approximations (`osrs_rl/sim/params.py` marks each).
* **Decisions, not ticks**: the policy acts at decision points (an action finished, something
  changed, or it has been busy 15 s) and picks an intent, a target entity and a delay in ms.
  Waits are invisible; every other intent is a logged input event.
* **Agents**: scripted bot (optimal choices, uniform 600–1800 ms sleeps); behaviour cloning
  (BC); PPO from BC with a per-step KL penalty to BC (coef 0.1), progress reward capped at
  the human 75th percentile rate, a penalty for reacting faster than the human 2nd percentile
  and for 10 minutes without a camera move; pure RL ablation (no BC, no KL, no shaping).
* **Detectors**, trained on 80 synthetic-human sessions (8 players) vs 50 bot sessions and
  tested on 40 sessions from 4 held-out players vs 50 fresh bot sessions:
  a rules floor (two of: even reaction times, never slow to react, superhuman reactions,
  no camera or tab use), gradient-boosted trees on 40 session features, and a GRU over the
  action sequence. AUC 0.5 is chance.

## Results

### gemcrab

| Agent | Progress/h | Actions per 30 min | Rules floor flags | GBT detector acc / AUC | GRU detector acc / AUC | Reaction-time KS |
|---|---|---|---|---|---|---|
| Held-out synthetic humans | 49,069 (IQR 39,238–53,393) | 12 | 0% (false flags) | – | – | – |
| Scripted bot | 51,478 | 10 | 100% | 1.00 / 1.00 | 0.97 / 0.97 | 0.48 |
| Behaviour cloning | 49,948 | 14 | 0% | 0.55 / 0.56 | 0.47 / 0.47 | 0.11 |
| PPO from BC + KL | 47,596 | 17 | 0% | 0.74 / 0.84 | 0.79 / 0.90 | 0.15 |
| Pure RL (no prior) | 52,398 | 84 | 6% | 1.00 / 1.00 | 1.00 / 1.00 | 0.34 |

### willows

| Agent | Progress/h | Actions per 30 min | Rules floor flags | GBT detector acc / AUC | GRU detector acc / AUC | Reaction-time KS |
|---|---|---|---|---|---|---|
| Held-out synthetic humans | 23,230 (IQR 18,774–25,881) | 79 | 0% (false flags) | – | – | – |
| Scripted bot | 29,228 | 72 | 100% | 1.00 / 1.00 | 1.00 / 1.00 | 0.50 |
| Behaviour cloning | 21,714 | 78 | 0% | 0.48 / 0.53 | 0.50 / 0.47 | 0.14 |
| PPO from BC + KL | 23,867 | 87 | 0% | 0.74 / 0.82 | 0.50 / 0.51 | 0.10 |
| Pure RL (no prior) | 20,110 | 218 | 70% | 1.00 / 1.00 | 1.00 / 1.00 | 0.90 |

## What it says

1. **The scripted bot is caught by everything**, including the rules floor. Uniform
   600–1800 ms sleeps give it a reaction-time distribution with no tail (KS 0.48–0.50
   against humans) and it never moves the camera.
2. **Pure RL is efficient-looking and obviously not a person.** It clicks 3–7 times as
   often as a human and almost never lapses (6% of gaps over 60 s at the crab vs 45% for
   humans). Every learned detector gets it at AUC 1.00. At willows it hasn't even reached
   the human XP rate after 5M ticks.
3. **BC sits at chance** (AUC 0.47–0.56). That is expected and mostly circular: it is cloned
   from the same synthetic model the held-out players come from. With real recordings this
   number becomes meaningful.
4. **PPO from BC drifts away from human** (best detector AUC 0.84–0.90 at the crab, 0.82 at
   willows) **without gaining much**: XP/h moved from 49.9k to 47.6k at the crab and from
   21.7k to 23.9k at willows, inside the human IQR. Part of the drift is reward hacking of a
   humanness term: the "no camera move in 10 minutes" penalty pushed camera moves at the crab
   to 9.9/h, three times the human 3.1/h. Gem crab has only about 12 observable inputs per
   30 minutes, so each extra click or camera move is a strong tell.

## Next

* Swap the synthetic humans for Jason's recorded gem crab sessions (recorder in `recorder/`).
* Drop the camera penalty in favour of the detector term (stage 3 of the plan: the detector's
  log-odds in the reward), and sweep the KL coefficient (0.1 / 0.3 / 1.0).
* Calibrate the simulator against tick traces from the RSMod server; script gem crabs and
  shell mining in RSMod (`docs/server-setup.md`).
