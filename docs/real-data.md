# Real recordings vs the simulator (Nov 2026)

Data: Jason's RuneLite recordings, 33 windows of 15 minutes at the gem crab from 8 files
(about 8 hours). Run with `python run_real.py --recordings <dir>`; crab statistics with
`python calibrate.py <files>`. Detectors are trained and tested with folds grouped by
recording file. AUC 0.5 means a detector can't tell an agent from Jason; brackets are
bootstrap 90% intervals.

## What the recordings showed (calibrate.py, 9.4 h in the crab area)

* ~99k combat XP/h while attacking: an XP drop every 4 ticks on 97% of attacks, 0–26
  damage per hit (98 Strength, dragon scimitar). The old simulator assumed max hits of
  10–20 and 85% accuracy, so it ran at ~50k/h.
* The crab lives ~1000 ticks and the shell stays ~155 ticks, matching the wiki.
* The shell was mined once in 9 hours, so real progress is almost all combat XP.
* The game window is unfocused 61% of the time. The next input after the crab burrows comes
  a median ~30 s later, while arriving at a new mine is answered in ~1.4 s.
* ~4.6 camera moves an hour and ~12.6 side-panel tab clicks an hour, often in quick runs.

## Calibration rounds

| Round | Change | Synthetic humans vs Jason (worst AUC) |
|---|---|---|
| 0 | none | 1.00 |
| 1 | damage, accuracy, AFK after burrow, tab checks, shell habit; converter maps tab clicks | 1.00 |
| 2 | converter emits "arrived" like the sim; tab clicks in quick runs | 1.00 |
| 3 | click rank filled for real clicks (it alone separated real from sim); shorter AFK | 0.95 [0.92, 0.98] |

## Round 3 results

| Agent | Progress/h | Worst AUC vs Jason | Reaction KS |
|---|---|---|---|
| Jason (real) | 86,158 | – | – |
| Synthetic humans | 78,948 | 0.95 [0.92, 0.98] | 0.08 |
| Scripted bot | 87,805 | 1.00 [1.00, 1.00] | 0.50 |
| Behaviour cloning | 78,869 | 0.87 [0.81, 0.92] | 0.10 |
| PPO from BC + KL 0.1 | 78,464 | 0.97 [0.94, 0.99] | 0.07 |
| Pure RL | 89,441 | 1.00 [1.00, 1.00] | 0.33 |

## What still separates simulated play from Jason

* **Long gaps.** Simulated players' longest gaps between inputs are ~5× Jason's, and 26–33%
  of gaps are over a minute vs his 20%.
* **Always the nearest target.** Synthetic players click the nearest crab, shell or cave
  every time; 84% of Jason's interactions are rank-0 (his "Walk here" clicks have no
  simulator equivalent).
* **Tab checks overshoot.** 25/h synthetic vs 12.6/h real after round 3's burst change.

BC still learns from the synthetic players, because the recordings don't yet carry the
simulator's observation. Training BC on the recordings directly is the next big step.

## Behaviour cloning on the recordings (`run_real.py --bc ...`)

`osrs_rl/real_bc.py` replays each recording through the env's decision rules and rebuilds
the simulator's observation from the tick snapshots, so BC can copy what Jason actually did.
About 9.9 h gave ~2,400 decisions, of which ~295 are inputs (the rest are 15 s waits).

| BC learns from | BC agent worst AUC | PPO + KL worst AUC | BC progress/h |
|---|---|---|---|
| Synthetic players only | 0.92 [0.87, 0.95] | 0.99 [0.98, 1.00] | 76,431 |
| Jason only | 0.97 [0.94, 0.99] | 0.98 [0.95, 1.00] | 31,093 |
| Both (Jason upweighted to ~half) | **0.90 [0.85, 0.94]** | **0.94 [0.91, 0.97]** | 67,125 |

* Jason-only BC overfits: ~295 inputs isn't enough, and the clone is too passive
  (31k/h, 29 s median reactions).
* Mixing helps most after RL. The KL leash to a prior that has seen Jason keeps PPO closer to
  him (0.94 vs 0.99), and that gap is outside the error bars.
* Caveat: the detectors' human side uses windows from the same files BC learned from, so
  these numbers may flatter the Jason-trained agents a little. More hours, and a held-out
  day, will settle that.
* Known remaining mismatch: the real crab is a large NPC, so the recorded tile is ~5 tiles
  from the player while the simulator puts it adjacent.

## Detector in the reward, and the leash (`--adv-weight`, `--kl`)

`osrs_rl/adversarial.py` adds a GAIL-style bonus: at the end of each PPO episode the current
GBT and GRU detectors score its 15-minute windows and the agent gets the clipped log-odds of
"human". The detectors are retrained on the agent's latest windows every 10 updates. All
agents are then scored by fresh detectors trained after training ends (same 4-fold, by-file
evaluation as above). Mixed BC, 5M ticks, 37 windows from 10 files.

| Agent | Progress/h | GBT | GRU | Worst AUC vs Jason |
|---|---|---|---|---|
| BC (run pair 1) | 68,427 | 0.82 | 0.92 | 0.92 [0.88, 0.96] |
| PPO, KL 0.03 | 70,229 | 0.98 | 0.99 | 0.99 [0.97, 1.00] |
| PPO, KL 0.1 | 74,835 | 0.99 | 0.96 | 0.99 [0.98, 1.00] |
| **PPO + detector reward, KL 0.1** | 72,853 | 0.93 | 0.86 | **0.93 [0.88, 0.97]** |
| BC (run pair 2) | 75,925 | 0.91 | 0.88 | 0.91 [0.86, 0.95] |
| PPO, KL 0.3 | 70,635 | 0.92 | 0.88 | 0.92 [0.87, 0.96] |
| PPO, KL 1.0 | 68,992 | 0.93 | 0.93 | 0.93 [0.89, 0.97] |

* The detector reward undoes most of the damage plain PPO does at the same leash (0.93 vs
  0.99), but doesn't get below BC.
* A tighter leash (0.3 or 1.0) does about as well on its own: PPO stays near BC.
* The in-loop detectors cycle: the agent pushes p(bot) down to ~0.65, the retrain brings it
  back to ~0.93, and the agent swings between ~10 and ~35 actions per episode.
* Noise: BC came out different in the two run pairs (recordings were still growing between
  launches), and PPO at KL 0.1 scored 0.94 in the previous round vs 0.99 here. Gaps under
  ~0.05 need several seeds on a frozen copy of the recordings.

### Three seeds, frozen recordings, KL 0.3, 8M ticks

Same 37 windows for every run (a snapshot of 10 finished files). BC is now identical for a
given seed; PPO still isn't bit-for-bit repeatable.

| Agent | Seed 0 | Seed 1 | Seed 2 | Mean worst AUC |
|---|---|---|---|---|
| BC | 0.92 | 0.92 | 0.90 | **0.91** |
| PPO, KL 0.3 (two runs per seed) | 0.96, 0.93 | 0.96, 0.96 | 0.95, 0.97 | 0.955 |
| PPO, KL 0.3 + detector reward (weight 1, retrain every 5) | 0.94 | 0.94 | 0.97 | 0.95 |

* With seeds averaged, neither PPO variant beats BC, and the detector reward adds nothing
  over the leash at this weight. The single-run 0.92 above was luck.
* The in-loop detectors win easily: the agent rarely gets p(bot) below ~0.8 between retrains.
* The bottleneck looks like the human data (~300 real inputs) rather than RL.

## How much of Jason's data does BC need? (`run_efficiency.py`)

BC trained on 100 / 50 / 25 / 10 / 5 % of the recording files (5 random file subsets per
fraction, one mid-sized file always held out), rolled out in the simulator and scored by the
usual by-file detectors against all 38 real windows from 14 files (11.3 h, 2 Nov snapshot). No
PPO. "AUC held" is the same score restricted to real windows from files BC never trained on.

| BC learns from | Fraction | Files | Real inputs | Progress/h | Worst AUC (all / held-out files) | Reaction KS |
|---|---|---|---|---|---|---|
| Jason only | 100% | 13 | 384 | 39,253 | 0.95 / 0.99 | 0.39 |
| Jason only | 50% | 6 | 182 | 34,217 | 0.97 / 0.97 | 0.31 |
| Jason only | 25% | 3 | 115 | 35,394 | 0.99 / 0.99 | 0.28 |
| Jason only | 10% | 1 | 29 | 8,035 | 0.97 / 0.97 | 0.41 |
| Jason only | 5% | 1 | 31 | 32,794 | 0.97 / 0.97 | 0.36 |
| Synthetic + Jason | 100% | 13 | 384 | 75,445 | 0.94 / 0.93 | 0.11 |
| Synthetic + Jason | 50% | 6 | 182 | 70,019 | 0.92 / 0.92 | 0.11 |
| Synthetic + Jason | 25% | 3 | 115 | 72,997 | 0.92 / 0.93 | 0.09 |
| Synthetic + Jason | 10% | 1 | 29 | 76,689 | 0.94 / 0.94 | 0.10 |
| Synthetic + Jason | 5% | 1 | 31 | 72,581 | 0.94 / 0.94 | 0.12 |
| Synthetic only | 0% | 0 | 0 | 76,431 | 0.93 / 0.93 | 0.07 |

Means over subsets; single runs at the same fraction range from 0.89 to 0.98, so anything
under ~0.05 is noise.

* **The curve is flat.** Mixed BC scores 0.92–0.94 whether it sees all of Jason's 384 inputs,
  29 of them, or none. Another 10 hours of the same kind of recording will not move this
  number by itself; what the detectors catch is not something BC is failing to learn from
  too few examples.
* **Jason-only BC is unstable, not just overfitted.** Subsets of the same size land anywhere
  between a clone that never clicks (0 progress/h) and one that checks tabs 468 times an
  hour; the 100 % clone reacts in a median 36 s and makes 39k/h. With one player and mostly
  WAIT labels, the intent head collapses to whichever habit dominates the chosen files.
* **Held-out and all-file AUC agree**, so the earlier worry that BC-trained files flatter the
  score was unfounded at this data size.
* What stays constant across the curve is the simulator side: the observation that BC sees
  (walk-here clicks dropped, crab adjacent instead of 5 tiles away, no bystanders, no focus
  flag) and the cut-off of reactions at 15 s chunks. Those, and the choice of observation
  features, are the next things to change; see `training-directions/plan.md` in the project.
