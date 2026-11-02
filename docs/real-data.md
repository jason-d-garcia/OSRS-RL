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
