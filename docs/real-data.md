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
* Not more data: a data-efficiency check (PR #12) found mixed BC scores 0.92–0.94 with all,
  29 or none of Jason's inputs. The limit is the simulator and observation: dropped Walk-here
  clicks, the crab adjacent instead of ~5 tiles away, no bystanders or focus flag in the
  observation, and reactions chunked at 15 s.

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

## Where the detector was reading the clone's rhythm (`rhythm` branch)

The flat curve above says the labels are not the bottleneck, so the next question was *which*
features the detector uses. Fitting the feature detector on all real windows against the
mixed-BC clone's windows and ranking features by permutation importance and single-feature
AUC gave a short list, none of it about reflexes:

| Feature (real vs clone) | Before | After |
|---|---|---|
| `gap_hist_6`, share of input gaps in the 13–29 s band | 2 % vs 15 % (AUC 0.83) | 2 % vs 8 % (not in the top ten) |
| `gap_p50`, median gap between inputs | 2.3 s vs 11.7 s (0.78) | 2.3 s vs 5.5 s (0.63 on the log mean) |
| `nearest_target_rate` | 0.85 vs 0.99 (0.77) | gone from the list |
| `walk_per_h`, Walk-here clicks | not a feature | 2.4 vs 0.4 (0.71) |
| `react_logstd`, spread of reaction times | 1.9 vs 1.4 (not in the top ten) | 1.9 vs 1.35 (0.67) |
| `check_tab_per_h` | 13 vs 24 | 13 vs 18 |

Each of the "before" tells was a clock we had put in the clone, not something Jason does:

* **The 13–29 s pile** came from two places. `real_bc.py` cut every long wait into 15 s
  WAIT chunks, so BC learned "wait 15 s" as the most common label; and the env re-asked the
  policy every 15 s while an action was in progress (`busy_check`), so even a policy that
  wanted to wait longer was quantised to that cadence. Jason's gaps are 54 % under a second
  (bursts of clicks) with a long tail past a minute, and almost nothing at 15 s.
* **`nearest_target_rate`** was 0.99 because the converter dropped Walk-here clicks: Jason
  walks a few tiles between attacks (2.4 times an hour, about a fifth of his crab-area
  clicks), and those are the only clicks that do not land on the nearest interactable.
* **Median gap** was five times too long for the same reason as the pile: the clone never
  produced the sub-second bursts.

What changed, in order of effect:

1. **Delays are continuous and interruptible.** The policy's delay can be anything from 50 ms
   to 300 s. A pending delay is cut short when an event in `INTERRUPTS` fires (inventory full,
   target gone, bank open, shell ready, arrived, idle); the policy then re-decides with that
   event as the trigger and the elapsed time recorded. There is no fixed clock any more: after
   a click the policy is asked straight away how long to wait before the next one. The
   recordings are labelled the same way: the next input with its true delay if it comes
   before the next event, else WAIT until the event.
2. **Walk-here is an intent.** `WALK` targets an entity slot and walks next to it without
   interacting (`Activity.walk_near`), fires `idle` on arrival, and is logged with kind `""`
   so the metrics can count it (`walk_per_h`).
3. **The observation carries burst context.** Two new context fields, log time since the
   last input and a one-hot of the last intent (`N_CTX = 8 + N_INTENTS`), so the policy can
   tell "I clicked 300 ms ago" from "I have been idle for a minute" and produce the burst
   second click.

Rerunning the 100 % points of the efficiency curve on the new env and converter
(`results/efficiency_rhythm/`, one subset, same recordings and validation file):

| BC learns from | Progress/h | Worst AUC all (90 % CI) | Worst AUC held-out file | Worst detector | Reaction KS |
|---|---|---|---|---|---|
| Synthetic + Jason, before | 75,445 | 0.94 | 0.93 | feature GBT | 0.11 |
| **Synthetic + Jason, after** | 80,235 | **0.84** [0.78, 0.90] | **0.78** [0.70, 0.85] | sequence GRU | 0.10 |
| Jason only, before | 39,253 | 0.95 | 0.99 | feature GBT | 0.39 |
| Jason only, after | 52,341 | 0.92 [0.87, 0.96] | 1.00 (1 window) | feature GBT | 0.17 |
| Synthetic only, before | 76,431 | 0.93 | 0.93 | feature GBT | 0.07 |
| Synthetic only, after | 83,812 | 0.93 [0.88, 0.96] | 0.93 | sequence GRU | 0.14 |

The mixed clone is the first one under 0.9 without PPO or a detector in the reward, and the
feature detector is no longer the stronger of the two: the GRU is, which is what you expect
once the session-level histograms stop separating. The synthetic-only floor does not move,
because the scripted players set their own delays and never had the chunk clock; the gain
comes from BC now being able to copy Jason's bursts and walks. The Jason-only clone remains
a one-player overfit (its held-out score is one window and means nothing).

What is left on the list is smaller and different in kind: the clone still walks too rarely
(0.4/h vs 2.4), its reaction times are too regular (log-std 1.35 vs 1.9, the mixture head
collapses toward one component on 411 real inputs), and it checks tabs a little too often.
Those are policy-capacity and data questions rather than simulator clocks, and the detector
now needs several of them together to score above 0.7 on any single feature.
