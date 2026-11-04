# Tutorial Island: first recorded run (4 Nov 2026)

One live run by Jason on a hardcore world, recorder schema 6, 10.2 minutes from login to
Death's office. Stage times come from the tutorial progress varp (`tut` on each tick).

| Section | Progress values | Time |
|---|---|---|
| Character creator (name and look) | 1 | 3 min 22 s |
| Gielinor Guide | 2–20 | 23 s |
| Survival Expert (fish, wood, fire, cook) | 30–120 | 49 s |
| Master Chef (dough, bread) | 130–200 | 42 s |
| Quest Guide | 220–250 | 16 s |
| Mining Instructor (mine, smelt, smith) | 260–360 | 61 s |
| Combat Instructor (melee, ranged) | 370–500 | 70 s |
| Bank and Account Guide | 510–550 | 57 s |
| Brother Brace (prayer) | 560–620 | 39 s |
| Magic Instructor | 630–671 | 33 s |
| Hardcore only: Death, then Death's office | 680–1000 | 16 s |

What it shows:

* **The dialogue logging works on live.** 136 dialogue records; 101 lines were moved past
  by a click, the space bar or a number key. The Tutorial Island instruction box
  (group 263) was not captured; the recorder now walks the whole interface tree for it.
* **An experienced player skips, not reads.** Median 320 ms per line (17 words), about
  20 ms per word, and time per line does not grow with length (correlation -0.27). Reading
  time is a signal for new players, not for Jason; for his runs the signal is the skip
  rhythm (space-bar cadence) and the action timing between guides.
* **Most continues are the space bar** (97 of 104), so the reader counts keys as well as
  clicks.
* **Bystanders:** 21 other players were seen even on a quiet island, up to 11 at once.

## The bot on the private server

`python run_tutorial.py` plays the first two sections on RSMod (`rsmod-content/tutorial-island`)
with the scripted bot in `osrs_rl/tutorial.py`, using delays fitted to the run above. 80 of 80
episodes finished, with no refused actions (`results/tutorial/runs.json`, seed 2).

| Section | Bot (median of 80) | Jason |
|---|---|---|
| Gielinor Guide | 12 s | 23 s |
| Survival Expert | 56 s | 49 s |

* **The guide is quicker for the bot** because the module's guide has fewer lines than live
  (the live guide talks more before and after the settings step), not because the bot skips
  faster: its median per line is 330 ms against Jason's 317 ms.
* **The Survival Expert is a little slower.** Chopping (7 s) and lighting the fire (7 s) are
  fitted to Jason's 10 s and 7 s. Most of the rest is fishing, a 19% roll every 4 ticks, so a
  run varies with luck.
