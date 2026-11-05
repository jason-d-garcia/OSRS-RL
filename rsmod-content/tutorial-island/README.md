# Tutorial Island for RSMod

Server content for the start of Tutorial Island, written as an RSMod v2 content module
(revision 233): the Gielinor Guide and the Survival Expert, up to the gate into the Master
Chef's area. RSMod has no Tutorial Island content of its own; the island's map, objects and
NPC types are in the game cache, so this module adds what makes it work.

* **Progress** is the cache's `tutorial` varp (281) with the live game's values, read off a
  recorded live run (`docs/tutorial-island.md`): 2 talk to the guide, 3 open settings, 7 talk
  again, 10 open the door, 20 talk to the Survival Expert, 30 open the inventory, 40 catch
  shrimp, 50 open skills, 60 talk again, 70 chop a tree, 80 light a fire, 90 cook the shrimp,
  120 go through the gate, 130 through it (the end of this module). The rules, stage values,
  hints, gates and success rates are in `TutorialRules.kt`, which has no RSMod types and is
  unit tested on its own.
* **NPCs**: the Gielinor Guide, the Survival Expert, two fishing spots and the Master Chef, at
  the tiles the live run found them (`npcs.toml`).
* **Dialogue**: the two instructors' lines from the live run, plus the guide's "how experienced
  are you?" choice. Instructors re-give a net, axe or tinderbox you've lost.
* **Doors and gates**: the guide's door opens from stage 10 and the Survival Expert's gate from
  stage 120. Going through opens it for 3 ticks and walks you across.
* **Skills, island-sized**: small-net fishing (shrimp), chopping the island's trees (logs),
  tinderbox on logs (a fire for 120 ticks; you step west, or east, south or north if west is blocked), raw shrimp on the fire (never burns).
  XP is the live game's. RSMod's own fishing, firemaking and cooking modules are empty, and its
  woodcutting doesn't cover the island's tree type.
* **Tabs**: opening settings, inventory and skills moves the matching stage on. The agent bridge
  does this with its `tab` action.

Not covered yet: the character creator (players start at stage 2), the instruction box and hint
arrows a real client shows (hints are sent as game messages instead), hiding tabs until they're
unlocked, and everything from the Master Chef on.

## Install

From the root of your RSMod clone:

```
cp -r <this repo>/rsmod-content/tutorial-island content/areas/tutorial-island
gradlew install --console=plain
gradlew run --console=plain
```

`install` packs the NPC spawns into the map, so it is needed again after editing `npcs.toml`.

## Playing it

Log in with any account and type `::tutorial` to go to the start of the island at stage 2, or
`::tutstage <value>` (for example `::tutstage 70`) to jump to a stage with the items it expects.
`::tutorial off` clears progress. Existing accounts are never moved to the island on login.

A headless bot plays it with `python run_tutorial.py` (see the agent bridge README for the
`tutorial` scene and the actions it needs).

## Not checked yet

* **With a real client.** Everything here was run headless through the agent bridge. Whether a
  real client's tab clicks reach the server (the module listens for the tab buttons on all three
  game frames) is untested; `::tutstage` gets you past a tab step if they don't.
* **Door and gate geometry.** Going through walks you to the tile across the wall the door is
  on; a gate whose hinge tile differs from the clicked half may need a tile's adjustment.
