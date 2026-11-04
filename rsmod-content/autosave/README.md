# Autosave for RSMod

RSMod saves a character only on logout, so a crash or a hard kill loses everything since the
last login. This module saves every logged-in player every few minutes of wall-clock time:
5 by default, or set `RSMOD_AUTOSAVE_MINUTES`. It uses wall-clock time, not game cycles, so it
behaves the same with `RSMOD_TICK_MS=0`.

The save runs on RSMod's saving thread and reads the live player, so it can catch a player
mid-change. On a single-player dev server that is a fair trade for not losing a session.
Varps and varbits (quest state, prayer unlocks) aren't in RSMod's save schema, so they don't
persist either way.

## Install

```
cp -r <this repo>/rsmod-content/autosave content/other/autosave
```
