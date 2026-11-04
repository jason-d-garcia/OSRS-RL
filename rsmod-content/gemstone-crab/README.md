# Gemstone crab for RSMod

Server content for the gemstone crab in the Tlati Rainforest, written as an RSMod v2
content module (revision 233). It covers:

* **The crab.** One crab is up at a time, at the north, east or south mine. It lives
  about 10 minutes (1000 ticks). Its health bar drains with time, not damage, and players
  can't kill it. It gives 12.5% less combat XP than a normal target.
* **Damage tracking.** Damage is tracked per player. The top 16 by damage (ties go to
  whoever hit first) can mine the shell, and the top crushers are announced.
* **Burrowing.** The crab burrows with the live game's messages, leaves its shell for
  90 seconds and comes up at one of the other two mines.
* **The shell.** Each eligible player can mine it once, using any pickaxe they can use.
  It gives three uncut gems: opal 9/32, jade 9/32, red topaz 6/32, sapphire 3/32,
  emerald 2/32, ruby 2/32 and diamond 1/32, plus a separate 1/500 dragonstone roll.
  Mining the shell gives no XP and needs no Mining level beyond the pickaxe's, so the
  Mining skill itself does not need to exist on the server.
* **The tunnels.** The tunnel at any mine takes the player to the mine where the crab is now.

The rules, timings, gem table and messages are in `GemstoneCrabRules.kt`, which has no
RSMod types, so it is unit tested on its own (`src/test`). The RSMod wiring is in
`scripts/GemstoneCrab.kt` and the type references are in `configs/GemstoneCrabConfigs.kt`.

## Install

From the root of your RSMod clone:

```
cp -r <this repo>/rsmod-content/gemstone-crab content/other/gemstone-crab
gradlew install --console=plain
gradlew run --console=plain
```

`settings.gradle.kts` picks up any folder under `content/` that has a `build.gradle.kts`,
so nothing else needs registering. Then log in and type `::tele 0 19 49 57 37` to go to
the north mine. RSMod's teleport takes the level, the map square, and the tile within
the square, so that is (1273, 3173).

## Not checked yet

* **Compiled and run headless (Nov 2026, RSMod main, revision 233).** `gradlew install`
  and `gradlew run` succeed, the unit tests pass, and with temporary logging the crab
  spawned at the north mine on startup, its timer ran every tick, its hitpoints fell
  by 1000 every 100 ticks down to the 2000 floor, and it burrowed at tick 1000 and came
  up at the east mine. Nothing has been checked with a client logged in yet: hitting
  the crab, mining the shell and the tunnels are still untested.
* **Spawns from the cache.** No cache-placed crab showed up in the headless run. If
  one ever does, the script deletes any crab it didn't spawn on that crab's first timer tick.
* **Played by a headless bot (Nov 2026).** With the `agent-bridge` module, a scripted bot
  hit the crab, got the burrow and eligibility messages, mined the shell for three gems and
  took the tunnel to the next mine, over several 30-minute runs at all three mines. That
  run found the first tunnel exits (one tile south of each tunnel) were cut off from the crab
  at the north and south mines ("I can't reach that!"). The exits are now the tiles a player
  walks to when using each tunnel from the crab's side, measured on the server.
* **Approximate coordinates.** The crab is centred on the mine coordinates from the RuneLite
  plugins and may be off by a tile or two.
* **Health bar.** Hitpoints are set from the timer every tick. Whether the client's bar
  follows that without a separate health-bar update is unchecked.
* **Not modelled.** The crab's own 1-damage attacks, the "pushed out of the way" message
  when it surfaces under a player, and the Varlamore achievement diary hooks.
