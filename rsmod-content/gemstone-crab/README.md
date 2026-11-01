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

* **It has never been compiled against RSMod.** My sandbox can't download RSMod's game
  cache. Every API call was checked against RSMod's source, but expect a few compile
  errors on the first build. Paste them to me and I'll fix them.
* **Spawns from the cache.** If the cache's map already places a gemstone crab, the
  script deletes any crab it didn't spawn on that crab's first timer tick.
* **Approximate coordinates.** The tunnel exit tile is set to one tile south of each
  tunnel, and the crab is centred on the mine coordinates from the RuneLite plugins.
  Both may be off by a tile or two.
* **Health bar.** Hitpoints are set from the timer every tick. Whether the client's bar
  follows that without a separate health-bar update is unchecked.
* **Not modelled.** The crab's own 1-damage attacks, the "pushed out of the way" message
  when it surfaces under a player, and the Varlamore achievement diary hooks.
