# PvP toggle for RSMod

A dev command for a private server. `::pvp` adds the "Attack" right-click option on other
players for whoever types it, and `::pvp off` removes it. RSMod's PvP combat already works
everywhere; its login script just hides that option. The command needs admin mod level, which
the `dev` realm gives everyone.

It isn't saved, so type it again after each login. The headless bots from `agent-bridge` don't
fight back, and a bot that dies respawns at the realm spawn, away from the crab, until the
agent's next `reset`.

## Install

```
cp -r <this repo>/rsmod-content/pvp-toggle content/other/pvp-toggle
```
