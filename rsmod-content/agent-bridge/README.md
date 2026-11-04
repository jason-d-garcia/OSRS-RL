# Agent bridge for RSMod

Server-side RSMod content module that lets a trained policy play on your private server. An
agent (Python, `osrs_rl/rsmod_bridge.py`) connects over a local socket, gets one player's state
every game cycle and sends back the inputs a client could send: click an NPC or object option,
or walk to a tile. Those inputs go through the same routing and interaction code as client
packets, so a bot can do nothing a player can't.

It is not a RuneLite plugin. RuneLite is only how you watch (see "Watching a bot" below).

Two ways to get a player:

* **Headless bot** (`spawn`). The bridge logs in an account with no client. It is a normal
  player on the server: it saves, levels up, and anyone logged in sees it walk and fight.
* **Your own character** (`attach`). The agent drives a player who is already logged in through
  a real client, for example you in RuneLite, so you watch the policy play from its own camera.

Tested on RSMod main (Sep 2025, revision 233) in a headless Linux run, Nov 2026: a scripted bot
and the synthetic human ran full 30-minute episodes through `OsrsEnv` (attack, burrow, mine the
shell, tunnel to the next mine), about 500 to 800 game cycles a second in lockstep. Not tested
yet: watching a headless bot from a real client, and `attach`.

## Install

From the root of your RSMod clone, with `gemstone-crab` installed as in its README:

```
cp -r <this repo>/rsmod-content/agent-bridge content/other/agent-bridge
git apply <this repo>/rsmod-content/patches/tick-interval.patch    # optional, see below
gradlew install --console=plain
gradlew run --console=plain
```

The log says `Agent bridge listening on 127.0.0.1:45600`. Environment variables:

| Variable | Default | |
|---|---|---|
| `OSRSRL_AGENT_PORT` | `45600` | socket port (not 43600: RSProx uses that one for its local web server) |
| `OSRSRL_AGENT_BIND` | `127.0.0.1` | keep it on localhost |
| `OSRSRL_LOCKSTEP_TIMEOUT_MS` | `30000` | a lockstep agent that goes quiet this long is switched to real time, so the server never hangs |
| `OSRSRL_AGENT_BRIDGE` | | `off` disables the module |
| `RSMOD_TICK_MS` | `600` | needs the patch; see below |

## Fast headless runs: the tick length

RSMod's game loop has the 600 ms tick hard-coded (`GameService.GAME_TICK_INTERVAL`).
`patches/tick-interval.patch` makes it read `RSMOD_TICK_MS` (or `-Drsmod.tickMs`), and 0
means "no sleep, next cycle straight away". Everything in the game logic counts cycles, not
milliseconds, so the rules don't change; only the wall clock does.

* For training or evaluation: `RSMOD_TICK_MS=0 gradlew run`, and connect the agent in lockstep.
  The server then waits for the agent at the end of every cycle and runs as fast as the agent
  answers. 30 minutes of game time takes 4 to 9 seconds with the scripted bot.
* With anyone logged in through a real client, leave it at 600. The client animates and
  interpolates on its own 600 ms clock, so a faster server looks like fast-forward at best.

## Running a policy against it

```
python run_server.py --controller scripted --episodes 4                      # plumbing check
python run_server.py --controller policy --policy results/gemcrab/policies.pt --which rl_kl \
    --episodes 8 --recordings ~/.runelite/osrs-rl-recordings                   # score vs your play
```

`run_server.py` plays 30-minute episodes, writes each one in the RuneLite recorder's format
(`results/server/recordings/bot-*.jsonl.gz`), and with `--recordings` trains the detectors on
your real windows against the server bot's windows (the same cross-validation as
`run_real.py`). In code, the server is a drop-in for the simulator:

```python
from osrs_rl.env import OsrsEnv
from osrs_rl.rsmod_bridge import RemoteGemCrab
env = OsrsEnv("gemcrab", sim=RemoteGemCrab(name="agent1", lockstep=True))
```

The world is the server's: `reset()` moves the bot back next to the crab and clears gems from
its inventory, but the crab keeps its own timer and other players are whoever is online.

## Watching a bot

1. Run the server at normal speed (`RSMOD_TICK_MS` unset).
2. Connect RuneLite to it through RSProx as in `docs/server-setup.md`. RSProx custom targets
   don't work on macOS, and the RSProx target file points the client at `localhost`, so on a
   Mac run the whole stack (server, agent, RSProx, RuneLite) on a Windows or Linux machine or
   VM. Headless training and scoring need none of this and run fine on the Mac.
3. `python run_server.py --controller policy --policy ... --realtime`
4. Log in with any name and go to the crab (`::tele 0 19 49 57 37` is the north mine; the bot
   spawns at whichever mine the crab is at). The bot is the player called `agent1`.

To watch from the bot's own camera instead, log in as yourself and run with `--attach <your
name>`: the policy then drives your character.

**Recording.** The RuneLite recorder records the client's own player, so it can't record a
headless bot, and with `--attach` it would see the tick snapshots but no clicks (the inputs come
from the server, not the mouse). That is why `run_server.py` writes the recorder-format file
itself, from the server side: `recordings.crab_windows` and the detectors read it exactly like
your own recordings. There is no mouse movement in it, because the policy doesn't produce any.

## Protocol

Newline-delimited JSON over TCP. One connection drives one player.

Agent to server (`op`):

| op | fields | |
|---|---|---|
| `hello` | | replies `hello` with the mines (crab, tunnel and exit tiles) and crab timings |
| `spawn` | `name`, `lockstep`, `mine`, `spread`, `attack`, `strength`, `defence`, `hitpoints`, `mining`, `weapon`, `pickaxe`, `run`, `xp_rate` | log in a headless bot; replies `spawned`. Defaults: 80 combat stats, dragon scimitar, rune pickaxe, run on, 1x xp |
| `attach` | `name`, `lockstep`, `loadout` | drive a logged-in player; replies `spawned` |
| `step` | `actions: [...]` | inputs for the next cycle; in lockstep this also lets the cycle run |
| `act` | `action` | one input for the next cycle without ending a lockstep wait |
| `reset` | `mine`, `spread`, `clear_gems` | back next to the crab; replies `reset_done` |
| `lockstep` | `on` | switch modes |
| `logout` | | a headless bot logs out; an attached player is left alone |

Actions (`type`): `attack_crab`, `mine_shell`, `tunnel`, `walk` (`x`, `z`), `walk_rel`
(`dx`, `dz`), `npc` (`slot`, `op` as a name like `"Attack"` or a number 1-5), and `camera`,
`check_tab`, `wait`, `noop`, which the server never sees and only accepts so the agent's log is
complete.

Server to agent: a `tick` message at the end of every cycle with `cycle`, `x`, `z`, `level`,
`run_energy` (0-100), `moving`, `delayed`, `attacking`, `interaction`, `levels`, `xp`, `inv`,
`gems`, `crab` (mine, centre tile, age in ticks, visible, your damage and rank), `shells`
(mine, ticks left, whether you may mine it and whether you have), `players` within 20 tiles,
the game `messages` the player got that cycle (headless bots only), and `results`, one per
action applied, with `ok` and a `why` when it did nothing.
