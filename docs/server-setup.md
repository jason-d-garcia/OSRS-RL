# Private server on your PC: RSMod + RSProx + RuneLite

This gets an RSMod server (OSRS revision 233) running on your own PC and your
normal RuneLite connected to it through RSProx. Windows or Linux; RSProx custom
targets do not work on macOS.

Everything stays on `localhost`. Do not open the port to the internet until you
want other players on it.

## 1. Java 21

```
winget install --id=EclipseAdoptium.Temurin.21.JDK -e
```

(On Linux: `sdk install java 21.0.7-tem` with SDKMAN.) Check with `java -version`.

## 2. RSMod

```
git clone https://github.com/rsmod/rsmod.git
cd rsmod
gradlew install --console=plain
gradlew run --console=plain
```

The first `install` downloads the revision-233 game cache and generates the
server's RSA key. It prints a block like:

```
Place these keys in the client (find BigInteger("10001" in client code):
--------------------
exponent: 10001
modulus: 8f3c...e1
--------------------
Additionally, these details have been written to .../client.key
```

Copy the **modulus** line; RSProx needs it. If you lose it, it is in that
`client.key` file. `gradlew run` then starts the game server on port 43594. Leave it
running.

## 3. RSProx

Download the installer from https://github.com/blurite/rsprox/releases and run it
once so it creates `%USERPROFILE%\.rsprox\`. Then create
`%USERPROFILE%\.rsprox\proxy-targets.yaml`:

```yaml
config:
  - name: OSRS-RL Local
    jav_config_url: https://client.blurite.io/jav_local_233.ws
    revision: 233
    modulus: PASTE_THE_MODULUS_FROM_STEP_2
```

The `jav_local_<revision>.ws` files are maintained by the RSProx author and point
the client at `localhost`. Restart RSProx after saving.

## 4. Connect

In RSProx: **New Session** > Client Type **RuneLite** > Proxy Target
**OSRS-RL Local** > **Launch Session**. The first launch takes a while to download
and patch. Log in with any username and password; RSMod creates the account.

## What works on the server today

From a look at RSMod's `content/` folder (late 2025 snapshot):

| Content | State |
|---|---|
| Woodcutting (all trees, axes, despawn timers) | implemented |
| Bank interface | implemented |
| Combat, special attacks | implemented |
| Gemstone crab, shell and tunnels | in this repo: `rsmod-content/gemstone-crab` (see its README to install) |
| Mining skill | not implemented, and not needed: the shell only needs a pickaxe |

So Draynor willows can run on the server as soon as it is up, and gem crabs once
the `gemstone-crab` module is copied in. The agent plugin that lets the trainer drive
players over a socket is `rsmod-content/agent-bridge`; see its README.

## Bots on your server, and watching them in RuneLite

The agent plugin runs inside the server, not in RuneLite. RuneLite stays an ordinary client
that you use to look at the world:

* Install `rsmod-content/agent-bridge` next to `gemstone-crab` and run the server.
* `python run_server.py --controller policy --policy <policies.pt> --realtime` spawns a bot
  called `agent1` at the crab and plays it at normal speed.
* Connect RuneLite through RSProx as above, log in as anyone, and walk or `::tele` to the
  crab; you see the bot fighting like any other player. With `--attach <your name>` the
  policy drives your own character instead.

For training and scoring you don't need a client at all: run the server with
`RSMOD_TICK_MS=0` (after applying `rsmod-content/patches/tick-interval.patch`) and the agent in
lockstep, and 30 minutes of game time take a few seconds. `run_server.py` writes each bot
session in the recorder's format itself, so it is scored with the same code as your
recordings.

## Notes

* RSMod downloads Jagex's game cache. Keep the server private and for your own
  research use.
* RSProx launches RuneLite with its own bootstrap. Whether it can also load the
  developer-mode recorder from `recorder/` is not checked yet. Until it is, record
  mouse data on live (which you already do). Bot sessions don't need it: the recorder
  only sees the client's own player, and a bot's inputs come from the server, so
  `run_server.py` writes the bot's session in the recorder format from the server side.
  Mouse movement can only come from a client, and the policy produces none.
