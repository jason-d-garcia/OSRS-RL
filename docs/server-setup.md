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
| Gemstone crab | **not scripted**: the NPC and objects exist in the cache only |
| Mining | **not implemented** (needed for the crab shell) |
| Varlamore / Tlati Rainforest area scripts (cave tunnels) | not implemented |

So Draynor willows can run on the server as soon as it is up. Gem crabs need new
RSMod content scripts: the crab's timer-based health and no-retaliation behaviour,
damage tracking with the top-16 rule, burrow and resurface at the next mine, the
shell (mine for 3 gems), and the cave shortcut. That is the next server task, along
with the headless agent plugin that lets the trainer drive bot players over a
socket with the same observation and action schema as `osrs_rl/env.py`.

## Notes

* RSMod downloads Jagex's game cache. Keep the server private and for your own
  research use.
* RSProx launches RuneLite with its own bootstrap. Whether it can also load the
  developer-mode recorder from `recorder/` is not checked yet. Until it is, record
  mouse data on live (which you already do), and the server-side agent plugin will
  log every connected player's ticks and clicks from the server side. Mouse
  movement can only come from the client.
