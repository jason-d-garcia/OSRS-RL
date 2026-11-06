# OSRS-RL Recorder (RuneLite plugin)

A passive RuneLite plugin that records how you play: mouse movement, clicks, keys,
the menu option each click picked, camera moves, window focus, and a snapshot of
the game state every tick. It is the human half of the training data for the
humanlike bots on the private server.

**It only listens.** It never sends input, never changes menus and never consumes
an event. Every listener hands the event back unchanged.

## What it records

One file per login session: `~/.runelite/osrs-rl-recordings/session-YYYYMMDD-HHMMSS.jsonl.gz`
(`%USERPROFILE%\.runelite\osrs-rl-recordings\` on Windows). Each line is one JSON object
with `t` (epoch milliseconds) and `type`:

| type | fields | when |
|---|---|---|
| `session_start` | `schema`, `world`, `canvas_w`, `canvas_h`, `stretched`, `mouse_hz`, `typed_text` | login |
| `ui_layout` | `canvas_w`, `canvas_h`, `resized`, `stretched`, and `[x, y, w, h]` boxes: `viewport`, `hover_text`, `xp_drops`, `xp_counter`, `xp_orb`, `minimap`, `inventory`, `chatbox` | session start, then whenever the layout changes |
| `mouse_move`, `mouse_drag` | `x`, `y` (canvas pixels) | every move, thinned to `mouse_hz` (100 Hz default) |
| `mouse_press`, `mouse_release` | `x`, `y`, `button` (1 left, 2 middle, 3 right) | every button |
| `wheel` | `x`, `y`, `rot` | scroll |
| `key_press`, `key_release` | `key` (`"char"` for letters/digits unless typed text is on), `code` | every key |
| `menu_click` | `option`, `target`, `action`, `id`, `p0`, `p1`, `item`, `x`, `y`, `tick`, plus `dx`, `dy` (target tile relative to you) for objects, ground items and NPCs, and `npc_index`, `npc_id` for NPCs | every menu action the client runs |
| `tick` | `tick`, `x`, `y`, `plane`, `anim`, `interacting`, `hp`, `prayer`, `stats`, `prayers`, `spec`, `spec_on`, `hover`, `run`, `yaw`, `pitch`, `camera_moved`, `focused`, `inv`, `worn`, `bank_open`, `tut`, `npcs[]`, `players[]`, `objects[]` | every 600 ms game tick |
| `dialogue` | `kind` (`npc`, `player`, `options`, `sprite`, `message`, `tutorial`, or null when it closes), `speaker`, `text`, `chars`, `words`, `options[]` | whenever the chatbox dialogue or Tutorial Island instruction changes |
| `xp` | `skill`, `xp`, `level` | XP drops |
| `hitsplat` | `on` (the NPC, `me`, or a bystander `pid`), `hit_type`, `amount`, `mine` | damage shown on anyone in view |
| `game_message` | `msg` | game/engine messages only, never player chat |
| `focus` | `focused` | alt-tab in or out |
| `widget_open`, `widget_close` | `group` | interfaces |
| `anim`, `interacting` | your animation / target changes | as they happen |
| `session_end` | `reason` | logout, pause, or plugin stopped |

`npcs[]` has `index` (matches `menu_click.npc_index`), `id`, `name`, `dx`, `dy` (tiles relative to you), `anim`, `hp_ratio`,
`hp_scale`, `targets_me`. `players[]` are anonymous: `pid` (a per-session number that stays
the same for one person from tick to tick, see below), offsets, animation and combat level,
plus `target_npc` (the `npcs[].index` they are interacting with) or `target_player` (-1 for
you, else their `pid`) when they are interacting with something, and from schema 5 what
everyone can see on them: `equip` (item ids for head, cape, amulet, weapon, torso, shield,
legs, hands, boots; 0 when empty), `overhead` (protect/overhead prayer icon), `skull`,
`hp_ratio`/`hp_scale` while their health bar shows, and `spot` (spot animations such as
spell impacts). Never names.

Own state added in schema 5: `stats` (boosted and real Attack, Strength, Defence, Ranged,
Magic), `prayers` (the active prayers by name), `spec` (special attack percent), `spec_on`,
and `hover`: the left-click action under the mouse at that tick (`option`, `target`, `id`
or `npc_index`, `item`, `kind`), with other players reduced to their `pid`. `objects[]` are game, wall and ground objects within the radius,
with `id`, `kind`, `dx`, `dy`; objects bigger than one tile give their south-west
corner (the tile a click on them reports) plus `w`, `h` in tiles.

Schema 2 (Nov 2026) added the click offsets and NPC index so every click can be matched
to the thing in the tick snapshot; schema 1 files only have scene `p0`/`p1`.
Schema 3 switched large objects in `objects[]` from their centre tile to their south-west
corner, so object clicks match exactly; in schema 2 files they are off by about half the
object's size.
Schema 4 (Nov 2026) added `pid` and `target_npc` / `target_player` to `players[]`, so the other
people at the crab can be followed across ticks and their state changes (started attacking,
hopped, left) used as observation-only trajectories. `pid` is assigned in order of first
sight within a session from the client's player slot and combat level; a slot reused by a
different person after a logout gets a new `pid` unless the combat levels match too. Files
before schema 4 have no `pid`, so bystanders there can only be matched by tile.
Schema 5 (Nov 2026) added the visible state of other players (`equip`, `overhead`, `skull`,
health bar, `spot`), `hitsplat` records, and the player's own `stats`, `prayers`, `spec` and
`hover`. Every new field is optional, so readers written for earlier schemas still work.
Schema 6 (Nov 2026) added `tut` (the Tutorial Island progress varp, 1000 once done) and
the `dialogue` record, so the time spent reading each line before clicking on can be
measured. Dialogue text is game content; your own name in it is replaced by `<name>`.
Schema 7 (Nov 2026) added the `ui_layout` record: where the client's interface sits on the
canvas, so video crops can be cut from the right place in any layout. Written right after
`session_start` and again whenever any value changes (checked each tick): `canvas_w`,
`canvas_h`, `resized` (resizable vs fixed mode), `stretched` (plus `stretched_w`/`stretched_h`
when on), and boxes as `[x, y, w, h]` in canvas pixels: `viewport` (the 3D game view),
`hover_text` (the mouseover text line at the viewport's top-left; its width is the viewport's,
the text itself is shorter), `xp_drops` (where drops rise), `xp_counter`, `xp_orb`, `minimap`,
`inventory`, `chatbox`. A box is left out while that interface is hidden or closed.

Privacy defaults: letters and digits are stored as `"char"` (timing only), other
players' chat and names are never stored, and the login screen is never recorded.
A 30-minute session is a few MB compressed.

## Running it

You need a JDK 11 or newer (Temurin 17 is fine). From this folder:

```
./gradlew run          # macOS / Linux
gradlew.bat run        # Windows
```

That builds the plugin and starts RuneLite in developer mode with the recorder
loaded. Turn it on under the plugin list ("OSRS-RL Recorder"); a game message
says where it is recording. Toggle "Recording" in its settings to pause.

**Jagex account login:** the developer-mode client cannot do the Jagex launcher
login by itself. Follow RuneLite's [Using Jagex Accounts](https://github.com/runelite/runelite/wiki/Using-Jagex-Accounts)
guide once: it launches RuneLite from the Jagex launcher with
`--insecure-write-credentials`, which saves `~/.runelite/credentials.properties`
for `gradlew run` to use. Keep that file private; delete it to revoke.

**What to check the first time:** the game message "OSRS-RL recorder: recording
to session-....jsonl.gz" appears after login; a file appears in the folder above
and grows while you play; logging out closes it. The plugin is passive and
follows RuneLite's plugin rules (no input injection, no menu changes, nothing
sent over the network, no other players' names).

## Recording Tutorial Island runs

Each run needs a fresh account on the live game. Create a new character under your Jagex
account, launch it once from the Jagex launcher with `--insecure-write-credentials` as above
so `gradlew run` logs into that character, turn the recorder on, and play the island
straight through the way you normally would. One run per sitting at a time (no second client
open); back-to-back runs are fine. `python -m osrs_rl.recordings <file>` shows the
tutorial progress reached and your median reading time per word.

The dialogue widget ids are raw ids (`231`, `217`, `219`, `193`, `229`, `263`) and have not
been checked on a live client yet. After the first run, check that the summary shows
`dialogues_read` above zero.

## What to record first

Your normal gem crab sessions, played the way you normally play: tabbing out,
AFKing, coming back late. That variety is the point. Aim for 5 to 10 hours in
sessions of 30 minutes or more. Then send the `.jsonl.gz` files (or drop them in
the project) and `python -m osrs_rl.recordings <file>` prints a summary.

## Notes

* Built against `net.runelite:client:latest.release`. `gradlew build` compiles
  cleanly against the RuneLite release of November 2026. It has not been run
  inside a live client yet.
* Mouse coordinates are canvas pixels as RuneLite reports them. With stretched
  mode on, `session_start.stretched` is true; keep the window size fixed within a
  session if you can.
