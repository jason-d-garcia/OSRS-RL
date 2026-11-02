"""Game constants used by the simulator.

Values marked RSMOD are read from RSMod v2 (github.com/rsmod/rsmod, revision 233
content): ``content/skills/woodcutting`` for willows. Values marked WIKI come from
the OSRS wiki. Values marked APPROX are guesses to be replaced by measurements
from tick traces on the real server; each one is a calibration target.
"""

# --- Woodcutting: willows -------------------------------------------------
WILLOW_LEVEL_REQ = 30  # RSMOD TreeLocs.willow levelrequire
WILLOW_XP = 67.5  # RSMOD skill_xp
WILLOW_DESPAWN_TICKS = 50  # RSMOD despawn_time (ticks of active cutting)
WILLOW_RESPAWN_TICKS = 15  # RSMOD respawn_time
WC_FIRST_ROLL_DELAY = 3  # RSMOD Woodcutting.kt: actionDelay = mapClock + 3
WC_ROLL_INTERVAL = 4  # RSMOD: one roll every 4 ticks
# RSMOD WoodcuttingRates.willow_tree_axes (low, high) for statRandom
WILLOW_AXE_RATES = {
    "steel": (32, 100),
    "mithril": (40, 125),
    "adamant": (48, 150),
    "rune": (56, 175),
}
INVENTORY_SLOTS = 28

# Draynor layout in a local frame, origin near the bank booths. APPROX: shape
# matches Draynor (booths in a row, willows along the river to the south) but
# tile positions need to be checked against the real map.
DRAYNOR_BOOTHS = [(0, 10), (1, 10), (2, 10), (3, 10)]
DRAYNOR_WILLOWS = [(-4, -6), (-1, -8), (2, -7), (-7, -9), (4, -10)]
DRAYNOR_START = (2, 8)

# --- Gemstone crab ---------------------------------------------------------
CRAB_ALIVE_TICKS = 1000  # WIKI: stays about 10 minutes before burrowing
CRAB_SHELL_TICKS = 150  # WIKI: shell stays 90 seconds
CRAB_TOP_N = 16  # WIKI: top 16 damage dealers can mine the shell
CRAB_GEMS_PER_SHELL = 3  # WIKI
CRAB_N_MINES = 3  # WIKI: three mines in the Tlati Rainforest
CAVE_TRANSIT_TICKS = 4  # APPROX
SHELL_MINE_TICKS = (3, 8)  # APPROX: range of ticks to mine the shell
ATTACK_SPEED_TICKS = 4  # REAL: 4 ticks between XP drops in 97% of cases (dragon scimitar)
CRAB_ACCURACY = 0.97  # REAL: 4-tick XP drops on 97% of attacks in 9.4 h of recordings (0-damage hits give none)
XP_PER_DAMAGE = 4.0 + 4.0 / 3.0  # combat style + hitpoints
GEM_VALUE = 150.0  # progress units per uncut gem (APPROX, roughly a mixed gem in XP-equivalent)
CRAB_OTHER_PLAYERS = (6, 30)  # APPROX: other players at the crab
# Local frame for each mine: crab position and cave entrance.
CRAB_MINES = [
    {"crab": (0, 0), "cave": (6, -5)},
    {"crab": (0, 0), "cave": (-7, 3)},
    {"crab": (0, 0), "cave": (5, 6)},
]
