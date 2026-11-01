package org.rsmod.content.other.gemstonecrab

/**
 * The gemstone crab's rules with no RSMod types in them, so they can be unit tested on their own.
 *
 * Sources: the OSRS wiki (Gemstone Crab page, Drop Rate Project numbers) and the coordinates used
 * by the Gemstone Crab RuneLite plugins and RuneLite's world map.
 */
object GemstoneCrabRules {
    /** Ticks the crab stays up before it burrows (about 10 minutes). */
    const val ALIVE_TICKS = 1000

    /** Ticks the shell stays after the crab burrows (90 seconds). */
    const val SHELL_TICKS = 150

    /** Players with the most damage on a crab who may mine its shell. */
    const val TOP_N = 16

    /** Players named as "top crab crushers" when the crab burrows. */
    const val ANNOUNCE_TOP = 3

    /** With fewer damage dealers than this, only the single top player is announced. */
    const val ANNOUNCE_MIN_PLAYERS = 5

    /** Gems from one shell. */
    const val GEMS_PER_SHELL = 3

    /** Combat xp multiplier, in thousandths: 12.5% less xp than a normal target. */
    const val XP_MULTIPLIER = 875

    /**
     * The crab's real hitpoints are hidden and player damage does not lower its health bar, so it
     * is given a pool large enough that no hit can finish it, and hits are refunded.
     */
    const val HITPOINTS_POOL = 10_000

    /** The lowest the pool is allowed to fall to, so xp is never capped by remaining hitpoints. */
    const val MIN_HITPOINTS = 2_000

    /** How often, in ticks, the crab's ai timer checks its own lifetime. */
    const val TIMER_INTERVAL = 1

    /** Ticks a player spends crawling through the tunnel (approximate). */
    const val TUNNEL_TICKS = 3

    /** Ticks of swinging a pick before the shell gives its gems. */
    const val MINE_TICKS = 4

    /** A mine in the Tlati Rainforest: where the crab surfaces and where its tunnel comes out. */
    data class Mine(val name: String, val crabX: Int, val crabZ: Int, val tunnelX: Int, val tunnelZ: Int)

    val MINES: List<Mine> =
        listOf(
            Mine("north", crabX = 1273, crabZ = 3173, tunnelX = 1278, tunnelZ = 3168),
            Mine("east", crabX = 1353, crabZ = 3112, tunnelX = 1351, tunnelZ = 3124),
            Mine("south", crabX = 1239, crabZ = 3043, tunnelX = 1246, tunnelZ = 3036),
        )

    /** The crab moves to one of the other two mines. [roll] is a random int in 0 until 2. */
    fun nextMine(current: Int, roll: Int): Int {
        require(roll in 0 until MINES.size - 1) { "roll must be in 0 until ${MINES.size - 1}" }
        return (current + 1 + roll) % MINES.size
    }

    /**
     * Health bar shown for the crab: it drains with time, not damage. Returns hitpoints out of
     * [HITPOINTS_POOL] for an age in ticks.
     */
    fun hitpointsForAge(age: Int): Int {
        val remaining = (ALIVE_TICKS - age).coerceIn(0, ALIVE_TICKS)
        val scaled = (HITPOINTS_POOL.toLong() * remaining / ALIVE_TICKS).toInt()
        return scaled.coerceAtLeast(MIN_HITPOINTS)
    }

    /**
     * Players allowed to mine the shell: the [TOP_N] with the most damage, ties broken by who
     * reached that damage first ([firstHitCycle], lower is earlier). Players with no damage are
     * never eligible.
     */
    fun <K> eligible(damage: Map<K, Int>, firstHitCycle: Map<K, Int>): List<K> =
        damage.entries
            .filter { it.value > 0 }
            .sortedWith(compareByDescending<Map.Entry<K, Int>> { it.value }.thenBy { firstHitCycle[it.key] ?: Int.MAX_VALUE })
            .take(TOP_N)
            .map { it.key }

    /** Who to name as top crab crushers, from an already ranked list. */
    fun <K> announced(ranked: List<K>): List<K> =
        if (ranked.size < ANNOUNCE_MIN_PLAYERS) ranked.take(1) else ranked.take(ANNOUNCE_TOP)

    enum class Gem(val objName: String, val displayName: String) {
        OPAL("uncut_opal", "opal"),
        JADE("uncut_jade", "jade"),
        RED_TOPAZ("uncut_red_topaz", "red topaz"),
        SAPPHIRE("uncut_sapphire", "sapphire"),
        EMERALD("uncut_emerald", "emerald"),
        RUBY("uncut_ruby", "ruby"),
        DIAMOND("uncut_diamond", "diamond"),
        DRAGONSTONE("uncut_dragonstone", "dragonstone"),
    }

    /** Weights out of 32 per gem (Drop Rate Project); dragonstone is a separate 1/500 roll. */
    private val GEM_WEIGHTS =
        listOf(
            Gem.OPAL to 9,
            Gem.JADE to 9,
            Gem.RED_TOPAZ to 6,
            Gem.SAPPHIRE to 3,
            Gem.EMERALD to 2,
            Gem.RUBY to 2,
            Gem.DIAMOND to 1,
        )

    const val DRAGONSTONE_ONE_IN = 500

    /**
     * Rolls one gem. [random] returns a uniform int in `0 until n` for the n it is given.
     */
    fun rollGem(random: (Int) -> Int): Gem {
        if (random(DRAGONSTONE_ONE_IN) == 0) {
            return Gem.DRAGONSTONE
        }
        var pick = random(GEM_WEIGHTS.sumOf { it.second })
        for ((gem, weight) in GEM_WEIGHTS) {
            if (pick < weight) {
                return gem
            }
            pick -= weight
        }
        error("unreachable: weights sum mismatch")
    }

    // Messages, as the live game words them.
    const val MSG_BURROW = "The gemstone crab burrows away, leaving a piece of its shell behind."
    const val MSG_ELIGIBLE = "You gained enough understanding of the crab to mine from its remains."
    const val MSG_NOT_ELIGIBLE = "Your understanding of the gemstone crab is not great enough to mine its shell."
    const val MSG_SWING = "You swing your pick at the crab shell."
    const val MSG_NO_PICK = "You need a pickaxe to mine the shell."
    const val MSG_NO_PICK_LEVEL = "You do not have a pickaxe which you have the Mining level to use."
    const val MSG_ALREADY_MINED = "You have already mined this shell."
    const val MSG_PUSHED = "The burrowing crab forces you out of its way.."

    fun msgGem(gem: Gem): String = "You mine an uncut ${gem.displayName} from the crab shell."

    fun msgTopCrushers(names: List<String>): String =
        when (names.size) {
            0 -> ""
            1 -> "Top crab crusher: ${names[0]}."
            else -> "Top crab crushers: ${names.joinToString(", ")}."
        }
}
