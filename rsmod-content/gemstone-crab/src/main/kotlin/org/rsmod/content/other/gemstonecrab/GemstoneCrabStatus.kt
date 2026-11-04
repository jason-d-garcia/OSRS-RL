package org.rsmod.content.other.gemstonecrab

import org.rsmod.game.entity.Npc

/**
 * Read-only view of the crab for other modules (the agent bridge reads it to build observations).
 *
 * Only the game thread writes or reads it: [scripts.GemstoneCrab] updates it as the crab spawns, is
 * hit and burrows, and readers look at it from game-cycle hooks.
 */
object GemstoneCrabStatus {
    /** Index into [GemstoneCrabRules.MINES] of the mine the crab is at now. */
    var mineIndex: Int = 0
        internal set

    /** The live crab, or null in the moment between burrowing and resurfacing. */
    var crab: Npc? = null
        internal set

    /** Map clock cycle the current crab spawned on. */
    var spawnCycle: Int = 0
        internal set

    /** Damage dealt to the current crab, by display name. */
    val damage: Map<String, Int>
        get() = mutableDamage

    /** Cycle of each player's most recent damaging hit on the current crab, by display name. */
    val lastHitCycle: Map<String, Int>
        get() = mutableLastHit

    /** Live shells: who may mine each one, who already has, and when it appeared. */
    val shells: List<Shell>
        get() = mutableShells

    class Shell(val npc: Npc, val mineIndex: Int, val eligible: Set<String>, val spawnCycle: Int) {
        val mined: Set<String>
            get() = mutableMined

        internal val mutableMined = mutableSetOf<String>()
    }

    internal val mutableDamage = mutableMapOf<String, Int>()
    internal val mutableLastHit = mutableMapOf<String, Int>()
    internal val mutableShells = mutableListOf<Shell>()
}
