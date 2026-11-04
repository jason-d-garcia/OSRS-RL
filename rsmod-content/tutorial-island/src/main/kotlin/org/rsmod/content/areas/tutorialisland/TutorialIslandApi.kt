package org.rsmod.content.areas.tutorialisland

import org.rsmod.game.entity.Player

/**
 * What other modules (the agent bridge) can do with the tutorial without depending on its script
 * class. The script fills these in at startup.
 */
object TutorialIslandApi {
    /** Puts [Player] at a stage: sets the varp and gives the items that stage expects to have. */
    internal var startHook: ((Player, Int) -> Unit)? = null

    /** Tells the tutorial a side-panel tab was opened (a client sends this as a button click). */
    internal var tabHook: ((Player, TutorialRules.Tab) -> Unit)? = null

    /** Reads the player's stage (the `tutorial` varp). */
    internal var stageHook: ((Player) -> Int)? = null

    fun start(player: Player, stage: Int) {
        startHook?.invoke(player, stage) ?: error("tutorial-island module is not loaded")
    }

    fun openTab(player: Player, tab: TutorialRules.Tab) {
        tabHook?.invoke(player, tab)
    }

    fun stage(player: Player): Int = stageHook?.invoke(player) ?: 0
}
