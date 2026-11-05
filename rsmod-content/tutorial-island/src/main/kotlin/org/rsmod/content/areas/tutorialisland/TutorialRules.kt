package org.rsmod.content.areas.tutorialisland

/**
 * Tutorial Island rules with no RSMod types, so they can be unit tested on their own.
 *
 * Progress lives in the cache's `tutorial` varp (281), the same one the live client reads. The
 * stage values are the ones a live run went through (recorded 4 Nov 2026, see
 * `docs/tutorial-island.md` in the OSRS-RL repo). This module covers the island up to leaving the
 * Survival Expert's area: the Gielinor Guide and the Survival Expert.
 */
object TutorialRules {
    // ---- stages (values of the `tutorial` varp) ------------------------------------------

    /** Character creator. This server skips it and starts players at [TALK_TO_GUIDE]. */
    const val CHARACTER_CREATOR = 1
    const val TALK_TO_GUIDE = 2
    const val OPEN_SETTINGS = 3
    const val TALK_TO_GUIDE_AGAIN = 7
    const val OPEN_DOOR = 10
    const val TALK_TO_SURVIVAL = 20
    const val OPEN_INVENTORY = 30
    const val CATCH_SHRIMP = 40
    const val OPEN_SKILLS = 50
    const val TALK_TO_SURVIVAL_AGAIN = 60
    const val CHOP_TREE = 70
    const val LIGHT_FIRE = 80
    const val COOK_SHRIMP = 90
    const val LEAVE_SURVIVAL = 120

    /** Through the gate, on the way to the Master Chef. The end of what this module covers. */
    const val MEET_CHEF = 130

    /** The live game's value once the island is done. */
    const val COMPLETE = 1000

    val STAGES =
        listOf(
            TALK_TO_GUIDE,
            OPEN_SETTINGS,
            TALK_TO_GUIDE_AGAIN,
            OPEN_DOOR,
            TALK_TO_SURVIVAL,
            OPEN_INVENTORY,
            CATCH_SHRIMP,
            OPEN_SKILLS,
            TALK_TO_SURVIVAL_AGAIN,
            CHOP_TREE,
            LIGHT_FIRE,
            COOK_SHRIMP,
            LEAVE_SURVIVAL,
            MEET_CHEF,
        )

    fun isOnIsland(stage: Int): Boolean = stage in TALK_TO_GUIDE until COMPLETE

    /** The stage after [stage] in this module, or [stage] itself at the end. */
    fun next(stage: Int): Int {
        val i = STAGES.indexOf(stage)
        return if (i < 0 || i == STAGES.size - 1) stage else STAGES[i + 1]
    }

    // ---- what the instruction box says --------------------------------------------------

    fun hint(stage: Int): String =
        when (stage) {
            TALK_TO_GUIDE ->
                "Getting started: talk to the Gielinor Guide. Click on him to talk to him."
            OPEN_SETTINGS -> "Options menu: click on the flashing spanner icon to open your settings."
            TALK_TO_GUIDE_AGAIN -> "Talk to the Gielinor Guide again to continue."
            OPEN_DOOR -> "Moving on: it's time to meet your first instructor. Open the door."
            TALK_TO_SURVIVAL -> "Talk to the Survival Expert by the pond to continue."
            OPEN_INVENTORY ->
                "Viewing the items you were given: click on the flashing backpack icon."
            CATCH_SHRIMP -> "Fishing: click on the sparkling fishing spot to catch some shrimp."
            OPEN_SKILLS -> "Your skill stats: click on the flashing bar graph icon."
            TALK_TO_SURVIVAL_AGAIN -> "Talk to the Survival Expert to continue."
            CHOP_TREE -> "Woodcutting: click on one of the trees to chop it down."
            LIGHT_FIRE ->
                "Firemaking: use the tinderbox on the logs. Click the tinderbox, then the logs."
            COOK_SHRIMP -> "Cooking: use the raw shrimp on the fire."
            LEAVE_SURVIVAL -> "Well done! Follow the path and open the gate to continue."
            MEET_CHEF -> "Find your next instructor: the Master Chef, through the door ahead."
            else -> ""
        }

    // ---- gates ---------------------------------------------------------------------------

    enum class Gate(val minStage: Int, val passedStage: Int) {
        /** The Gielinor Guide's front door. */
        GUIDE_DOOR(OPEN_DOOR, TALK_TO_SURVIVAL),
        /** The gate out of the Survival Expert's area. */
        SURVIVAL_GATE(LEAVE_SURVIVAL, MEET_CHEF),
    }

    fun canPass(gate: Gate, stage: Int): Boolean = stage >= gate.minStage

    /** The stage after going through [gate] at [stage]. Going through again changes nothing. */
    fun afterPassing(gate: Gate, stage: Int): Int =
        if (stage == gate.minStage) gate.passedStage else stage

    fun refusal(gate: Gate): String =
        when (gate) {
            Gate.GUIDE_DOOR -> "You need to talk to the Gielinor Guide before you go outside."
            Gate.SURVIVAL_GATE -> "You need to finish the Survival Expert's lessons first."
        }

    // ---- tabs ----------------------------------------------------------------------------

    enum class Tab(val stone: Int) {
        COMBAT(0),
        SKILLS(1),
        QUESTS(2),
        INVENTORY(3),
        EQUIPMENT(4),
        PRAYER(5),
        MAGIC(6),
        SETTINGS(11),
    }

    /** The stage after opening [tab] at [stage]. */
    fun afterTab(tab: Tab, stage: Int): Int =
        when {
            tab == Tab.SETTINGS && stage == OPEN_SETTINGS -> TALK_TO_GUIDE_AGAIN
            tab == Tab.INVENTORY && stage == OPEN_INVENTORY -> CATCH_SHRIMP
            tab == Tab.SKILLS && stage == OPEN_SKILLS -> TALK_TO_SURVIVAL_AGAIN
            else -> stage
        }

    // ---- places --------------------------------------------------------------------------

    data class Tile(val x: Int, val z: Int)

    /** Where `::tutstage` and the agent bridge put a player for each stage. */
    fun startTile(stage: Int): Tile =
        when {
            stage <= OPEN_DOOR -> Tile(3094, 3107)
            stage < LEAVE_SURVIVAL -> Tile(3102, 3096)
            stage == LEAVE_SURVIVAL -> Tile(3097, 3093)
            else -> Tile(3087, 3091)
        }

    // ---- items ---------------------------------------------------------------------------

    enum class Item {
        NET,
        AXE,
        TINDERBOX,
        LOGS,
        RAW_SHRIMPS,
    }

    /**
     * What a player starting at [stage] should be carrying, as if they had done the stages before
     * it: the net from the first lesson, the shrimp they caught, the axe and tinderbox from the
     * second, and logs to light a fire with.
     */
    fun itemsFor(stage: Int): List<Item> {
        val items = ArrayList<Item>()
        if (stage in OPEN_INVENTORY until MEET_CHEF) {
            items += Item.NET
        }
        if (stage in TALK_TO_SURVIVAL_AGAIN..COOK_SHRIMP) {
            items += Item.RAW_SHRIMPS
        }
        if (stage in CHOP_TREE until MEET_CHEF) {
            items += Item.AXE
            items += Item.TINDERBOX
        }
        if (stage == LIGHT_FIRE || stage == COOK_SHRIMP) {
            items += Item.LOGS
        }
        return items
    }

    // ---- skilling rolls ------------------------------------------------------------------

    /** Ticks between rolls while fishing, chopping or lighting a fire. */
    const val ROLL_TICKS = 4

    /**
     * Chance out of 256 of a success on each roll at level 1. Fishing is the low end of the live
     * game's level-1 shrimp rate. Chopping and lighting the fire are fitted to the live run (10 s
     * from the axe to the logs and 7 s from the logs to the fire, reaction time included): the
     * island's tree and logs go faster than the normal ones the live rates are for.
     */
    const val FISH_CHANCE = 48
    const val CHOP_CHANCE = 100
    const val FIRE_CHANCE = 110

    /** Cooking on the island never burns: the live tutorial gives you more shrimp if it does. */
    const val COOK_TICKS = 4

    fun rollSucceeds(chance256: Int, roll0to255: Int): Boolean = roll0to255 < chance256

    // xp per action, as on the live island
    const val FISHING_XP = 10.0
    const val WOODCUTTING_XP = 25.0
    const val FIREMAKING_XP = 40.0
    const val COOKING_XP = 30.0

    /** How long a lit fire stays, in ticks (the live game uses 60 to 90 seconds). */
    const val FIRE_TICKS = 120
}
