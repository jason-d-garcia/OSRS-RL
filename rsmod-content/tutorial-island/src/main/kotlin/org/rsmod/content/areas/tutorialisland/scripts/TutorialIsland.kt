package org.rsmod.content.areas.tutorialisland.scripts

import jakarta.inject.Inject
import org.rsmod.api.config.refs.modlevels
import org.rsmod.api.config.refs.stats
import org.rsmod.api.invtx.invAdd
import org.rsmod.api.player.dialogue.Dialogue
import org.rsmod.api.player.output.mes
import org.rsmod.api.player.protect.ProtectedAccess
import org.rsmod.api.player.protect.ProtectedAccessLauncher
import org.rsmod.api.player.vars.intVarp
import org.rsmod.api.repo.loc.LocRepository
import org.rsmod.api.route.StepFactory
import org.rsmod.api.script.onCommand
import org.rsmod.api.script.onIfOverlayButton
import org.rsmod.api.script.onOpHeldU
import org.rsmod.api.script.onOpLoc1
import org.rsmod.api.script.onOpLocU
import org.rsmod.api.script.onOpNpc1
import org.rsmod.api.script.onPlayerLogin
import org.rsmod.content.areas.tutorialisland.TutorialIslandApi
import org.rsmod.content.areas.tutorialisland.TutorialRules
import org.rsmod.content.areas.tutorialisland.TutorialRules.Gate
import org.rsmod.content.areas.tutorialisland.TutorialRules.Tab
import org.rsmod.content.areas.tutorialisland.configs.tut_components
import org.rsmod.content.areas.tutorialisland.configs.tut_locs
import org.rsmod.content.areas.tutorialisland.configs.tut_npcs
import org.rsmod.content.areas.tutorialisland.configs.tut_objs
import org.rsmod.content.areas.tutorialisland.configs.tut_seqs
import org.rsmod.content.areas.tutorialisland.configs.tut_varps
import org.rsmod.game.MapClock
import org.rsmod.game.entity.Npc
import org.rsmod.game.entity.Player
import org.rsmod.game.loc.BoundLocInfo
import org.rsmod.game.loc.LocAngle
import org.rsmod.game.loc.LocShape
import org.rsmod.game.type.loc.LocType
import org.rsmod.game.type.obj.ObjType
import org.rsmod.map.CoordGrid
import org.rsmod.plugin.scripts.PluginScript
import org.rsmod.plugin.scripts.ScriptContext

/**
 * Tutorial Island, from the Gielinor Guide to the gate out of the Survival Expert's area.
 *
 * Progress is the `tutorial` varp, with the live game's values (see [TutorialRules]). Players get
 * here with `::tutorial` (or `::tutstage <value>` to jump to a stage); the agent bridge puts
 * headless bots here with its `tutorial` scene. Existing accounts are never moved here on login.
 */
class TutorialIsland
@Inject
constructor(
    private val locRepo: LocRepository,
    private val mapClock: MapClock,
    private val protectedAccess: ProtectedAccessLauncher,
    private val steps: StepFactory,
) : PluginScript() {
    private var Player.tutorial by intVarp(tut_varps.tutorial)

    override fun ScriptContext.startup() {
        TutorialIslandApi.startHook = { player, stage -> setUp(player, stage) }
        TutorialIslandApi.tabHook = { player, tab -> tabOpened(player, tab) }
        TutorialIslandApi.stageHook = { player -> player.tutorial }

        onCommand("tutorial") {
            modLevel = modlevels.admin
            desc = "Start Tutorial Island (::tutorial off to leave it)"
            cheat {
                if (args.firstOrNull()?.lowercase() == "off") {
                    player.tutorial = 0
                    player.mes("Tutorial progress cleared.")
                } else {
                    startAt(player, TutorialRules.TALK_TO_GUIDE)
                }
            }
        }
        onCommand("tutstage") {
            modLevel = modlevels.admin
            desc = "Jump to a Tutorial Island stage, e.g. ::tutstage 70"
            invalidArgs = "Use as ::tutstage <value>, one of ${TutorialRules.STAGES}"
            cheat {
                val stage = args.firstOrNull()?.toIntOrNull()
                if (stage == null || stage !in TutorialRules.STAGES) {
                    player.mes("Stages: ${TutorialRules.STAGES.joinToString()}")
                } else {
                    startAt(player, stage)
                }
            }
        }
        onPlayerLogin {
            if (TutorialRules.isOnIsland(player.tutorial)) {
                player.mes(TutorialRules.hint(player.tutorial))
            }
        }

        onOpNpc1(tut_npcs.gielinor_guide) { startDialogue(it.npc) { guide() } }
        onOpNpc1(tut_npcs.survival_expert) { startDialogue(it.npc) { survivalExpert() } }
        onOpNpc1(tut_npcs.master_chef) {
            startDialogue(it.npc) {
                chatNpc(neutral, "Ah, a newcomer! Come back when the rest of my kitchen is built.")
            }
        }
        onOpNpc1(tut_npcs.fishing_spot) { fish(it.npc) }

        onOpLoc1(tut_locs.guide_door) { passGate(it.loc, Gate.GUIDE_DOOR, null) }
        onOpLoc1(tut_locs.survival_gate_left) {
            passGate(it.loc, Gate.SURVIVAL_GATE, tut_locs.survival_gate_right)
        }
        onOpLoc1(tut_locs.survival_gate_right) {
            passGate(it.loc, Gate.SURVIVAL_GATE, tut_locs.survival_gate_left)
        }
        onOpLoc1(tut_locs.tree) { chop(it.loc) }
        onOpHeldU(tut_objs.tinderbox, tut_objs.logs) { lightFire() }
        onOpLocU(tut_locs.fire, tut_objs.raw_shrimps) { cook() }

        for (tab in Tab.entries) {
            val stones =
                listOf(tut_components.fixed, tut_components.resizable, tut_components.resizable_bottom)
            for (frame in stones) {
                onIfOverlayButton(frame[tab.stone]) { tabOpened(player, tab) }
            }
        }
    }

    // ---- stages ---------------------------------------------------------------------------

    private fun setStage(player: Player, stage: Int) {
        if (player.tutorial == stage) {
            return
        }
        player.tutorial = stage
        val hint = TutorialRules.hint(stage)
        if (hint.isNotEmpty()) {
            player.mes(hint)
        }
    }

    /** Sets [stage] and makes sure the player has what that stage expects. Doesn't move them. */
    private fun setUp(player: Player, stage: Int) {
        player.tutorial = 0
        setStage(player, stage)
        for (obj in TutorialRules.itemsFor(stage).map(::objFor)) {
            if (player.inv.none { it?.id == obj.id }) {
                player.invAdd(player.inv, obj)
            }
        }
    }

    private fun startAt(player: Player, stage: Int) {
        setUp(player, stage)
        val tile = TutorialRules.startTile(stage)
        protectedAccess.launch(player) { telejump(CoordGrid(tile.x, tile.z, 0)) }
    }

    private fun objFor(item: TutorialRules.Item): ObjType =
        when (item) {
            TutorialRules.Item.NET -> tut_objs.small_fishing_net
            TutorialRules.Item.AXE -> tut_objs.bronze_axe
            TutorialRules.Item.TINDERBOX -> tut_objs.tinderbox
            TutorialRules.Item.LOGS -> tut_objs.logs
            TutorialRules.Item.RAW_SHRIMPS -> tut_objs.raw_shrimps
        }

    private fun tabOpened(player: Player, tab: Tab) {
        val stage = player.tutorial
        val next = TutorialRules.afterTab(tab, stage)
        if (next != stage) {
            setStage(player, next)
        }
    }

    // ---- the Gielinor Guide ---------------------------------------------------------------

    private suspend fun Dialogue.guide() {
        val stage = player.tutorial
        when {
            stage < TutorialRules.TALK_TO_GUIDE -> {
                chatNpc(neutral, "Welcome! Type ::tutorial if you'd like to start the tutorial.")
            }
            stage == TutorialRules.TALK_TO_GUIDE -> {
                chatNpc(
                    happy,
                    "Greetings! I see you are a new arrival to the world of Gielinor. My job is " +
                        "to welcome all new visitors. So welcome!",
                )
                chatNpc(
                    neutral,
                    "You have already learned the first thing needed to succeed in this world: " +
                        "talking to other people!",
                )
                val experience =
                    choice3(
                        "I am brand new! This is my first time here.",
                        0,
                        "I've played in the past, but not recently.",
                        1,
                        "I am an experienced player.",
                        2,
                    )
                if (experience == 2) {
                    chatPlayer(neutral, "I am an experienced player.")
                    chatNpc(laugh, "Then you'll know your way around. I'll keep this short.")
                }
                setStage(player, TutorialRules.OPEN_SETTINGS)
                chatNpc(neutral, "Now then, let's start by looking at your settings menu.")
            }
            stage == TutorialRules.OPEN_SETTINGS -> {
                chatNpc(neutral, "Click on the flashing spanner icon to open your settings menu.")
            }
            stage == TutorialRules.TALK_TO_GUIDE_AGAIN -> {
                chatNpc(
                    happy,
                    "Looks like you're making good progress! The menu you've just opened is one " +
                        "of many. You'll learn about the rest as you progress through the tutorial.",
                )
                setStage(player, TutorialRules.OPEN_DOOR)
                chatNpc(neutral, "Anyway, I'd say it's time for you to go and meet your first instructor!")
            }
            else -> {
                chatNpc(neutral, "Through the door you'll find your first instructor. Good luck!")
            }
        }
    }

    // ---- the Survival Expert --------------------------------------------------------------

    private suspend fun Dialogue.survivalExpert() {
        val stage = player.tutorial
        when {
            stage < TutorialRules.TALK_TO_SURVIVAL -> {
                chatNpc(neutral, "Hello there. Talk to the Gielinor Guide in the house first.")
            }
            stage == TutorialRules.TALK_TO_SURVIVAL -> {
                chatNpc(
                    happy,
                    "Hello there, newcomer. My name is Brynna. My job is to teach you about the " +
                        "skills you can use to survive in this world.",
                )
                chatNpc(
                    neutral,
                    "The first skill we're going to look at is Fishing. There's some shrimp in " +
                        "this pond here. Let's try and catch some.",
                )
                give(tut_objs.small_fishing_net)
                setStage(player, TutorialRules.OPEN_INVENTORY)
                objbox(tut_objs.small_fishing_net, "The Survival Expert gives you a small fishing net.")
            }
            stage < TutorialRules.TALK_TO_SURVIVAL_AGAIN -> {
                if (give(tut_objs.small_fishing_net)) {
                    objbox(tut_objs.small_fishing_net, "The Survival Expert gives you a spare net.")
                }
                chatNpc(neutral, "Use your net on the fishing spot in the pond to catch some shrimp.")
            }
            stage == TutorialRules.TALK_TO_SURVIVAL_AGAIN -> {
                chatPlayer(happy, "I've managed to catch some shrimp.")
                chatNpc(
                    happy,
                    "Excellent work. Next you'll want to cook those shrimp. To do that, you'll " +
                        "need a fire. This brings us on to the Woodcutting and Firemaking skills.",
                )
                give(tut_objs.bronze_axe)
                give(tut_objs.tinderbox)
                setStage(player, TutorialRules.CHOP_TREE)
                doubleobjbox(
                    tut_objs.bronze_axe,
                    tut_objs.tinderbox,
                    "The Survival Expert gives you a bronze axe and a tinderbox.",
                )
            }
            stage < TutorialRules.LEAVE_SURVIVAL -> {
                var gave = give(tut_objs.bronze_axe)
                gave = give(tut_objs.tinderbox) || gave
                if (stage >= TutorialRules.COOK_SHRIMP) {
                    gave = give(tut_objs.raw_shrimps) || gave
                }
                if (gave) {
                    mesbox("The Survival Expert gives you what you were missing.")
                }
                val what =
                    when (stage) {
                        TutorialRules.CHOP_TREE -> "Chop down one of the trees to get some logs."
                        TutorialRules.LIGHT_FIRE -> "Use the tinderbox on your logs to light a fire."
                        else -> "Use your raw shrimp on the fire to cook them."
                    }
                chatNpc(neutral, what)
            }
            else -> {
                chatNpc(happy, "Well done! Follow the path and go through the gate to continue.")
            }
        }
    }

    /** Gives one of [obj] if the player has none. Returns true if it gave one. */
    private fun Dialogue.give(obj: ObjType): Boolean {
        if (player.inv.any { it?.id == obj.id }) {
            return false
        }
        player.invAdd(player.inv, obj)
        return true
    }

    // ---- doors and gates ------------------------------------------------------------------

    private suspend fun ProtectedAccess.passGate(loc: BoundLocInfo, gate: Gate, pair: LocType?) {
        val stage = player.tutorial
        if (TutorialRules.isOnIsland(stage) && !TutorialRules.canPass(gate, stage)) {
            mes(TutorialRules.refusal(gate))
            return
        }
        val (a, b) = wallSides(loc)
        val dest =
            when (coords) {
                a -> b
                b -> a
                else -> if (coords.chebyshevDistance(a) > coords.chebyshevDistance(b)) a else b
            }
        locRepo.del(loc, GATE_OPEN_TICKS)
        if (pair != null) {
            for ((dx, dz) in NEIGHBOURS) {
                val other = locRepo.findExact(loc.coords.translate(dx, dz), pair) ?: continue
                locRepo.del(other, GATE_OPEN_TICKS)
                break
            }
        }
        playerWalk(dest)
        delay(1)
        val next = TutorialRules.afterPassing(gate, stage)
        if (next != stage) {
            setStage(player, next)
        }
    }

    /** The two tiles a wall loc separates: the one it's on and the one across the wall. */
    private fun wallSides(loc: BoundLocInfo): Pair<CoordGrid, CoordGrid> {
        val on = loc.coords
        val across =
            when (loc.angle) {
                LocAngle.West -> on.translateX(-1)
                LocAngle.North -> on.translateZ(1)
                LocAngle.East -> on.translateX(1)
                LocAngle.South -> on.translateZ(-1)
            }
        return on to across
    }

    // ---- fishing --------------------------------------------------------------------------

    private fun ProtectedAccess.fish(spot: Npc) {
        if (player.inv.none { it?.id == tut_objs.small_fishing_net.id }) {
            mes("You need a small fishing net to catch these fish.")
            return
        }
        if (inv.isFull()) {
            mes("You can't carry any more fish.")
            return
        }
        if (actionDelay < mapClock) {
            actionDelay = mapClock + TutorialRules.ROLL_TICKS
            anim(tut_seqs.fishing_net)
            spam("You cast out your net...")
            opNpc1(spot)
            return
        }
        if (actionDelay == mapClock) {
            if (TutorialRules.rollSucceeds(TutorialRules.FISH_CHANCE, random.of(0, 255))) {
                invAdd(inv, tut_objs.raw_shrimps)
                statAdvance(stats.fishing, TutorialRules.FISHING_XP)
                resetAnim()
                spam("You catch some shrimps.")
                if (player.tutorial == TutorialRules.CATCH_SHRIMP) {
                    setStage(player, TutorialRules.OPEN_SKILLS)
                }
                return
            }
            actionDelay = mapClock + TutorialRules.ROLL_TICKS
            anim(tut_seqs.fishing_net)
        }
        opNpc1(spot)
    }

    // ---- woodcutting ----------------------------------------------------------------------

    private fun ProtectedAccess.chop(tree: BoundLocInfo) {
        if (player.inv.none { it?.id == tut_objs.bronze_axe.id }) {
            mes("You need an axe to chop down this tree.")
            return
        }
        if (inv.isFull()) {
            mes("Your inventory is too full to hold any more logs.")
            return
        }
        if (actionDelay < mapClock) {
            actionDelay = mapClock + TutorialRules.ROLL_TICKS
            anim(tut_seqs.woodcutting_bronze)
            spam("You swing your axe at the tree.")
            opLoc1(tree)
            return
        }
        if (actionDelay == mapClock) {
            if (TutorialRules.rollSucceeds(TutorialRules.CHOP_CHANCE, random.of(0, 255))) {
                invAdd(inv, tut_objs.logs)
                statAdvance(stats.woodcutting, TutorialRules.WOODCUTTING_XP)
                resetAnim()
                spam("You get some logs.")
                if (player.tutorial == TutorialRules.CHOP_TREE) {
                    setStage(player, TutorialRules.LIGHT_FIRE)
                }
                return
            }
            actionDelay = mapClock + TutorialRules.ROLL_TICKS
            anim(tut_seqs.woodcutting_bronze)
        }
        opLoc1(tree)
    }

    // ---- firemaking -----------------------------------------------------------------------

    private suspend fun ProtectedAccess.lightFire() {
        if (locRepo.findAll(coords).any()) {
            mes("You can't light a fire here.")
            return
        }
        spam("You attempt to light the logs.")
        var lit = false
        repeat(MAX_FIRE_ROLLS) {
            if (lit) {
                return@repeat
            }
            anim(tut_seqs.light_fire)
            delay(TutorialRules.ROLL_TICKS)
            lit = TutorialRules.rollSucceeds(TutorialRules.FIRE_CHANCE, random.of(0, 255))
        }
        if (!lit) {
            // The live game keeps trying until it lights; a cap keeps a stuck bot from hanging.
            mes("You fail to light the logs.")
            return
        }
        val spot = coords
        invDel(inv, tut_objs.logs)
        locRepo.add(spot, tut_locs.fire, TutorialRules.FIRE_TICKS, LocAngle.West, LocShape.CentrepieceStraight)
        statAdvance(stats.firemaking, TutorialRules.FIREMAKING_XP)
        resetAnim()
        spam("The fire catches and the logs begin to burn.")
        stepOffFire(spot)?.let { playerWalk(it) }
        if (player.tutorial == TutorialRules.LIGHT_FIRE) {
            setStage(player, TutorialRules.COOK_SHRIMP)
        }
    }

    /** Where you step after lighting a fire: west, else east, south or north, as on live. */
    private fun stepOffFire(spot: CoordGrid): CoordGrid? =
        listOf(spot.translateX(-1), spot.translateX(1), spot.translateZ(-1), spot.translateZ(1))
            .firstOrNull { steps.validated(spot, it) == it }

    // ---- cooking --------------------------------------------------------------------------

    private suspend fun ProtectedAccess.cook() {
        anim(tut_seqs.cook_on_fire)
        delay(TutorialRules.COOK_TICKS)
        invDel(inv, tut_objs.raw_shrimps)
        invAdd(inv, tut_objs.shrimps)
        statAdvance(stats.cooking, TutorialRules.COOKING_XP)
        resetAnim()
        spam("The shrimps are now nicely cooked.")
        if (player.tutorial == TutorialRules.COOK_SHRIMP) {
            setStage(player, TutorialRules.LEAVE_SURVIVAL)
        }
    }

    private companion object {
        /** How long a door or gate stays open after a player goes through. */
        const val GATE_OPEN_TICKS = 3
        const val MAX_FIRE_ROLLS = 10
        val NEIGHBOURS = listOf(1 to 0, -1 to 0, 0 to 1, 0 to -1)
    }
}
