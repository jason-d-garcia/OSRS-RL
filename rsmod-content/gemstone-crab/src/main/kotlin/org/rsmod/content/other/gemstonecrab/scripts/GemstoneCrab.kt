package org.rsmod.content.other.gemstonecrab.scripts

import jakarta.inject.Inject
import org.rsmod.api.player.output.mes
import org.rsmod.api.player.protect.ProtectedAccess
import org.rsmod.api.player.righthand
import org.rsmod.api.player.stat.miningLvl
import org.rsmod.api.random.GameRandom
import org.rsmod.api.repo.npc.NpcRepository
import org.rsmod.api.repo.obj.ObjRepository
import org.rsmod.api.script.onAiTimer
import org.rsmod.api.script.onGameStartup
import org.rsmod.api.script.onNpcHit
import org.rsmod.api.script.onOpLoc1
import org.rsmod.api.script.onOpNpc1
import org.rsmod.content.other.gemstonecrab.GemstoneCrabRules
import org.rsmod.content.other.gemstonecrab.GemstoneCrabRules.Gem
import org.rsmod.content.other.gemstonecrab.configs.gem_crab_locs
import org.rsmod.content.other.gemstonecrab.configs.gem_crab_npcs
import org.rsmod.content.other.gemstonecrab.configs.gem_crab_objs
import org.rsmod.content.other.gemstonecrab.configs.gem_crab_seqs
import org.rsmod.game.MapClock
import org.rsmod.game.entity.Npc
import org.rsmod.game.entity.Player
import org.rsmod.game.entity.PlayerList
import org.rsmod.game.hit.Hit
import org.rsmod.game.type.npc.NpcTypeList
import org.rsmod.game.type.obj.ObjType
import org.rsmod.game.type.obj.isAssociatedWith
import org.rsmod.game.type.seq.SeqType
import org.rsmod.map.CoordGrid
import org.rsmod.plugin.scripts.PluginScript
import org.rsmod.plugin.scripts.ScriptContext

/**
 * Gemstone crab in the Tlati Rainforest.
 *
 * One crab is up at a time, at one of three mines. It lives for [GemstoneCrabRules.ALIVE_TICKS],
 * its health bar drains with time rather than damage, and it never dies to players. When it
 * burrows it leaves a shell for [GemstoneCrabRules.SHELL_TICKS]; the top 16 players by damage can
 * mine the shell once each for three uncut gems. The crab then surfaces at another mine, and the
 * tunnel at each mine takes players to wherever the crab is now.
 *
 * Players are keyed by display name for the length of one crab, because [Player.uid] is not stable
 * across a relog.
 */
class GemstoneCrab
@Inject
constructor(
    private val npcTypes: NpcTypeList,
    private val npcRepo: NpcRepository,
    private val objRepo: ObjRepository,
    private val playerList: PlayerList,
    private val random: GameRandom,
    private val mapClock: MapClock,
) : PluginScript() {
    private var mineIndex = 0
    private var crab: Npc? = null
    private var crabSpawnCycle = 0
    private val damage = mutableMapOf<String, Int>()
    private val firstHitCycle = mutableMapOf<String, Int>()
    private val shells = mutableMapOf<Npc, ShellState>()

    private class ShellState(val eligible: Set<String>, val spawnCycle: Int) {
        val mined = mutableSetOf<String>()
    }

    private class Pickaxe(val obj: ObjType, val level: Int, val seq: SeqType)

    private val pickaxes =
        listOf(
            Pickaxe(gem_crab_objs.dragon_pickaxe, 61, gem_crab_seqs.mining_dragon),
            Pickaxe(gem_crab_objs.rune_pickaxe, 41, gem_crab_seqs.mining_rune),
            Pickaxe(gem_crab_objs.adamant_pickaxe, 31, gem_crab_seqs.mining_adamant),
            Pickaxe(gem_crab_objs.mithril_pickaxe, 21, gem_crab_seqs.mining_mithril),
            Pickaxe(gem_crab_objs.black_pickaxe, 11, gem_crab_seqs.mining_black),
            Pickaxe(gem_crab_objs.steel_pickaxe, 6, gem_crab_seqs.mining_steel),
            Pickaxe(gem_crab_objs.iron_pickaxe, 1, gem_crab_seqs.mining_iron),
            Pickaxe(gem_crab_objs.bronze_pickaxe, 1, gem_crab_seqs.mining_bronze),
        )

    override fun ScriptContext.startup() {
        onGameStartup { spawnCrab(random.of(GemstoneCrabRules.MINES.size)) }
        onAiTimer(gem_crab_npcs.crab) { npc.crabTimer() }
        onNpcHit(gem_crab_npcs.crab) { crabHit(npc, hit) }
        onOpNpc1(gem_crab_npcs.shell) { mineShell(it.npc) }
        onOpLoc1(gem_crab_locs.tunnel) { useTunnel() }
    }

    // ---- crab lifecycle ----

    private fun spawnCrab(index: Int) {
        mineIndex = index
        val mine = GemstoneCrabRules.MINES[index]
        // Mine coordinates are the crab's centre tile; npcs are placed by their south-west tile.
        val half = npcTypes[gem_crab_npcs.crab].size / 2
        val coords = CoordGrid(mine.crabX - half, mine.crabZ - half, 0)
        val npc = Npc(npcTypes[gem_crab_npcs.crab], coords)
        npcRepo.add(npc, Int.MAX_VALUE)
        crab = npc
        crabSpawnCycle = mapClock.cycle
        damage.clear()
        firstHitCycle.clear()
        npc.aiTimer(GemstoneCrabRules.TIMER_INTERVAL)
    }

    private fun age(): Int = mapClock.cycle - crabSpawnCycle

    private fun Npc.crabTimer() {
        if (this !== crab) {
            // A crab placed by the cache's own map spawns; this script owns the only live crab.
            npcRepo.del(this, Int.MAX_VALUE)
            return
        }
        val age = age()
        hitpoints = GemstoneCrabRules.hitpointsForAge(age)
        if (age >= GemstoneCrabRules.ALIVE_TICKS) {
            burrow(this)
            return
        }
        aiTimer(GemstoneCrabRules.TIMER_INTERVAL)
    }

    private fun crabHit(npc: Npc, hit: Hit) {
        if (npc !== crab) {
            return
        }
        // The health bar follows the timer, not damage; undo whatever this hit took.
        npc.hitpoints = GemstoneCrabRules.hitpointsForAge(age())
        if (!hit.isFromPlayer || hit.damage <= 0) {
            return
        }
        val player = hit.resolvePlayerSource(playerList) ?: return
        val key = player.displayName
        damage[key] = (damage[key] ?: 0) + hit.damage
        firstHitCycle.putIfAbsent(key, mapClock.cycle)
    }

    private fun burrow(npc: Npc) {
        val ranked = GemstoneCrabRules.eligible(damage, firstHitCycle)
        val eligible = ranked.toSet()
        val nearby = playersNear(npc.coords, NEARBY_RADIUS)

        val shell = Npc(npcTypes[gem_crab_npcs.shell], npc.coords)
        npcRepo.add(shell, GemstoneCrabRules.SHELL_TICKS)
        shells.entries.removeIf { mapClock.cycle - it.value.spawnCycle > GemstoneCrabRules.SHELL_TICKS }
        shells[shell] = ShellState(eligible, mapClock.cycle)

        val crushers = GemstoneCrabRules.msgTopCrushers(GemstoneCrabRules.announced(ranked))
        for (player in nearby) {
            player.mes(GemstoneCrabRules.MSG_BURROW)
            if (crushers.isNotEmpty()) {
                player.mes(crushers)
            }
            val name = player.displayName
            when {
                name in eligible -> player.mes(GemstoneCrabRules.MSG_ELIGIBLE)
                name in damage -> player.mes(GemstoneCrabRules.MSG_NOT_ELIGIBLE)
            }
        }

        npcRepo.del(npc, Int.MAX_VALUE)
        crab = null
        spawnCrab(GemstoneCrabRules.nextMine(mineIndex, random.of(GemstoneCrabRules.MINES.size - 1)))
    }

    private fun playersNear(coords: CoordGrid, radius: Int): List<Player> =
        playerList.filter { it.coords.level == coords.level && it.coords.chebyshevDistance(coords) <= radius }

    // ---- shell ----

    private suspend fun ProtectedAccess.mineShell(shell: Npc) {
        val state = shells[shell] ?: return
        val pick = bestPickaxe(player)
        if (pick == null) {
            val hasAny = pickaxes.any { it.obj in inv || it.obj.isAssociatedWith(player.righthand) }
            mes(if (hasAny) GemstoneCrabRules.MSG_NO_PICK_LEVEL else GemstoneCrabRules.MSG_NO_PICK)
            return
        }
        val name = player.displayName
        if (name !in state.eligible) {
            mes(GemstoneCrabRules.MSG_NOT_ELIGIBLE)
            return
        }
        if (name in state.mined) {
            mes(GemstoneCrabRules.MSG_ALREADY_MINED)
            return
        }
        faceEntitySquare(shell)
        mes(GemstoneCrabRules.MSG_SWING)
        anim(pick.seq)
        delay(GemstoneCrabRules.MINE_TICKS)
        // The shell may have despawned, or the player mined it from another click, while swinging.
        if (shells[shell] !== state || name in state.mined) {
            return
        }
        state.mined += name
        repeat(GemstoneCrabRules.GEMS_PER_SHELL) {
            val gem = GemstoneCrabRules.rollGem { n -> random.of(n) }
            invAddOrDrop(objRepo, gemObj(gem))
            mes(GemstoneCrabRules.msgGem(gem))
        }
        resetAnim()
    }

    private fun bestPickaxe(player: Player): Pickaxe? =
        pickaxes.firstOrNull { pick ->
            player.miningLvl >= pick.level &&
                (pick.obj in player.inv || pick.obj.isAssociatedWith(player.righthand))
        }

    private fun gemObj(gem: Gem): ObjType =
        when (gem) {
            Gem.OPAL -> gem_crab_objs.uncut_opal
            Gem.JADE -> gem_crab_objs.uncut_jade
            Gem.RED_TOPAZ -> gem_crab_objs.uncut_red_topaz
            Gem.SAPPHIRE -> gem_crab_objs.uncut_sapphire
            Gem.EMERALD -> gem_crab_objs.uncut_emerald
            Gem.RUBY -> gem_crab_objs.uncut_ruby
            Gem.DIAMOND -> gem_crab_objs.uncut_diamond
            Gem.DRAGONSTONE -> gem_crab_objs.uncut_dragonstone
        }

    // ---- tunnel ----

    private suspend fun ProtectedAccess.useTunnel() {
        val target = GemstoneCrabRules.MINES[mineIndex]
        val dest = CoordGrid(target.tunnelX, target.tunnelZ - 1, 0)
        if (coords.chebyshevDistance(dest) <= NEARBY_RADIUS) {
            mes("The crab is already at this mine.")
            return
        }
        delay(GemstoneCrabRules.TUNNEL_TICKS)
        telejump(dest)
    }

    private companion object {
        /** Players this close to the crab hear its burrow messages. */
        const val NEARBY_RADIUS = 15
    }
}
