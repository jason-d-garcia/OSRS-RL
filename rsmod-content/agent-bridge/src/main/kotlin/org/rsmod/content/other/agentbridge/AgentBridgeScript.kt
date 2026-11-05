package org.rsmod.content.other.agentbridge

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.node.ArrayNode
import com.fasterxml.jackson.databind.node.ObjectNode
import com.github.michaelbull.logging.InlineLogger
import jakarta.inject.Inject
import net.rsprot.protocol.api.NetworkService
import net.rsprot.protocol.common.client.OldSchoolClientType
import org.rsmod.api.account.AccountManager
import org.rsmod.api.account.loader.request.AccountLoadAuth
import org.rsmod.api.account.loader.request.AccountLoadResponse
import org.rsmod.api.config.refs.stats
import org.rsmod.api.config.refs.varps
import org.rsmod.api.game.process.GameLifecycle
import org.rsmod.api.player.interact.LocInteractions
import org.rsmod.api.player.interact.NpcInteractions
import org.rsmod.api.player.protect.clearPendingAction
import org.rsmod.api.player.back
import org.rsmod.api.player.feet
import org.rsmod.api.player.front
import org.rsmod.api.player.hands
import org.rsmod.api.player.hat
import org.rsmod.api.player.legs
import org.rsmod.api.player.lefthand
import org.rsmod.api.player.righthand
import org.rsmod.api.player.ring
import org.rsmod.api.player.torso
import org.rsmod.api.player.vars.VarPlayerIntMapSetter
import org.rsmod.api.player.stat.statAdvance
import org.rsmod.api.player.stat.statBase
import org.rsmod.api.player.vars.varMoveSpeed
import org.rsmod.api.pw.hash.PasswordHashing
import org.rsmod.api.random.GameRandom
import org.rsmod.api.realm.Realm
import org.rsmod.api.registry.account.AccountRegistry
import org.rsmod.api.registry.loc.LocRegistry
import org.rsmod.api.registry.player.PlayerRegistry
import org.rsmod.api.registry.player.isSuccess
import org.rsmod.api.script.onEvent
import org.rsmod.api.script.onGameStartup
import org.rsmod.content.other.gemstonecrab.GemstoneCrabRules
import org.rsmod.content.other.gemstonecrab.GemstoneCrabStatus
import org.rsmod.content.other.gemstonecrab.configs.gem_crab_locs
import org.rsmod.content.other.gemstonecrab.configs.gem_crab_objs
import org.rsmod.events.EventBus
import org.rsmod.game.MapClock
import org.rsmod.game.entity.Npc
import org.rsmod.game.entity.NpcList
import org.rsmod.game.entity.Player
import org.rsmod.game.entity.PlayerList
import org.rsmod.game.entity.player.SessionStateEvent
import org.rsmod.game.entity.util.PathingEntityCommon
import org.rsmod.game.interact.InteractionLocOp
import org.rsmod.game.interact.InteractionNpcOp
import org.rsmod.game.interact.InteractionOp
import org.rsmod.game.inv.InvObj
import org.rsmod.game.loc.BoundLocInfo
import org.rsmod.game.loc.LocInfo
import org.rsmod.game.map.collision.isZoneValid
import org.rsmod.game.movement.MoveSpeed
import org.rsmod.game.movement.RouteRequestCoord
import org.rsmod.game.movement.RouteRequestLoc
import org.rsmod.game.movement.RouteRequestPathingEntity
import org.rsmod.game.stat.PlayerSkillXPTable
import org.rsmod.game.type.loc.LocTypeList
import org.rsmod.game.type.obj.ObjType
import org.rsmod.game.type.obj.ObjTypeList
import org.rsmod.game.type.stat.StatType
import org.rsmod.map.CoordGrid
import org.rsmod.map.zone.ZoneKey
import org.rsmod.plugin.scripts.PluginScript
import org.rsmod.plugin.scripts.ScriptContext
import org.rsmod.routefinder.collision.CollisionFlagMap
import org.rsmod.routefinder.flag.CollisionFlag

/**
 * Lets a trained policy drive players on this server over a local socket.
 *
 * An agent connects to `127.0.0.1:45600` (set `OSRSRL_AGENT_PORT` to change it), then either spawns
 * a headless bot (`spawn`) or takes over a player who is already logged in through a real client
 * (`attach`). Every game cycle the bridge sends that player's state as one JSON line; the agent
 * answers with the inputs to apply on the next cycle. In lockstep mode the game waits for the
 * agent's `step` at the end of each cycle, so with a short tick (see the README) the world runs as
 * fast as the agent can decide.
 *
 * The inputs are the ones a client can send: click an npc or loc option, or walk to a tile. They go
 * through the same routing and interaction code as client packets, so a bot can't do anything a
 * player can't. The protocol is in the module README.
 */
class AgentBridgeScript
@Inject
constructor(
    private val eventBus: EventBus,
    private val mapClock: MapClock,
    private val playerList: PlayerList,
    private val npcList: NpcList,
    private val objTypes: ObjTypeList,
    private val locTypes: LocTypeList,
    private val locRegistry: LocRegistry,
    private val npcInteractions: NpcInteractions,
    private val locInteractions: LocInteractions,
    private val collision: CollisionFlagMap,
    private val accountManager: AccountManager,
    private val accountRegistry: AccountRegistry,
    private val playerRegistry: PlayerRegistry,
    private val passwordHashing: PasswordHashing,
    private val service: NetworkService<Player>,
    private val realm: Realm,
    private val random: GameRandom,
) : PluginScript() {
    private val logger = InlineLogger()
    private val connections = ArrayList<AgentConnection>()
    private val server =
        BridgeServer(
            port = System.getenv("OSRSRL_AGENT_PORT")?.toIntOrNull() ?: DEFAULT_PORT,
            bindAddress = System.getenv("OSRSRL_AGENT_BIND") ?: "127.0.0.1",
        )
    private val lockstepTimeoutMs =
        System.getenv("OSRSRL_LOCKSTEP_TIMEOUT_MS")?.toLongOrNull() ?: 30_000L

    private val mapper = AgentConnection.MAPPER

    override fun ScriptContext.startup() {
        if (System.getenv("OSRSRL_AGENT_BRIDGE") == "off") {
            return
        }
        onGameStartup { server.start() }
        onEvent<GameLifecycle.StartCycle> { startCycle() }
        onEvent<GameLifecycle.EndCycle> { endCycle() }
        onEvent<GameLifecycle.Shutdown> { server.stop() }
        onEvent<SessionStateEvent.Delete> { onPlayerDeleted(player) }
    }

    // ---- cycle hooks ----------------------------------------------------------------------

    private fun startCycle() {
        while (true) {
            val conn = server.accepted.poll() ?: break
            connections += conn
        }
        val iterator = connections.iterator()
        while (iterator.hasNext()) {
            val conn = iterator.next()
            drainInbox(conn)
            if (!conn.open) {
                release(conn)
                iterator.remove()
                continue
            }
            conn.pendingReset?.let { deferred ->
                if (conn.player?.isDelayed == false) {
                    conn.pendingReset = null
                    reset(conn, deferred)
                }
            }
            applyPending(conn)
        }
    }

    private fun endCycle() {
        for (conn in connections) {
            val player = conn.player ?: continue
            if (!conn.open) {
                continue
            }
            conn.send(snapshot(conn, player))
            conn.results.clear()
        }
        for (conn in connections) {
            if (conn.lockstep && conn.open && conn.player != null) {
                awaitStep(conn)
            }
        }
    }

    /** Blocks the game thread until [conn] sends `step` (lockstep), handling other commands. */
    private fun awaitStep(conn: AgentConnection) {
        // The agent is blocked waiting for `reset_done`, so it will not send `step`: let the
        // cycle run until the delay ends and the deferred reset goes through.
        if (conn.pendingReset != null) {
            return
        }
        val deadline = System.currentTimeMillis() + lockstepTimeoutMs
        while (true) {
            val left = deadline - System.currentTimeMillis()
            if (left <= 0) {
                conn.lockstep = false
                conn.send(event("warning").put("warning", "lockstep timed out; now real time"))
                logger.warn { "Agent ${conn.id} lockstep timed out; switched to real time." }
                return
            }
            val cmd = conn.take(left) ?: continue
            if (handle(conn, cmd) || conn.pendingReset != null) {
                return
            }
            if (!conn.open || !conn.lockstep || conn.player == null) {
                return
            }
        }
    }

    private fun drainInbox(conn: AgentConnection) {
        while (true) {
            val cmd = conn.inbox.poll() ?: return
            handle(conn, cmd)
        }
    }

    // ---- commands -------------------------------------------------------------------------

    /** Handles one command from the agent. Returns true for `step`, which ends a lockstep wait. */
    private fun handle(conn: AgentConnection, cmd: JsonNode): Boolean {
        when (val op = cmd.path("op").asText()) {
            "__closed__" -> return true
            "hello" -> conn.send(hello())
            "spawn" -> spawnBot(conn, cmd)
            "attach" -> attach(conn, cmd)
            "act" -> conn.pending.add(cmd.path("action"))
            "step" -> {
                cmd.path("actions").forEach { conn.pending.add(it) }
                return true
            }
            "lockstep" -> {
                conn.lockstep = cmd.path("on").asBoolean(true)
                conn.send(event("lockstep").put("on", conn.lockstep))
            }
            "reset" -> reset(conn, cmd)
            "logout" -> release(conn)
            "__login_failed__" -> {
                conn.loginPending = false
                conn.send(event("error").put("error", cmd.path("error").asText()))
            }
            else -> conn.send(event("error").put("error", "unknown op '$op'"))
        }
        return false
    }

    private fun hello(): ObjectNode {
        val node = event("hello")
        node.put("protocol", PROTOCOL_VERSION)
        node.put("cycle", mapClock.cycle)
        node.put("alive_ticks", GemstoneCrabRules.ALIVE_TICKS)
        node.put("shell_ticks", GemstoneCrabRules.SHELL_TICKS)
        node.put("tunnel_ticks", GemstoneCrabRules.TUNNEL_TICKS)
        node.put("mine_ticks", GemstoneCrabRules.MINE_TICKS)
        val mines = node.putArray("mines")
        for (mine in GemstoneCrabRules.MINES) {
            mines
                .addObject()
                .put("name", mine.name)
                .put("crab_x", mine.crabX)
                .put("crab_z", mine.crabZ)
                .put("tunnel_x", mine.tunnelX)
                .put("tunnel_z", mine.tunnelZ)
                .put("exit_x", mine.exitX)
                .put("exit_z", mine.exitZ)
        }
        return node
    }

    // ---- spawning and attaching -----------------------------------------------------------

    private fun spawnBot(conn: AgentConnection, cmd: JsonNode) {
        if (conn.player != null || conn.loginPending) {
            conn.send(event("error").put("error", "this connection already has a player"))
            return
        }
        val name = cmd.path("name").asText("agent${conn.id}").take(12)
        val username = name.lowercase()
        if (playerList.any { it.username.equals(username, ignoreCase = true) }) {
            conn.send(event("error").put("error", "'$name' is already logged in"))
            return
        }
        conn.loginPending = true
        conn.lockstep = cmd.path("lockstep").asBoolean(false)
        // Bots get a fixed password derived from their name; they are never logged in by hand.
        val hashed = passwordHashing.hash("osrs-rl:$username".toCharArray())
        val submitted =
            accountManager.loadOrCreate(AccountLoadAuth.UnknownDevice, username, { hashed }) {
                response ->
                // Called on the account loader thread.
                when (response) {
                    is AccountLoadResponse.Ok -> {
                        val player = createPlayer(response, name)
                        accountRegistry.queueLogin(player, response) { p, _ ->
                            finishLogin(conn, cmd, p)
                        }
                    }
                    is AccountLoadResponse.Err -> {
                        val err = mapper.createObjectNode()
                        err.put("op", "__login_failed__")
                        err.put("error", "login failed: ${response::class.simpleName}")
                        conn.inbox.put(err)
                    }
                }
            }
        if (!submitted) {
            conn.loginPending = false
            conn.send(event("error").put("error", "account loader is busy"))
        }
    }

    private fun createPlayer(response: AccountLoadResponse.Ok, name: String): Player {
        val player = Player()
        for (transform in response.transforms) {
            transform.apply(player)
        }
        player.ui.setWindowStatus(width = 765, height = 503, resizable = false)
        if (response is AccountLoadResponse.Ok.NewAccount) {
            val config = realm.config
            player.coords = config.spawnCoord
            player.xpRate = config.baseXpRate
        }
        if (player.displayName.isBlank()) {
            player.displayName = name
        }
        return player
    }

    /** Runs on the game thread, from [AccountRegistry.handleLogins]. */
    private fun finishLogin(conn: AgentConnection, cmd: JsonNode, player: Player) {
        conn.loginPending = false
        if (!conn.open) {
            return
        }
        val slot = playerRegistry.nextFreeSlot()
        if (slot == null) {
            conn.send(event("error").put("error", "server full"))
            return
        }
        player.slotId = slot
        val playerInfo = service.playerInfoProtocol.alloc(slot, OldSchoolClientType.DESKTOP)
        val client = BotClient()
        val cycle = BotCycle(service, playerInfo, objTypes)
        player.client = client
        player.clientCycle = cycle
        val register = playerRegistry.add(player)
        if (!register.isSuccess()) {
            cycle.dealloc()
            conn.send(event("error").put("error", "could not register bot: $register"))
            return
        }
        eventBus.publish(SessionStateEvent.Login(player))
        eventBus.publish(SessionStateEvent.EngineLogin(player))
        conn.player = player
        conn.headless = true
        conn.botClient = client
        conn.botCycle = cycle
        applyLoadout(player, cmd)
        teleportToCrab(player, cmd.path("mine").asInt(-1), cmd.path("spread").asInt(6))
        conn.send(
            event("spawned")
                .put("name", player.displayName)
                .put("slot", slot)
                .put("headless", true)
                .put("lockstep", conn.lockstep)
        )
        logger.info { "Agent ${conn.id} spawned headless bot '${player.displayName}' (slot $slot)" }
    }

    private fun attach(conn: AgentConnection, cmd: JsonNode) {
        if (conn.player != null || conn.loginPending) {
            conn.send(event("error").put("error", "this connection already has a player"))
            return
        }
        val name = cmd.path("name").asText()
        val player =
            playerList.firstOrNull {
                it.displayName.equals(name, ignoreCase = true) ||
                    it.username.equals(name, ignoreCase = true)
            }
        if (player == null) {
            conn.send(event("error").put("error", "no player named '$name' is logged in"))
            return
        }
        if (connections.any { it !== conn && it.player === player }) {
            conn.send(event("error").put("error", "'$name' is already driven by another agent"))
            return
        }
        conn.player = player
        conn.headless = false
        conn.lockstep = cmd.path("lockstep").asBoolean(false)
        if (cmd.has("loadout")) {
            applyLoadout(player, cmd.path("loadout"))
        }
        conn.send(
            event("spawned")
                .put("name", player.displayName)
                .put("slot", player.slotId)
                .put("headless", false)
                .put("lockstep", conn.lockstep)
        )
        logger.info { "Agent ${conn.id} attached to '${player.displayName}'" }
    }

    /** Lets go of the player: a headless bot logs out, an attached player is left as it is. */
    private fun release(conn: AgentConnection) {
        val player = conn.player
        if (player != null && conn.headless) {
            player.forceDisconnect = true
        }
        conn.player = null
        conn.lockstep = false
        conn.pending.clear()
        if (!conn.open) {
            conn.close()
        }
    }

    private fun onPlayerDeleted(player: Player) {
        (player.clientCycle as? BotCycle)?.dealloc()
        for (conn in connections) {
            if (conn.player === player) {
                conn.player = null
                conn.send(event("logged_out"))
            }
        }
    }

    // ---- loadout and placement ------------------------------------------------------------

    /**
     * Sets combat stats and gear. Defaults are a mid-level melee setup like the simulator's: 80
     * attack, strength and defence, a dragon scimitar, and a rune pickaxe for the shell.
     *
     * Optional: `exact_levels` (set levels exactly, also lowering them; otherwise a level only
     * goes up), worn items by name or obj id in `hat`, `back`, `front`, `shield`, `body`, `legs`,
     * `hands`, `feet`, `ring` ("none" empties the slot), `stance` (0-3, the weapon's combat
     * option: 1 is the scimitar's Slash, aggressive, strength xp), and `pickaxe: "none"` also
     * takes any pickaxe out of the inventory.
     */
    private fun applyLoadout(player: Player, cmd: JsonNode) {
        val exact = cmd.path("exact_levels").asBoolean(false)
        setLevel(player, stats.attack, cmd.path("attack").asInt(80), exact)
        setLevel(player, stats.strength, cmd.path("strength").asInt(80), exact)
        setLevel(player, stats.defence, cmd.path("defence").asInt(80), exact)
        setLevel(player, stats.hitpoints, cmd.path("hitpoints").asInt(80), exact)
        setLevel(player, stats.mining, cmd.path("mining").asInt(41), exact)
        val weapon = objByName(cmd.path("weapon").asText("dragon_scimitar"))
        if (weapon != null) {
            player.righthand = InvObj(weapon)
        }
        wear(cmd, "hat") { player.hat = it }
        wear(cmd, "back") { player.back = it }
        wear(cmd, "front") { player.front = it }
        wear(cmd, "shield") { player.lefthand = it }
        wear(cmd, "body") { player.torso = it }
        wear(cmd, "legs") { player.legs = it }
        wear(cmd, "hands") { player.hands = it }
        wear(cmd, "feet") { player.feet = it }
        wear(cmd, "ring") { player.ring = it }
        if (cmd.has("stance")) {
            VarPlayerIntMapSetter.set(player, varps.com_mode, cmd.path("stance").asInt(0).coerceIn(0, 3))
        }
        val pickName = cmd.path("pickaxe").asText("rune_pickaxe")
        if (pickName == "none") {
            val picks = setOf(gem_crab_objs.rune_pickaxe.id, gem_crab_objs.dragon_pickaxe.id, gem_crab_objs.bronze_pickaxe.id)
            for (slot in 0 until player.inv.size) {
                if (player.inv[slot]?.id in picks) {
                    player.inv[slot] = null
                }
            }
        }
        val pickaxe = objByName(pickName)
        if (pickaxe != null && player.inv.none { it?.id == pickaxe.id }) {
            val free = player.inv.indexOfFirst { it == null }
            if (free >= 0) {
                player.inv[free] = InvObj(pickaxe)
            }
        }
        if (cmd.path("run").asBoolean(true)) {
            player.varMoveSpeed = MoveSpeed.Run
        }
        // Live-game xp (the dev realm's default rate is 150x), so progress matches the simulator.
        player.xpRate = cmd.path("xp_rate").asDouble(1.0)
    }

    /** Puts the named or numbered obj in a worn slot when `cmd` has that field. */
    private inline fun wear(cmd: JsonNode, field: String, set: (InvObj?) -> Unit) {
        if (!cmd.has(field)) {
            return
        }
        val name = cmd.path(field).asText()
        if (name == "none" || name.isEmpty()) {
            set(null)
            return
        }
        val type = objByName(name)
        if (type == null) {
            logger.warn { "Loadout: unknown obj '$name' for $field" }
            return
        }
        set(InvObj(type))
    }

    private fun objByName(name: String): ObjType? =
        when (name) {
            "",
            "none" -> null
            "dragon_scimitar" -> AgentBridgeObjs.dragon_scimitar
            "abyssal_whip" -> AgentBridgeObjs.abyssal_whip
            "dragon_dagger" -> AgentBridgeObjs.dragon_dagger
            "rune_pickaxe" -> gem_crab_objs.rune_pickaxe
            "dragon_pickaxe" -> gem_crab_objs.dragon_pickaxe
            "bronze_pickaxe" -> gem_crab_objs.bronze_pickaxe
            else -> name.toIntOrNull()?.let { objTypes[it] }
        }

    private fun setLevel(player: Player, stat: StatType, level: Int, exact: Boolean = false) {
        val target = level.coerceIn(1, 99)
        if (exact) {
            // Straight to the level's xp, down as well as up, with no xp drop (it isn't progress).
            player.statMap.setFineXP(stat, PlayerSkillXPTable.getFineXPFromLevel(target))
            player.statMap.setBaseLevel(stat, target.toByte())
            player.statMap.setCurrentLevel(stat, target.toByte())
            return
        }
        val xp = PlayerSkillXPTable.getXPFromLevel(target)
        val delta = xp - player.statMap.getXP(stat)
        if (delta > 0) {
            player.statMap.setCurrentLevel(stat, target.toByte())
            player.statAdvance(stat, delta.toDouble(), rate = 1.0, globalRate = 1.0)
        }
    }

    /** Puts the player a few tiles from the crab (or a given mine) with nothing in progress. */
    private fun teleportToCrab(player: Player, mineArg: Int, spread: Int) {
        val index =
            if (mineArg in GemstoneCrabRules.MINES.indices) mineArg
            else GemstoneCrabStatus.mineIndex
        val mine = GemstoneCrabRules.MINES[index]
        val crabSize = GemstoneCrabStatus.crab?.type?.size ?: 1
        val reach = crabSize / 2 + 2
        val range = maxOf(spread, reach + 2)
        repeat(TELEPORT_TRIES) {
            val dx = random.of(-range, range)
            val dz = random.of(-range, range)
            if (maxOf(kotlin.math.abs(dx), kotlin.math.abs(dz)) < reach) {
                return@repeat
            }
            val dest = CoordGrid(mine.crabX + dx, mine.crabZ + dz, 0)
            val flags = collision[dest.x, dest.z, dest.level]
            if (collision.isZoneValid(dest) && flags and BLOCKED_TILE == 0) {
                player.clearPendingAction(eventBus)
                PathingEntityCommon.telejump(player, collision, dest)
                return
            }
        }
        val fallback = CoordGrid(mine.exitX, mine.exitZ, 0)
        player.clearPendingAction(eventBus)
        PathingEntityCommon.telejump(player, collision, fallback)
    }

    private fun reset(conn: AgentConnection, cmd: JsonNode) {
        val player = conn.player
        if (player == null) {
            conn.send(event("error").put("error", "no player to reset"))
            return
        }
        if (player.isDelayed) {
            conn.pendingReset = cmd
            return
        }
        conn.pending.clear()
        conn.lastActionType = ""
        if (cmd.path("clear_gems").asBoolean(true)) {
            // Like a bank trip between episodes; a full inventory would drop new gems on the floor.
            for (slot in 0 until player.inv.size) {
                if (player.inv[slot]?.id in gemIds) {
                    player.inv[slot] = null
                }
            }
        }
        if (cmd.has("loadout")) {
            applyLoadout(player, cmd.path("loadout"))
        }
        teleportToCrab(player, cmd.path("mine").asInt(-1), cmd.path("spread").asInt(6))
        conn.send(event("reset_done").put("cycle", mapClock.cycle))
    }

    // ---- inputs ---------------------------------------------------------------------------

    private fun applyPending(conn: AgentConnection) {
        val player = conn.player ?: return
        if (conn.pending.isEmpty()) {
            return
        }
        for (action in conn.pending) {
            val why = apply(conn, player, action)
            val result = mapper.createObjectNode()
            result.set<JsonNode>("action", action)
            result.put("ok", why == null)
            if (why != null) {
                result.put("why", why)
            }
            conn.results.add(result)
        }
        conn.pending.clear()
    }

    /** Applies one input. Returns null when it went through, else why it did nothing. */
    private fun apply(conn: AgentConnection, player: Player, action: JsonNode): String? {
        val type = action.path("type").asText()
        val why =
            when (type) {
                "attack_crab" -> {
                    val crab = GemstoneCrabStatus.crab ?: return "no crab"
                    if (!inView(player, crab)) "crab not in view" else opNpc(player, crab, "Attack")
                }
                "mine_shell" -> {
                    val shell = nearestShell(player) ?: return "no shell"
                    opNpc(player, shell, "Mine")
                }
                "tunnel" -> {
                    val tunnel = nearestTunnel(player) ?: return "no tunnel nearby"
                    opLoc(player, tunnel, InteractionOp.Op1)
                }
                "npc" -> {
                    val npc = npcList[action.path("slot").asInt(-1)] ?: return "no such npc"
                    val op = action.path("op")
                    if (op.isInt) opNpc(player, npc, op.asInt())
                    else opNpc(player, npc, op.asText())
                }
                "walk" -> walk(player, action.path("x").asInt(), action.path("z").asInt())
                "walk_rel" ->
                    walk(
                        player,
                        player.x + action.path("dx").asInt(),
                        player.z + action.path("dz").asInt(),
                    )
                // Inputs the server never sees; accepted so the agent's log stays complete.
                "camera",
                "check_tab",
                "wait",
                "noop" -> null
                else -> "unknown action type '$type'"
            }
        if (why == null) {
            conn.lastActionType = type
            conn.lastActionCycle = mapClock.cycle
            if (type == "tunnel") {
                conn.lastTunnelCycle = mapClock.cycle
            }
        }
        return why
    }

    private fun opNpc(player: Player, npc: Npc, opName: String): String? {
        val slot = npc.visType.op.indexOfFirst { it.equals(opName, ignoreCase = true) }
        if (slot < 0) {
            return "npc has no '$opName' option"
        }
        return opNpc(player, npc, slot + 1)
    }

    /** Same steps as RSMod's OpNpcHandler for a client click. */
    private fun opNpc(player: Player, npc: Npc, opSlot: Int): String? {
        val op = InteractionOp.entries.firstOrNull { it.slot == opSlot } ?: return "bad op $opSlot"
        if (player.isDelayed) {
            return "busy"
        }
        if (npc.isDelayed) {
            return "npc busy"
        }
        if (!npcInteractions.hasOp(npc, player.vars, op)) {
            return "npc has no op $opSlot"
        }
        val interaction =
            InteractionNpcOp(
                target = npc,
                op = op,
                hasOpTrigger = npcInteractions.hasOpTrigger(player, npc, op),
                hasApTrigger = npcInteractions.hasApTrigger(player, npc, op),
            )
        player.clearPendingAction(eventBus)
        player.faceNpc(npc)
        player.interaction = interaction
        player.routeRequest = RouteRequestPathingEntity(npc.avatar, clientRequest = true)
        player.tempMoveSpeed = null
        return null
    }

    /** Same steps as RSMod's OpLocHandler for a client click. */
    private fun opLoc(player: Player, loc: LocInfo, op: InteractionOp): String? {
        if (player.isDelayed) {
            return "busy"
        }
        val type = locTypes[loc.id] ?: return "unknown loc"
        val bound = BoundLocInfo(loc, type)
        if (!locInteractions.hasOp(bound, type, player.vars, op)) {
            return "loc has no op ${op.slot}"
        }
        val interaction =
            InteractionLocOp(
                target = bound,
                op = op,
                hasOpTrigger = locInteractions.hasOpTrigger(player, bound, op, type),
                hasApTrigger = locInteractions.hasApTrigger(player, bound, op, type),
            )
        player.clearPendingAction(eventBus)
        player.resetFaceEntity()
        player.faceLoc(loc, type.width, type.length)
        player.interaction = interaction
        player.routeRequest =
            RouteRequestLoc(
                destination = loc.coords,
                width = type.width,
                length = type.length,
                shape = loc.entity.shape,
                angle = loc.entity.angle,
                forceApproachFlags = type.forceApproachFlags,
                clientRequest = true,
            )
        player.tempMoveSpeed = null
        return null
    }

    /** Same steps as RSMod's MoveGameClickHandler ("Walk here"). */
    private fun walk(player: Player, x: Int, z: Int): String? {
        if (player.isDelayed) {
            return "busy"
        }
        val dest = CoordGrid(x, z, player.level)
        if (dest.chebyshevDistance(player.coords) > VIEW_RADIUS) {
            return "too far"
        }
        player.clearPendingAction(eventBus)
        player.resetFaceEntity()
        player.routeRequest = RouteRequestCoord(dest, clientRequest = true)
        player.tempMoveSpeed = null
        return null
    }

    private fun inView(player: Player, npc: Npc): Boolean =
        npc.level == player.level && npc.coords.chebyshevDistance(player.coords) <= VIEW_RADIUS

    private fun nearestShell(player: Player): Npc? =
        GemstoneCrabStatus.shells
            .asSequence()
            .map { it.npc }
            .filter { it.isSlotAssigned && npcList[it.slotId] === it && inView(player, it) }
            .minByOrNull { it.coords.chebyshevDistance(player.coords) }

    private fun nearestTunnel(player: Player): LocInfo? {
        val tunnelId = gem_crab_locs.tunnel.id
        var best: LocInfo? = null
        var bestDistance = Int.MAX_VALUE
        for (mine in GemstoneCrabRules.MINES) {
            val near = CoordGrid(mine.tunnelX, mine.tunnelZ, 0)
            if (near.chebyshevDistance(player.coords) > VIEW_RADIUS + 8) {
                continue
            }
            for (dx in -1..1) {
                for (dz in -1..1) {
                    val zone = ZoneKey.from(CoordGrid(near.x + dx * 8, near.z + dz * 8, 0))
                    for (loc in locRegistry.findAll(zone)) {
                        if (loc.id != tunnelId) {
                            continue
                        }
                        val d = loc.coords.chebyshevDistance(player.coords)
                        if (d < bestDistance) {
                            best = loc
                            bestDistance = d
                        }
                    }
                }
            }
        }
        return best
    }

    // ---- observation ----------------------------------------------------------------------

    private fun snapshot(conn: AgentConnection, player: Player): ObjectNode {
        val cycle = mapClock.cycle
        val node = event("tick")
        node.put("cycle", cycle)
        node.put("x", player.x)
        node.put("z", player.z)
        node.put("level", player.level)
        node.put("run_energy", player.runEnergy / 100.0)
        node.put("running", player.varMoveSpeed == MoveSpeed.Run)
        val moved = conn.lastCoords != null && player.coords != conn.lastCoords
        conn.lastCoords = player.coords
        node.put("moving", moved || player.routeDestination.isNotEmpty())
        node.put("delayed", player.isDelayed)
        node.put("last_action", conn.lastActionType)
        node.put("last_action_cycle", conn.lastActionCycle)
        node.put("last_tunnel_cycle", conn.lastTunnelCycle)

        val interaction = player.interaction
        when (interaction) {
            is InteractionNpcOp ->
                node
                    .putObject("interaction")
                    .put("kind", "npc")
                    .put("slot", interaction.target.slotId)
                    .put("op", interaction.op.slot)
            is InteractionLocOp ->
                node
                    .putObject("interaction")
                    .put("kind", "loc")
                    .put("id", interaction.target.id)
                    .put("op", interaction.op.slot)
            else -> node.putNull("interaction")
        }
        node.put("face_npc", if (player.faceEntity.isNpc) player.faceEntity.npcSlot else -1)

        val levels = node.putObject("levels")
        val xp = node.putObject("xp")
        for ((name, stat) in TRACKED_STATS) {
            levels.put(name, player.statBase(stat))
            xp.put(name, player.statMap.getXP(stat))
        }

        val inv = node.putArray("inv")
        var gems = 0
        for (obj in player.inv) {
            if (obj == null) {
                continue
            }
            inv.addArray().add(obj.id).add(obj.count)
            if (obj.id in gemIds) {
                gems += obj.count
            }
        }
        node.put("gems", gems)

        val name = player.displayName
        val crab = GemstoneCrabStatus.crab
        if (crab != null) {
            val mine = GemstoneCrabRules.MINES[GemstoneCrabStatus.mineIndex]
            val crabNode = node.putObject("crab")
            crabNode.put("slot", crab.slotId)
            crabNode.put("mine", GemstoneCrabStatus.mineIndex)
            crabNode.put("x", mine.crabX)
            crabNode.put("z", mine.crabZ)
            crabNode.put("age", cycle - GemstoneCrabStatus.spawnCycle)
            crabNode.put("visible", inView(player, crab))
            val damage = GemstoneCrabStatus.damage
            val mineDamage = damage[name] ?: 0
            crabNode.put("my_damage", mineDamage)
            crabNode.put("rank", 1 + damage.values.count { it > mineDamage })
            crabNode.put("attackers", damage.size)
            val lastHit = GemstoneCrabStatus.lastHitCycle[name]
            crabNode.put("my_last_hit_cycle", lastHit ?: -1)
            val targetsCrab =
                (interaction is InteractionNpcOp && interaction.target === crab) ||
                    (player.faceEntity.isNpc &&
                        player.faceEntity.npcSlot == crab.slotId &&
                        lastHit != null &&
                        cycle - lastHit <= ATTACKING_GRACE_CYCLES)
            node.put("attacking", targetsCrab)
        } else {
            node.putNull("crab")
            node.put("attacking", false)
        }

        val shells = node.putArray("shells")
        for (shell in GemstoneCrabStatus.shells) {
            val left = GemstoneCrabRules.SHELL_TICKS - (cycle - shell.spawnCycle)
            val npc = shell.npc
            if (left <= 0 || !npc.isSlotAssigned || npcList[npc.slotId] !== npc) {
                continue
            }
            val mine = GemstoneCrabRules.MINES[shell.mineIndex]
            shells
                .addObject()
                .put("slot", npc.slotId)
                .put("mine", shell.mineIndex)
                .put("x", mine.crabX)
                .put("z", mine.crabZ)
                .put("ticks_left", left)
                .put("visible", inView(player, npc))
                .put("eligible", name in shell.eligible)
                .put("mined", name in shell.mined)
        }

        val others = node.putArray("players")
        for (other in playerList) {
            if (other === player || other.level != player.level) {
                continue
            }
            if (other.coords.chebyshevDistance(player.coords) > VIEW_RADIUS) {
                continue
            }
            val o = others.addObject().put("slot", other.slotId).put("x", other.x).put("z", other.z)
            val oi = other.interaction
            o.put("target_npc", if (oi is InteractionNpcOp) oi.target.slotId else -1)
        }

        val messages = node.putArray("messages")
        conn.botClient?.drain()?.forEach { messages.add(it) }

        val results: ArrayNode = node.putArray("results")
        conn.results.forEach { results.add(it) }
        return node
    }

    private val gemIds: Set<Int> by lazy {
        setOf(
            gem_crab_objs.uncut_opal.id,
            gem_crab_objs.uncut_jade.id,
            gem_crab_objs.uncut_red_topaz.id,
            gem_crab_objs.uncut_sapphire.id,
            gem_crab_objs.uncut_emerald.id,
            gem_crab_objs.uncut_ruby.id,
            gem_crab_objs.uncut_diamond.id,
            gem_crab_objs.uncut_dragonstone.id,
        )
    }

    private fun event(type: String): ObjectNode = mapper.createObjectNode().put("type", type)

    private companion object {
        const val PROTOCOL_VERSION = 1
        const val DEFAULT_PORT = 45600

        /** How far the agent can see and click, in tiles (the simulator uses 20 too). */
        const val VIEW_RADIUS = 20

        /**
         * Still counts as attacking this many cycles after the last hit (4-tick weapon + slack).
         */
        const val ATTACKING_GRACE_CYCLES = 6

        const val TELEPORT_TRIES = 40

        /** A tile a player can stand on has none of these (walls on its edges are fine). */
        const val BLOCKED_TILE =
            CollisionFlag.LOC or CollisionFlag.BLOCK_WALK or CollisionFlag.BLOCK_NPCS

        val TRACKED_STATS =
            listOf(
                "attack" to stats.attack,
                "strength" to stats.strength,
                "defence" to stats.defence,
                "hitpoints" to stats.hitpoints,
                "ranged" to stats.ranged,
                "magic" to stats.magic,
                "mining" to stats.mining,
            )
    }
}
