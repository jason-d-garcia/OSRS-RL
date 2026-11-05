package org.rsmod.content.other.agentbridge

import net.rsprot.protocol.api.NetworkService
import net.rsprot.protocol.game.outgoing.info.playerinfo.PlayerAvatarExtendedInfo
import net.rsprot.protocol.game.outgoing.info.playerinfo.PlayerInfo
import net.rsprot.protocol.game.outgoing.info.util.BuildArea
import net.rsprot.protocol.game.outgoing.misc.player.MessageGame
import net.rsprot.protocol.game.outgoing.misc.player.RunClientScript
import org.rsmod.api.config.refs.baseanimsets
import org.rsmod.api.config.refs.params
import org.rsmod.api.player.righthand
import org.rsmod.game.client.Client
import org.rsmod.game.client.ClientCycle
import org.rsmod.game.entity.Player
import org.rsmod.game.headbar.Headbar
import org.rsmod.game.hit.Hitmark
import org.rsmod.game.movement.MoveSpeed
import org.rsmod.game.seq.EntitySeq
import org.rsmod.game.type.obj.ObjTypeList
import org.rsmod.game.type.obj.Wearpos
import org.rsmod.map.CoordGrid
import org.rsmod.map.zone.ZoneKey

/**
 * The "client" of a headless bot. Nothing is sent anywhere; game messages are kept so the bridge
 * can pass them to the agent ("You don't have enough damage to mine this", burrow messages).
 */
class BotClient : Client<Any, Any> {
    val messages = ArrayDeque<String>()

    /**
     * The options of the last multiple-choice dialogue the server opened (clientscript 58,
     * `chatbox_multi_init`, whose second argument is the options joined with `|`).
     */
    var lastChoices: List<String> = emptyList()
        private set

    override fun close() {}

    override fun write(message: Any) {
        if (message is RunClientScript && message.id == CHATBOX_MULTI_INIT) {
            val joined = message.values.getOrNull(1) as? String
            lastChoices = joined?.split('|') ?: emptyList()
        }
        if (message is MessageGame) {
            messages.addLast(message.message)
            while (messages.size > MAX_MESSAGES) {
                messages.removeFirst()
            }
        }
    }

    override fun read(player: Player) {}

    override fun flush() {}

    override fun flushHighPriority() {}

    override fun unregister(service: Any, player: Player) {}

    fun drain(): List<String> {
        val out = messages.toList()
        messages.clear()
        return out
    }

    private companion object {
        const val MAX_MESSAGES = 64
        const val CHATBOX_MULTI_INIT = 58
    }
}

/**
 * Keeps a headless bot's rsprot player-info avatar in step with the player, so real clients near it
 * (your RuneLite) see it walk, animate and fight like anyone else.
 *
 * This is the avatar half of RSMod's `RspCycle` (api/net) without the packets that only matter to a
 * connected client: map rebuilds, npc info and the player's own view. The bot's own player info is
 * still computed by rsprot every cycle, so [flush] consumes and releases it.
 */
@OptIn(ExperimentalUnsignedTypes::class)
class BotCycle(
    private val service: NetworkService<Player>,
    private val playerInfo: PlayerInfo,
    private val objTypes: ObjTypeList,
) : ClientCycle {
    private var knownCoords: CoordGrid = CoordGrid.ZERO
    private var knownBuildArea: CoordGrid = CoordGrid.NULL
    private var knownCachedSpeed: MoveSpeed = MoveSpeed.Stationary
    private var knownFaceEntity: Int? = -1
    private var released = false

    private val info: PlayerAvatarExtendedInfo
        get() = playerInfo.avatar.extendedInfo

    override fun update(player: Player) {
        player.updateMoveSpeed()
        player.updateCoords()
        player.updateBuildArea()
        player.applyFacePathingEntity()
        player.applyAnim()
        player.applyHeadbars()
        player.applyHitmarks()
        player.syncAppearance()
    }

    override fun flush(player: Player) {
        playerInfo.toPacket().safeRelease()
    }

    override fun release() {
        if (!released) {
            playerInfo.toPacket().safeRelease()
        }
    }

    /** Hands the avatar back to rsprot when the bot logs out. */
    fun dealloc() {
        if (!released) {
            released = true
            service.playerInfoProtocol.dealloc(playerInfo)
        }
    }

    private fun Player.updateMoveSpeed() {
        if (knownCachedSpeed != cachedMoveSpeed) {
            info.setMoveSpeed(cachedMoveSpeed.steps)
            knownCachedSpeed = cachedMoveSpeed
        }
        val moveSpeed =
            when {
                pendingTelejump -> MoveSpeed.Stationary
                pendingTeleport -> MoveSpeed.Walk
                pendingStepCount == 1 -> MoveSpeed.Walk
                pendingStepCount == 2 -> MoveSpeed.Run
                else -> moveSpeed
            }
        if (moveSpeed != cachedMoveSpeed && coords != knownCoords) {
            info.setTempMoveSpeed(moveSpeed.steps)
        }
    }

    private fun Player.updateCoords() {
        playerInfo.updateCoord(level, x, z)
        playerInfo.updateRenderCoord(WORLD_ID, level, x, z)
        knownCoords = coords
    }

    private fun Player.updateBuildArea() {
        if (knownBuildArea != buildArea && buildArea != CoordGrid.NULL) {
            val zone = ZoneKey.from(buildArea)
            playerInfo.updateBuildArea(WORLD_ID, BuildArea(zone.x, zone.z))
            knownBuildArea = buildArea
        }
    }

    private fun Player.applyFacePathingEntity() {
        val slot = faceEntity.entitySlot
        if (knownFaceEntity != slot) {
            info.setFacePathingEntity(slot)
            knownFaceEntity = slot
        }
    }

    private fun Player.applyAnim() {
        when (pendingSequence) {
            EntitySeq.NULL -> return
            EntitySeq.ZERO -> info.setSequence(-1, 0)
            else -> info.setSequence(pendingSequence.id, pendingSequence.delay)
        }
    }

    private fun Player.applyHeadbars() {
        for (packed in activeHeadbars.longIterator()) {
            val headbar = Headbar(packed)
            info.addHeadBar(
                sourceIndex = if (headbar.isNoSource) -1 else headbar.sourceSlot,
                selfType = headbar.self,
                otherType = if (headbar.isPrivate) -1 else headbar.public,
                startFill = headbar.startFill,
                endFill = headbar.endFill,
                startTime = headbar.startTime,
                endTime = headbar.endTime,
            )
        }
    }

    private fun Player.applyHitmarks() {
        for (packed in activeHitmarks.longIterator()) {
            val hitmark = Hitmark(packed)
            info.addHitMark(
                sourceIndex = if (hitmark.isNoSource) -1 else hitmark.sourceSlot,
                selfType = hitmark.self,
                sourceType = hitmark.source,
                otherType = if (hitmark.isPrivate) -1 else hitmark.public,
                value = hitmark.damage,
                delay = hitmark.delay,
            )
        }
    }

    private fun Player.syncAppearance() {
        if (!appearance.rebuild) {
            return
        }
        val colours = appearance.coloursSnapshot()
        for (i in colours.indices) {
            info.setColour(i, colours[i].toInt())
        }
        val identKit = appearance.identKitSnapshot()
        for (i in identKit.indices) {
            info.setIdentKit(i, identKit[i].toInt())
        }
        info.setName(displayName)
        info.setOverheadIcon(overheadIcon ?: -1)
        info.setSkullIcon(skullIcon ?: -1)
        info.setCombatLevel(combatLevel)
        info.setBodyType(appearance.bodyType)
        info.setPronoun(appearance.pronoun)
        info.setHidden(appearance.softHidden)
        info.setNameExtras(beforeName = "", afterName = "", afterCombatLevel = "")

        val weapon = righthand
        if (weapon != null) {
            val type = objTypes[weapon]
            info.setBaseAnimationSet(
                readyAnim = type.param(params.bas_readyanim).id,
                turnAnim = type.param(params.bas_turnonspot).id,
                walkAnim = type.param(params.bas_walk_f).id,
                walkAnimBack = type.param(params.bas_walk_b).id,
                walkAnimLeft = type.param(params.bas_walk_l).id,
                walkAnimRight = type.param(params.bas_walk_r).id,
                runAnim = type.param(params.bas_running).id,
            )
        } else {
            val default = baseanimsets.human_default
            info.setBaseAnimationSet(
                readyAnim = default.readyAnim.id,
                turnAnim = default.turnOnSpot.id,
                walkAnim = default.walkForward.id,
                walkAnimBack = default.walkBack.id,
                walkAnimLeft = default.walkLeft.id,
                walkAnimRight = default.walkRight.id,
                runAnim = default.running.id,
            )
        }
        info.setTransmogrification(-1)
        for (wearpos in Wearpos.visibleWearpos) {
            val obj = worn[wearpos.slot]
            if (obj == null) {
                info.setWornObj(wearpos.slot, -1, -1, -1)
                continue
            }
            val objType = objTypes[obj]
            info.setWornObj(wearpos.slot, obj.id, objType.wearpos2, objType.wearpos3)
        }
    }

    private companion object {
        const val WORLD_ID = -1
    }
}
