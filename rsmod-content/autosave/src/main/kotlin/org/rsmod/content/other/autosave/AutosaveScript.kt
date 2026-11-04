package org.rsmod.content.other.autosave

import com.github.michaelbull.logging.InlineLogger
import jakarta.inject.Inject
import org.rsmod.api.account.AccountManager
import org.rsmod.api.game.process.GameLifecycle
import org.rsmod.api.script.onEvent
import org.rsmod.game.entity.PlayerList
import org.rsmod.plugin.scripts.PluginScript
import org.rsmod.plugin.scripts.ScriptContext

/**
 * Local dev module: saves every logged-in player every few minutes of wall-clock time, so a crash
 * or a hard kill loses at most that much. RSMod otherwise saves only on logout.
 *
 * Wall-clock, not cycles, so it behaves the same with `RSMOD_TICK_MS=0`. The save runs on RSMod's
 * saving thread and reads the live player, so a save can catch a player mid-change; on this
 * single-player dev server that is an acceptable trade for not losing a session.
 */
class AutosaveScript
@Inject
constructor(private val playerList: PlayerList, private val accountManager: AccountManager) :
    PluginScript() {
    private val logger = InlineLogger()
    private val intervalMs =
        (System.getenv("RSMOD_AUTOSAVE_MINUTES")?.toLongOrNull() ?: 5L).coerceAtLeast(1L) * 60_000L
    private var nextSave = System.currentTimeMillis() + intervalMs

    override fun ScriptContext.startup() {
        onEvent<GameLifecycle.EndCycle> { maybeSave() }
    }

    private fun maybeSave() {
        val now = System.currentTimeMillis()
        if (now < nextSave) return
        nextSave = now + intervalMs
        var count = 0
        for (player in playerList) {
            accountManager.save(player) {}
            count++
        }
        if (count > 0) logger.info { "Autosave queued for $count player(s)" }
    }
}
