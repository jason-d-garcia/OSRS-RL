package org.rsmod.content.other.pvptoggle

import org.rsmod.api.config.refs.modlevels
import org.rsmod.api.player.output.MiscOutput
import org.rsmod.api.player.output.mes
import org.rsmod.api.script.onCommand
import org.rsmod.plugin.scripts.PluginScript
import org.rsmod.plugin.scripts.ScriptContext

/**
 * Local dev command: `::pvp` toggles the "Attack" right-click option on other players for the
 * player who types it. RSMod's PvP combat already works everywhere; the login script just hides
 * the option. Not persisted: it resets on relog. Stateless on purpose: an
 * earlier toggle kept per-player state across relogs and silently turned the option off.
 */
class PvpToggleScript : PluginScript() {

    override fun ScriptContext.startup() {
        onCommand("pvp") {
            modLevel = modlevels.admin
            desc = "Show the Attack option on players (::pvp off to hide it)"
            cheat {
                val on = args.firstOrNull()?.lowercase() != "off"
                MiscOutput.setPlayerOp(player, slot = 2, op = if (on) "Attack" else null)
                player.mes(if (on) "PvP on: you can attack players." else "PvP off.")
            }
        }
    }
}
