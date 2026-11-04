package org.rsmod.content.other.agentbridge

import org.rsmod.api.type.refs.obj.ObjReferences

/** Weapons a bot can be given with `spawn {"weapon": ...}`. */
object AgentBridgeObjs : ObjReferences() {
    val dragon_scimitar = find("dragon_scimitar")
    val abyssal_whip = find("abyssal_whip")
    val dragon_dagger = find("dragon_dagger")
}
