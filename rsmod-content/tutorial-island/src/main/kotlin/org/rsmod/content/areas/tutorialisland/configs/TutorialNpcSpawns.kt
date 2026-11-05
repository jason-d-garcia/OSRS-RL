package org.rsmod.content.areas.tutorialisland.configs

import org.rsmod.api.type.builders.map.npc.MapNpcSpawnBuilder
import org.rsmod.content.areas.tutorialisland.scripts.TutorialIsland

object TutorialNpcSpawns : MapNpcSpawnBuilder() {
    override fun onPackMapTask() {
        resourceFile<TutorialIsland>("npcs.toml")
    }
}
