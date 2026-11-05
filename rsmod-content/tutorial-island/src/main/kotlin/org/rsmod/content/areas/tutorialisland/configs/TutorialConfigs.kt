@file:Suppress("unused", "SpellCheckingInspection")

package org.rsmod.content.areas.tutorialisland.configs

import org.rsmod.api.type.editors.npc.NpcEditor
import org.rsmod.api.type.refs.comp.ComponentReferences
import org.rsmod.api.type.refs.loc.LocReferences
import org.rsmod.api.type.refs.npc.NpcReferences
import org.rsmod.api.type.refs.obj.ObjReferences
import org.rsmod.api.type.refs.seq.SeqReferences
import org.rsmod.api.type.refs.varp.VarpReferences

typealias tut_npcs = TutorialNpcs

typealias tut_locs = TutorialLocs

typealias tut_objs = TutorialObjs

typealias tut_seqs = TutorialSeqs

typealias tut_varps = TutorialVarps

typealias tut_components = TutorialComponents

// The island's own types are the "newbie" ones in the cache; the ids match a live run's clicks.

object TutorialNpcs : NpcReferences() {
    val gielinor_guide = find("newbie_basics_instructor")
    val survival_expert = find("newbie_survival_instructor")
    val fishing_spot = find("0_48_48_newbiefishing")
    val master_chef = find("newbie_cook_instructor")
}

object TutorialLocs : LocReferences() {
    val guide_door = find("newbie_door1")
    val survival_gate_left = find("newbiegateclosedl2")
    val survival_gate_right = find("newbiegateclosedr2")
    val tree = find("newbietree")
    val fire = find("fire")
}

object TutorialObjs : ObjReferences() {
    val small_fishing_net = find("net")
    val bronze_axe = find("bronze_axe")
    val tinderbox = find("tinderbox")
    val logs = find("newbielogs")
    val raw_shrimps = find("newbieraw_shrimp")
    val shrimps = find("shrimp")
}

object TutorialSeqs : SeqReferences() {
    val fishing_net = find("human_smallnet")
    val woodcutting_bronze = find("human_woodcutting_bronze_axe")
    val light_fire = find("human_createfire")
    val cook_on_fire = find("human_firecooking")
}

object TutorialVarps : VarpReferences() {
    val tutorial = find("tutorial")
}

/** The side-panel tab buttons ("stones") on each of the three game frames. */
object TutorialComponents : ComponentReferences() {
    val fixed = (0..13).map { find("toplevel:stone$it") }
    val resizable = (0..13).map { find("toplevel_osrs_stretch:stone$it") }
    val resizable_bottom = (0..13).map { find("toplevel_pre_eoc:stone$it") }
}

internal object TutorialNpcEditor : NpcEditor() {
    init {
        edit(tut_npcs.gielinor_guide) {
            wanderRange = 2
            maxRange = 4
        }
        edit(tut_npcs.survival_expert) {
            wanderRange = 2
            maxRange = 4
        }
        edit(tut_npcs.master_chef) {
            wanderRange = 2
            maxRange = 4
        }
        edit(tut_npcs.fishing_spot) {
            wanderRange = 0
            maxRange = 0
        }
    }
}
