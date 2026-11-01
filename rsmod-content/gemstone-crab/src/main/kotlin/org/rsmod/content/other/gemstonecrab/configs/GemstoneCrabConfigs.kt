package org.rsmod.content.other.gemstonecrab.configs

import org.rsmod.api.config.refs.params
import org.rsmod.api.type.editors.npc.NpcEditor
import org.rsmod.api.type.refs.loc.LocReferences
import org.rsmod.api.type.refs.npc.NpcReferences
import org.rsmod.api.type.refs.obj.ObjReferences
import org.rsmod.api.type.refs.seq.SeqReferences
import org.rsmod.content.other.gemstonecrab.GemstoneCrabRules

typealias gem_crab_npcs = GemstoneCrabNpcs

typealias gem_crab_locs = GemstoneCrabLocs

typealias gem_crab_objs = GemstoneCrabObjs

typealias gem_crab_seqs = GemstoneCrabSeqs

object GemstoneCrabNpcs : NpcReferences() {
    val crab = find("gemstone_crab")
    val shell = find("gemstone_crab_remains")
}

object GemstoneCrabLocs : LocReferences() {
    val tunnel = find("cave_rock02_entrance01_gemstone")
}

object GemstoneCrabObjs : ObjReferences() {
    val uncut_opal = find("uncut_opal")
    val uncut_jade = find("uncut_jade")
    val uncut_red_topaz = find("uncut_red_topaz")
    val uncut_sapphire = find("uncut_sapphire")
    val uncut_emerald = find("uncut_emerald")
    val uncut_ruby = find("uncut_ruby")
    val uncut_diamond = find("uncut_diamond")
    val uncut_dragonstone = find("uncut_dragonstone")

    val bronze_pickaxe = find("bronze_pickaxe")
    val iron_pickaxe = find("iron_pickaxe")
    val steel_pickaxe = find("steel_pickaxe")
    val black_pickaxe = find("black_pickaxe")
    val mithril_pickaxe = find("mithril_pickaxe")
    val adamant_pickaxe = find("adamant_pickaxe")
    val rune_pickaxe = find("rune_pickaxe")
    val dragon_pickaxe = find("dragon_pickaxe")
}

object GemstoneCrabSeqs : SeqReferences() {
    val mining_bronze = find("human_mining_bronze_pickaxe")
    val mining_iron = find("human_mining_iron_pickaxe")
    val mining_steel = find("human_mining_steel_pickaxe")
    val mining_black = find("human_mining_black_pickaxe")
    val mining_mithril = find("human_mining_mithril_pickaxe")
    val mining_adamant = find("human_mining_adamant_pickaxe")
    val mining_rune = find("human_mining_rune_pickaxe")
    val mining_dragon = find("human_mining_dragon_pickaxe")
}

internal object GemstoneCrabNpcEditor : NpcEditor() {
    init {
        edit(gem_crab_npcs.crab) {
            hitpoints = GemstoneCrabRules.HITPOINTS_POOL
            timer = GemstoneCrabRules.TIMER_INTERVAL
            respawnRate = Int.MAX_VALUE
            giveChase = false
            defaultMode = none
            param[params.npc_com_xp_multiplier] = GemstoneCrabRules.XP_MULTIPLIER
        }
        edit(gem_crab_npcs.shell) {
            defaultMode = none
        }
    }
}
