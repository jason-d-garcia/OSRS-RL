package org.rsmod.content.areas.tutorialisland

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class TutorialRulesTest {
    @Test
    fun `stages are in increasing order and every stage has a hint`() {
        val stages = TutorialRules.STAGES
        assertEquals(stages.sorted(), stages)
        assertEquals(stages.size, stages.toSet().size)
        for (stage in stages) {
            assertTrue(TutorialRules.hint(stage).isNotBlank(), "no hint for $stage")
        }
    }

    @Test
    fun `stage values match the live run`() {
        // From the recorded run of 4 Nov 2026 (docs/tutorial-island.md).
        assertEquals(
            listOf(2, 3, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 120, 130),
            TutorialRules.STAGES,
        )
    }

    @Test
    fun `next walks the stage list and stops at the end`() {
        assertEquals(TutorialRules.OPEN_SETTINGS, TutorialRules.next(TutorialRules.TALK_TO_GUIDE))
        assertEquals(TutorialRules.MEET_CHEF, TutorialRules.next(TutorialRules.MEET_CHEF))
    }

    @Test
    fun `the guide's door opens only once the guide is done`() {
        val door = TutorialRules.Gate.GUIDE_DOOR
        assertFalse(TutorialRules.canPass(door, TutorialRules.TALK_TO_GUIDE_AGAIN))
        assertTrue(TutorialRules.canPass(door, TutorialRules.OPEN_DOOR))
        assertTrue(TutorialRules.canPass(door, TutorialRules.CHOP_TREE))
        assertEquals(TutorialRules.TALK_TO_SURVIVAL, TutorialRules.afterPassing(door, TutorialRules.OPEN_DOOR))
        // Going back out later doesn't move progress.
        assertEquals(TutorialRules.CHOP_TREE, TutorialRules.afterPassing(door, TutorialRules.CHOP_TREE))
    }

    @Test
    fun `the survival gate opens only after cooking`() {
        val gate = TutorialRules.Gate.SURVIVAL_GATE
        assertFalse(TutorialRules.canPass(gate, TutorialRules.COOK_SHRIMP))
        assertTrue(TutorialRules.canPass(gate, TutorialRules.LEAVE_SURVIVAL))
        assertEquals(TutorialRules.MEET_CHEF, TutorialRules.afterPassing(gate, TutorialRules.LEAVE_SURVIVAL))
    }

    @Test
    fun `tabs advance only their own stage`() {
        assertEquals(TutorialRules.TALK_TO_GUIDE_AGAIN, TutorialRules.afterTab(TutorialRules.Tab.SETTINGS, TutorialRules.OPEN_SETTINGS))
        assertEquals(TutorialRules.CATCH_SHRIMP, TutorialRules.afterTab(TutorialRules.Tab.INVENTORY, TutorialRules.OPEN_INVENTORY))
        assertEquals(TutorialRules.TALK_TO_SURVIVAL_AGAIN, TutorialRules.afterTab(TutorialRules.Tab.SKILLS, TutorialRules.OPEN_SKILLS))
        assertEquals(TutorialRules.OPEN_SETTINGS, TutorialRules.afterTab(TutorialRules.Tab.INVENTORY, TutorialRules.OPEN_SETTINGS))
        assertEquals(TutorialRules.CHOP_TREE, TutorialRules.afterTab(TutorialRules.Tab.SETTINGS, TutorialRules.CHOP_TREE))
    }

    @Test
    fun `a player starting at a stage carries what the earlier stages gave`() {
        assertEquals(emptyList<TutorialRules.Item>(), TutorialRules.itemsFor(TutorialRules.TALK_TO_GUIDE))
        assertEquals(listOf(TutorialRules.Item.NET), TutorialRules.itemsFor(TutorialRules.CATCH_SHRIMP))
        assertTrue(TutorialRules.Item.LOGS in TutorialRules.itemsFor(TutorialRules.LIGHT_FIRE))
        assertTrue(TutorialRules.Item.RAW_SHRIMPS in TutorialRules.itemsFor(TutorialRules.COOK_SHRIMP))
        assertFalse(TutorialRules.Item.RAW_SHRIMPS in TutorialRules.itemsFor(TutorialRules.LEAVE_SURVIVAL))
    }

    @Test
    fun `start tiles are on the island`() {
        for (stage in TutorialRules.STAGES) {
            val tile = TutorialRules.startTile(stage)
            assertTrue(tile.x in 3072..3135 && tile.z in 3072..3135, "$stage starts at $tile")
        }
    }
}
