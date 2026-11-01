package org.rsmod.content.other.gemstonecrab

import kotlin.random.Random
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class GemstoneCrabRulesTest {
    @Test
    fun `next mine is always a different mine and both others are reachable`() {
        for (current in GemstoneCrabRules.MINES.indices) {
            val next = (0 until GemstoneCrabRules.MINES.size - 1).map { GemstoneCrabRules.nextMine(current, it) }.toSet()
            assertFalse(current in next)
            assertEquals(GemstoneCrabRules.MINES.size - 1, next.size)
        }
    }

    @Test
    fun `health bar drains with age and never reaches zero`() {
        assertEquals(GemstoneCrabRules.HITPOINTS_POOL, GemstoneCrabRules.hitpointsForAge(0))
        assertEquals(GemstoneCrabRules.HITPOINTS_POOL / 2, GemstoneCrabRules.hitpointsForAge(GemstoneCrabRules.ALIVE_TICKS / 2))
        assertEquals(GemstoneCrabRules.MIN_HITPOINTS, GemstoneCrabRules.hitpointsForAge(GemstoneCrabRules.ALIVE_TICKS))
        assertEquals(GemstoneCrabRules.MIN_HITPOINTS, GemstoneCrabRules.hitpointsForAge(GemstoneCrabRules.ALIVE_TICKS * 3))
        assertEquals(GemstoneCrabRules.HITPOINTS_POOL, GemstoneCrabRules.hitpointsForAge(-5))
    }

    @Test
    fun `top 16 by damage are eligible, ties go to the earlier hitter`() {
        val damage = (1..20).associate { "p$it" to 100 }.toMutableMap()
        damage["zero"] = 0
        val first = (1..20).associate { "p$it" to it }
        val ranked = GemstoneCrabRules.eligible(damage, first)
        assertEquals(GemstoneCrabRules.TOP_N, ranked.size)
        assertEquals((1..16).map { "p$it" }, ranked)
        assertFalse("zero" in ranked)
    }

    @Test
    fun `higher damage beats an earlier first hit`() {
        val ranked = GemstoneCrabRules.eligible(mapOf("a" to 5, "b" to 9), mapOf("a" to 1, "b" to 50))
        assertEquals(listOf("b", "a"), ranked)
    }

    @Test
    fun `announcement names one player in small fights and three otherwise`() {
        assertEquals(listOf("a"), GemstoneCrabRules.announced(listOf("a", "b", "c", "d")))
        assertEquals(listOf("a", "b", "c"), GemstoneCrabRules.announced(listOf("a", "b", "c", "d", "e")))
        assertEquals("Top crab crusher: a.", GemstoneCrabRules.msgTopCrushers(listOf("a")))
        assertEquals("Top crab crushers: a, b, c.", GemstoneCrabRules.msgTopCrushers(listOf("a", "b", "c")))
    }

    @Test
    fun `gem roll covers every weight bucket exactly`() {
        // Feed every value of the 32-wide table with the dragonstone roll missed.
        val counts = mutableMapOf<GemstoneCrabRules.Gem, Int>()
        for (v in 0 until 32) {
            var call = 0
            val gem = GemstoneCrabRules.rollGem { n -> if (call++ == 0) n - 1 else v }
            counts[gem] = (counts[gem] ?: 0) + 1
        }
        assertEquals(mapOf(
            GemstoneCrabRules.Gem.OPAL to 9, GemstoneCrabRules.Gem.JADE to 9, GemstoneCrabRules.Gem.RED_TOPAZ to 6,
            GemstoneCrabRules.Gem.SAPPHIRE to 3, GemstoneCrabRules.Gem.EMERALD to 2, GemstoneCrabRules.Gem.RUBY to 2,
            GemstoneCrabRules.Gem.DIAMOND to 1,
        ), counts)
        assertEquals(GemstoneCrabRules.Gem.DRAGONSTONE, GemstoneCrabRules.rollGem { 0 })
    }

    @Test
    fun `dragonstone rate is close to one in five hundred`() {
        val rng = Random(1)
        val n = 500_000
        val dragons = (0 until n).count { GemstoneCrabRules.rollGem { k -> rng.nextInt(k) } == GemstoneCrabRules.Gem.DRAGONSTONE }
        assertTrue(dragons in 850..1150) { "got $dragons" }
    }

    @Test
    fun `gem message matches the game`() {
        assertEquals("You mine an uncut red topaz from the crab shell.", GemstoneCrabRules.msgGem(GemstoneCrabRules.Gem.RED_TOPAZ))
    }
}
