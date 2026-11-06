package com.osrsrl.recorder;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNull;

import java.awt.Rectangle;
import org.junit.Test;

public class LayoutBoxTest
{
	@Test
	public void boxIsXYWidthHeight()
	{
		assertArrayEquals(new int[]{516, 4, 160, 200}, RecorderPlugin.box(new Rectangle(516, 4, 160, 200)));
	}

	@Test
	public void emptyOrMissingBoundsGiveNull()
	{
		assertNull(RecorderPlugin.box(null));
		assertNull(RecorderPlugin.box(new Rectangle(10, 10, 0, 50)));
		assertNull(RecorderPlugin.box(new Rectangle(10, 10, 50, -1)));
	}
}
