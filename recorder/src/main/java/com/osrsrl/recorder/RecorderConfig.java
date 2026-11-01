package com.osrsrl.recorder;

import net.runelite.client.config.Config;
import net.runelite.client.config.ConfigGroup;
import net.runelite.client.config.ConfigItem;
import net.runelite.client.config.Range;

@ConfigGroup("osrsrlrecorder")
public interface RecorderConfig extends Config
{
	@ConfigItem(
		keyName = "recording",
		name = "Recording",
		description = "Write a session file while logged in. Turn off to pause.",
		position = 0
	)
	default boolean recording()
	{
		return true;
	}

	@ConfigItem(
		keyName = "mouseHz",
		name = "Max mouse sample rate (Hz)",
		description = "Mouse moves faster than this are thinned out",
		position = 1
	)
	@Range(min = 20, max = 250)
	default int mouseHz()
	{
		return 100;
	}

	@ConfigItem(
		keyName = "recordTypedText",
		name = "Record typed characters",
		description = "Off: letters and digits are stored as 'char' (timing only), so chat and messages are not saved",
		position = 2
	)
	default boolean recordTypedText()
	{
		return false;
	}

	@ConfigItem(
		keyName = "radius",
		name = "Scene radius (tiles)",
		description = "NPCs, players and objects within this many tiles are logged each tick",
		position = 3
	)
	@Range(min = 5, max = 30)
	default int radius()
	{
		return 15;
	}
}
