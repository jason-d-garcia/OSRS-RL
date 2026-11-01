package com.osrsrl.recorder;

import net.runelite.client.RuneLite;
import net.runelite.client.externalplugins.ExternalPluginManager;

/** Starts RuneLite with the recorder loaded, in developer mode. Used by `gradlew run`. */
public class RecorderPluginLauncher
{
	public static void main(String[] args) throws Exception
	{
		ExternalPluginManager.loadBuiltin(RecorderPlugin.class);
		RuneLite.main(args);
	}
}
