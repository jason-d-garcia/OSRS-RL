package com.osrsrl.recorder;

import com.google.gson.Gson;
import com.google.inject.Provides;
import java.awt.Dimension;
import java.awt.Rectangle;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.inject.Inject;
import lombok.extern.slf4j.Slf4j;
import net.runelite.api.Actor;
import net.runelite.api.ChatMessageType;
import net.runelite.api.Client;
import net.runelite.api.GameObject;
import net.runelite.api.GameState;
import net.runelite.api.GroundObject;
import net.runelite.api.HeadIcon;
import net.runelite.api.Hitsplat;
import net.runelite.api.Item;
import net.runelite.api.ItemContainer;
import net.runelite.api.MenuEntry;
import net.runelite.api.NPC;
import net.runelite.api.Player;
import net.runelite.api.PlayerComposition;
import net.runelite.api.Prayer;
import net.runelite.api.Scene;
import net.runelite.api.Skill;
import net.runelite.api.VarPlayer;
import net.runelite.api.Tile;
import net.runelite.api.TileObject;
import net.runelite.api.WallObject;
import net.runelite.api.WorldView;
import net.runelite.api.coords.LocalPoint;
import net.runelite.api.coords.WorldPoint;
import net.runelite.api.events.AnimationChanged;
import net.runelite.api.events.ChatMessage;
import net.runelite.api.events.FocusChanged;
import net.runelite.api.events.GameStateChanged;
import net.runelite.api.events.GameTick;
import net.runelite.api.events.HitsplatApplied;
import net.runelite.api.events.InteractingChanged;
import net.runelite.api.events.MenuOptionClicked;
import net.runelite.api.events.StatChanged;
import net.runelite.api.events.WidgetClosed;
import net.runelite.api.events.WidgetLoaded;
import net.runelite.api.ActorSpotAnim;
import net.runelite.api.gameval.InterfaceID;
import net.runelite.api.gameval.InventoryID;
import net.runelite.api.widgets.Widget;
import net.runelite.client.RuneLite;
import net.runelite.client.config.ConfigManager;
import net.runelite.client.eventbus.Subscribe;
import net.runelite.client.events.ConfigChanged;
import net.runelite.client.input.KeyListener;
import net.runelite.client.input.KeyManager;
import net.runelite.client.input.MouseAdapter;
import net.runelite.client.input.MouseManager;
import net.runelite.client.input.MouseWheelListener;
import net.runelite.client.plugins.Plugin;
import net.runelite.client.plugins.PluginDescriptor;
import net.runelite.client.util.Text;

/**
 * Passive session recorder for the OSRS-RL project.
 *
 * It only listens. It never sends input, never changes menus and never consumes
 * an event: every listener returns the event it was given, unchanged.
 *
 * Output: ~/.runelite/osrs-rl-recordings/session-YYYYMMDD-HHMMSS.jsonl.gz, one
 * JSON object per line, each with "t" (epoch ms) and "type". See
 * recorder/README.md for the schema.
 */
@Slf4j
@PluginDescriptor(
	name = "OSRS-RL Recorder",
	description = "Records mouse, keys, menu clicks and per-tick game state to a local file",
	tags = {"recording", "research"}
)
public class RecorderPlugin extends Plugin
{
	static final int SCHEMA_VERSION = 7;

	/** Tutorial Island progress varp (0 at the start, 1000 once the island is done). */
	static final int VARP_TUTORIAL = 281;

	// Chatbox dialogue interfaces: {group, child holding the text}. Raw ids so the
	// plugin does not depend on gameval constant names that move between releases.
	static final int GROUP_NPC_CHAT = 231;
	static final int GROUP_PLAYER_CHAT = 217;
	static final int GROUP_OPTIONS = 219;
	static final int GROUP_SPRITE = 193;
	static final int GROUP_MESBOX = 229;
	static final int GROUP_TUTORIAL_TEXT = 263;
	static final File OUT_DIR = new File(RuneLite.RUNELITE_DIR, "osrs-rl-recordings");

	// Screen regions written to ui_layout records, each tried in order (fixed, resizable classic,
	// resizable modern) and the first visible one kept. Component ids from gameval.
	static final int[] XP_DROPS = {InterfaceID.XpDrops.CONTAINER, InterfaceID.Toplevel.XP_DROPS,
		InterfaceID.ToplevelOsrsStretch.XP_DROPS, InterfaceID.ToplevelPreEoc.XP_DROPS};
	static final int[] XP_COUNTER = {InterfaceID.XpDrops.COUNTER_CONTAINER};
	static final int[] XP_ORB = {InterfaceID.Orbs.XP_DROPS};
	static final int[] MINIMAP = {InterfaceID.Toplevel.MINIMAP, InterfaceID.ToplevelOsrsStretch.MINIMAP,
		InterfaceID.ToplevelPreEoc.MINIMAP};
	static final int[] INVENTORY = {InterfaceID.Inventory.ITEMS};
	static final int[] CHATBOX = {InterfaceID.Chatbox.CHATAREA};
	/** Height of the mouseover text line drawn at the top-left of the game view. */
	static final int HOVER_TEXT_H = 24;

	@Inject
	private Client client;

	@Inject
	private RecorderConfig config;

	@Inject
	private MouseManager mouseManager;

	@Inject
	private KeyManager keyManager;

	@Inject
	private Gson gson;

	private volatile SessionWriter writer;
	private volatile long lastMoveMs;
	private int lastYaw = -1;
	private int lastPitch = -1;
	private String lastDialogue = "";
	private String lastLayout = "";
	// Per-session anonymous ids for other players, so one person can be followed from tick to
	// tick without storing anything that identifies them. Keyed by the client's player slot and
	// combat level; both reset when a session starts. Never written anywhere but this map.
	private final Map<Long, Integer> bystanderIds = new LinkedHashMap<>();
	private static final Skill[] COMBAT_STATS = {Skill.ATTACK, Skill.STRENGTH, Skill.DEFENCE, Skill.RANGED, Skill.MAGIC};
	// KitType ordinals of the slots whose item is visible on another player, in the order written to equip[]:
	// head, cape, amulet, weapon, torso, shield, legs, hands, boots.
	private static final int[] EQUIP_SLOTS = {0, 1, 2, 3, 4, 5, 7, 9, 10};

	private final MouseAdapter mouseListener = new MouseAdapter()
	{
		@Override
		public MouseEvent mouseMoved(MouseEvent e)
		{
			move(e, "mouse_move");
			return e;
		}

		@Override
		public MouseEvent mouseDragged(MouseEvent e)
		{
			move(e, "mouse_drag");
			return e;
		}

		@Override
		public MouseEvent mousePressed(MouseEvent e)
		{
			button(e, "mouse_press");
			return e;
		}

		@Override
		public MouseEvent mouseReleased(MouseEvent e)
		{
			button(e, "mouse_release");
			return e;
		}

		@Override
		public MouseEvent mouseEntered(MouseEvent e)
		{
			emit(rec("mouse_enter"));
			return e;
		}

		@Override
		public MouseEvent mouseExited(MouseEvent e)
		{
			emit(rec("mouse_exit"));
			return e;
		}
	};

	private final MouseWheelListener wheelListener = e ->
	{
		Map<String, Object> r = rec("wheel");
		r.put("x", e.getX());
		r.put("y", e.getY());
		r.put("rot", e.getWheelRotation());
		emit(r);
		return e;
	};

	private final KeyListener keyListener = new KeyListener()
	{
		@Override
		public void keyTyped(KeyEvent e)
		{
		}

		@Override
		public void keyPressed(KeyEvent e)
		{
			key(e, "key_press");
		}

		@Override
		public void keyReleased(KeyEvent e)
		{
			key(e, "key_release");
		}
	};

	@Provides
	RecorderConfig provideConfig(ConfigManager configManager)
	{
		return configManager.getConfig(RecorderConfig.class);
	}

	@Override
	protected void startUp()
	{
		mouseManager.registerMouseListener(mouseListener);
		mouseManager.registerMouseWheelListener(wheelListener);
		keyManager.registerKeyListener(keyListener);
		if (client.getGameState() == GameState.LOGGED_IN)
		{
			openSession();
		}
	}

	@Override
	protected void shutDown()
	{
		mouseManager.unregisterMouseListener(mouseListener);
		mouseManager.unregisterMouseWheelListener(wheelListener);
		keyManager.unregisterKeyListener(keyListener);
		closeSession("plugin_stopped");
	}

	// --- session lifecycle -------------------------------------------------

	@Subscribe
	public void onGameStateChanged(GameStateChanged e)
	{
		GameState s = e.getGameState();
		if (s == GameState.LOGGED_IN && writer == null)
		{
			openSession();
		}
		else if (s == GameState.LOGIN_SCREEN || s == GameState.CONNECTION_LOST)
		{
			closeSession(s.name().toLowerCase());
		}
		Map<String, Object> r = rec("game_state");
		r.put("state", s.name());
		emit(r);
	}

	@Subscribe
	public void onConfigChanged(ConfigChanged e)
	{
		if (!"osrsrlrecorder".equals(e.getGroup()) || !"recording".equals(e.getKey()))
		{
			return;
		}
		if (config.recording() && client.getGameState() == GameState.LOGGED_IN)
		{
			openSession();
		}
		else if (!config.recording())
		{
			closeSession("paused");
		}
	}

	private synchronized void openSession()
	{
		if (writer != null || !config.recording())
		{
			return;
		}
		try
		{
			OUT_DIR.mkdirs();
			String stamp = new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date());
			File f = new File(OUT_DIR, "session-" + stamp + ".jsonl.gz");
			writer = new SessionWriter(gson, f);
			bystanderIds.clear();
			Map<String, Object> r = rec("session_start");
			r.put("schema", SCHEMA_VERSION);
			r.put("world", client.getWorld());
			r.put("canvas_w", client.getCanvasWidth());
			r.put("canvas_h", client.getCanvasHeight());
			r.put("stretched", client.isStretchedEnabled());
			r.put("mouse_hz", config.mouseHz());
			r.put("typed_text", config.recordTypedText());
			emit(r);
			lastLayout = "";
			layout();
			client.addChatMessage(ChatMessageType.GAMEMESSAGE, "", "OSRS-RL recorder: recording to " + f.getName(), null);
		}
		catch (IOException ex)
		{
			log.warn("could not open recording", ex);
			writer = null;
		}
	}

	private synchronized void closeSession(String reason)
	{
		SessionWriter w = writer;
		if (w == null)
		{
			return;
		}
		Map<String, Object> r = rec("session_end");
		r.put("reason", reason);
		w.write(r);
		writer = null;
		w.close();
		log.info("osrs-rl recording saved: {}", w.getFile());
	}

	// --- input -----------------------------------------------------------

	private void move(MouseEvent e, String type)
	{
		long now = System.currentTimeMillis();
		if (now - lastMoveMs < 1000L / Math.max(1, config.mouseHz()))
		{
			return;
		}
		lastMoveMs = now;
		Map<String, Object> r = rec(type);
		r.put("x", e.getX());
		r.put("y", e.getY());
		emit(r);
	}

	private void button(MouseEvent e, String type)
	{
		Map<String, Object> r = rec(type);
		r.put("x", e.getX());
		r.put("y", e.getY());
		r.put("button", e.getButton());
		emit(r);
	}

	private void key(KeyEvent e, String type)
	{
		Map<String, Object> r = rec(type);
		int code = e.getKeyCode();
		boolean printable = Character.isLetterOrDigit(e.getKeyChar())
			|| (code >= KeyEvent.VK_A && code <= KeyEvent.VK_Z)
			|| (code >= KeyEvent.VK_0 && code <= KeyEvent.VK_9)
			|| (code >= KeyEvent.VK_NUMPAD0 && code <= KeyEvent.VK_NUMPAD9);
		if (printable && !config.recordTypedText())
		{
			r.put("key", "char");
		}
		else
		{
			r.put("key", KeyEvent.getKeyText(code));
			r.put("code", code);
		}
		emit(r);
	}

	// --- game events -------------------------------------------------------

	@Subscribe
	public void onMenuOptionClicked(MenuOptionClicked e)
	{
		Map<String, Object> r = rec("menu_click");
		r.put("option", Text.removeTags(e.getMenuOption()));
		r.put("target", Text.removeTags(e.getMenuTarget()));
		r.put("action", e.getMenuAction() != null ? e.getMenuAction().name() : null);
		r.put("id", e.getId());
		r.put("p0", e.getParam0());
		r.put("p1", e.getParam1());
		r.put("item", e.getItemId());
		clickTarget(r, e);
		net.runelite.api.Point m = client.getMouseCanvasPosition();
		if (m != null)
		{
			r.put("x", m.getX());
			r.put("y", m.getY());
		}
		r.put("tick", client.getTickCount());
		emit(r);
	}

	/**
	 * Adds where the clicked thing is, in the same frame as the tick snapshots: {@code dx}/{@code dy}
	 * tile offsets from the player for tile targets (objects and ground items), and the NPC's
	 * {@code npc_index} (the same key as {@code npcs[].index}) and type id for NPC targets.
	 */
	private void clickTarget(Map<String, Object> r, MenuOptionClicked e)
	{
		if (e.getMenuAction() == null || client.getLocalPlayer() == null)
		{
			return;
		}
		String a = e.getMenuAction().name();
		WorldView wv = client.getTopLevelWorldView();
		WorldPoint me = client.getLocalPlayer().getWorldLocation();
		if (a.contains("NPC"))
		{
			NPC n = wv.npcs().byIndex(e.getId());
			r.put("npc_index", e.getId());
			if (n != null)
			{
				WorldPoint p = n.getWorldLocation();
				r.put("npc_id", n.getId());
				r.put("dx", p.getX() - me.getX());
				r.put("dy", p.getY() - me.getY());
			}
		}
		else if (a.contains("GAME_OBJECT") || a.contains("GROUND_ITEM") || a.equals("EXAMINE_OBJECT"))
		{
			WorldPoint p = WorldPoint.fromScene(wv, e.getParam0(), e.getParam1(), wv.getPlane());
			r.put("dx", p.getX() - me.getX());
			r.put("dy", p.getY() - me.getY());
		}
	}

	@Subscribe
	public void onStatChanged(StatChanged e)
	{
		Map<String, Object> r = rec("xp");
		r.put("skill", e.getSkill().getName());
		r.put("xp", e.getXp());
		r.put("level", e.getLevel());
		emit(r);
	}

	@Subscribe
	public void onChatMessage(ChatMessage e)
	{
		ChatMessageType t = e.getType();
		// game and engine messages only; other players' chat is never stored
		if (t != ChatMessageType.GAMEMESSAGE && t != ChatMessageType.SPAM && t != ChatMessageType.ENGINE)
		{
			return;
		}
		Map<String, Object> r = rec("game_message");
		r.put("msg", Text.removeTags(e.getMessage()));
		emit(r);
	}

	@Subscribe
	public void onFocusChanged(FocusChanged e)
	{
		Map<String, Object> r = rec("focus");
		r.put("focused", e.isFocused());
		emit(r);
	}

	@Subscribe
	public void onWidgetLoaded(WidgetLoaded e)
	{
		Map<String, Object> r = rec("widget_open");
		r.put("group", e.getGroupId());
		emit(r);
	}

	@Subscribe
	public void onWidgetClosed(WidgetClosed e)
	{
		Map<String, Object> r = rec("widget_close");
		r.put("group", e.getGroupId());
		emit(r);
	}

	@Subscribe
	public void onAnimationChanged(AnimationChanged e)
	{
		if (e.getActor() != client.getLocalPlayer())
		{
			return;
		}
		Map<String, Object> r = rec("anim");
		r.put("anim", e.getActor().getAnimation());
		emit(r);
	}

	@Subscribe
	public void onInteractingChanged(InteractingChanged e)
	{
		if (e.getSource() != client.getLocalPlayer())
		{
			return;
		}
		Map<String, Object> r = rec("interacting");
		r.put("target", describe(e.getTarget()));
		emit(r);
	}

	@Subscribe
	public void onGameTick(GameTick e)
	{
		if (writer == null)
		{
			return;
		}
		Player me = client.getLocalPlayer();
		if (me == null)
		{
			return;
		}
		WorldPoint wp = me.getWorldLocation();
		Map<String, Object> r = rec("tick");
		r.put("tick", client.getTickCount());
		r.put("x", wp.getX());
		r.put("y", wp.getY());
		r.put("plane", wp.getPlane());
		r.put("anim", me.getAnimation());
		r.put("interacting", describe(me.getInteracting()));
		r.put("hp", client.getBoostedSkillLevel(Skill.HITPOINTS));
		r.put("prayer", client.getBoostedSkillLevel(Skill.PRAYER));
		// schema 5: boosted/real combat stats, active prayers, special attack, and what the mouse hovers
		Map<String, Object> stats = new LinkedHashMap<>();
		for (Skill sk : COMBAT_STATS)
		{
			stats.put(sk.getName().toLowerCase(), new int[]{client.getBoostedSkillLevel(sk), client.getRealSkillLevel(sk)});
		}
		r.put("stats", stats);
		List<String> prayers = new ArrayList<>();
		for (Prayer pr : Prayer.values())
		{
			if (client.isPrayerActive(pr))
			{
				prayers.add(pr.name().toLowerCase());
			}
		}
		r.put("prayers", prayers);
		r.put("spec", client.getVarpValue(VarPlayer.SPECIAL_ATTACK_PERCENT) / 10);
		if (client.getVarpValue(VarPlayer.SPECIAL_ATTACK_ENABLED) == 1)
		{
			r.put("spec_on", true);
		}
		Object hover = hover();
		if (hover != null)
		{
			r.put("hover", hover);
		}
		r.put("run", client.getEnergy() / 100);
		int yaw = client.getCameraYaw();
		int pitch = client.getCameraPitch();
		r.put("yaw", yaw);
		r.put("pitch", pitch);
		if (lastYaw >= 0 && (Math.abs(yaw - lastYaw) > 8 || Math.abs(pitch - lastPitch) > 4))
		{
			r.put("camera_moved", true);
		}
		lastYaw = yaw;
		lastPitch = pitch;
		r.put("focused", client.getCanvas() != null && client.getCanvas().isFocusOwner());
		r.put("inv", items(client.getItemContainer(InventoryID.INV)));
		r.put("worn", items(client.getItemContainer(InventoryID.WORN)));
		r.put("bank_open", client.getItemContainer(InventoryID.BANK) != null);
		r.put("tut", client.getVarpValue(VARP_TUTORIAL));
		scene(r, wp);
		emit(r);
		dialogue(me);
		layout();
	}

	/**
	 * Schema 7: where things are on screen, so video crops (XP drops, hover text, inventory...)
	 * can be cut from the right place in any client layout. Written at session start and
	 * again whenever anything moves (checked every tick).
	 */
	private void layout()
	{
		if (writer == null)
		{
			return;
		}
		Map<String, Object> r = rec("ui_layout");
		r.put("canvas_w", client.getCanvasWidth());
		r.put("canvas_h", client.getCanvasHeight());
		r.put("resized", client.isResized());
		r.put("stretched", client.isStretchedEnabled());
		if (client.isStretchedEnabled())
		{
			Dimension d = client.getStretchedDimensions();
			if (d != null)
			{
				r.put("stretched_w", d.width);
				r.put("stretched_h", d.height);
			}
		}
		int vx = client.getViewportXOffset();
		int vy = client.getViewportYOffset();
		int vw = client.getViewportWidth();
		r.put("viewport", new int[]{vx, vy, vw, client.getViewportHeight()});
		r.put("hover_text", new int[]{vx, vy, vw, HOVER_TEXT_H});
		putBox(r, "xp_drops", XP_DROPS);
		putBox(r, "xp_counter", XP_COUNTER);
		putBox(r, "xp_orb", XP_ORB);
		putBox(r, "minimap", MINIMAP);
		putBox(r, "inventory", INVENTORY);
		putBox(r, "chatbox", CHATBOX);
		Map<String, Object> key = new LinkedHashMap<>(r);
		key.remove("t");
		String s = gson.toJson(key);
		if (!s.equals(lastLayout))
		{
			lastLayout = s;
			emit(r);
		}
	}

	private void putBox(Map<String, Object> r, String name, int[] componentIds)
	{
		for (int id : componentIds)
		{
			Widget w = client.getWidget(id);
			if (w != null && !w.isHidden())
			{
				int[] b = box(w.getBounds());
				if (b != null)
				{
					r.put(name, b);
					return;
				}
			}
		}
	}

	/** [x, y, w, h] on the canvas, or null for an empty rectangle. */
	static int[] box(Rectangle b)
	{
		if (b == null || b.width <= 0 || b.height <= 0)
		{
			return null;
		}
		return new int[]{b.x, b.y, b.width, b.height};
	}

	// --- dialogue ------------------------------------------------------------

	/**
	 * Emits a "dialogue" record whenever the chatbox dialogue or the Tutorial Island
	 * instruction text changes. Reading time before "continue" is the main new
	 * humanness signal on Tutorial Island, so the text length is what matters; the
	 * text itself is game content, with the player's own name replaced by "<name>".
	 */
	private void dialogue(Player me)
	{
		String kind = null;
		String speaker = null;
		String text = null;
		List<String> options = null;
		if (visible(GROUP_NPC_CHAT, 6))
		{
			kind = "npc";
			speaker = widgetText(GROUP_NPC_CHAT, 4);
			text = widgetText(GROUP_NPC_CHAT, 6);
		}
		else if (visible(GROUP_PLAYER_CHAT, 6))
		{
			kind = "player";
			text = widgetText(GROUP_PLAYER_CHAT, 6);
		}
		else if (visible(GROUP_OPTIONS, 1))
		{
			kind = "options";
			options = new ArrayList<>();
			Widget w = client.getWidget(GROUP_OPTIONS, 1);
			Widget[] kids = w.getDynamicChildren();
			if (kids != null)
			{
				for (Widget k : kids)
				{
					String t = clean(k.getText());
					if (!t.isEmpty())
					{
						options.add(t);
					}
				}
			}
			// the first line is the "Select an option" title
			text = options.isEmpty() ? "" : options.remove(0);
		}
		else if (visible(GROUP_SPRITE, 2))
		{
			kind = "sprite";
			text = widgetText(GROUP_SPRITE, 2);
		}
		else if (visible(GROUP_MESBOX, 1))
		{
			kind = "message";
			text = widgetText(GROUP_MESBOX, 1);
		}
		else if (groupText(GROUP_TUTORIAL_TEXT) != null)
		{
			kind = "tutorial";
			text = groupText(GROUP_TUTORIAL_TEXT);
		}

		String name = me.getName();
		if (text != null && name != null && !name.isEmpty())
		{
			text = text.replace(name, "<name>");
		}
		String key = kind == null ? "" : kind + "|" + text + "|" + options;
		if (key.equals(lastDialogue))
		{
			return;
		}
		lastDialogue = key;
		Map<String, Object> r = rec("dialogue");
		r.put("kind", kind); // null: the dialogue closed
		if (kind != null)
		{
			r.put("speaker", speaker);
			r.put("text", text);
			r.put("chars", text == null ? 0 : text.length());
			r.put("words", text == null || text.isEmpty() ? 0 : text.trim().split("\\s+").length);
			if (options != null)
			{
				r.put("options", options);
			}
		}
		emit(r);
	}

	private boolean visible(int group, int child)
	{
		Widget w = client.getWidget(group, child);
		return w != null && !w.isHidden();
	}

	private String widgetText(int group, int child)
	{
		Widget w = client.getWidget(group, child);
		return w == null ? "" : clean(w.getText());
	}

	/**
	 * All visible text in an interface group, joined, or null if there is none. Walks every
	 * component's static, dynamic and nested children, because the Tutorial Island
	 * instruction text (group 263) is not a direct static child.
	 */
	private String groupText(int group)
	{
		List<String> parts = new ArrayList<>();
		Set<Widget> seen = Collections.newSetFromMap(new IdentityHashMap<>());
		for (int child = 0; child < 64; child++)
		{
			collectText(client.getWidget(group, child), parts, seen, 0);
		}
		return parts.isEmpty() ? null : String.join(" ", parts);
	}

	private void collectText(Widget w, List<String> parts, Set<Widget> seen, int depth)
	{
		if (w == null || depth > 6 || !seen.add(w) || w.isHidden())
		{
			return;
		}
		String t = clean(w.getText());
		if (!t.isEmpty() && !parts.contains(t))
		{
			parts.add(t);
		}
		for (Widget[] kids : new Widget[][]{w.getStaticChildren(), w.getDynamicChildren(), w.getNestedChildren()})
		{
			if (kids != null)
			{
				for (Widget k : kids)
				{
					collectText(k, parts, seen, depth + 1);
				}
			}
		}
	}

	private static String clean(String s)
	{
		return s == null ? "" : Text.removeTags(s.replace("<br>", " ")).trim();
	}

	private void scene(Map<String, Object> r, WorldPoint me)
	{
		int radius = config.radius();
		WorldView wv = client.getTopLevelWorldView();
		List<Object> npcs = new ArrayList<>();
		for (NPC n : wv.npcs())
		{
			WorldPoint p = n.getWorldLocation();
			if (p.getPlane() != me.getPlane() || p.distanceTo2D(me) > radius)
			{
				continue;
			}
			Map<String, Object> m = new LinkedHashMap<>();
			m.put("index", n.getIndex());
			m.put("id", n.getId());
			m.put("name", n.getName());
			m.put("dx", p.getX() - me.getX());
			m.put("dy", p.getY() - me.getY());
			m.put("anim", n.getAnimation());
			m.put("hp_ratio", n.getHealthRatio());
			m.put("hp_scale", n.getHealthScale());
			m.put("targets_me", n.getInteracting() == client.getLocalPlayer());
			npcs.add(m);
		}
		r.put("npcs", npcs);

		List<Object> players = new ArrayList<>();
		for (Player p : wv.players())
		{
			if (p == client.getLocalPlayer())
			{
				continue;
			}
			WorldPoint q = p.getWorldLocation();
			if (q.getPlane() != me.getPlane() || q.distanceTo2D(me) > radius)
			{
				continue;
			}
			// no names: other players are stored only as anonymous positions
			Map<String, Object> m = new LinkedHashMap<>();
			m.put("pid", bystanderId(p));
			m.put("dx", q.getX() - me.getX());
			m.put("dy", q.getY() - me.getY());
			m.put("anim", p.getAnimation());
			m.put("combat", p.getCombatLevel());
			// schema 5: what everyone can see on them
			PlayerComposition comp = p.getPlayerComposition();
			if (comp != null)
			{
				int[] ids = comp.getEquipmentIds();
				int[] equip = new int[EQUIP_SLOTS.length];
				for (int i = 0; i < EQUIP_SLOTS.length; i++)
				{
					int id = EQUIP_SLOTS[i] < ids.length ? ids[EQUIP_SLOTS[i]] : 0;
					equip[i] = id >= PlayerComposition.ITEM_OFFSET ? id - PlayerComposition.ITEM_OFFSET : 0;
				}
				m.put("equip", equip);
			}
			HeadIcon icon = p.getOverheadIcon();
			if (icon != null)
			{
				m.put("overhead", icon.name().toLowerCase());
			}
			if (p.getSkullIcon() != -1)
			{
				m.put("skull", p.getSkullIcon());
			}
			if (p.getHealthRatio() >= 0)
			{
				m.put("hp_ratio", p.getHealthRatio());
				m.put("hp_scale", p.getHealthScale());
			}
			List<Integer> spots = spotAnims(p);
			if (!spots.isEmpty())
			{
				m.put("spot", spots);
			}
			Actor t = p.getInteracting();
			if (t instanceof NPC)
			{
				m.put("target_npc", ((NPC) t).getIndex());
			}
			else if (t != null)
			{
				m.put("target_player", t == client.getLocalPlayer() ? -1 : bystanderId((Player) t));
			}
			players.add(m);
		}
		r.put("players", players);

		List<Object> objects = new ArrayList<>();
		Scene scene = wv.getScene();
		LocalPoint lp = client.getLocalPlayer().getLocalLocation();
		Tile[][][] tiles = scene.getTiles();
		int plane = wv.getPlane();
		if (lp != null && tiles != null && plane >= 0 && plane < tiles.length)
		{
			int sx = lp.getSceneX();
			int sy = lp.getSceneY();
			Set<TileObject> seen = Collections.newSetFromMap(new IdentityHashMap<>());
			for (int x = Math.max(0, sx - radius); x <= Math.min(tiles[plane].length - 1, sx + radius); x++)
			{
				for (int y = Math.max(0, sy - radius); y <= Math.min(tiles[plane][x].length - 1, sy + radius); y++)
				{
					Tile t = tiles[plane][x][y];
					if (t == null)
					{
						continue;
					}
					GameObject[] gos = t.getGameObjects();
					if (gos != null)
					{
						for (GameObject go : gos)
						{
							addObject(objects, seen, go, me, "game", wv);
						}
					}
					addObject(objects, seen, t.getWallObject(), me, "wall", wv);
					addObject(objects, seen, t.getGroundObject(), me, "ground", wv);
				}
			}
		}
		r.put("objects", objects);
	}

	private static void addObject(List<Object> out, Set<TileObject> seen, TileObject o, WorldPoint me, String kind, WorldView wv)
	{
		if (o == null || !seen.add(o))
		{
			return;
		}
		WorldPoint p = o.getWorldLocation();
		int w = 1;
		int h = 1;
		if (o instanceof GameObject)
		{
			// Large objects: use the south-west corner, the tile a click on the object reports.
			GameObject go = (GameObject) o;
			p = WorldPoint.fromScene(wv, go.getSceneMinLocation().getX(), go.getSceneMinLocation().getY(), wv.getPlane());
			w = go.sizeX();
			h = go.sizeY();
		}
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("id", o.getId());
		m.put("kind", kind);
		m.put("dx", p.getX() - me.getX());
		m.put("dy", p.getY() - me.getY());
		if (w != 1 || h != 1)
		{
			m.put("w", w);
			m.put("h", h);
		}
		out.add(m);
	}

	private static List<int[]> items(ItemContainer c)
	{
		List<int[]> out = new ArrayList<>();
		if (c == null)
		{
			return out;
		}
		for (Item it : c.getItems())
		{
			out.add(new int[]{it.getId(), it.getQuantity()});
		}
		return out;
	}

	/** Damage shown on anyone in the scene: ties hits to the player or NPC that took them. */
	@Subscribe
	public void onHitsplatApplied(HitsplatApplied e)
	{
		if (writer == null)
		{
			return;
		}
		Hitsplat h = e.getHitsplat();
		Map<String, Object> r = rec("hitsplat");
		r.put("on", describe(e.getActor()));
		r.put("hit_type", h.getHitsplatType());
		r.put("amount", h.getAmount());
		r.put("mine", h.isMine());
		emit(r);
	}

	/** The left-click action the mouse is over right now (what a click would do), name-free. */
	private Object hover()
	{
		MenuEntry[] entries = client.getMenuEntries();
		if (entries == null || entries.length == 0)
		{
			return null;
		}
		MenuEntry top = entries[entries.length - 1];
		String option = top.getOption();
		if (option == null || option.isEmpty() || "Cancel".equals(option))
		{
			return null;
		}
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("option", Text.removeTags(option));
		if (top.getPlayer() != null)
		{
			m.put("player", top.getPlayer() == client.getLocalPlayer() ? -1 : bystanderId(top.getPlayer()));
		}
		else if (top.getNpc() != null)
		{
			m.put("npc_index", top.getNpc().getIndex());
			m.put("target", Text.removeTags(top.getTarget()));
		}
		else
		{
			m.put("target", Text.removeTags(top.getTarget()));
			m.put("id", top.getIdentifier());
		}
		if (top.getItemId() > 0)
		{
			m.put("item", top.getItemId());
		}
		m.put("kind", top.getType().name().toLowerCase());
		return m;
	}

	private static List<Integer> spotAnims(Actor a)
	{
		List<Integer> out = new ArrayList<>();
		for (ActorSpotAnim sa : a.getSpotAnims())
		{
			out.add(sa.getId());
		}
		return out;
	}

	/**
	 * A small integer that stays the same for one other player for the rest of the session.
	 * The client reuses a player slot when someone logs out and someone else logs in, so the
	 * key also carries the combat level; a different person in the same slot almost always gets
	 * a new id. Nothing identifying is stored or derivable from it.
	 */
	private int bystanderId(Player p)
	{
		long key = ((long) p.getId() << 16) | (p.getCombatLevel() & 0xffff);
		Integer id = bystanderIds.get(key);
		if (id == null)
		{
			id = bystanderIds.size();
			bystanderIds.put(key, id);
		}
		return id;
	}

	private Object describe(Actor a)
	{
		if (a == null)
		{
			return null;
		}
		Map<String, Object> m = new LinkedHashMap<>();
		if (a instanceof NPC)
		{
			m.put("npc", ((NPC) a).getId());
			m.put("name", a.getName());
		}
		else if (a == client.getLocalPlayer())
		{
			m.put("player", true);
			m.put("me", true);
		}
		else
		{
			m.put("player", true);
			m.put("pid", bystanderId((Player) a));
		}
		return m;
	}

	// --- plumbing --------------------------------------------------------

	private static Map<String, Object> rec(String type)
	{
		Map<String, Object> r = new LinkedHashMap<>();
		r.put("t", System.currentTimeMillis());
		r.put("type", type);
		return r;
	}

	private void emit(Map<String, Object> r)
	{
		SessionWriter w = writer;
		if (w != null)
		{
			w.write(r);
		}
	}
}
