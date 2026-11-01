package com.osrsrl.recorder;

import com.google.gson.Gson;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPOutputStream;
import lombok.extern.slf4j.Slf4j;

/**
 * Writes one gzipped JSON-lines file per session on a background thread, so the
 * client and AWT threads only ever enqueue.
 */
@Slf4j
class SessionWriter
{
	private static final Map<String, Object> POISON = Map.of();

	private final LinkedBlockingQueue<Map<String, Object>> queue = new LinkedBlockingQueue<>();
	private final Gson gson;
	private final File file;
	private final Thread thread;
	private volatile boolean closed;

	SessionWriter(Gson gson, File file) throws IOException
	{
		this.gson = gson;
		this.file = file;
		final BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
			new GZIPOutputStream(new FileOutputStream(file), 1 << 16, true), StandardCharsets.UTF_8));
		thread = new Thread(() -> run(out), "osrs-rl-recorder");
		thread.setDaemon(true);
		thread.start();
	}

	File getFile()
	{
		return file;
	}

	void write(Map<String, Object> record)
	{
		if (!closed)
		{
			queue.offer(record);
		}
	}

	void close()
	{
		if (closed)
		{
			return;
		}
		closed = true;
		queue.offer(POISON);
		try
		{
			thread.join(5000);
		}
		catch (InterruptedException e)
		{
			Thread.currentThread().interrupt();
		}
	}

	private void run(BufferedWriter out)
	{
		long lastFlush = System.currentTimeMillis();
		try (out)
		{
			while (true)
			{
				Map<String, Object> rec = queue.poll(1, TimeUnit.SECONDS);
				if (rec == POISON)
				{
					break;
				}
				if (rec != null)
				{
					out.write(gson.toJson(rec));
					out.write('\n');
				}
				long now = System.currentTimeMillis();
				if (now - lastFlush > 2000)
				{
					out.flush();
					lastFlush = now;
				}
			}
		}
		catch (IOException | InterruptedException e)
		{
			log.warn("osrs-rl recorder stopped writing {}", file, e);
		}
	}
}
