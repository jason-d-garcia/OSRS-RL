package org.rsmod.content.other.agentbridge

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.node.ObjectNode
import com.github.michaelbull.logging.InlineLogger
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketException
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.rsmod.game.entity.Player

/**
 * The bridge's socket side: one TCP connection per agent, newline-delimited JSON both ways.
 *
 * Sockets are read on their own threads and every command is queued for the game thread; nothing
 * here touches game state. Writes come from the game thread.
 */
class BridgeServer(private val port: Int, private val bindAddress: String) {
    private val logger = InlineLogger()
    private val ids = AtomicInteger()
    private var serverSocket: ServerSocket? = null

    /** New connections waiting for the game thread to pick them up. */
    val accepted = ConcurrentLinkedQueue<AgentConnection>()

    fun start() {
        val socket = ServerSocket(port, 16, InetAddress.getByName(bindAddress))
        serverSocket = socket
        val thread = Thread({ acceptLoop(socket) }, "agent-bridge-accept").apply { isDaemon = true }
        thread.start()
        logger.info { "Agent bridge listening on $bindAddress:$port" }
    }

    fun stop() {
        serverSocket?.close()
    }

    private fun acceptLoop(server: ServerSocket) {
        while (!server.isClosed) {
            val socket =
                try {
                    server.accept()
                } catch (e: SocketException) {
                    return
                }
            socket.tcpNoDelay = true
            val conn = AgentConnection(ids.incrementAndGet(), socket)
            accepted += conn
            Thread({ conn.readLoop() }, "agent-bridge-${conn.id}").apply { isDaemon = true }.start()
            logger.info { "Agent ${conn.id} connected from ${socket.remoteSocketAddress}" }
        }
    }
}

class AgentConnection(val id: Int, private val socket: Socket) {
    private val logger = InlineLogger()
    private val reader = BufferedReader(InputStreamReader(socket.getInputStream(), Charsets.UTF_8))
    private val writer =
        BufferedWriter(OutputStreamWriter(socket.getOutputStream(), Charsets.UTF_8))

    /** Commands from the agent, in order. A [CLOSED] node marks the end of the stream. */
    val inbox = LinkedBlockingQueue<JsonNode>()

    @Volatile
    var open: Boolean = true
        private set

    // ---- game-thread state ----

    /**
     * The player this agent drives: a headless bot it spawned, or a logged-in player it attached.
     */
    var player: Player? = null
    var headless: Boolean = false
    var botClient: BotClient? = null
    var botCycle: BotCycle? = null
    var loginPending: Boolean = false

    /** In lockstep the game waits at the end of every cycle until this agent sends `step`. */
    var lockstep: Boolean = false

    /** Actions to apply at the start of the next cycle. */
    val pending = ArrayList<JsonNode>()

    /** Outcomes of the actions applied this cycle, sent with the next tick. */
    val results = ArrayList<ObjectNode>()

    /** The last action of each kind the agent sent, used to tell attacking from mining. */
    var lastActionType: String = ""
    var lastActionCycle: Int = -1
    var lastCoords: org.rsmod.map.CoordGrid? = null
    var lastTunnelCycle: Int = Int.MIN_VALUE

    fun readLoop() {
        try {
            while (true) {
                val line = reader.readLine() ?: break
                if (line.isBlank()) {
                    continue
                }
                val node =
                    try {
                        MAPPER.readTree(line)
                    } catch (e: Exception) {
                        send(
                            MAPPER.createObjectNode().put("type", "error").put("error", "bad json")
                        )
                        continue
                    }
                inbox.put(node)
            }
        } catch (e: Exception) {
            logger.debug { "Agent $id read ended: ${e.message}" }
        } finally {
            open = false
            inbox.put(CLOSED)
        }
    }

    fun send(node: JsonNode) {
        if (!open) {
            return
        }
        try {
            synchronized(writer) {
                writer.write(MAPPER.writeValueAsString(node))
                writer.write("\n")
                writer.flush()
            }
        } catch (e: Exception) {
            open = false
            inbox.put(CLOSED)
        }
    }

    fun take(timeoutMs: Long): JsonNode? = inbox.poll(timeoutMs, TimeUnit.MILLISECONDS)

    fun close() {
        open = false
        try {
            socket.close()
        } catch (_: Exception) {}
    }

    companion object {
        val MAPPER = ObjectMapper()
        val CLOSED: JsonNode = MAPPER.createObjectNode().put("op", "__closed__")
    }
}
