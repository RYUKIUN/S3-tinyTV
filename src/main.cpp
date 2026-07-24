/*
 * ESP32-S3  |  ILI9341  |  8-bit Parallel i80  |  320x240
 *
 * PIPELINE ARCHITECTURE
 * ─────────────────────
 *  Four shared slots replace the old 2-slot ping-pong scheme.
 *  Depth-4 decodeQueue lets networkTask post all 4 tiles without blocking,
 *  eliminating the UDP dead-zone that drove ABRT rates at high JPEG quality.
 *
 *  Memory layout:
 *    slot[0..3].assembly  SRAM  33 KB each (4 × 33.6 KB = 134.4 KB total)
 *                                ─ decoder reads every byte → must be fast
 *    frameFb[0]        PSRAM 150 KB  ─┐ full 320×240 frame; DMA source ONLY
 *    frameFb[1]        PSRAM 150 KB  ─┘ double-buffered; display pushes one atomic frame
 *    chunkStorage[4]   PSRAM 134 KB    chunk staging; network writes, not decode-critical
 *
 *  JPEGDEC's MCU callback now writes directly into frameFb[writeSet] at the
 *  tile's offset (BIG_ENDIAN pixel type matches the ILI9341's native byte
 *  order), eliminating the old SRAM decode-scratch buffer + byte-swap pass
 *  entirely. One PSRAM write per tile instead of SRAM write + SRAM read +
 *  PSRAM write.
 *
 *  Total SRAM for buffers: ~66 KB (was ~104 KB before removing decodeTemp)
 *
 * OFFLINE MODE
 * ────────────
 *  Trigger: WiFi connected but no stream packet received within OFFLINE_TRIGGER_MS (5 s).
 *           OR: WiFi fails to associate within WIFI_CONNECT_TIMEOUT_MS (15 s).
 *
 *  Behaviour:
 *   1. enterOfflineMode() — sets g_offlineMode, tears down WiFi, drops CPU to 80 MHz.
 *   2. All streaming tasks (networkTask, wifiWatchdogTask, displayTask, otaTask) see
 *      g_offlineMode on their next iteration and call vTaskDelete(NULL).
 *   3. 500 ms grace period for tasks to exit before LCD is claimed by the player.
 *   4. runOfflinePlayer() — called from Arduino loop() (Core 1); loops forever.
 *      Scans SPIFFS for .mjpeg files, plays them in sequence, cycles on end.
 *
 *  Power savings in offline mode:
 *   • WiFi radio completely off  (WiFi.mode(WIFI_OFF) + esp_wifi_stop())
 *   • CPU frequency 240 → 80 MHz
 *   • No FreeRTOS tasks except the idle task and loop() itself
 *   • vTaskDelay() used for frame pacing — Core idle between frames
 */

#define LGFX_USE_V1
#include <LovyanGFX.hpp>
#include <JPEGDEC.h>
#include <Arduino.h>
#include <WiFi.h>
#include <ArduinoOTA.h>
#include <esp_wifi.h>
#include <esp_attr.h>
#include <esp_heap_caps.h>
#include <SPIFFS.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/queue.h"
#include <lwip/sockets.h>
#include <lwip/netdb.h>
#include <fcntl.h>
#include <math.h>
#include "shared.h"
#include "display.h"
#include "network.h"
#include "jpeg_decode.h"
#include "offline_player.h"
#include "esp_freertos_hooks.h"


// ─────────────────────────────────────────────
//  CONFIG DEFINITIONS
// ─────────────────────────────────────────────
const char* WIFI_SSID  = "Endmin";
const char* WIFI_PASS  = "987654321";

// ─────────────────────────────────────────────
//  PIPELINE SLOTS DEFINITIONS
// ─────────────────────────────────────────────
PipeSlot slot[MM_MAX_JPEG_SLOTS];   // only [0..g_numJpegSlots-1] used at runtime

// Framebuffers: sized to maximum; only [0..g_numDisplayBufs-1] allocated at runtime.
uint16_t* frameFb[MM_MAX_DISPLAY_BUFS] = {};
uint8_t   writeSet = 0;  // Core 1 exclusive — no sync needed

// Runtime counts (set in setup() after fallback allocation)
uint8_t g_numJpegSlots   = CFG_NUM_JPEG_SLOTS;
uint8_t g_numDisplayBufs = CFG_NUM_DISPLAY_BUFS;

// Pipeline synchronisation
QueueHandle_t     decodeQueue;                       // depth = g_numJpegSlots
SemaphoreHandle_t slotFree[MM_MAX_JPEG_SLOTS];       // only [0..g_numJpegSlots-1] used

// Display pipeline: Core 1 posts here when all 4 tiles are ready.
QueueHandle_t displayQueue;      // depth = g_numDisplayBufs

// ─────────────────────────────────────────────
//  PER-CORE CPU UTILISATION (TASK 1)
// ─────────────────────────────────────────────
// The idle hook fires in a tight loop inside the FreeRTOS idle task — it can
// be called tens of thousands of times per second, NOT once per tick.  So the
// old "count iterations vs tick count" formula always evaluated idle >> ticks
// and pinned CPU% at 0.
//
// Fix: accumulate ACTUAL MICROSECONDS of idle time using esp_timer_get_time().
// s_idleLastUs[] tracks the timestamp of the previous hook call for each core.
// If consecutive calls are < IDLE_GAP_THRESH_US apart, the idle task ran
// uninterrupted — add the delta as genuine idle time.
// If the gap is ≥ thresh, a real task preempted the idle task in between —
// don't count that gap, only reset the baseline.
//
// g_cpuIdleUs[] is read by networkTask (Core 0) and written by idle hooks.
// Core 0's hook writes index [0] from the same core as networkTask — no
// simultaneous write+read since idle priority < networkTask priority.
// Core 1's hook writes index [1] read cross-core; uint32_t reads are atomic
// on Xtensa, and volatile ensures no stale cache.

#define IDLE_GAP_THRESH_US  1000u   // 1 ms — one FreeRTOS tick

volatile uint32_t g_cpuIdleUs[2] = { 0, 0 };

// Per-core last-call timestamps: written and read only from their own core.
static uint32_t IRAM_ATTR s_idleLastUs[2] = { 0, 0 };

static bool IRAM_ATTR idleHookCore0() {
    uint32_t now  = (uint32_t)esp_timer_get_time();
    uint32_t last = s_idleLastUs[0];
    s_idleLastUs[0] = now;
    if (last && (now - last) < IDLE_GAP_THRESH_US)
        g_cpuIdleUs[0] += (now - last);
    return false;
}

static bool IRAM_ATTR idleHookCore1() {
    uint32_t now  = (uint32_t)esp_timer_get_time();
    uint32_t last = s_idleLastUs[1];
    s_idleLastUs[1] = now;
    if (last && (now - last) < IDLE_GAP_THRESH_US)
        g_cpuIdleUs[1] += (now - last);
    return false;
}

// ─────────────────────────────────────────────
//  CHUNK REASSEMBLY STATE DEFINITIONS
// ─────────────────────────────────────────────
TileState tiles[NUM_TILES];
uint8_t*  tileChunkStorage[NUM_TILES] = {};

// ─────────────────────────────────────────────
//  CROSS-CORE STATS DEFINITIONS
// ─────────────────────────────────────────────
volatile uint32_t g_avgDecodeUs     = 0;  // avg tile decode us (excl. pushImage)
volatile uint32_t g_presentedFrames = 0;  // frames fully pushed to LCD
volatile uint32_t g_abortedFrames   = 0;  // partial frames dropped (UDP reorder)

// ─────────────────────────────────────────────
//  GLOBAL STATE DEFINITIONS
// ─────────────────────────────────────────────
bool     debugEnabled      = false;
char     debugBuf[320];
int      g_sock            = -1;
struct   sockaddr_in g_remoteAddr;
bool     g_remoteAddrValid = false;
float    stat_jitter       = 0.0f;
uint32_t stat_prevMs       = 0;

// ── WiFi watchdog & RSSI display ─────────────────────────────────────────────
volatile bool     g_streaming        = false;
volatile bool     g_wifiOk           = false;
volatile uint32_t g_lastPktMs        = 0;
volatile uint32_t g_wifiConnectedMs  = 0;   // millis() when WiFi first associated
volatile uint32_t g_wifiDisconnectedMs = 0; // millis() when WiFi last disconnected

// ── Offline mode ──────────────────────────────────────────────────────────────
volatile bool g_offlineMode = false;   // set by enterOfflineMode(), never cleared


// ─────────────────────────────────────────────
//  OTA TASK  (Core 0, priority 1)
// ─────────────────────────────────────────────
static void otaTask(void* /*pv*/) {
    ArduinoOTA.setHostname("esp32s3-display");

    ArduinoOTA.onStart([]() {
        String type = (ArduinoOTA.getCommand() == U_FLASH) ? "firmware" : "filesystem";
        Serial.printf("[OTA] Start: %s\n", type.c_str());
    });
    ArduinoOTA.onEnd([]() {
        Serial.println("[OTA] Done — rebooting");
    });
    ArduinoOTA.onProgress([](unsigned int prog, unsigned int total) {
        Serial.printf("[OTA] %u%%\r", (prog * 100) / total);
    });
    ArduinoOTA.onError([](ota_error_t err) {
        const char* reason = "unknown";
        if      (err == OTA_AUTH_ERROR)    reason = "auth failed";
        else if (err == OTA_BEGIN_ERROR)   reason = "begin failed";
        else if (err == OTA_CONNECT_ERROR) reason = "connect failed";
        else if (err == OTA_RECEIVE_ERROR) reason = "receive failed";
        else if (err == OTA_END_ERROR)     reason = "end failed";
        Serial.printf("[OTA] Error[%u]: %s\n", err, reason);
    });

    ArduinoOTA.begin();
    Serial.printf("[OTA] Ready — hostname: esp32s3-display  IP: %s\n",
                  WiFi.localIP().toString().c_str());

    for (;;) {
        // Self-terminate when offline mode is active (WiFi is off, OTA is useless)
        if (g_offlineMode) {
            Serial.println("[OTA] Offline mode — otaTask exiting");
            vTaskDelete(NULL);
            return;
        }
        ArduinoOTA.handle();
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ─────────────────────────────────────────────
//  ENTER OFFLINE MODE
//  Can be called from setup() (no tasks running) or loop() (tasks running).
//  Sets g_offlineMode so all tasks self-terminate, then kills WiFi and
//  reduces the CPU frequency for power saving.
//  After this returns, call runOfflinePlayer() (which never returns).
// ─────────────────────────────────────────────
static void enterOfflineMode() {
    Serial.println("[OFFLINE] Entering offline mode");

    // 1. Signal all streaming tasks to stop.
    //    They poll g_offlineMode on each iteration and call vTaskDelete(NULL).
    g_offlineMode = true;

    // 2. Close the UDP socket so networkTask's recvfrom() / select() unblocks
    //    immediately rather than waiting on its 1 ms timeout.
    if (g_sock >= 0) {
        close(g_sock);
        g_sock = -1;
    }

    // 3. Give tasks up to 500 ms to see the flag and exit.
    //    displayTask has a 100 ms xQueueReceive timeout, so it will exit
    //    within that window.  networkTask unblocks immediately from the closed
    //    socket.  This prevents concurrent LCD access between displayTask and
    //    runOfflinePlayer().
    vTaskDelay(pdMS_TO_TICKS(500));

    // 4. Shut WiFi hardware down completely.
    WiFi.disconnect(true);
    vTaskDelay(pdMS_TO_TICKS(50));
    esp_wifi_stop();
    WiFi.mode(WIFI_OFF);
    g_wifiOk = false;
    Serial.println("[OFFLINE] WiFi radio OFF");

    // 5. Reduce CPU frequency: 240 → 80 MHz.
    //    Halves dynamic power consumption; 80 MHz is plenty for SPIFFS reads
    //    and JPEGDEC at 15 fps.
    setCpuFrequencyMhz(160);
    Serial.printf("[OFFLINE] CPU frequency: %d MHz\n", getCpuFrequencyMhz());

    // 6. Update status display.
    statusLine(3, "WiFi:", "OFF (power save)", TFT_YELLOW);
    statusLine(4, "OTA:", "Disabled",          TFT_DARKGREY);
    statusLine(5, "Mode:", "OFFLINE PLAYER",   TFT_CYAN);
    statusLine(6, "Status:", "Loading...",     TFT_YELLOW);
}


// ─────────────────────────────────────────────
//  SETUP
// ─────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    uint32_t t0 = millis();
    while (!Serial && (millis() - t0) < 2000) delay(10);
    Serial.println("\n[BOOT] ping-pong pipeline (direct-to-PSRAM decode, BE pixels)");

    lcd.init(); lcd.setRotation(3); lcd.setColorDepth(16);
    lcd.setTextFont(2); lcd.setTextSize(1);
    drawBootHeader();
    statusLine(0, "Display:", "OK", TFT_GREEN);

    bool psramOk = psramFound();
    statusLine(1, "PSRAM:", psramOk ? "Found" : "MISSING!", psramOk ? TFT_GREEN : TFT_RED);
    if (!psramOk) { while (1) delay(1000); }

    // ── JPEG assembly slots (SRAM) — fallback from CFG down to 1 ─────────────
    // Each slot holds one reassembled tile JPEG (MAX_TILE_JPEG = 33.6 KB SRAM).
    g_numJpegSlots = CFG_NUM_JPEG_SLOTS;
    while (g_numJpegSlots >= 1) {
        // Release any partially-attempted allocations from a previous pass
        for (int s = 0; s < MM_MAX_JPEG_SLOTS; s++) {
            if (slot[s].assembly) { heap_caps_free(slot[s].assembly); slot[s].assembly = nullptr; }
        }
        bool ok = true;
        for (int s = 0; s < g_numJpegSlots; s++) {
            slot[s].assembly = (uint8_t*)heap_caps_aligned_alloc(
                16, MAX_TILE_JPEG, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
            if (!slot[s].assembly) { ok = false; break; }
        }
        if (ok) break;
        Serial.printf("[WARN] JPEG slots %d failed — retrying with %d\n",
                      g_numJpegSlots, g_numJpegSlots - 1);
        g_numJpegSlots--;
    }
    if (g_numJpegSlots == 0) {
        Serial.println("[ERROR] Cannot allocate even 1 JPEG slot — halting");
        statusLine(2, "Buffers:", "SRAM ALLOC FAILED!", TFT_RED);
        while (1) delay(1000);
    }
    Serial.printf("[MEM] JPEG slots: %d / %d (CFG=%d)\n",
                  g_numJpegSlots, MM_MAX_JPEG_SLOTS, CFG_NUM_JPEG_SLOTS);

    // ── Display framebuffers (PSRAM) — fallback from CFG down to 2 ───────────
    // Each buffer is a full 320×240 RGB565 frame (150 KB PSRAM).
    // Minimum 2 required for tear-free DMA ping-pong.
    g_numDisplayBufs = CFG_NUM_DISPLAY_BUFS;
    while (g_numDisplayBufs >= 2) {
        for (int s = 0; s < MM_MAX_DISPLAY_BUFS; s++) {
            if (frameFb[s]) { heap_caps_free(frameFb[s]); frameFb[s] = nullptr; }
        }
        bool ok = true;
        for (int s = 0; s < g_numDisplayBufs; s++) {
            frameFb[s] = (uint16_t*)heap_caps_aligned_alloc(
                16, SCREEN_W * SCREEN_H * 2, MALLOC_CAP_SPIRAM);
            if (!frameFb[s]) { ok = false; break; }
        }
        if (ok) break;
        Serial.printf("[WARN] Display bufs %d failed — retrying with %d\n",
                      g_numDisplayBufs, g_numDisplayBufs - 1);
        g_numDisplayBufs--;
    }
    if (g_numDisplayBufs < 2) {
        Serial.println("[ERROR] Cannot allocate 2 display framebuffers — halting");
        statusLine(2, "Buffers:", "PSRAM ALLOC FAILED!", TFT_RED);
        while (1) delay(1000);
    }
    Serial.printf("[MEM] Display bufs: %d / %d (CFG=%d)\n",
                  g_numDisplayBufs, MM_MAX_DISPLAY_BUFS, CFG_NUM_DISPLAY_BUFS);

    // ── Tile chunk storage (PSRAM) ────────────────────────────────────────────
    bool allocOk = true;
    for (int t = 0; t < NUM_TILES && allocOk; t++) {
        tileChunkStorage[t] = (uint8_t*)heap_caps_malloc(
            (size_t)MAX_TILE_CHUNKS * CHUNK_DATA_SIZE, MALLOC_CAP_SPIRAM);
        if (!tileChunkStorage[t]) {
            Serial.printf("[ERROR] tile[%d] chunkStorage PSRAM alloc failed\n", t);
            allocOk = false; break;
        }
        for (int c = 0; c < MAX_TILE_CHUNKS; c++)
            tiles[t].chunkBuf[c] = tileChunkStorage[t] + (size_t)c * CHUNK_DATA_SIZE;
    }
    if (!allocOk) {
        statusLine(2, "Buffers:", "CHUNK ALLOC FAILED!", TFT_RED);
        while (1) delay(1000);
    }

    // ── Pipeline sync primitives ──────────────────────────────────────────────
    decodeQueue  = xQueueCreate(g_numJpegSlots,   sizeof(DecodeMsg));
    displayQueue = xQueueCreate(g_numDisplayBufs,  sizeof(DisplayMsg));
    for (int s = 0; s < g_numJpegSlots; s++) {
        slotFree[s] = xSemaphoreCreateBinary();
        xSemaphoreGive(slotFree[s]);
    }

    {
        char bufMsg[40];
        snprintf(bufMsg, sizeof(bufMsg), "J:%d D:%d (max J:%d D:%d)",
                 g_numJpegSlots, g_numDisplayBufs,
                 MM_MAX_JPEG_SLOTS, MM_MAX_DISPLAY_BUFS);
        statusLine(2, "Buffers:", bufMsg, TFT_GREEN);
    }

    Serial.printf("[MEM] free SRAM  : %lu KB / %lu KB\n",
        heap_caps_get_free_size(MALLOC_CAP_INTERNAL)  / 1024,
        heap_caps_get_total_size(MALLOC_CAP_INTERNAL) / 1024);
    Serial.printf("[MEM] free PSRAM : %lu KB / %lu KB\n",
        heap_caps_get_free_size(MALLOC_CAP_SPIRAM)  / 1024,
        heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024);

    // ── SPIFFS — mounted now so initOfflinePlayer() is available later ────────
    // The partition table (partitions_16MB_ota.csv) must include a spiffs entry.
    // Use PlatformIO "Upload Filesystem Image" to upload .mjpeg files.
    // Mount failure is non-fatal here; offline mode will report no files found.
    bool spiffsOk = SPIFFS.begin(false);
    Serial.printf("[SPIFFS] %s\n", spiffsOk ? "mounted OK" : "mount failed (no SPIFFS partition?)");

    // ── WiFi — try for up to WIFI_CONNECT_TIMEOUT_MS ─────────────────────────
    statusLine(3, "WiFi:", "Connecting...", TFT_YELLOW);
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    esp_wifi_set_max_tx_power(40);   // ~10 dBm — the community fix
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11N);
    esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT40);

    uint32_t ws = millis();
    uint8_t  tick = 0;
    bool wifiConnected = false;

    while (WiFi.status() != WL_CONNECTED) {
        delay(250);
        tick++;
        char buf[24];
        snprintf(buf, sizeof(buf), "Connecting%.*s", tick % 4, "....");
        statusLine(3, "WiFi:", buf, TFT_YELLOW);

        if (millis() - ws > WIFI_CONNECT_TIMEOUT_MS) {
            Serial.println("[WIFI] Association timeout");
            statusLine(3, "WiFi:", "Timeout", TFT_RED);

            // Attempt offline mode
            if (initOfflinePlayer()) {
                enterOfflineMode();
                // loop() will call runOfflinePlayer() — skip task creation
                return;
            }
            // No files either → restart and try again
            statusLine(3, "WiFi:", "No offline files — restart", TFT_RED);
            delay(3000);
            ESP.restart();
        }
    }

    wifiConnected = true;
    esp_wifi_set_ps(WIFI_PS_NONE);
    g_wifiOk = true;
    g_wifiConnectedMs = millis();   // start the offline-trigger countdown

    String ip = WiFi.localIP().toString();
    char ipBuf[36];
    snprintf(ipBuf, sizeof(ipBuf), "%s (%ddBm)", ip.c_str(), WiFi.RSSI());
    statusLine(3, "WiFi:", ipBuf, TFT_GREEN);
    statusLine(5, "Mode:",   "Wireless display",  TFT_CYAN);
    statusLine(6, "Status:", "Waiting for PC...", TFT_YELLOW);
    Serial.printf("[OK] WiFi: %s\n", ip.c_str());

    // Pre-scan SPIFFS so initOfflinePlayer is fast if needed later
    initOfflinePlayer();

    // ── OTA ───────────────────────────────────────────────────────────────────
    xTaskCreatePinnedToCore(otaTask, "OTA", 8192, NULL, 1, NULL, 0);
    statusLine(4, "OTA:", "Ready (esp32s3-display)", TFT_CYAN);

    // ── Streaming tasks ───────────────────────────────────────────────────────
    xTaskCreatePinnedToCore(networkTask,     "NetTask",     10240, NULL, 3, NULL, 0);
    xTaskCreatePinnedToCore(wifiWatchdogTask,"WifiWatchdog", 4096, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(displayTask,     "DispTask",     4096, NULL, 2, NULL, 0);

    // ── Per-core CPU utilisation idle hooks ───────────────────────────────────
    // Core 0 runs NetTask / OTA / displayTask; Core 1 runs loop() (renderer).
    // Each hook increments its core's idle-tick counter; networkTask computes
    // CPU% = 100 - (idleDelta * 100 / expectedTicks) over its 400 ms window.
    esp_register_freertos_idle_hook_for_cpu(idleHookCore0, 0);
    esp_register_freertos_idle_hook_for_cpu(idleHookCore1, 1);

    Serial.println("[OK] Ready.");
}


// ─────────────────────────────────────────────
//  MAIN LOOP  (Core 1 — renderer)
// ─────────────────────────────────────────────
void loop() {

    // ── Offline mode: hand Core 1 to the MJPEG player forever ────────────────
    // runOfflinePlayer() is an infinite loop; it never returns.
    // Reaching this point means enterOfflineMode() was called and all streaming
    // tasks have (or are in the process of) self-terminating.
    if (g_offlineMode) {
        runOfflinePlayer();
        return;   // unreachable — satisfies the compiler
    }

    // ── Offline-trigger watchdog ───────────────────────────────────────────────
    // If WiFi is up but no stream packet has arrived within OFFLINE_TRIGGER_MS
    // of WiFi association, switch to offline mode.
    // g_wifiConnectedMs is set when WiFi first connects in setup().
    // g_streaming becomes true when the first tile is decoded.
    if (!g_streaming
        && g_wifiOk
        && g_wifiConnectedMs > 0
        && (millis() - g_wifiConnectedMs) > OFFLINE_TRIGGER_MS) {

        Serial.println("[MAIN] No stream after 5 s — checking for offline files...");
        // initOfflinePlayer() was called speculatively in setup(); s_fileCount is ready.
        enterOfflineMode();
        // Next call to loop() hits the g_offlineMode branch above and starts playback.
        return;
    }

    // ── WiFi disconnection watchdog ───────────────────────────────────────────
    // If WiFi is disconnected for more than 10 seconds, enter offline mode.
    if (!g_wifiOk
        && g_wifiDisconnectedMs > 0
        && (millis() - g_wifiDisconnectedMs) > 10000) {

        Serial.println("[MAIN] WiFi disconnected for 10 s — entering offline mode");
        enterOfflineMode();
        return;
    }

    // ── Normal streaming decode path ──────────────────────────────────────────
    static bool     streamStarted = false;
    static uint32_t decodeAcc     = 0;
    static uint32_t decodeCount   = 0;

    static uint8_t  pendingFrame = 0xFF;
    static uint8_t  readyMask    = 0;
    static uint32_t frameStartMs = 0;

    DecodeMsg msg;
    if (xQueueReceive(decodeQueue, &msg, pdMS_TO_TICKS(40)) != pdTRUE) return;

    if (pendingFrame != 0xFF && frameStartMs > 0 && (millis() - frameStartMs) > 150) {
        pendingFrame = 0xFF;
        readyMask    = 0;
        frameStartMs = 0;
    }

    if (pendingFrame == 0xFF || msg.frameId != pendingFrame) {
        if (pendingFrame != 0xFF && readyMask != 0)
            g_abortedFrames++;
        pendingFrame = msg.frameId;
        readyMask    = 0;
        frameStartMs = millis();
    }

    uint32_t decUs = 0;
    bool ok = decodeSlot(msg, decUs);

    xSemaphoreGive(slotFree[msg.slotIdx]);

    if (ok) {
        tiles[msg.tId].stat_decoded++;
        decodeAcc   += decUs;
        decodeCount++;

        readyMask |= (uint8_t)(1u << msg.tId);

        if (decodeCount >= 16) {
            g_avgDecodeUs = decodeAcc / decodeCount;
            decodeAcc = 0; decodeCount = 0;
        }

        if (!streamStarted) {
            streamStarted = true;
            g_streaming   = true;
            g_lastPktMs   = millis();
            statusLine(6, "Status:", "STREAMING!", TFT_GREEN);
            delay(200);
        }

        if (readyMask == 0x0F) {
            DisplayMsg dmsg = { msg.frameId, writeSet };
            if (xQueueSend(displayQueue, &dmsg, pdMS_TO_TICKS(20)) != pdTRUE)
                g_abortedFrames++;
            writeSet = (writeSet + 1) % g_numDisplayBufs;
            readyMask    = 0;
            pendingFrame = 0xFF;
            frameStartMs = 0;
        }
    }

    uint32_t now = millis();
    if (stat_prevMs > 0) {
        static uint32_t lastIv = 0;
        uint32_t iv = now - stat_prevMs;
        if (lastIv > 0) {
            int32_t d = (int32_t)iv - (int32_t)lastIv;
            stat_jitter += (fabsf((float)d) - stat_jitter) / 16.0f;
        }
        lastIv = iv;
    }
    stat_prevMs = now;
}