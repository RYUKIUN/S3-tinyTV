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


// ─────────────────────────────────────────────
//  CONFIG DEFINITIONS
// ─────────────────────────────────────────────
const char* WIFI_SSID  = "Endmin";
const char* WIFI_PASS  = "987654321";

// ─────────────────────────────────────────────
//  PIPELINE SLOTS DEFINITIONS
// ─────────────────────────────────────────────
PipeSlot slot[NUM_SLOTS];

// Double-buffered full-frame framebuffers in PSRAM.
uint16_t* frameFb[2] = { nullptr, nullptr };
uint8_t writeSet = 0;  // Core 1 exclusive — no sync needed

// Pipeline synchronisation
QueueHandle_t     decodeQueue;                // depth-4 queue: net -> renderer
SemaphoreHandle_t slotFree[NUM_SLOTS];        // given when renderer finishes slot

// Display pipeline: Core 1 posts here when all 4 tiles are ready.
QueueHandle_t displayQueue;      // depth-2 queue: renderer -> display task (Core 0)

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

volatile uint32_t g_avgQueueWaitUs  = 0;
volatile uint32_t g_avgDmaPushUs    = 0;
volatile uint32_t g_avgFrameUs      = 0;
volatile uint32_t g_cpu0SpinHz      = 0;
volatile uint32_t g_cpu1SpinHz      = 0;

// ─────────────────────────────────────────────
//  GLOBAL STATE DEFINITIONS
// ─────────────────────────────────────────────
bool     debugEnabled      = false;
char     debugBuf[256];
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
//  CPU LOAD PROXY — REMOVED
// ─────────────────────────────────────────────
// A previous version of this file used a dedicated per-core task that spun
// a counter with taskYIELD() to approximate CPU load (since the FreeRTOS
// run-time-stats API isn't linked in on this Arduino-framework build).
//
// THIS WAS WRONG AND CRASHED THE BOARD: taskYIELD() only yields to tasks
// that are READY at the same or higher priority. The real FreeRTOS IDLE
// task is priority 0 — lower than our spinner's priority 1 — so the
// scheduler had no reason to ever switch to it. IDLE never got to run,
// the Task Watchdog (which IDLE feeds) never got reset, and the board
// hit TWDT abort within seconds.
//
// g_cpu0SpinHz / g_cpu1SpinHz are left defined as always-zero stubs (see
// below) so the rest of the pipeline (debug packet, dashboard) doesn't
// need further changes. If real CPU load visibility is wanted later, the
// safe paths are: (a) get configGENERATE_RUN_TIME_STATS actually linked
// in via sdkconfig.defaults + a clean rebuild, and use the real
// ulTaskGetIdleRunTimeCounter() API, or (b) instrument loop()/networkTask
// iteration time directly (no new tasks, no watchdog risk).
static void sampleIdleSpinRates() {
    // Intentionally a no-op. g_cpu0SpinHz/g_cpu1SpinHz stay at their
    // initialized value of 0 — visibly "unwired" rather than silently wrong.
}


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

    // ── Allocate SRAM tile-assembly slots ────────────────────────────────────
    bool allocOk = true;
    for (int s = 0; s < NUM_SLOTS; s++) {
        slot[s].assembly = (uint8_t*)heap_caps_aligned_alloc(
            16, MAX_TILE_JPEG, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!slot[s].assembly) {
            Serial.printf("[ERROR] slot[%d].assembly SRAM alloc failed\n", s);
            allocOk = false; break;
        }
    }

    for (int s = 0; s < 2 && allocOk; s++) {
        frameFb[s] = (uint16_t*)heap_caps_aligned_alloc(
            16, SCREEN_W * SCREEN_H * 2, MALLOC_CAP_SPIRAM);
        if (!frameFb[s]) {
            Serial.printf("[ERROR] frameFb[%d] PSRAM alloc failed\n", s);
            allocOk = false;
        }
    }

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
        statusLine(2, "Buffers:", "ALLOC FAILED!", TFT_RED);
        while (1) delay(1000);
    }

    // ── Pipeline sync primitives ──────────────────────────────────────────────
    decodeQueue  = xQueueCreate(NUM_SLOTS, sizeof(DecodeMsg));
    displayQueue = xQueueCreate(2, sizeof(DisplayMsg));
    for (int s = 0; s < NUM_SLOTS; s++) {
        slotFree[s] = xSemaphoreCreateBinary();
        xSemaphoreGive(slotFree[s]);
    }

    statusLine(2, "Buffers:", "OK (2-frame depth)", TFT_GREEN);

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
    Serial.println("[OK] Ready.");
}


// ─────────────────────────────────────────────
//  MAIN LOOP  (Core 1 — renderer)
// ─────────────────────────────────────────────
void loop() {

    sampleIdleSpinRates();  // cheap (internally rate-limited to 1 Hz) — see definition above

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
    static uint32_t queueWaitAcc  = 0;
    static uint32_t queueWaitCnt  = 0;

    static uint8_t  pendingFrame = 0xFF;
    static uint8_t  readyMask    = 0;
    static uint32_t frameStartMs = 0;
    static uint32_t frameStartUs = 0;   // micros() version, for g_avgFrameUs
    static uint32_t frameTimeAcc = 0;
    static uint32_t frameTimeCnt = 0;

    DecodeMsg msg;
    if (xQueueReceive(decodeQueue, &msg, pdMS_TO_TICKS(40)) != pdTRUE) return;

    // Queue-wait: time between network.cpp finishing tile reassembly
    // (DecodeMsg.readyMs, stamped at xQueueSend) and decodeSlot() starting here.
    // Large/growing values mean Core 1 is backed up — tiles are arriving faster
    // than they're being consumed, not that decode itself is slow.
    uint32_t queueWaitMs = millis() - msg.readyMs;
    queueWaitAcc += queueWaitMs;
    queueWaitCnt++;
    if (queueWaitCnt >= 16) {
        g_avgQueueWaitUs = (queueWaitAcc * 1000) / queueWaitCnt;  // ms->us for consistent units with DEC
        queueWaitAcc = 0; queueWaitCnt = 0;
    }

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
        frameStartUs = micros();
    }

    uint32_t decUs = 0;
    uint16_t mcuCalls = 0;
    bool ok = decodeSlot(msg, decUs, &mcuCalls);

    static uint32_t mcuLogCounter = 0;
    // if ((++mcuLogCounter & 0x3F) == 0) {  // every 64th tile, ~once every couple seconds at speed
    //     Serial.printf("[MCU] tile=%u mcuCallback invocations=%u (decode=%lu us)\n",
    //                   msg.tId, mcuCalls, decUs);
    // }

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
            frameTimeAcc += (micros() - frameStartUs);
            frameTimeCnt++;
            if (frameTimeCnt >= 16) {
                g_avgFrameUs = frameTimeAcc / frameTimeCnt;
                frameTimeAcc = 0; frameTimeCnt = 0;
            }

            DisplayMsg dmsg = { msg.frameId, writeSet };
            if (xQueueSend(displayQueue, &dmsg, pdMS_TO_TICKS(20)) != pdTRUE)
                g_abortedFrames++;
            writeSet ^= 1;
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