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
 *  If WiFi never associates within WIFI_CONNECT_TIMEOUT_MS, the ESP just
 *  restarts and tries again (no offline/local-playback fallback anymore).
 */
// nexus.h already pulls in Arduino.h, WiFi.h, esp_wifi.h, JPEGDEC.h,
// LovyanGFX.hpp, FreeRTOS/queue/semphr, lwip sockets/netdb, fcntl, math —
// only listing what's uniquely needed here.
#include "nexus.h"
#include "display.h"
#include "network.h"
#include "jpeg_decode.h"
#include <ArduinoOTA.h>
#include <esp_attr.h>
#include <esp_heap_caps.h>
#include "freertos/task.h"
#include "esp_freertos_hooks.h"
#include <cstring>

// WIFI_SSID / WIFI_PASS / OTA_HOSTNAME are #defines in nexus.h (Zone 1)
// — edit them there, not here.

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
//  PER-CORE CPU UTILISATION
// ─────────────────────────────────────────────
// The idle hook fires in a tight loop — potentially thousands of times per
// tick.  Calling esp_timer_get_time() (hardware 64-bit register read) on
// every invocation created enough overhead to stutter the decode/network loop.
//
// Fix: gate on the FreeRTOS tick counter, which is just a global variable
// read (< 5 ns).  We add one tick-period (1000 µs) to the idle accumulator
// at most once per 1 ms tick — zero hardware access, zero stutter.
// Resolution is ±1 tick (1 ms), which is plenty for xx.x% display.

volatile uint32_t g_cpuIdleUs[2] = { 0, 0 };

static bool IRAM_ATTR idleHookCore0() {
    static TickType_t s_lastTick = 0;
    TickType_t tick = xTaskGetTickCount();
    if (tick == s_lastTick) return false;
    s_lastTick = tick;
    g_cpuIdleUs[0] += portTICK_PERIOD_MS * 1000u;
    return false;
}

static bool IRAM_ATTR idleHookCore1() {
    static TickType_t s_lastTick = 0;
    TickType_t tick = xTaskGetTickCount();
    if (tick == s_lastTick) return false;
    s_lastTick = tick;
    g_cpuIdleUs[1] += portTICK_PERIOD_MS * 1000u;
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

// Forward decl — defined after setup() (was the body of Arduino loop());
// setup() needs it to spin the task up.
static void decodeTask(void*);


// ─────────────────────────────────────────────
//  OTA TASK  (Core 0, priority 1)
// ─────────────────────────────────────────────
static void otaTask(void* /*pv*/) {
    ArduinoOTA.setHostname(OTA_HOSTNAME);

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
    Serial.printf("[OTA] Ready — hostname: " OTA_HOSTNAME "  IP: %s\n",
                  WiFi.localIP().toString().c_str());

    for (;;) {
        ArduinoOTA.handle();
        vTaskDelay(pdMS_TO_TICKS(10));
    }
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

    // ── WiFi — try for up to WIFI_CONNECT_TIMEOUT_MS ─────────────────────────
    statusLine(3, "WiFi:", "Connecting...", TFT_YELLOW);
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    esp_wifi_set_max_tx_power(80);   // units are 0.25 dBm steps → requests 20 dBm;
                                      // actual radiated power may be clamped lower
                                      // by regional regulatory limits
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11N);
    esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT40);

    uint32_t ws = millis();
    uint8_t  tick = 0;

    while (WiFi.status() != WL_CONNECTED) {
        delay(250);
        tick++;
        char buf[24];
        snprintf(buf, sizeof(buf), "Connecting%.*s", tick % 4, "....");
        statusLine(3, "WiFi:", buf, TFT_YELLOW);

        if (millis() - ws > WIFI_CONNECT_TIMEOUT_MS) {
            Serial.println("[WIFI] Association timeout — restarting");
            statusLine(3, "WiFi:", "Timeout — restart", TFT_RED);
            delay(3000);
            ESP.restart();
        }
    }

    esp_wifi_set_ps(WIFI_PS_NONE);
    g_wifiOk = true;
    g_wifiConnectedMs = millis();

    String ip = WiFi.localIP().toString();
    char ipBuf[36];
    snprintf(ipBuf, sizeof(ipBuf), "%s (%ddBm)", ip.c_str(), WiFi.RSSI());
    statusLine(3, "WiFi:", ipBuf, TFT_GREEN);
    statusLine(5, "Mode:",   "Wireless display",  TFT_CYAN);
    statusLine(6, "Status:", "Waiting for PC...", TFT_YELLOW);
    Serial.printf("[OK] WiFi: %s\n", ip.c_str());

    // ── OTA ───────────────────────────────────────────────────────────────────
    xTaskCreatePinnedToCore(otaTask, "OTA", 8192, NULL, 1, NULL, 0);
    statusLine(4, "OTA:", "Ready (" OTA_HOSTNAME ")", TFT_CYAN);

    // ── Streaming tasks ───────────────────────────────────────────────────────
    xTaskCreatePinnedToCore(networkTask,     "NetTask",     10240, NULL, 3, NULL, 0);
    xTaskCreatePinnedToCore(wifiWatchdogTask,"WifiWatchdog", 4096, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(displayTask,     "DispTask",     4096, NULL, 2, NULL, 0);

    // ── Decode task (was Arduino loop()) ──────────────────────────────────────
    // Same 8 KB the Arduino loop task used to run on, same core (Core 1),
    // now an explicit named task instead of the framework's implicit one.
    xTaskCreatePinnedToCore(decodeTask,      "DecodeTask",   8192, NULL, 2, NULL, 1);

    // ── Per-core CPU utilisation idle hooks ───────────────────────────────────
    // Core 0 runs NetTask / OTA / displayTask; Core 1 runs DecodeTask.
    // Each hook increments its core's idle-tick counter; networkTask computes
    // CPU% = 100 - (idleDelta * 100 / expectedTicks) over its 400 ms window.
    esp_register_freertos_idle_hook_for_cpu(idleHookCore0, 0);
    esp_register_freertos_idle_hook_for_cpu(idleHookCore1, 1);

    Serial.println("[OK] Ready.");

    // Everything from here on runs in dedicated tasks. This setup task
    // (the Arduino "loop task") has nothing left to do — delete it and
    // reclaim its 8 KB stack instead of leaving it idling in an empty loop().
    vTaskDelete(NULL);
}


// ─────────────────────────────────────────────
//  DECODE TASK  (Core 1) — formerly Arduino loop()
//  Everything below used to be one loop() iteration; the Arduino framework
//  called it repeatedly forever. Now it's an explicit task with its own
//  for(;;), created in setup() and pinned to Core 1 same as before — the
//  only behavioral difference is this task has a name FreeRTOS/debuggers
//  can see, instead of being the anonymous implicit loop task.
// ─────────────────────────────────────────────
static void decodeTask(void*) {

    bool     streamStarted = false;
    uint32_t decodeAcc     = 0;
    uint32_t decodeCount   = 0;

    uint8_t  pendingFrame = 0xFF;
    uint8_t  readyMask    = 0;
    uint32_t frameStartMs = 0;

    uint32_t lastIv = 0;

    for (;;) {

        // ── Normal streaming decode path ────────────────────────────────────────
        DecodeMsg msg;
        if (xQueueReceive(decodeQueue, &msg, pdMS_TO_TICKS(40)) != pdTRUE) continue;

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

                uint8_t justCompleted = writeSet;
                writeSet = (writeSet + 1) % g_numDisplayBufs;

                // Seed the next write buffer with the frame we just finished,
                // before any tile of the upcoming frame gets decoded into it.
                // Without this, a tile that fails mid-decode (corrupt/truncated
                // bitstream) or never arrives in time (network/decode
                // backpressure) leaves whatever partial MCU rows/garbage was
                // last written there — visible as torn/"shredded" blocks.
                // With the seed, a stalled tile falls back to the last good
                // frame's pixels instead: stale at worst, never garbage.
                // Safe to overwrite here — same assumption the original
                // 2-buffer design already relied on (decode is Core-1
                // exclusive between writeSet flips, and by the time writeSet
                // cycles back to this index displayTask's DMA out of it has
                // long since finished).
                memcpy(frameFb[writeSet], frameFb[justCompleted],
                       (size_t)SCREEN_W * SCREEN_H * 2);

                readyMask    = 0;
                pendingFrame = 0xFF;
                frameStartMs = 0;
            }
        }

        uint32_t now = millis();
        if (stat_prevMs > 0) {
            uint32_t iv = now - stat_prevMs;
            if (lastIv > 0) {
                int32_t d = (int32_t)iv - (int32_t)lastIv;
                stat_jitter += (fabsf((float)d) - stat_jitter) / 16.0f;
            }
            lastIv = iv;
        }
        stat_prevMs = now;
    }
}

// Arduino framework still requires loop() to exist and link, but nothing
// calls it anymore — setup() creates every task explicitly (including
// decodeTask, which replaces what loop() used to do) and then deletes its
// own task at the end instead of falling through to loop().
void loop() {}