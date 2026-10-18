/*
 * ESP32-S3  |  ILI9341  |  SPI  |  320x240
 *
 * PIPELINE ARCHITECTURE — unchanged from the original project; see
 * PROJECT_STRUCTURE_AND_EXPLANATION.md for the full writeup. This file is
 * the direct port of the old main.cpp: setup() -> app_main(), the Arduino
 * loop() task -> decodeTask() (already was its own pinned task, nothing
 * changes there), and ArduinoOTA's own polling task -> a single otaStart()
 * call once WiFi is up (esp_http_server runs its own task internally, so
 * there's no task to spin up by hand anymore).
 *
 * Decode pipeline: esp_new_jpeg now does the tile decode (jpeg_decode.cpp)
 * instead of JPEGDEC. decodeTask() itself — the frame-assembly/ready-mask/
 * fallback-to-last-good-frame state machine below — did not need to change
 * at all: decodeSlot() kept the exact same (DecodeMsg, uint32_t&) ->
 * bool signature specifically so this wouldn't have to move.
 */
#include "nexus.h"
#include "display.h"
#include "network.h"
#include "jpeg_decode.h"
#include "touch.h"
#include "wifi.h"
#include "ota.h"

#include "esp_heap_caps.h"
#include "esp_psram.h"
#include "esp_system.h"
#include "esp_freertos_hooks.h"
#include "freertos/task.h"
#include <cstring>
#include <cstdio>

// WIFI_SSID / WIFI_PASS / OTA_HOSTNAME are #defines in nexus.h (Zone 1)
// — edit them there, not here.

// ─────────────────────────────────────────────
//  PIPELINE SLOTS DEFINITIONS
// ─────────────────────────────────────────────
PipeSlot slot[MM_MAX_JPEG_SLOTS];

uint16_t* frameFb[MM_MAX_DISPLAY_BUFS] = {};
uint8_t   writeSet = 0;

uint8_t g_numJpegSlots   = CFG_NUM_JPEG_SLOTS;
uint8_t g_numDisplayBufs = CFG_NUM_DISPLAY_BUFS;

QueueHandle_t     decodeQueue;
SemaphoreHandle_t slotFree[MM_MAX_JPEG_SLOTS];
QueueHandle_t displayQueue;

SemaphoreHandle_t g_spiMutex = nullptr;

QueueHandle_t     touchQueue        = nullptr;
volatile bool     g_touchEnabled    = true;
volatile bool     g_touchCalibrated = false;

// ─────────────────────────────────────────────
//  PER-CORE CPU UTILISATION (esp_freertos_hooks.h is a genuine ESP-IDF
//  freertos-component API, not an Arduino wrapper — unchanged from before)
// ─────────────────────────────────────────────
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
volatile uint32_t g_avgDecodeUs     = 0;
volatile uint32_t g_presentedFrames = 0;
volatile uint32_t g_abortedFrames   = 0;

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

volatile bool     g_streaming        = false;
volatile bool     g_wifiOk           = false;
volatile uint32_t g_lastPktMs        = 0;
volatile uint32_t g_wifiConnectedMs  = 0;

static void decodeTask(void*);

// ─────────────────────────────────────────────
//  app_main()  — was setup()
// ─────────────────────────────────────────────
extern "C" void app_main(void) {
    printf("\n[BOOT] ping-pong pipeline (esp_new_jpeg decode, BE pixels)\n");

    g_spiMutex = xSemaphoreCreateMutex();
    touchQueue = xQueueCreate(TOUCH_EVENT_QUEUE_LEN, sizeof(TouchEvent));

    lcd.init(); lcd.setRotation(3); lcd.setColorDepth(16);
    lcd.setTextFont(2); lcd.setTextSize(1);

#if TOUCH_FEATURE_ENABLED
    touchBegin();
#endif

    drawBootHeader();
    statusLine(0, "Display:", g_touchCalibrated ? "OK + Touch" : "OK (no touch cal)",
               g_touchCalibrated ? TFT_GREEN : TFT_YELLOW);

    bool psramOk = esp_psram_is_initialized();
    statusLine(1, "PSRAM:", psramOk ? "Found" : "MISSING!", psramOk ? TFT_GREEN : TFT_RED);
    if (!psramOk) { while (1) delay(1000); }

    initJpegDecoder();

    // ── JPEG assembly slots (SRAM) — fallback from CFG down to 1 ─────────────
    g_numJpegSlots = CFG_NUM_JPEG_SLOTS;
    while (g_numJpegSlots >= 1) {
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
        printf("[WARN] JPEG slots %d failed — retrying with %d\n",
               g_numJpegSlots, g_numJpegSlots - 1);
        g_numJpegSlots--;
    }
    if (g_numJpegSlots == 0) {
        printf("[ERROR] Cannot allocate even 1 JPEG slot — halting\n");
        statusLine(2, "Buffers:", "SRAM ALLOC FAILED!", TFT_RED);
        while (1) delay(1000);
    }
    printf("[MEM] JPEG slots: %d / %d (CFG=%d)\n",
           g_numJpegSlots, MM_MAX_JPEG_SLOTS, CFG_NUM_JPEG_SLOTS);

    // ── Display framebuffers (PSRAM) — fallback from CFG down to 2 ───────────
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
        printf("[WARN] Display bufs %d failed — retrying with %d\n",
               g_numDisplayBufs, g_numDisplayBufs - 1);
        g_numDisplayBufs--;
    }
    if (g_numDisplayBufs < 2) {
        printf("[ERROR] Cannot allocate 2 display framebuffers — halting\n");
        statusLine(2, "Buffers:", "PSRAM ALLOC FAILED!", TFT_RED);
        while (1) delay(1000);
    }
    printf("[MEM] Display bufs: %d / %d (CFG=%d)\n",
           g_numDisplayBufs, MM_MAX_DISPLAY_BUFS, CFG_NUM_DISPLAY_BUFS);

    // ── Tile chunk storage (PSRAM) ────────────────────────────────────────────
    bool allocOk = true;
    for (int t = 0; t < NUM_TILES && allocOk; t++) {
        tileChunkStorage[t] = (uint8_t*)heap_caps_malloc(
            (size_t)MAX_TILE_CHUNKS * CHUNK_DATA_SIZE, MALLOC_CAP_SPIRAM);
        if (!tileChunkStorage[t]) {
            printf("[ERROR] tile[%d] chunkStorage PSRAM alloc failed\n", t);
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

    printf("[MEM] free SRAM  : %lu KB / %lu KB\n",
        (unsigned long)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)  / 1024),
        (unsigned long)(heap_caps_get_total_size(MALLOC_CAP_INTERNAL) / 1024));
    printf("[MEM] free PSRAM : %lu KB / %lu KB\n",
        (unsigned long)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)  / 1024),
        (unsigned long)(heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024));

    // ── WiFi — try for up to WIFI_CONNECT_TIMEOUT_MS ─────────────────────────
    statusLine(3, "WiFi:", "Connecting...", TFT_YELLOW);
    wifiInit();   // native esp_wifi station bring-up (wifi.cpp) — sets g_wifiOk
                  // asynchronously via its own event handler from here on.

    uint32_t ws = millis();
    uint8_t  tick = 0;

    while (!g_wifiOk) {
        delay(250);
        tick++;
        char buf[24];
        snprintf(buf, sizeof(buf), "Connecting%.*s", tick % 4, "....");
        statusLine(3, "WiFi:", buf, TFT_YELLOW);

        if (millis() - ws > WIFI_CONNECT_TIMEOUT_MS) {
            printf("[WIFI] Association timeout — restarting\n");
            statusLine(3, "WiFi:", "Timeout — restart", TFT_RED);
            delay(3000);
            esp_restart();
        }
    }

    g_wifiConnectedMs = millis();

    char ipStr[16];
    wifiGetIPString(ipStr, sizeof(ipStr));
    char ipBuf[36];
    snprintf(ipBuf, sizeof(ipBuf), "%s (%ddBm)", ipStr, (int)wifiGetRSSI());
    statusLine(3, "WiFi:", ipBuf, TFT_GREEN);
    statusLine(5, "Mode:",   "Wireless display",  TFT_CYAN);
    statusLine(6, "Status:", "Waiting for PC...", TFT_YELLOW);
    printf("[OK] WiFi: %s\n", ipStr);

    // ── OTA (HTTP, see ota.h) ─────────────────────────────────────────────────
    // esp_http_server runs its own internal task — nothing to pin/create here.
    otaStart();
    statusLine(4, "OTA:", "HTTP :" OTA_HOSTNAME, TFT_CYAN);

    // ── Streaming tasks ───────────────────────────────────────────────────────
    xTaskCreatePinnedToCore(networkTask,     "NetTask",     10240, NULL, 3, NULL, 0);
    xTaskCreatePinnedToCore(wifiWatchdogTask,"WifiWatchdog", 4096, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(displayTask,     "DispTask",     4096, NULL, 2, NULL, 0);

#if TOUCH_FEATURE_ENABLED
    xTaskCreatePinnedToCore(touchTask,       "TouchTask",    6144, NULL, 1, NULL, 0);
#endif

    // ── Decode task (was Arduino loop()) ──────────────────────────────────────
    xTaskCreatePinnedToCore(decodeTask,      "DecodeTask",   8192, NULL, 2, NULL, 1);

    // ── Per-core CPU utilisation idle hooks ───────────────────────────────────
    esp_register_freertos_idle_hook_for_cpu(idleHookCore0, 0);
    esp_register_freertos_idle_hook_for_cpu(idleHookCore1, 1);

    printf("[OK] Ready.\n");

    // Everything from here on runs in dedicated tasks. app_main()'s own task
    // is torn down automatically by the IDF FreeRTOS port when this function
    // returns — no explicit vTaskDelete(NULL) needed (that was an Arduino-
    // loop-task-specific cleanup; app_main has no "loop" to fall through to).
}

// ─────────────────────────────────────────────
//  DECODE TASK  (Core 1) — formerly Arduino loop()
//  Unchanged: decodeSlot() keeps the same signature regardless of which
//  JPEG library sits behind it, so none of this state machine moved.
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
