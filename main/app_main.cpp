/*
 * ESP32-S3  |  ILI9341  |  SPI  |  320x240
 *
 * Full-frame pipeline: networkTask reassembles complete JPEGs in PSRAM,
 * decodeTask decodes block-by-block into a free PSRAM framebuffer, and
 * displayTask owns that framebuffer until asynchronous SPI DMA completes.
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
#include "esp_log.h"
#include "freertos/task.h"
#include <cstring>
#include <cstdio>

// WIFI_SSID / WIFI_PASS / OTA_HOSTNAME are #defines in nexus.h (Zone 1)
// — edit them there, not here.

uint16_t* frameFb[MM_MAX_DISPLAY_BUFS] = {};
JpegFrameSlot jpegFrames[MM_MAX_JPEG_SLOTS] = {};

uint8_t g_numJpegSlots   = CFG_NUM_JPEG_SLOTS;
uint8_t g_numDisplayBufs = CFG_NUM_DISPLAY_BUFS;

QueueHandle_t     jpegFreeQueue;
QueueHandle_t     decodeQueue;
QueueHandle_t     displayQueue;
QueueHandle_t     framebufferFreeQueue;

SemaphoreHandle_t g_spiMutex = nullptr;

#if TOUCH_FEATURE_ENABLED
QueueHandle_t     touchQueue        = nullptr;
volatile bool     g_touchEnabled    = true;
volatile bool     g_touchCalibrated = false;
#endif

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
//  CROSS-CORE STATS DEFINITIONS
// ─────────────────────────────────────────────
volatile uint32_t g_avgDecodeUs     = 0;
volatile uint32_t g_presentedFrames = 0;
volatile uint32_t g_abortedFrames   = 0;
volatile uint32_t g_networkDrops    = 0;

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

static void resetJpegMetadata(JpegFrameSlot& frame) {
    memset(frame.chunkReceived, 0, sizeof(frame.chunkReceived));
    frame.frameId = 0xFF;
    frame.totalChunks = 0;
    frame.chunksReceived = 0;
    frame.frameSize = 0;
    frame.firstChunkMs = 0;
    frame.receiving = false;
    frame.queued = false;
    frame.inUse = false;
}

// ─────────────────────────────────────────────
//  app_main()  — was setup()
// ─────────────────────────────────────────────
extern "C" void app_main(void) {
    printf("\n[BOOT] full-frame pipeline (esp_new_jpeg block decode, BE pixels)\n");

    g_spiMutex = xSemaphoreCreateMutex();
#if TOUCH_FEATURE_ENABLED
    touchQueue = xQueueCreate(TOUCH_EVENT_QUEUE_LEN, sizeof(TouchEvent));
#endif

    lcd.init(); lcd.setRotation(3); lcd.setColorDepth(16);
    lcd.setTextFont(2); lcd.setTextSize(1);
    displayTuneBus();

    nvsInit();

#if TOUCH_FEATURE_ENABLED
    touchBegin();
    displayTuneBus();   // touch init re-inits the shared bus pads
#endif

    drawBootHeader();
#if TOUCH_FEATURE_ENABLED
    statusLine(0, "Display:", g_touchCalibrated ? "OK + Touch" : "OK (no touch cal)",
               g_touchCalibrated ? TFT_GREEN : TFT_YELLOW);
#else
    statusLine(0, "Display:", "OK (touch disabled)", TFT_GREEN);
#endif

    bool psramOk = esp_psram_is_initialized();
    statusLine(1, "PSRAM:", psramOk ? "Found" : "MISSING!", psramOk ? TFT_GREEN : TFT_RED);
    if (!psramOk) { while (1) delay(1000); }

    if (!initJpegDecoder()) {
        statusLine(2, "Decoder:", "INIT FAILED!", TFT_RED);
        while (1) delay(1000);
    }

    // ── Full-frame JPEG slots: use SRAM for two hot-path buffers, then PSRAM.
    g_numJpegSlots = CFG_NUM_JPEG_SLOTS;
    while (g_numJpegSlots >= 1) {
        for (int s = 0; s < MM_MAX_JPEG_SLOTS; s++) {
            if (jpegFrames[s].data) {
                heap_caps_free(jpegFrames[s].data);
                jpegFrames[s].data = nullptr;
            }
        }
        bool ok = true;
        for (int s = 0; s < g_numJpegSlots; s++) {
            uint32_t caps = s < 2
                ? MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
                : MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
            jpegFrames[s].data = (uint8_t*)heap_caps_aligned_alloc(
                16, MAX_JPEG_FRAME, caps);
            if (jpegFrames[s].data == nullptr && s < 2) {
                jpegFrames[s].data = (uint8_t*)heap_caps_aligned_alloc(
                    16, MAX_JPEG_FRAME, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            }
            if (!jpegFrames[s].data) { ok = false; break; }
        }
        if (ok) break;
        printf("[WARN] Full-frame JPEG buffers %d failed — retrying with %d\n",
               g_numJpegSlots, g_numJpegSlots - 1);
        g_numJpegSlots--;
    }
    if (g_numJpegSlots == 0) {
        printf("[ERROR] Cannot allocate even 1 full-frame JPEG buffer — halting\n");
        statusLine(2, "Buffers:", "JPEG ALLOC FAILED!", TFT_RED);
        while (1) delay(1000);
    }
    printf("[MEM] Full-frame JPEG buffers: %d / %d (up to %u bytes each, first 2 prefer SRAM, CFG=%d)\n",
           g_numJpegSlots, MM_MAX_JPEG_SLOTS, (unsigned)MAX_JPEG_FRAME, CFG_NUM_JPEG_SLOTS);

    // ── Display framebuffers (PSRAM) — fallback from CFG down to 2 ───────────
    g_numDisplayBufs = CFG_NUM_DISPLAY_BUFS;
    while (g_numDisplayBufs >= 2) {
        for (int s = 0; s < MM_MAX_DISPLAY_BUFS; s++) {
            if (frameFb[s]) { heap_caps_free(frameFb[s]); frameFb[s] = nullptr; }
        }
        bool ok = true;
        for (int s = 0; s < g_numDisplayBufs; s++) {
            frameFb[s] = (uint16_t*)heap_caps_aligned_alloc(
                32, SCREEN_W * SCREEN_H * 2, MALLOC_CAP_SPIRAM);  // 32 = S3 D-cache line
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

    // ── Pipeline sync primitives ──────────────────────────────────────────────
    jpegFreeQueue = xQueueCreate(g_numJpegSlots, sizeof(uint8_t));
    decodeQueue  = xQueueCreate(g_numJpegSlots,   sizeof(DecodeMsg));
    displayQueue = xQueueCreate(g_numDisplayBufs,  sizeof(DisplayMsg));
    framebufferFreeQueue = xQueueCreate(g_numDisplayBufs, sizeof(uint8_t));
    if (!jpegFreeQueue || !decodeQueue || !displayQueue || !framebufferFreeQueue) {
        printf("[ERROR] Pipeline queue allocation failed\n");
        statusLine(2, "Buffers:", "QUEUE ALLOC FAILED!", TFT_RED);
        while (1) delay(1000);
    }
    for (int s = 0; s < g_numJpegSlots; s++) {
        uint8_t idx = (uint8_t)s;
        resetJpegMetadata(jpegFrames[s]);
        if (xQueueSend(jpegFreeQueue, &idx, 0) != pdTRUE) {
            printf("[ERROR] Failed to initialize JPEG free queue\n");
            while (1) delay(1000);
        }
    }
    for (int s = 0; s < g_numDisplayBufs; s++) {
        uint8_t idx = (uint8_t)s;
        if (xQueueSend(framebufferFreeQueue, &idx, 0) != pdTRUE) {
            printf("[ERROR] Failed to initialize framebuffer free queue\n");
            while (1) delay(1000);
        }
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
    xTaskCreatePinnedToCore(displayTask,     "DispTask",     4096, NULL, 3, NULL, 0);

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
//  DECODE TASK (Core 1): full-frame JPEG to a free framebuffer.
// ─────────────────────────────────────────────
static void decodeTask(void*) {
    for (;;) {
        DecodeMsg msg;
        if (xQueueReceive(decodeQueue, &msg, pdMS_TO_TICKS(40)) != pdTRUE) continue;

        uint8_t jpegIdx = msg.jpegSlot;
        uint8_t framebufferIdx = 0xFF;
        uint32_t decodeUs = 0;
        bool gotFramebuffer = xQueueReceive(
            framebufferFreeQueue, &framebufferIdx, pdMS_TO_TICKS(50)) == pdTRUE;
        bool ok = false;
        if (gotFramebuffer) {
            ok = decodeJpegFrame(msg, framebufferIdx, decodeUs);
        }

        JpegFrameSlot& jpeg = jpegFrames[jpegIdx];
        jpeg.frameId = 0xFF;
        jpeg.totalChunks = 0;
        jpeg.chunksReceived = 0;
        jpeg.frameSize = 0;
        jpeg.firstChunkMs = 0;
        memset(jpeg.chunkReceived, 0, sizeof(jpeg.chunkReceived));
        jpeg.receiving = false;
        jpeg.queued = false;
        jpeg.inUse = false;
        if (xQueueSend(jpegFreeQueue, &jpegIdx, 0) != pdTRUE) {
            ESP_LOGE("pipeline", "JPEG slot %u could not be returned", jpegIdx);
        }

        if (!ok) {
            g_abortedFrames = g_abortedFrames + 1;
            if (gotFramebuffer &&
                xQueueSend(framebufferFreeQueue, &framebufferIdx, 0) != pdTRUE) {
                ESP_LOGE("pipeline", "framebuffer %u could not be returned after decode failure",
                         framebufferIdx);
            }
        } else {
            DisplayMsg dmsg = { msg.frameId, framebufferIdx };
            if (xQueueSend(displayQueue, &dmsg, 0) != pdTRUE) {
                g_abortedFrames = g_abortedFrames + 1;
                if (xQueueSend(framebufferFreeQueue, &framebufferIdx, 0) != pdTRUE) {
                    ESP_LOGE("pipeline", "framebuffer %u could not be returned after queue failure",
                             framebufferIdx);
                }
            } else {
                g_avgDecodeUs = decodeUs;
                if (!g_streaming) {
                    g_streaming = true;
                    statusLine(6, "Status:", "STREAMING!", TFT_GREEN);
                }
            }
        }

        uint32_t now = millis();
        if (stat_prevMs > 0) {
            uint32_t interval = now - stat_prevMs;
            static uint32_t lastInterval = 0;
            if (lastInterval > 0) {
                int32_t delta = (int32_t)interval - (int32_t)lastInterval;
                stat_jitter += (fabsf((float)delta) - stat_jitter) / 16.0f;
            }
            lastInterval = interval;
        }
        stat_prevMs = now;
    }
}