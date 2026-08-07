#pragma once

#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <JPEGDEC.h>
#include <LovyanGFX.hpp>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include <lwip/sockets.h>
#include <lwip/netdb.h>
#include <fcntl.h>
#include <math.h>

// ── Streaming timeouts ────────────────────────────────────────────────────────
#define PKT_TIMEOUT_MS   3000
#define OVERLAY_FLASH_MS 1000

// ── Offline-mode timeouts ─────────────────────────────────────────────────────
// How long to wait for WiFi association before giving up and going offline.
#define WIFI_CONNECT_TIMEOUT_MS  150000
// How long after WiFi connects with no stream packet before going offline.
// Gives the PC sender time to discover the ESP via beacon.
#define OFFLINE_TRIGGER_MS       500000

// ── Display geometry ──────────────────────────────────────────────────────────
#define SCREEN_W         320
#define SCREEN_H         240
#define NUM_TILES        4
#define TILE_W           160
#define TILE_H           120
#define TILE_PIXELS      (TILE_W * TILE_H)
#define CHUNK_DATA_SIZE  1400
#define MAX_TILE_CHUNKS  24
#define MAX_TILE_JPEG    (MAX_TILE_CHUNKS * CHUNK_DATA_SIZE)
#define TILE_TIMEOUT_MS  200

static const int16_t TILE_X[NUM_TILES] = {  0, 160,   0, 160 };
static const int16_t TILE_Y[NUM_TILES] = {  0,   0, 120, 120 };
//  CFG_NUM_DISPLAY_BUFS (2–6)  — PSRAM ping-pong buffers for the display DMA.
//    Minimum 2 required for tear-free output.  3+ reduces frame drops when
//    decoding is slower than the display push (e.g. complex scenes at high quality).
// ═══════════════════════════════════════════════════════════════════════════════
#define CFG_NUM_JPEG_SLOTS    4   // desired JPEG slots  (1–6)
#define CFG_NUM_DISPLAY_BUFS  2   // desired display bufs (2–6)

// ── Hard array caps — do NOT exceed ──────────────────────────────────────────
#define MM_MAX_JPEG_SLOTS     6
#define MM_MAX_DISPLAY_BUFS   6

// Backward-compat alias (used by legacy code paths that don't need the runtime val)
#define NUM_SLOTS MM_MAX_JPEG_SLOTS

// ── Runtime actual counts (set by setup() after fallback allocation) ──────────
extern uint8_t g_numJpegSlots;    // actual slots allocated  (1–MM_MAX_JPEG_SLOTS)
extern uint8_t g_numDisplayBufs;  // actual display bufs allocated (2–MM_MAX_DISPLAY_BUFS)

// ── Pipeline structs ──────────────────────────────────────────────────────────
struct PipeSlot {
    uint8_t* assembly;
};

struct DecodeMsg {
    uint8_t  frameId;
    uint8_t  tId;
    uint8_t  slotIdx;
    uint16_t len;
};

struct DisplayMsg {
    uint8_t frameId;
    uint8_t bufSet;
};

struct TileState {
    uint8_t* chunkBuf[MAX_TILE_CHUNKS];
    uint16_t chunkLen[MAX_TILE_CHUNKS];
    bool     chunkGot[MAX_TILE_CHUNKS];
    uint8_t  frameId      = 0xFF;
    uint8_t  totalChunks  = 0;
    uint16_t frameSize    = 0;
    uint8_t  chunksGot    = 0;
    uint32_t firstChunkMs = 0;
    uint32_t stat_decoded = 0;
    uint32_t stat_corrupt = 0;
    uint32_t stat_timeout = 0;
};

// ── FreeRTOS handles ──────────────────────────────────────────────────────────
extern QueueHandle_t     decodeQueue;
extern QueueHandle_t     displayQueue;
extern SemaphoreHandle_t slotFree[MM_MAX_JPEG_SLOTS];

// ── Shared buffers ────────────────────────────────────────────────────────────
extern uint16_t* frameFb[MM_MAX_DISPLAY_BUFS];   // only [0..g_numDisplayBufs-1] allocated
extern uint8_t*  tileChunkStorage[NUM_TILES];
extern PipeSlot  slot[MM_MAX_JPEG_SLOTS];         // only [0..g_numJpegSlots-1] allocated
extern TileState tiles[NUM_TILES];

// ── Cross-core stats ──────────────────────────────────────────────────────────
extern volatile uint32_t g_avgDecodeUs;
extern volatile uint32_t g_presentedFrames;
extern volatile uint32_t g_abortedFrames;

// ── Streaming / WiFi state ────────────────────────────────────────────────────
extern volatile bool     g_streaming;        // true once first tile decoded
extern volatile bool     g_wifiOk;           // mirrors WiFi.status() == WL_CONNECTED
extern volatile uint32_t g_lastPktMs;        // updated by networkTask on each valid UDP packet
extern volatile uint32_t g_wifiConnectedMs;  // millis() when WiFi first associated

// ── Offline mode flag ─────────────────────────────────────────────────────────
// Set true by enterOfflineMode() in main.cpp.  All streaming tasks check this
// on each iteration and call vTaskDelete(NULL) when they see it set.
// Once set, never cleared — offline mode persists until reboot.
extern volatile bool g_offlineMode;

// ── Config ────────────────────────────────────────────────────────────────────
extern const char* WIFI_SSID;
extern const char* WIFI_PASS;
const int UDP_PORT = 12345;

// ── Per-core CPU utilisation (updated by FreeRTOS idle hooks in main.cpp) ─────
// Accumulates actual microseconds the idle task spent running on each core.
// Between consecutive idle-hook calls that are <1 ms apart (idle task running
// continuously), the elapsed µs are added.  If the gap is ≥1 ms a real task
// preempted — that gap is NOT counted as idle time.
// Network debug task divides window idle-µs by total-µs to get CPU%.
extern volatile uint32_t g_cpuIdleUs[2];

// ── Debug / stats ─────────────────────────────────────────────────────────────
extern bool  debugEnabled;
extern char  debugBuf[320];   // 320 B: original ~125 B + CPU0/CPU1/extra headroom
extern int   g_sock;
extern struct sockaddr_in g_remoteAddr;
extern bool  g_remoteAddrValid;
extern float stat_jitter;

// ── Display double-buffer write index (Core-1 exclusive) ─────────────────────
extern uint8_t writeSet;