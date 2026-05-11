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
#define WIFI_CONNECT_TIMEOUT_MS  15000
// How long after WiFi connects with no stream packet before going offline.
// Gives the PC sender time to discover the ESP via beacon.
#define OFFLINE_TRIGGER_MS       5000

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

#define NUM_SLOTS 4

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
extern SemaphoreHandle_t slotFree[NUM_SLOTS];

// ── Shared buffers ────────────────────────────────────────────────────────────
extern uint16_t* decodeTemp;
extern uint16_t* frameFb[2];
extern uint8_t*  tileChunkStorage[NUM_TILES];
extern PipeSlot  slot[NUM_SLOTS];
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

// ── Debug / stats ─────────────────────────────────────────────────────────────
extern bool  debugEnabled;
extern char  debugBuf[256];
extern int   g_sock;
extern struct sockaddr_in g_remoteAddr;
extern bool  g_remoteAddrValid;
extern float stat_jitter;

// ── Display double-buffer write index (Core-1 exclusive) ─────────────────────
extern uint8_t writeSet;