#ifndef SHARED_H
#define SHARED_H

#include <Arduino.h>
#include <WiFi.h>
#include <esp_attr.h>
#include <esp_heap_caps.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/queue.h"
#include <lwip/sockets.h>
#include <lwip/netdb.h>

// ─────────────────────────────────────────────
//  CONFIG
// ─────────────────────────────────────────────
extern const char* WIFI_SSID;
extern const char* WIFI_PASS;
const int UDP_PORT = 12345;

#define SCREEN_W         320
#define SCREEN_H         240
#define NUM_TILES        4
#define TILE_W           160
#define TILE_H           120
#define TILE_PIXELS      (TILE_W * TILE_H)          // 19 200
#define CHUNK_DATA_SIZE  1400
#define MAX_TILE_CHUNKS  24          // 24 x 1400 = 33.6 KB max JPEG per tile
#define MAX_TILE_JPEG    (MAX_TILE_CHUNKS * CHUNK_DATA_SIZE)
#define TILE_TIMEOUT_MS  200

// Screen position of each tile: TL TR BL BR
extern const int16_t TILE_X[];
extern const int16_t TILE_Y[];

// ─────────────────────────────────────────────
//  PIPELINE SLOTS  (4 shared decode/display buffers)
// ─────────────────────────────────────────────
#define NUM_SLOTS 4
struct PipeSlot {
    uint8_t* assembly;   // SRAM — JPEGDEC reads here; single-cycle access critical
};
extern PipeSlot slot[];

// SRAM scratch for Core-1 decode.
extern uint16_t* decodeTemp;

// Double-buffered full-frame framebuffers in PSRAM.
extern uint16_t* frameFb[2];
extern uint8_t writeSet;  // Core 1 exclusive — no sync needed

// Message passed through the decode queue
struct DecodeMsg {
    uint8_t  frameId;   // frame sequence (0-255) for frame-sync presentation
    uint8_t  tId;       // which tile position (0-3) -> determines screen XY
    uint8_t  slotIdx;   // which PipeSlot holds the assembled JPEG
    uint16_t len;       // JPEG byte count in slot[slotIdx].assembly
};

// Pipeline synchronisation
extern QueueHandle_t     decodeQueue;                // depth-4 queue: net -> renderer
extern SemaphoreHandle_t slotFree[];        // given when renderer finishes slot

// Display pipeline: Core 1 posts here when all 4 tiles are ready.
struct DisplayMsg {
    uint8_t frameId;   // for stats / debug
    uint8_t bufSet;    // which frameFb[bufSet] to push (0 or 1)
};
extern QueueHandle_t displayQueue;      // depth-2 queue: renderer -> display task (Core 0)

// ─────────────────────────────────────────────
//  CHUNK REASSEMBLY STATE  (one per tile position)
// ─────────────────────────────────────────────
struct TileState {
    uint8_t* chunkBuf[MAX_TILE_CHUNKS]; // -> PSRAM chunkStorage slab
    uint16_t chunkLen[MAX_TILE_CHUNKS];
    bool     chunkGot[MAX_TILE_CHUNKS];
    uint8_t  frameId      = 0xFF;
    uint8_t  totalChunks  = 0;
    uint16_t frameSize    = 0;
    uint8_t  chunksGot    = 0;
    uint32_t firstChunkMs = 0;
    // Stats — written by Core 0, reset by Core 0 after each stat window
    uint32_t stat_decoded = 0;
    uint32_t stat_corrupt = 0;
    uint32_t stat_timeout = 0;
};
extern TileState tiles[];
extern uint8_t*  tileChunkStorage[];

// ─────────────────────────────────────────────
//  CROSS-CORE STATS  (Core 1 writes, Core 0 reads for UDP report)
// ─────────────────────────────────────────────
extern volatile uint32_t g_avgDecodeUs;     // avg tile decode us (excl. pushImage)
extern volatile uint32_t g_presentedFrames; // frames fully pushed to LCD
extern volatile uint32_t g_abortedFrames;   // partial frames dropped (UDP reorder)

// ─────────────────────────────────────────────
//  GLOBAL STATE
// ─────────────────────────────────────────────
extern bool     debugEnabled;
extern char     debugBuf[256];
extern int      g_sock;
extern struct   sockaddr_in g_remoteAddr;
extern bool     g_remoteAddrValid;
extern float    stat_jitter;  // Core 1 writes, Core 0 reads — float OK
extern uint32_t stat_prevMs;

// ── WiFi watchdog & RSSI display ─────────────────────────────────────────
extern volatile bool g_streaming;   // set true by loop() on first decoded tile
extern volatile bool g_wifiOk;      // mirrors WiFi.status() == WL_CONNECTED
extern volatile uint32_t g_wifiConnectedMs;   // millis() when WiFi first associated
extern volatile uint32_t g_wifiDisconnectedMs; // millis() when WiFi last disconnected

// ── Streaming timeout overlay ─────────────────────────────────────────────
#define PKT_TIMEOUT_MS   3000   // 3 s silence → "loose connection" overlay
#define OVERLAY_FLASH_MS 1000   // 1 s flash period (white ↔ red)
extern volatile uint32_t g_lastPktMs;     // written by networkTask on every valid packet

// Function declarations
extern void initDisplay();
extern void statusLine(uint8_t row, const char* label, const char* value, uint32_t col = TFT_WHITE);
extern void drawBootHeader();
extern void displayTask(void*);

extern void initJpegDecoder();
extern bool decodeSlot(const DecodeMsg& msg, uint32_t& decodeUs);

extern void networkTask(void*);

#endif // SHARED_H