#pragma once
/*
 * nexus.h — the center of this project.
 * ───────────────────────────────────────
 * Every setting you'd ever need to touch to make this run on YOUR network
 * and YOUR wiring lives in this one file, at the top, in the order you're
 * most likely to need it. Everything below the internal-structures line
 * is plumbing — pipeline structs, cross-task globals, FreeRTOS handles —
 * that the rest of the firmware depends on, but that you shouldn't need
 * to edit just to get this running.
 *
 * (This file used to be called shared.h. Renamed because "nexus" is what
 * it actually is: the single point everything else in this codebase
 * connects through.)
 */

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

// ═══════════════════════════════════════════════════════════════════════════
//  ZONE 1 — EDIT THIS BEFORE YOU FLASH
//  Nothing works until these match your network and your wiring.
// ═══════════════════════════════════════════════════════════════════════════

// ── WiFi ──────────────────────────────────────────────────────────────────
// Your network's credentials. This is the one thing every single person
// who uses this firmware has to change.
#define WIFI_SSID  "Endmin"
#define WIFI_PASS  "987654321"

// Hostname the ESP advertises for OTA (wireless flashing) discovery.
// If you change this, also update `upload_port` in platformio.ini to match
// (it's currently `esp32s3-display.local`).
#define OTA_HOSTNAME  "esp32s3-display"

// ── Display wiring — ILI9341 over SPI ────────────────────────────────────
// If your panel is wired to different pins, this is the only place you
// need to change it — display.h reads these, nothing is hardcoded there.
#define LCD_SPI_HOST    SPI2_HOST   // ESP32-S3 has SPI2_HOST / SPI3_HOST available
#define LCD_WRITE_HZ    80000000    // SPI write clock

#define LCD_PIN_SCLK    12
#define LCD_PIN_MOSI    13
#define LCD_PIN_MISO    -1          // -1 = not connected (display is write-only)
#define LCD_PIN_DC      4           // data/command
#define LCD_PIN_CS      10          // chip select
#define LCD_PIN_RST     5           // reset
#define LCD_PIN_BUSY    -1          // -1 = not used by this panel

// Panel's native resolution, before rotation. Most 2.4"-2.8" ILI9341
// boards report portrait (240x320) natively and get rotated to landscape
// in software — see `lcd.setRotation()` in display.cpp if you need to
// change orientation.
#define LCD_PANEL_W     240
#define LCD_PANEL_H     320

// ═══════════════════════════════════════════════════════════════════════════
//  ZONE 2 — CHANGE THESE IF YOUR HARDWARE SETUP DIFFERS
//  Different panel resolution, different memory budget, etc. The defaults
//  here match a 320x240 panel on an N16R8 (16 MB flash / 8 MB PSRAM) board.
// ═══════════════════════════════════════════════════════════════════════════

// ── Display geometry (logical, post-rotation) ────────────────────────────
#define SCREEN_W         320
#define SCREEN_H         240
#define NUM_TILES        4
#define TILE_W           160
#define TILE_H           120
#define TILE_PIXELS      (TILE_W * TILE_H)

static const int16_t TILE_X[NUM_TILES] = {  0, 160,   0, 160 };
static const int16_t TILE_Y[NUM_TILES] = {  0,   0, 120, 120 };

// ── Network chunking ──────────────────────────────────────────────────────
#define CHUNK_DATA_SIZE  1400              // bytes of JPEG payload per UDP packet
#define MAX_TILE_CHUNKS  24                // hard cap on chunks per tile
#define MAX_TILE_JPEG    (MAX_TILE_CHUNKS * CHUNK_DATA_SIZE)  // = 33,600 B/tile ceiling

// ── Pipeline depth (memory budget) ────────────────────────────────────────
// CFG_NUM_JPEG_SLOTS lives in SRAM (decoder reads it, needs to be fast) —
// SRAM is the scarce resource here, don't raise this casually.
// CFG_NUM_DISPLAY_BUFS lives in PSRAM (8 MB, much more headroom) — safe to
// raise a buffer or two if you want more slack absorbing display jitter.
#define CFG_NUM_JPEG_SLOTS    4   // desired JPEG slots  (1–6)
#define CFG_NUM_DISPLAY_BUFS  3   // desired display bufs (2–6)

// Hard array caps — do NOT exceed. These size the actual arrays; the
// CFG_ values above are the *desired* runtime count within that cap.
#define MM_MAX_JPEG_SLOTS     6
#define MM_MAX_DISPLAY_BUFS   6

// Backward-compat alias (used by legacy code paths that don't need the runtime val)
#define NUM_SLOTS MM_MAX_JPEG_SLOTS

const int UDP_PORT = 12345;

// ═══════════════════════════════════════════════════════════════════════════
//  ZONE 3 — TIMING / BEHAVIOR TUNABLES
//  Safe defaults already set. Change these only if you know why.
// ═══════════════════════════════════════════════════════════════════════════
#define PKT_TIMEOUT_MS           3000
#define OVERLAY_FLASH_MS         1000
#define TILE_TIMEOUT_MS          200
#define WIFI_CONNECT_TIMEOUT_MS  150000   // how long to wait for WiFi before restarting

// ═══════════════════════════════════════════════════════════════════════════
//  ZONE 4 — INTERNAL PLUMBING
//  Pipeline structs, cross-task globals, FreeRTOS handles. The rest of the
//  firmware depends on these; you shouldn't need to touch anything below
//  this line just to get up and running.
// ═══════════════════════════════════════════════════════════════════════════

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
