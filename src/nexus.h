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

// SPI write clock. 80MHz is right at the edge of what jumper-wire/breadboard
// wiring can carry reliably on this panel — random bit corruption at that
// rate shows up as unpredictable tearing (top/bottom of frame visibly
// split or shifted, position varies frame to frame). Dropped to a safer
// default; raise it back up in ~10MHz steps and watch for glitches
// returning if you want to reclaim some of that margin once wiring is
// solid (short leads, common ground, ideally soldered rather than
// breadboarded).
#define LCD_WRITE_HZ    75000000

#define LCD_PIN_SCLK    12
#define LCD_PIN_MOSI    13
#define LCD_PIN_MISO    -1          // the PANEL is write-only and never read.
                                    // The shared BUS does get a MISO line, but
                                    // it belongs to the touch controller — see
                                    // TOUCH_PIN_MISO below and display.h.
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

// ── Touch wiring — XPT2046 on the SAME SPI bus as the display ────────────
// The touch controller shares SCLK/MOSI with the panel and has its own CS.
// This is deliberate: LovyanGFX's bus lock already serialises access, and a
// touch read is 57 bytes at 1 MHz (~456 us) against the display's 150 KB
// DMA pushes — sharing costs far less than burning a second SPI host.
//
// MISO matters here. The panel is write-only (LCD_PIN_MISO = -1), but the
// XPT2046 has to be *read*, so the shared bus needs a MISO line after all.
// LovyanGFX's touch init adds it to the host; the panel simply never uses it.
#define TOUCH_PIN_CS    9
#define TOUCH_PIN_IRQ   8           // PENIRQ — low while the panel is pressed

// ── PENIRQ: wired or not? ────────────────────────────────────────────────
// 0 = PENIRQ is NOT usable (not connected / shorted). The driver is told
//     pin_int = -1 and touchTask polls instead.
// 1 = PENIRQ is properly wired. The driver gates on it and touchTask sleeps
//     on the interrupt, which costs literally nothing while untouched.
//
// This MUST be 0 while the line is unwired. It is not merely an
// optimisation flag: Touch_XPT2046::getTouchRaw() opens with
//     if (_cfg.pin_int >= 0 && gpio_in(_cfg.pin_int)) return 0;
// so a configured-but-floating PENIRQ reads high and makes every single read
// report "not touched" — touch would appear completely dead, not just less
// efficient. Flipping this to 1 is the ONLY change needed once it's fixed.
#define TOUCH_USE_IRQ   0
#define TOUCH_PIN_MISO  11          // shared-bus MISO (display doesn't use it)
#define TOUCH_SPI_HZ    1000000     // XPT2046 max is ~2 MHz; 1 MHz is the safe default

// BOOT button (GPIO0 on every ESP32-S3 devkit). Double-press it before the
// stream starts to force a touch recalibration — see touch.cpp.
#define BOOT_PIN        0

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

// ── Touch → PC mouse ──────────────────────────────────────────────────────
// Budget note: the video stream runs ~300 packets/s and ~420 KB/s. Touch is
// deliberately kept ~3 orders of magnitude below that. Events are sent ONLY
// while a finger is actually down, capped at TOUCH_REPORT_HZ, and suppressed
// entirely when the point hasn't moved. A 9-byte payload at 30 Hz is ~1.5 KB/s
// on the wire — 0.35% of the stream's bandwidth, and 0% when nobody's touching.
#define TOUCH_REPORT_HZ        30    // max MOVE reports/sec while pressed (DOWN/UP always sent)
// Idle sampling rate, used ONLY when TOUCH_USE_IRQ is 0. With PENIRQ wired,
// idle detection is interrupt-driven and this is ignored entirely.
// Each idle poll is one full 57-byte read (~456 us at 1 MHz), because without
// PENIRQ the driver has no cheap way to know the panel is untouched. 20 Hz
// costs ~0.9% SPI bus occupancy at idle and puts touch-down latency at up to
// 50 ms. Raise for snappier presses, lower to give the display more slack.
#define TOUCH_IDLE_POLL_HZ     20
#define TOUCH_MOVE_EPS         2     // skip a MOVE report if it moved fewer than this many px
// Force a MOVE report this often while pressed, even if the finger hasn't moved.
// Without it, "finger held perfectly still" and "the link just died" look
// identical to the PC — both are silence — and the PC's stuck-button watchdog
// cannot tell them apart. That matters during a hold-drag: pausing while
// dragging a window would otherwise trip the watchdog and drop it. Two packets
// per second during a stationary press is nothing next to the video stream.
#define TOUCH_KEEPALIVE_MS     500
#define TOUCH_RELEASE_SAMPLES  2     // consecutive empty reads before declaring release (debounce)
// Consecutive VALID reads before declaring a press. 1 = fire immediately.
// Only worth raising while PENIRQ is unwired: with no interrupt to corroborate
// it, press detection rests entirely on the XPT2046's pressure reading, and a
// single spurious sample would become a real click on the PC. Setting this to 2
// makes a stray click essentially impossible, at the cost of one extra poll
// period (~50 ms at TOUCH_IDLE_POLL_HZ) before a press registers.
#define TOUCH_PRESS_SAMPLES    1

// ── Calibration ───────────────────────────────────────────────────────────
// Grid resolution: 3 -> 9 targets, 4 -> 16. Simulating a twisted resistive
// panel with realistic tap and ADC noise, mean error across the panel came out
// at 1.93 px for the old 4-corner affine fit, 1.68 px for a 9-point bilinear
// fit, and 1.26 px at 16 points (worst-case 6.05 -> 4.82 -> 4.43 px). Most of
// the residual is human tap scatter rather than model error, which is exactly
// what more points average away. 16 taps is a slower one-time setup for a
// meaningfully steadier result; drop to 3 if you would rather it were quicker.
#define TOUCH_CAL_GRID         4
// Inset of the outermost targets from the panel edge, in pixels. Resistive
// panels get noisy and non-linear right at the very edge, so don't put targets
// in the last few pixels — but don't pull them too far in either, or the fit is
// extrapolating everywhere near the border.
#define TOUCH_CAL_INSET        26
// Raw ADC reads averaged per target (median-filtered, settling reads dropped).
#define TOUCH_CAL_SAMPLES      16
// If the fit's RMS residual is worse than this many pixels, one of the taps was
// probably bad — redo the whole run rather than saving a calibration that will
// annoy you every day. Bounded by TOUCH_CAL_MAX_RETRY so it can't loop forever.
#define TOUCH_CAL_MAX_RMS      6.0f
#define TOUCH_CAL_MAX_RETRY    2
#define TOUCH_BUS_WAIT_MS      25    // max wait for the SPI bus; display always holds priority
#define TOUCH_EVENT_QUEUE_LEN  8     // touchTask -> networkTask handoff depth

// Touch event kinds — must match the PC side's TOUCH_EV_* in captureJpeg.py.
#define TOUCH_EV_DOWN  0
#define TOUCH_EV_MOVE  1
#define TOUCH_EV_UP    2

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

// ── Shared SPI bus arbitration ───────────────────────────────────────────────
// The display and the touch controller sit on the same SPI host, but the
// display's transfer is an ASYNC DMA push: pushPixelsDMA() returns while bytes
// are still going out, and displayTask keeps its startWrite() open across
// several loop iterations until dmaBusy() clears. LovyanGFX's own bus lock does
// not cover that window, and Panel_Device::getTouchRaw() would happily call
// endTransaction() mid-DMA from another task — tearing the frame and desyncing
// displayTask's startWrite/endWrite pairing.
//
// So bus ownership is arbitrated explicitly: displayTask holds this mutex for
// the ENTIRE push (startWrite -> DMA -> waitDMA -> endWrite), and touchTask
// takes it around its read. It's a real mutex, not a binary semaphore, so
// priority inheritance applies: if the display wants the bus while touch holds
// it, touch is boosted and hands it back within one 456 us read. That's the
// hard upper bound on how much touch can ever delay a frame.
extern SemaphoreHandle_t g_spiMutex;

// ── Touch pipeline ───────────────────────────────────────────────────────────
// touchTask produces these; networkTask drains the queue and puts them on the
// wire. Touch never calls sendto() itself — the socket stays owned by the one
// task that already runs on the core the LWIP stack lives on, so there's no
// concurrent-socket question and no second task blocking in the network stack.
struct TouchEvent {
    uint8_t  kind;    // TOUCH_EV_DOWN / _MOVE / _UP
    uint8_t  touchId; // increments on every new press; lets the PC spot a lost DOWN
    uint8_t  seq;     // per-touch sequence; lets the PC drop UDP-reordered stragglers
    uint16_t x, y;    // panel coordinates, 0..SCREEN_W-1 / 0..SCREEN_H-1
};

extern QueueHandle_t     touchQueue;
extern volatile bool     g_touchEnabled;   // set by the PC over the 0xAA 0xCC control channel
extern volatile bool     g_touchCalibrated;

// ── Display double-buffer write index (Core-1 exclusive) ─────────────────────
extern uint8_t writeSet;
