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
 * REBASE NOTE (pure ESP-IDF): this file used to pull in <Arduino.h>,
 * <WiFi.h> and <JPEGDEC.h>. None of those are used by this project
 * anymore — WiFi is now the native esp_wifi/esp_netif stack (see wifi.h),
 * JPEG decode is esp_new_jpeg (see jpeg_decode.h), and millis()/micros()/
 * delay()/constrain() come from port.h instead of the Arduino core.
 */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "port.h"

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"

#include <lwip/sockets.h>
#include <lwip/netdb.h>
#include <fcntl.h>

#define LGFX_USE_V1
#include <LovyanGFX.hpp>

// ═══════════════════════════════════════════════════════════════════════════
//  ZONE 1 — EDIT THIS BEFORE YOU FLASH
//  Nothing works until these match your network and your wiring.
// ═══════════════════════════════════════════════════════════════════════════

// ── WiFi ──────────────────────────────────────────────────────────────────
// Your network's credentials. This is the one thing every single person
// who uses this firmware has to change.
#define WIFI_SSID  "Endmin"
#define WIFI_PASS  "987654321"

// Maximum WiFi transmit power in 0.25 dBm units (80 = 20 dBm requested).
#define WIFI_TX_POWER  80

// Hostname the ESP advertises on the network and uses for OTA discovery.
#define OTA_HOSTNAME  "esp32s3-display"

// Port the HTTP OTA server listens on (see ota.h). Push a new firmware
// with:  curl -X POST --data-binary @firmware.bin http://<device-ip>:3232/update
#define OTA_HTTP_PORT  3232

// ── Display selection ────────────────────────────────────────────────────
// Pick which physical display this firmware drives. Exactly one:
//   DISPLAY_ILI9341_SPI  — 2.4"-2.8" ILI9341 module, SPI, optional XPT2046 touch
//   DISPLAY_TFT24_I80    — 2.4" 240x320 TFT LCD *shield*, 8-bit parallel (i80),
//                          NO touch (touch is force-disabled for this option)
#define DISPLAY_ILI9341_SPI   1
#define DISPLAY_TFT24_I80     2

#define DISPLAY_SELECT        DISPLAY_ILI9341_SPI

#if (DISPLAY_SELECT != DISPLAY_ILI9341_SPI) && (DISPLAY_SELECT != DISPLAY_TFT24_I80)
#error "DISPLAY_SELECT must be DISPLAY_ILI9341_SPI or DISPLAY_TFT24_I80"
#endif

// ── Display wiring — ILI9341 over SPI (DISPLAY_ILI9341_SPI) ────────────────────────────────────
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
#define LCD_WRITE_HZ    80000000    // NOTE: GP-SPI on the S3 can only produce 80 or 40 MHz
                                    // (80M PLL / integer divider). Anything between is
                                    // silently snapped to one of them, so say what you mean.

// LCD_DRIVE_CAP: pad drive strength for SCLK/MOSI, 0 (weakest ~5 mA) .. 3 (~40 mA).
//   2 is the silicon default (no change). Lower values made the picture worse
//   on the reference setup; 3 reduced glitches but made the backlight flicker
//   (supply/ground noise), so it is a trade-off, not a fix.
#define LCD_DRIVE_CAP   2

#define LCD_PIN_SCLK    12
#define LCD_PIN_MOSI    11
#define LCD_PIN_MISO    -1          // the PANEL is write-only and never read.
                                    // The shared BUS does get a MISO line, but
                                    // it belongs to the touch controller — see
                                    // TOUCH_PIN_MISO below and display.h.
#define LCD_PIN_DC      4           // data/command
#define LCD_PIN_CS      10          // chip select
#define LCD_PIN_RST     5           // reset
#define LCD_PIN_BUSY    -1          // -1 = not used by this panel

// ── Display wiring — 2.4" TFT LCD shield over i80 / 8-bit parallel ───────
// Used when DISPLAY_SELECT == DISPLAY_TFT24_I80. These are PLACEHOLDER pins —
// change them to match your real wiring. Any free GPIO works (the S3 LCD_CAM
// peripheral routes through the GPIO matrix). Avoid: GPIO0/3/45/46 (strapping),
// GPIO19/20 (USB), GPIO26-37 (flash/PSRAM on N16R8 boards).
// Shield signal names: LCD_D0..D7, LCD_WR, LCD_RD, LCD_RS (=DC), LCD_CS, LCD_RST.
#define LCD_I80_PIN_D0   4
#define LCD_I80_PIN_D1   5
#define LCD_I80_PIN_D2   6
#define LCD_I80_PIN_D3   7
#define LCD_I80_PIN_D4   15
#define LCD_I80_PIN_D5   16
#define LCD_I80_PIN_D6   17
#define LCD_I80_PIN_D7   18
#define LCD_I80_PIN_WR   8      // write strobe
#define LCD_I80_PIN_RD   12     // read strobe — held high by the driver (panel is write-only)
#define LCD_I80_PIN_RS   9      // register select = data/command (DC)
#define LCD_I80_PIN_CS   10     // chip select
#define LCD_I80_PIN_RST  11     // reset (-1 if tied to the board reset)

// i80 write clock. 20 MHz is a safe start for a shield on jumper wires;
// the ILI9341 write cycle allows up to ~ 20-40 MHz depending on wiring.
#define LCD_I80_WRITE_HZ 20000000

// Pad drive strength for WR + D0..D7, 0 (weakest) .. 3 (strongest). 2 = default.
#define LCD_I80_DRIVE_CAP 2

#if DISPLAY_SELECT == DISPLAY_TFT24_I80
  #define LCD_MODEL_NAME  "ILI9341 i80 320x240"
#else
  #define LCD_MODEL_NAME  "ILI9341  320x240"
#endif

// Panel's native resolution, before rotation. Most 2.4"-2.8" ILI9341
// boards report portrait (240x320) natively and get rotated to landscape
// in the panel (MADCTL) — see LCD_MADCTL below if you need to change
// orientation. The software canvas is always 320x240 landscape.
#define LCD_PANEL_W     240
#define LCD_PANEL_H     320

// ── esp_lcd transfer settings ────────────────────────────────────────────────
// A frame goes out as strips of LCD_STRIP_ROWS rows, queued back-to-back
// (RAMWR then RAMWRC). Each strip is one queued SPI/i80 transaction, so the
// queue depth must be >= ceil(SCREEN_H / LCD_STRIP_ROWS) = 15.
#define LCD_STRIP_ROWS     16
#define LCD_TRANS_QUEUE    16
// Buffers handed to the DMA engine from PSRAM are aligned to this. 64 covers
// both 32 B and 64 B data-cache lines; sizes used here are multiples of it.
#define LCD_FB_ALIGN       64
// ILI9341 MADCTL (0x36) for landscape 320x240: MV (0x20) + BGR (0x08) = 0x28.
// 0xE8 (MX|MY|MV|BGR) is the same picture rotated 180 degrees, so to flip the
// orientation toggle BOTH 0x80 and 0x40. A single one of them mirrors the
// image. If red and blue are swapped toggle 0x08.
// NOTE: changing this changes where a touch lands, so bump TOUCH_CAL_VERSION
// (touch.cpp) to force a recalibration.
#define LCD_MADCTL         0x28
// Give up waiting for a frame's DMA after this long (pathological case only).
#define DISPLAY_DMA_TIMEOUT_MS 1000

// ── Touch wiring — XPT2046 on the SAME SPI bus as the display ────────────
// The touch controller shares SCLK/MOSI with the panel and has its own CS.
// This is deliberate: g_spiMutex serialises access, and a touch read is a
// few short transfers at 1 MHz (well under 1 ms) against the display's 150 KB
// DMA pushes — sharing costs far less than burning a second SPI host.
//
// MISO matters here. The panel is write-only (LCD_PIN_MISO = -1), but the
// XPT2046 has to be *read*, so the shared bus needs a MISO line after all.
// initDisplay() puts it on the host (spi_bus_initialize); the panel never uses it.
#define TOUCH_PIN_CS    9
#define TOUCH_PIN_IRQ   8           // PENIRQ — low while the panel is pressed

// ── PENIRQ: wired or not? ────────────────────────────────────────────────
// 0 = PENIRQ is NOT usable (not connected / shorted). The driver is told
//     pin_int = -1 and touchTask polls instead.
// 1 = PENIRQ is properly wired. The driver gates on it and touchTask sleeps
//     on the interrupt, which costs literally nothing while untouched.
//
// This MUST be 0 while the line is unwired. It is not merely an
// optimisation flag: xptReadRaw() (touch.cpp) returns "not touched" whenever
// the PENIRQ pin reads high, so a configured-but-floating PENIRQ would make
// every single read report "not touched" — touch would appear completely
// dead, not just less efficient. Flipping this to 1 is the ONLY change needed once it's fixed.
#define TOUCH_USE_IRQ   1
#define TOUCH_PIN_MISO  13          // shared-bus MISO (display doesn't use it)
#define TOUCH_SPI_HZ    1000000     // XPT2046 max is ~2 MHz; 1 MHz is the safe default
// XPT2046 pressure gate: z = Z1 + 4095 - Z2. Readings below this count as "not
// touched". Raise it if you get ghost touches, lower it if light taps are missed.
#define TOUCH_PRESSURE_MIN 200

// BOOT button (GPIO0 on every ESP32-S3 devkit). Double-press it before the
// stream starts to force a touch recalibration — see touch.cpp.
#define BOOT_PIN        0

// Set to 0 to disable touch setup, calibration, sampling, and network events.
// This is only your *request*: the effective TOUCH_FEATURE_ENABLED below is
// forced to 0 whenever the selected display has no touch panel (i80 shield).
#define TOUCH_FEATURE_REQUESTED 1

#if DISPLAY_SELECT == DISPLAY_TFT24_I80
  #define TOUCH_FEATURE_ENABLED 0   // i80 shield has no touch — always off
#else
  #define TOUCH_FEATURE_ENABLED TOUCH_FEATURE_REQUESTED
#endif

// ═══════════════════════════════════════════════════════════════════════════
//  ZONE 2 — CHANGE THESE IF YOUR HARDWARE SETUP DIFFERS
//  Different panel resolution, different memory budget, etc. The defaults
//  here match a 320x240 panel on an N16R8 (16 MB flash / 8 MB PSRAM) board.
// ═══════════════════════════════════════════════════════════════════════════

// ── Display geometry (logical, post-rotation) ────────────────────────────
#define SCREEN_W         320
#define SCREEN_H         240

// ── Network chunking ──────────────────────────────────────────────────────
#define DEBUGINTERVAL 300                   // ms between debug stats packets
#define CHUNK_DATA_SIZE  1400              // bytes of JPEG payload per UDP packet
#define MAX_JPEG_FRAME   (35 * 1024)       // per-frame JPEG allocation/transport cap: 35 KiB
#define MAX_FRAME_CHUNKS ((MAX_JPEG_FRAME + CHUNK_DATA_SIZE - 1) / CHUNK_DATA_SIZE)

// ── Pipeline depth (memory budget) ────────────────────────────────────────
// Pipeline depth — deliberately minimal (low latency, small RAM footprint):
//   * 2 JPEG slots: slot 0 in internal SRAM, slot 1 in PSRAM (see app_main).
//   * 2 display framebuffers (PSRAM): exactly one ping-pong pair — one is being
//     sent to the panel while the decoder fills the other.
// Fewer slots = less buffered (stale) video and less RAM, but also less slack
// to absorb network/decode jitter: any hiccup now shows up as a dropped frame.
#define CFG_NUM_JPEG_SLOTS    2   // desired full-frame JPEG buffers (1–6)
#define CFG_NUM_DISPLAY_BUFS  2   // desired display bufs (2–6)

// Hard array caps — do NOT exceed. These size the actual arrays; the
// CFG_ values above are the *desired* runtime count within that cap.
#define MM_MAX_JPEG_SLOTS     6
#define MM_MAX_DISPLAY_BUFS   6

const int UDP_PORT = 12345;

// ═══════════════════════════════════════════════════════════════════════════
//  ZONE 3 — TIMING / BEHAVIOR TUNABLES
//  Safe defaults already set. Change these only if you know why.
// ═══════════════════════════════════════════════════════════════════════════
#define PKT_TIMEOUT_MS           3000
#define OVERLAY_FLASH_MS         1000
#define FRAME_ASSEMBLY_TIMEOUT_MS 80
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

// ── Runtime actual counts (set by app_main() after fallback allocation) ──────
extern uint8_t g_numJpegSlots;    // actual slots allocated  (1–MM_MAX_JPEG_SLOTS)
extern uint8_t g_numDisplayBufs;  // actual display bufs allocated (2–MM_MAX_DISPLAY_BUFS)

// ── Pipeline structs ──────────────────────────────────────────────────────────
struct JpegFrameSlot {
    uint8_t* data = nullptr;
    uint8_t frameId = 0xFF;
    uint8_t totalChunks = 0;
    uint8_t chunksReceived = 0;
    uint16_t frameSize = 0;
    uint32_t firstChunkMs = 0;
    bool chunkReceived[MAX_FRAME_CHUNKS] = {};
    bool receiving = false;
    bool queued = false;
    bool inUse = false;
};

struct DecodeMsg {
    uint8_t  frameId;
    uint8_t  jpegSlot;
    uint16_t len;
};

struct DisplayMsg {
    uint8_t frameId;
    uint8_t bufSet;
};

// ── FreeRTOS handles ──────────────────────────────────────────────────────────
extern QueueHandle_t     jpegFreeQueue;
extern QueueHandle_t     decodeQueue;
extern QueueHandle_t     displayQueue;
extern QueueHandle_t     framebufferFreeQueue;

// ── Shared buffers ────────────────────────────────────────────────────────────
extern uint16_t* frameFb[MM_MAX_DISPLAY_BUFS];   // only [0..g_numDisplayBufs-1] allocated
extern JpegFrameSlot jpegFrames[MM_MAX_JPEG_SLOTS]; // only [0..g_numJpegSlots-1] allocated

// ── Cross-core stats ──────────────────────────────────────────────────────────
extern volatile uint32_t g_avgDecodeUs;
extern volatile uint32_t g_presentedFrames;
extern volatile uint32_t g_abortedFrames;
extern volatile uint32_t g_networkDrops;

// ── Streaming / WiFi state ────────────────────────────────────────────────────
extern volatile bool     g_streaming;        // true once first full frame decoded
extern volatile bool     g_wifiOk;           // mirrors the IP_EVENT_STA_GOT_IP / WIFI_EVENT_STA_DISCONNECTED state
extern volatile uint32_t g_lastPktMs;        // updated by networkTask on each valid UDP packet
extern volatile uint32_t g_wifiConnectedMs;  // millis() when WiFi first associated

// ── Per-core CPU utilisation (updated by FreeRTOS idle hooks in app_main.cpp)
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
// The display (esp_lcd panel IO) and the XPT2046 touch controller are two
// devices on the same SPI host. A frame is sent as a chain of queued DMA strips
// that completes asynchronously (see display.cpp), so displayTask holds this
// mutex from the first strip being queued until the last-strip completion
// callback has fired. touchTask takes it around each read; so does every UI
// push. It is a real mutex (priority inheritance): if the display wants the
// bus while touch holds it, touch is boosted and hands it back within one
// bounded read. FreeRTOS requires the SAME task to take and give it.
extern SemaphoreHandle_t g_spiMutex;

// ── Touch pipeline ───────────────────────────────────────────────────────────
// touchTask produces these; networkTask drains the queue and puts them on the
// wire. Touch never calls sendto() itself — the socket stays owned by the one
// task that already runs on the core the LWIP stack lives on, so there's no
// concurrent-socket question and no second task blocking in the network stack.
#if TOUCH_FEATURE_ENABLED
struct TouchEvent {
    uint8_t  kind;    // TOUCH_EV_DOWN / _MOVE / _UP
    uint8_t  touchId; // increments on every new press; lets the PC spot a lost DOWN
    uint8_t  seq;     // per-touch sequence; lets the PC drop UDP-reordered stragglers
    uint16_t x, y;    // panel coordinates, 0..SCREEN_W-1 / 0..SCREEN_H-1
};

extern QueueHandle_t     touchQueue;
extern volatile bool     g_touchEnabled;   // set by the PC over the 0xAA 0xCC control channel
extern volatile bool     g_touchCalibrated;
#endif