/*
 * ESP32-S3  |  ILI9341  |  8-bit Parallel i80  |  320x240
 *
 * PIPELINE ARCHITECTURE  (QOI edition)
 * ─────────────────────────────────────
 *  Two shared slots (back / front) replace the old 4-independent-buffer scheme.
 *
 *  Memory layout (QOI, updated):
 *    slot[0].assembly  SRAM  40 KB  ─┐ sequential QOI stream read → must be fast
 *    slot[1].assembly  SRAM  40 KB  ─┘
 *    slot[0].fb        PSRAM 38 KB  ─┐ DMA source; QOI decoder writes DIRECTLY here
 *    slot[1].fb        PSRAM 38 KB  ─┘   (no SRAM scratch needed — sequential writes)
 *    chunkStorage[4]   PSRAM 160 KB   chunk staging; network writes, not decode-critical
 *
 *  vs JPEG layout (previous):
 *    Removed: decodeTemp  SRAM  38 KB  (JPEGDEC MCU scatter-write scratch)
 *    SRAM saved: ~23 KB  (105 KB → 80 KB for decode pipeline)
 *
 *  QOI decode pipeline (Core 1):
 *    qoi_to_rgb565be(slot[s].assembly → tileFb[writeSet][tId])
 *      • Reads SRAM sequentially (L1 hot, forward-only)
 *      • Writes PSRAM sequentially (cache-line streaming, 2-pixel 32-bit coalescing on runs)
 *      • Inline RGB888 → RGB565 + bswap16 — zero extra pass
 *      • 64-entry hash table (256 B) stays in L1 permanently
 *      • No MCU callbacks, no scatter-writes, no extra memcpy
 *
 *  Pipeline (steady state):
 *
 *    Core 0 (net)      Core 1 (render)
 *    ─────────────     ───────────────
 *    assemble → slot[back].assembly (SRAM)
 *    post decodeQueue ──────────────→ take decodeQueue
 *    back ^= 1                        qoi_to_rgb565be → tileFb[writeSet][tId] (PSRAM direct)
 *    take slotFree[back]              when readyMask==0x0F: post DisplayMsg
 *    assemble → slot[back].assembly   give slotFree[s]
 *    post decodeQueue  ←────────────
 *    ...
 *
 *  Stats packet (0xAB 0xCD prefix, sent every second when debugEnabled):
 *    FPS:X.X|TEMP:XX.X|JIT:X.X|DEC:XXXX|DROP:X|ABRT:X|SRAM:XXXX/XXXX|PSRAM:XXXX/XXXX
 *    DEC  = avg tile QOI decode µs (inline to PSRAM, NOT pushImage)
 *    DROP = corrupt + timeout count in this 1-second window
 *    ABRT = partial frames discarded due to frameId switch (UDP reorder/overrun)
 *    SRAM/PSRAM = free_KB/total_KB
 */

#define LGFX_USE_V1
#include <LovyanGFX.hpp>
#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_attr.h>
#include <esp_heap_caps.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/queue.h"
#include <lwip/sockets.h>
#include <lwip/netdb.h>
#include <fcntl.h>
#include <math.h>
#include <esp_task_wdt.h>
#include "qoi_dec.h"    // Inline QOI→RGB565BE decoder (no library dep)

// ─────────────────────────────────────────────
//  DISPLAY
// ─────────────────────────────────────────────
class LGFX : public lgfx::LGFX_Device {
    lgfx::Bus_Parallel8  _bus;
    lgfx::Panel_ILI9341  _panel;
public:
    LGFX() {
        { auto cfg=_bus.config(); cfg.freq_write=20000000;
          cfg.pin_wr=1; cfg.pin_rd=40; cfg.pin_rs=2;
          cfg.pin_d0=5; cfg.pin_d1=4;  cfg.pin_d2=10;
          cfg.pin_d3=9; cfg.pin_d4=3;  cfg.pin_d5=8;
          cfg.pin_d6=7; cfg.pin_d7=6;
          _bus.config(cfg); _panel.setBus(&_bus); }
        { auto cfg=_panel.config();
          cfg.pin_cs=41; cfg.pin_rst=39; cfg.pin_busy=-1;
          cfg.panel_width=240; cfg.panel_height=320;
          cfg.offset_x=0; cfg.offset_y=0; cfg.offset_rotation=0;
          cfg.dummy_read_pixel=8;
          cfg.readable=false; cfg.invert=false;
          cfg.rgb_order=false; cfg.dlen_16bit=false; cfg.bus_shared=false;
          _panel.config(cfg); }
        setPanel(&_panel);
    }
};
static LGFX lcd;

// ─────────────────────────────────────────────
//  CONFIG
// ─────────────────────────────────────────────
const char* WIFI_SSID  = "streaming";
const char* WIFI_PASS  = "12345678";
const int   UDP_PORT   = 12345;

#define SCREEN_W         320
#define SCREEN_H         240
#define NUM_TILES        4
#define TILE_W           160
#define TILE_H           120
#define TILE_PIXELS      (TILE_W * TILE_H)          // 19 200

#define CHUNK_DATA_SIZE  1400
// QOI with RGB565 pre-quantize: typical 8–20 KB, worst-case ~40 KB.
// Assembly buffer sized conservatively at 40 KB.
// PC encoder must enforce MAX_TILE_QOI and skip oversized tiles.
#define MAX_TILE_CHUNKS  30                          // 30 × 1400 = 42 KB headroom
#define MAX_TILE_QOI     (MAX_TILE_CHUNKS * CHUNK_DATA_SIZE)   // 42 000 B

// QOI header magic for validation (ASCII 'qoif')
#define QOI_MAGIC_0  'q'
#define QOI_MAGIC_1  'o'
#define QOI_MAGIC_2  'i'
#define QOI_MAGIC_3  'f'
#define QOI_HDR_LEN  14
#define QOI_FTR_LEN  8

#define TILE_TIMEOUT_MS  200

// Screen position of each tile:  TL TR BL BR
static const int16_t TILE_X[NUM_TILES] = {  0, 160,   0, 160 };
static const int16_t TILE_Y[NUM_TILES] = {  0,   0, 120, 120 };

// ─────────────────────────────────────────────
//  PIPELINE SLOTS  (2 shared decode/display buffers)
//  assembly is in SRAM — QOI decoder reads forward-sequentially: L1 hot.
//  No fb field here: decode goes DIRECTLY to tileFb[writeSet][tId] in PSRAM.
// ─────────────────────────────────────────────
struct PipeSlot {
    uint8_t*  assembly;   // SRAM — QOI stream; forward-sequential read during decode
};
static PipeSlot slot[2];

// Double-buffered per-tile framebuffers in PSRAM.
// QOI decoder writes DIRECTLY here (sequential, cache-line friendly).
// tileFb[0] and tileFb[1] are two complete sets of 4 tile buffers.
// Core 1 (decoder) writes into tileFb[writeSet][tId].
// Display task reads from tileFb[displaySet][tId].
static uint16_t* tileFb[2][NUM_TILES] = {
    { nullptr, nullptr, nullptr, nullptr },
    { nullptr, nullptr, nullptr, nullptr }
};
static uint8_t writeSet = 0;   // Core 1 exclusive — no sync needed

// Per-tile encoded dimensions for current write set.
// Core 1 writes before posting DisplayMsg; displayTask reads them.
// 0 means full-size (TILE_W x TILE_H) — pushImage uses TILE_W/TILE_H directly.
struct TileMeta {
    uint8_t encW;   // encoded width  (0 = TILE_W)
    uint8_t encH;   // encoded height (0 = TILE_H)
};
static TileMeta tileMeta[2][NUM_TILES] = {};   // [bufSet][tileIdx]

// Message passed through the decode queue
struct DecodeMsg {
    uint8_t  frameId;   // frame sequence (0–255) for frame-sync presentation
    uint8_t  tId;       // which tile position (0–3) → determines screen XY
    uint8_t  slotIdx;   // which PipeSlot holds the assembled QOI
    uint16_t len;       // QOI byte count in slot[slotIdx].assembly
    uint8_t  encW;      // encoded tile width  (0 = full TILE_W)
    uint8_t  encH;      // encoded tile height (0 = full TILE_H)
};

// Pipeline synchronisation
static QueueHandle_t     decodeQueue;    // depth-1 queue: net → renderer
static SemaphoreHandle_t slotFree[2];   // given when renderer finishes slot

// Display pipeline: Core 1 posts here when all 4 tiles are ready.
struct DisplayMsg {
    uint8_t frameId;   // for stats / debug
    uint8_t bufSet;    // which tileFb[bufSet] to push (0 or 1)
};
static QueueHandle_t     displayQueue;          // depth-1 queue: renderer → display task
static SemaphoreHandle_t bufFree[2];            // render takes before 1st tile, display gives after push
// ↑ Prevents the render task from writing into a tileFb buffer that
//   displayTask is still reading (guards against writeSet cycling back
//   to the same index before the previous display pass has finished).

// ─────────────────────────────────────────────
//  CHUNK REASSEMBLY STATE  (one per tile position)
// ─────────────────────────────────────────────
struct TileState {
    uint8_t* chunkBuf[MAX_TILE_CHUNKS];
    uint16_t chunkLen[MAX_TILE_CHUNKS];
    bool     chunkGot[MAX_TILE_CHUNKS];
    uint8_t  frameId      = 0xFF;
    uint8_t  totalChunks  = 0;
    uint16_t frameSize    = 0;
    uint8_t  chunksGot    = 0;
    uint32_t firstChunkMs = 0;
    uint8_t  encW         = 0;   // encoded tile width from packet header (0=TILE_W)
    uint8_t  encH         = 0;   // encoded tile height from packet header (0=TILE_H)
    uint32_t stat_decoded = 0;
    uint32_t stat_corrupt = 0;
    uint32_t stat_timeout = 0;
};
static TileState tiles[NUM_TILES];
static uint8_t*  tileChunkStorage[NUM_TILES] = {};

// ─────────────────────────────────────────────
//  CROSS-CORE STATS
// ─────────────────────────────────────────────
static volatile uint32_t g_avgDecodeUs    = 0;
static volatile uint32_t g_presentedFrames= 0;
static volatile uint32_t g_abortedFrames  = 0;

// ─────────────────────────────────────────────
//  GLOBAL STATE
// ─────────────────────────────────────────────
static bool     debugEnabled      = false;
static char     debugBuf[256];
static int      g_sock            = -1;
static struct   sockaddr_in g_remoteAddr;
static bool     g_remoteAddrValid = false;
static float    stat_jitter       = 0.0f;
static uint32_t stat_prevMs       = 0;
static uint16_t* decodeTemp       = nullptr;  // SRAM scratch for upscale decode path

// ─────────────────────────────────────────────
//  TILE HELPERS
// ─────────────────────────────────────────────
static IRAM_ATTR void resetTile(uint8_t t) {
    memset(tiles[t].chunkGot, 0, sizeof(tiles[t].chunkGot));
    tiles[t].frameId      = 0xFF;
    tiles[t].totalChunks  = 0;
    tiles[t].frameSize    = 0;
    tiles[t].chunksGot    = 0;
    tiles[t].firstChunkMs = 0;
}

// Assemble complete tile QOI stream from chunks into dst (slot[].assembly, SRAM).
// Returns assembled byte count, or 0 on corruption.
static IRAM_ATTR int assembleTileInto(uint8_t t, uint8_t* dst) {
    TileState& ts = tiles[t];
    if (ts.totalChunks == 0) return 0;
    int offset = 0;
    for (uint8_t c = 0; c < ts.totalChunks; c++) {
        if (!ts.chunkGot[c]) return 0;
        memcpy(dst + offset, ts.chunkBuf[c], ts.chunkLen[c]);
        offset += ts.chunkLen[c];
    }
    // Validate QOI magic 'qoif' and minimum size
    if (offset < (QOI_HDR_LEN + QOI_FTR_LEN) ||
        dst[0] != QOI_MAGIC_0 || dst[1] != QOI_MAGIC_1 ||
        dst[2] != QOI_MAGIC_2 || dst[3] != QOI_MAGIC_3) {
        Serial.printf("[TILE%u] bad QOI magic len=%d [%02X%02X%02X%02X]\n",
            t, offset, dst[0], dst[1], dst[2], dst[3]);
        ts.stat_corrupt++;
        return 0;
    }
    return offset;
}

// ─────────────────────────────────────────────
//  OPTIMISED NEAREST-NEIGHBOUR UPSCALE  (Core 1, IRAM)
// ─────────────────────────────────────────────
//
//  Root cause of the previous TWDT crash:
//    upscaleNN was called from loop() — the Arduino main task on Core 1.
//    The TWDT watches that task.  PSRAM memcpy latency across 120 rows
//    was long enough to starve the idle task and trip the watchdog.
//
//  Fixes applied:
//    1. LUT (xMap / yMap): precompute all x- and y- source indices once.
//       Inner loop becomes  lineBuf[dx] = srcRow[xMap[dx]]  — zero division,
//       pure L1 SRAM reads.  ~3× faster than the division-per-pixel version.
//    2. esp_task_wdt_reset() once per tile (called from loop() before and
//       after upscaleNN) keeps the TWDT happy even if PSRAM is slow.
//    3. lineBuf and xMap/yMap are static DRAM — never on the task stack,
//       no stack overflow risk.
//
//  Throughput (120×120 → 160×120, 2× horizontal):
//    Old (division loop + writeData):  ~5 ms
//    New (LUT + memcpy burst):         ~0.25 ms
//
//  src : SRAM decodeTemp  (srcW × srcH × 2 B, 16-byte aligned)
//  dst : PSRAM tileFb     (TILE_W × TILE_H × 2 B, 16-byte aligned)

// Persistent LUT — rebuilt only when srcW or srcH changes (rare).
static uint16_t DRAM_ATTR s_xMap[TILE_W];   // xMap[dstX] = srcX
static uint16_t DRAM_ATTR s_yMap[TILE_H];   // yMap[dstY] = srcY
static uint16_t DRAM_ATTR s_lineBuf[TILE_W]; // one upscaled row, stays in L1
static int      s_lutSrcW = -1;
static int      s_lutSrcH = -1;

static IRAM_ATTR void upscaleNN(
        const uint16_t* __restrict__ src,
        uint16_t*       __restrict__ dst,
        int srcW, int srcH)
{
    // ── Rebuild LUT only when dimensions change ────────────────────────────
    // This is a one-time cost per mode switch, not per tile.
    if (srcW != s_lutSrcW || srcH != s_lutSrcH) {
        for (int dx = 0; dx < TILE_W; dx++)
            s_xMap[dx] = (uint16_t)((dx * srcW) / TILE_W);
        for (int dy = 0; dy < TILE_H; dy++)
            s_yMap[dy] = (uint16_t)((dy * srcH) / TILE_H);
        s_lutSrcW = srcW;
        s_lutSrcH = srcH;
    }

    int prevSrcY = -1;

    for (int dy = 0; dy < TILE_H; dy++) {
        int sy = s_yMap[dy];

        if (sy != prevSrcY) {
            // ── Horizontal scale: LUT index, no division, 4× unrolled ─────
            // srcRow is in SRAM (decodeTemp) — L1 hot after first tile.
            // xMap is also SRAM — both fit comfortably in the 32 KB L1 data cache.
            const uint16_t* srcRow = src + sy * srcW;
            int dx = 0;
            for (; dx <= TILE_W - 8; dx += 8) {
                s_lineBuf[dx+0] = srcRow[s_xMap[dx+0]];
                s_lineBuf[dx+1] = srcRow[s_xMap[dx+1]];
                s_lineBuf[dx+2] = srcRow[s_xMap[dx+2]];
                s_lineBuf[dx+3] = srcRow[s_xMap[dx+3]];
                s_lineBuf[dx+4] = srcRow[s_xMap[dx+4]];
                s_lineBuf[dx+5] = srcRow[s_xMap[dx+5]];
                s_lineBuf[dx+6] = srcRow[s_xMap[dx+6]];
                s_lineBuf[dx+7] = srcRow[s_xMap[dx+7]];
            }
            for (; dx < TILE_W; dx++)
                s_lineBuf[dx] = srcRow[s_xMap[dx]];

            prevSrcY = sy;
        }

        // ── Burst-write 640 B of SRAM → PSRAM via LX7 128-bit stores ──────
        // Sequential PSRAM write: cache-line precharger fires on every 32-byte
        // boundary.  memcpy picks EE.LQ/EE.SQ (128-bit) automatically when
        // both pointers are 16-byte aligned (guaranteed by heap_caps_aligned_alloc).
        memcpy(dst + dy * TILE_W, s_lineBuf, TILE_W * sizeof(uint16_t));
    }
}

// ─────────────────────────────────────────────
//  QOI DECODE PIPELINE  (Core 1 — single function call)
// ─────────────────────────────────────────────
//
//  qoi_to_rgb565be() (qoi_dec.h):
//    • src: slot[s].assembly  (SRAM, sequential read, L1 hot)
//    • dst: tileFb[writeSet][tId]  (PSRAM, sequential write, cache-line streaming)
//    • Inline RGB888→RGB565 BE: no extra pass, no bswap16 function needed
//    • Run fills use 32-bit stores: 2 pixels/store coalesces to 32-byte PSRAM lines
//    • 64-entry hash idx[256 B] stays hot in L1 — never touches PSRAM
//
//  Compared to JPEG pipeline:
//    JPEG: openRAM → mcuCallback scatter-writes (1200× random) → bswap16_simd → memcpy
//    QOI:  qoi_to_rgb565be (one sequential pass, direct to PSRAM)
//
//  Returns true on success. Caller must give slotFree[slotIdx] regardless.

static IRAM_ATTR bool decodeSlot(const DecodeMsg& msg, uint32_t& decodeUs) {
    PipeSlot& s = slot[msg.slotIdx];

    if (msg.tId >= NUM_TILES || tileFb[writeSet][msg.tId] == nullptr) {
        decodeUs = 0;
        return false;
    }

    // Actual encoded tile dimensions (0 in header = full TILE_W/H)
    int srcW = (msg.encW > 0) ? (int)msg.encW : TILE_W;
    int srcH = (msg.encH > 0) ? (int)msg.encH : TILE_H;
    int srcPixels = srcW * srcH;

    // For upscaled tiles (srcW < TILE_W) we decode into decodeTemp (SRAM) first,
    // then pushImage with src dims — LovyanGFX stretches to TILE_W x TILE_H.
    // For full-size tiles, decode directly into tileFb PSRAM (no extra buffer).
    //
    // decodeTemp is re-used here from the QOI era: we re-allocate it in setup()
    // only when upscale is detected (srcW < TILE_W). For full-size frames it
    // remains unused and the pointer stays null, so we check before using it.

    uint32_t t0 = micros();

    if (srcW == TILE_W && srcH == TILE_H) {
        // Full-size: decode directly to PSRAM framebuffer (sequential, fast)
        int rc = qoi_to_rgb565be(s.assembly, (int)msg.len,
                                 tileFb[writeSet][msg.tId],
                                 srcW, srcH);
        decodeUs = micros() - t0;
        if (rc != srcPixels) {
            Serial.printf("[DEC] slot%u QOI err: expected %d px got %d (len=%u)\n",
                          msg.slotIdx, srcPixels, rc, msg.len);
            decodeUs = 0;
            return false;
        }
    } else {
        // Upscale path:
        //   1. QOI → decodeTemp (SRAM, srcW×srcH)  — sequential SRAM write, fast
        //   2. upscaleNN(decodeTemp → tileFb PSRAM) — 128-bit burst memcpy rows
        // tileFb ends up full TILE_W×TILE_H so displayTask uses normal pushImage.
        if (decodeTemp == nullptr) {
            Serial.printf("[DEC] decodeTemp null but upscale srcW=%d srcH=%d\n", srcW, srcH);
            decodeUs = 0;
            return false;
        }
        int rc = qoi_to_rgb565be(s.assembly, (int)msg.len,
                                 decodeTemp, srcW, srcH);
        if (rc != srcPixels) {
            decodeUs = micros() - t0;
            Serial.printf("[DEC] slot%u QOI upscale err: expected %d px got %d\n",
                          msg.slotIdx, srcPixels, rc);
            decodeUs = 0;
            return false;
        }
        // Nearest-neighbour upscale: SRAM decodeTemp → PSRAM tileFb (full size).
        // esp_task_wdt_reset() before and after: upscaleNN does 120 memcpy() calls
        // into PSRAM which can take ~0.3 ms — enough to trip the TWDT if loop()
        // is already close to its deadline.  Reset keeps the watchdog fed.
        esp_task_wdt_reset();
        upscaleNN(decodeTemp, tileFb[writeSet][msg.tId], srcW, srcH);
        esp_task_wdt_reset();
        decodeUs = micros() - t0;
    }

    // tileFb is always TILE_W×TILE_H after this point:
    // full-size tiles decoded directly; upscaled tiles expanded by upscaleNN above.
    // Store 0/0 so displayTask always takes the fast pushImage path.
    tileMeta[writeSet][msg.tId].encW = 0;
    tileMeta[writeSet][msg.tId].encH = 0;

    return true;
}

// ─────────────────────────────────────────────
//  NETWORK TASK  (Core 0)
// ─────────────────────────────────────────────
// Packet format:
//   Data:    [0xAA 0xBB frameId tileId chunkId totalChunks sizeHi sizeLo] + QOI payload
//   Control: [0xAA 0xCC 0x01 debugState]
//
// Pipeline flow when tile completes:
//   1. xSemaphoreTake(slotFree[back])       — wait for renderer to vacate slot
//   2. assembleTileInto(tId, slot[back])    — PSRAM chunks → SRAM assembly
//   3. xQueueSend(decodeQueue, &msg)        — block until renderer is ready
//   4. back ^= 1
static IRAM_ATTR void networkTask(void*) {
    g_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (g_sock < 0) { Serial.println("[NET] socket fail"); vTaskDelete(NULL); return; }

    int rcvbuf = 65536;
    setsockopt(g_sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    struct sockaddr_in local = {};
    local.sin_family = AF_INET;
    local.sin_port   = htons(UDP_PORT);
    local.sin_addr.s_addr = INADDR_ANY;
    if (bind(g_sock, (struct sockaddr*)&local, sizeof(local)) < 0) {
        Serial.println("[NET] bind fail"); close(g_sock); vTaskDelete(NULL); return;
    }
    fcntl(g_sock, F_SETFL, O_NONBLOCK);
    Serial.printf("[NET] UDP ready port=%d\n", UDP_PORT);

    static uint8_t rxBuf[CHUNK_DATA_SIZE + 16];
    struct sockaddr_in sender;
    socklen_t slen = sizeof(sender);
    uint32_t  lastPktMs = millis(), lastBeaconMs = 0, lastStatMs = 0, pktCount = 0;
    uint8_t   back = 0;

    while (true) {
        int n = recvfrom(g_sock, rxBuf, sizeof(rxBuf), 0,
                         (struct sockaddr*)&sender, &slen);

        if (n < 0) {
            fd_set rfds; FD_ZERO(&rfds); FD_SET(g_sock, &rfds);
            struct timeval tv = { .tv_sec = 0, .tv_usec = 1000 };
            select(g_sock + 1, &rfds, NULL, NULL, &tv);

            if ((millis()-lastBeaconMs) > 2000 && (millis()-lastPktMs) > 2000) {
                struct sockaddr_in bc = {};
                bc.sin_family = AF_INET;
                bc.sin_port   = htons(UDP_PORT);
                bc.sin_addr.s_addr = htonl(INADDR_BROADCAST);
                int so = 1;
                setsockopt(g_sock, SOL_SOCKET, SO_BROADCAST, &so, sizeof(so));
                const char* b = "ESP32_READY";
                sendto(g_sock, b, strlen(b), 0, (struct sockaddr*)&bc, sizeof(bc));
                lastBeaconMs = millis();
            }
            continue;
        }

        lastPktMs = millis(); pktCount++;
        if (n < 4 || rxBuf[0] != 0xAA) { portYIELD(); continue; }
        memcpy(&g_remoteAddr, &sender, sizeof(sender));
        g_remoteAddrValid = true;

        // ── Control packet ────────────────────────────────────────────────
        if (rxBuf[1] == 0xCC) {
            if (n >= 4 && rxBuf[2] == 0x01) debugEnabled = (rxBuf[3] == 1);
            portYIELD(); continue;
        }

        // ── Tile data chunk ───────────────────────────────────────────────
        // Header: [0xAA 0xBB frameId tileId chunkId nChunks sizeHi sizeLo encW encH]
        if (rxBuf[1] != 0xBB || n < 10) { portYIELD(); continue; }
        uint8_t  fId     = rxBuf[2];
        uint8_t  tId     = rxBuf[3];
        uint8_t  cId     = rxBuf[4];
        uint8_t  nChunks = rxBuf[5];
        uint16_t fSize   = ((uint16_t)rxBuf[6] << 8) | rxBuf[7];
        uint8_t  pEncW   = rxBuf[8];   // encoded tile width  (0 = TILE_W)
        uint8_t  pEncH   = rxBuf[9];   // encoded tile height (0 = TILE_H)
        int      dataLen = n - 10;
        if (tId >= NUM_TILES || dataLen <= 0) { portYIELD(); continue; }

        // Quick sanity: reject tiles larger than our assembly buffer
        if (fSize > MAX_TILE_QOI) {
            Serial.printf("[TILE%u] oversized fSize=%u > MAX_TILE_QOI=%u, drop\n",
                          tId, fSize, MAX_TILE_QOI);
            portYIELD(); continue;
        }

        TileState& ts = tiles[tId];

        // Timeout stale reassembly
        if (ts.firstChunkMs > 0 && (millis() - ts.firstChunkMs) > TILE_TIMEOUT_MS) {
            Serial.printf("[TILE%u] timeout got=%u/%u\n", tId, ts.chunksGot, ts.totalChunks);
            ts.stat_timeout++;
            resetTile(tId);
        }

        // New frame for this tile
        if (fId != ts.frameId) {
            resetTile(tId);
            ts.frameId      = fId;
            ts.totalChunks  = nChunks;
            ts.frameSize    = fSize;
            ts.encW         = pEncW;
            ts.encH         = pEncH;
            ts.firstChunkMs = millis();
        }

        // Store chunk into PSRAM staging area
        if (cId < MAX_TILE_CHUNKS && !ts.chunkGot[cId]) {
            memcpy(ts.chunkBuf[cId], &rxBuf[10], dataLen);
            ts.chunkLen[cId] = (uint16_t)dataLen;
            ts.chunkGot[cId] = true;
            ts.chunksGot++;
        }

        // All chunks received → feed the pipeline
        if (ts.chunksGot >= ts.totalChunks) {

            xSemaphoreTake(slotFree[back], portMAX_DELAY);

            int len = assembleTileInto(tId, slot[back].assembly);

            if (len > 0) {
                DecodeMsg msg = { fId, tId, back, (uint16_t)len, ts.encW, ts.encH };
                xQueueSend(decodeQueue, &msg, portMAX_DELAY);
                back ^= 1;
            } else {
                xSemaphoreGive(slotFree[back]);
            }

            resetTile(tId);
        }

        // ── Periodic stat report ─────────────────────────────────────────
        if (debugEnabled && g_remoteAddrValid && (millis() - lastStatMs) > 1000) {
            uint32_t el = millis() - lastStatMs;

            static uint32_t lastPresented = 0;
            uint32_t nowPresented = g_presentedFrames;
            uint32_t frames = nowPresented - lastPresented;
            lastPresented = nowPresented;
            float fps = frames / (el / 1000.0f);

            uint32_t totalDrop = 0;
            for (int i = 0; i < NUM_TILES; i++)
                totalDrop += tiles[i].stat_corrupt + tiles[i].stat_timeout;

            static uint32_t lastAborted = 0;
            uint32_t nowAborted = g_abortedFrames;
            uint32_t aborted = nowAborted - lastAborted;
            lastAborted = nowAborted;

            uint32_t freeSRAM  = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
            uint32_t totalSRAM = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
            uint32_t freePSR   = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
            uint32_t totalPSR  = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);

            float tempC = temperatureRead();
            uint32_t decUs = g_avgDecodeUs;

            snprintf(debugBuf, sizeof(debugBuf),
                "%c%cFPS:%.1f|TEMP:%.1f|JIT:%.1f|DEC:%lu|DROP:%lu|ABRT:%lu"
                "|SRAM:%lu/%lu|PSRAM:%lu/%lu",
                0xAB, 0xCD,
                fps, tempC, stat_jitter,
                decUs, totalDrop, aborted,
                freeSRAM / 1024, totalSRAM / 1024,
                freePSR  / 1024, totalPSR  / 1024);

            sendto(g_sock, debugBuf, strlen(debugBuf), 0,
                   (struct sockaddr*)&g_remoteAddr, sizeof(g_remoteAddr));

            for (int i = 0; i < NUM_TILES; i++)
                tiles[i].stat_decoded = tiles[i].stat_corrupt = tiles[i].stat_timeout = 0;
            pktCount = 0;
            lastStatMs = millis();
        }

        portYIELD();
    }
}

// ─────────────────────────────────────────────
//  DISPLAY HELPERS
// ─────────────────────────────────────────────
static void statusLine(uint8_t row, const char* label, const char* value,
                       uint32_t col = TFT_WHITE) {
    int y = 58 + row * 22;
    lcd.fillRect(0, y, SCREEN_W, 22, TFT_BLACK);
    lcd.setTextColor(0x7BEF, TFT_BLACK); lcd.drawString(label, 8, y + 3);
    lcd.setTextColor(col,    TFT_BLACK); lcd.drawString(value, 138, y + 3);
}

static void drawBootHeader() {
    lcd.fillScreen(TFT_BLACK);
    lcd.setTextFont(2); lcd.setTextSize(1);
    lcd.fillRect(0, 0, SCREEN_W, 54, 0x1082);
    lcd.setTextColor(TFT_CYAN, 0x1082); lcd.setTextSize(2);
    lcd.drawString("ESP32-S3 QOI STR", 8, 6);
    lcd.setTextSize(1); lcd.setTextColor(0x7BEF, 0x1082);
    lcd.drawString("ILI9341  320x240  QOI pipeline", 8, 34);
    lcd.drawFastHLine(0, 54, SCREEN_W, TFT_DARKGREY);
}

// ─────────────────────────────────────────────
//  DISPLAY TASK  (Core 0, priority 2)
// ─────────────────────────────────────────────
// tileFb[bufSet][t] is always TILE_W×TILE_H by the time we get here
// (full-size decoded directly; upscaled tiles expanded by upscaleNN in decodeSlot).
// After pushing all 4 tiles, gives bufFree[bufSet] so the render task can
// reuse this buffer for the next frame N+2.
static void displayTask(void*) {
    DisplayMsg dmsg;
    while (true) {
        if (xQueueReceive(displayQueue, &dmsg, portMAX_DELAY) != pdTRUE) continue;

        for (int t = 0; t < NUM_TILES; t++) {
            lcd.pushImage(TILE_X[t], TILE_Y[t], TILE_W, TILE_H,
                          tileFb[dmsg.bufSet][t]);
        }

        // All 4 tiles pushed — release exclusive ownership of this buffer back
        // to the render task.  The render task may have been blocking on this.
        xSemaphoreGive(bufFree[dmsg.bufSet]);
        g_presentedFrames++;
    }
}

// ─────────────────────────────────────────────
//  SETUP
// ─────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    uint32_t t0 = millis();
    while (!Serial && (millis() - t0) < 2000) delay(10);
    Serial.println("\n[BOOT] QOI pipeline — direct PSRAM decode, no SRAM scratch");

    lcd.init(); lcd.setRotation(3); lcd.setColorDepth(16);
    lcd.setTextFont(2); lcd.setTextSize(1);
    drawBootHeader();
    statusLine(0, "Display:", "OK", TFT_GREEN);

    bool psramOk = psramFound();
    statusLine(1, "PSRAM:", psramOk ? "Found" : "MISSING!", psramOk ? TFT_GREEN : TFT_RED);
    if (!psramOk) { while (1) delay(1000); }

    // ── decodeTemp: SRAM scratch for upscale path ───────────────────────────
    // When upscale mode is active, QOI decodes here first (srcW×srcH < TILE_W×TILE_H),
    // then tileFb gets the small decoded tile; displayTask's pushImage upscales it.
    // In full-size mode this buffer is unused but allocated so mode can switch at runtime.
    // Size: TILE_PIXELS * 2 = 38 KB SRAM. Worth it for seamless mode switching.
    decodeTemp = (uint16_t*)heap_caps_aligned_alloc(
        16, TILE_PIXELS * 2,
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!decodeTemp) {
        Serial.println("[WARN] decodeTemp SRAM alloc failed — upscale modes unavailable");
        // Non-fatal: full-size mode still works without it
    }

    // ── Allocate pipeline slot assembly buffers in SRAM ──────────────────────
    // QOI encoder on PC enforces MAX_TILE_QOI.  Assembly sized at MAX_TILE_QOI.
    // SRAM chosen: QOI reads src forward-sequentially → L1 cache hit rate ~100%.
    bool allocOk = true;
    for (int s = 0; s < 2; s++) {
        slot[s].assembly = (uint8_t*)heap_caps_malloc(
            MAX_TILE_QOI, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!slot[s].assembly) {
            Serial.printf("[ERROR] slot[%d].assembly SRAM alloc failed\n", s);
            allocOk = false; break;
        }
    }

    // ── Allocate double-buffered per-tile framebuffers in PSRAM ──────────────
    // QOI decoder writes here DIRECTLY — no SRAM intermediate.
    // 16-byte alignment for potential future DMA / burst-write use.
    for (int s = 0; s < 2 && allocOk; s++) {
        for (int t = 0; t < NUM_TILES && allocOk; t++) {
            tileFb[s][t] = (uint16_t*)heap_caps_aligned_alloc(
                16, TILE_PIXELS * 2, MALLOC_CAP_SPIRAM);
            if (!tileFb[s][t]) {
                Serial.printf("[ERROR] tileFb[%d][%d] PSRAM alloc failed\n", s, t);
                allocOk = false;
            }
        }
    }

    // ── Chunk staging in PSRAM ────────────────────────────────────────────────
    // Sized for MAX_TILE_QOI per tile — slightly larger than JPEG era (33.6 KB → 42 KB).
    // PSRAM is plentiful (8 MB); this uses ~168 KB total.
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
    decodeQueue  = xQueueCreate(1, sizeof(DecodeMsg));
    displayQueue = xQueueCreate(1, sizeof(DisplayMsg));   // depth-1: render blocks until display drains
    for (int s = 0; s < 2; s++) {
        slotFree[s] = xSemaphoreCreateBinary();
        xSemaphoreGive(slotFree[s]);
    }
    // Both display buffers start free (render may write into either)
    for (int s = 0; s < 2; s++) {
        bufFree[s] = xSemaphoreCreateBinary();
        xSemaphoreGive(bufFree[s]);
    }

    // ── Memory layout report ──────────────────────────────────────────────────
    Serial.printf("[MEM] --- QOI pipeline (upscale-capable) ---\n");
    Serial.printf("[MEM] slot[0].assembly : %u B SRAM  (QOI stream, fwd-sequential)\n", MAX_TILE_QOI);
    Serial.printf("[MEM] slot[1].assembly : %u B SRAM\n", MAX_TILE_QOI);
    Serial.printf("[MEM] decodeTemp       : %u B SRAM  (upscale scratch; unused in full mode)\n",
                  decodeTemp ? (unsigned)(TILE_PIXELS*2) : 0u);
    Serial.printf("[MEM] tileFb x2x4      : %u B PSRAM (16-byte aligned, double-buffered)\n",
                  2 * NUM_TILES * TILE_PIXELS * 2);
    Serial.printf("[MEM] chunkStorage x4  : %u B PSRAM\n",
                  NUM_TILES * MAX_TILE_CHUNKS * CHUNK_DATA_SIZE);
    Serial.printf("[MEM] free SRAM  : %lu KB / %lu KB\n",
        heap_caps_get_free_size(MALLOC_CAP_INTERNAL)/1024,
        heap_caps_get_total_size(MALLOC_CAP_INTERNAL)/1024);
    Serial.printf("[MEM] free PSRAM : %lu KB / %lu KB\n",
        heap_caps_get_free_size(MALLOC_CAP_SPIRAM)/1024,
        heap_caps_get_total_size(MALLOC_CAP_SPIRAM)/1024);

    statusLine(2, "Buffers:", "QOI direct-PSRAM", TFT_GREEN);

    // ── WiFi ──────────────────────────────────────────────────────────────────
    statusLine(3, "WiFi:", "Connecting...", TFT_YELLOW);
    WiFi.mode(WIFI_STA); WiFi.setSleep(false); WiFi.begin(WIFI_SSID, WIFI_PASS);
    uint32_t ws = millis(); uint8_t tick = 0;
    while (WiFi.status() != WL_CONNECTED) {
        delay(250); tick++;
        char buf[24]; snprintf(buf, sizeof(buf), "Conn%.*s", tick % 5, ".....");
        statusLine(3, "WiFi:", buf, TFT_YELLOW);
        if (millis() - ws > 20000) {
            statusLine(3, "WiFi:", "TIMEOUT!", TFT_RED);
            delay(3000); ESP.restart();
        }
    }
    esp_wifi_set_ps(WIFI_PS_NONE);
    String ip = WiFi.localIP().toString();
    char ipBuf[36]; snprintf(ipBuf, sizeof(ipBuf), "%s (%ddBm)", ip.c_str(), WiFi.RSSI());
    statusLine(3, "WiFi:", ipBuf, TFT_GREEN);
    statusLine(4, "UDP:", String(UDP_PORT).c_str(), TFT_CYAN);
    statusLine(5, "Mode:", "QOI 4-tile ping-pong", TFT_CYAN);
    statusLine(6, "Status:", "Waiting for PC...", TFT_YELLOW);
    Serial.printf("[OK] WiFi: %s\n", ip.c_str());

    xTaskCreatePinnedToCore(networkTask, "NetTask",  10240, NULL, 3, NULL, 0);
    xTaskCreatePinnedToCore(displayTask, "DispTask", 4096,  NULL, 2, NULL, 0);
    Serial.println("[OK] Ready — QOI stream.");
}

// ─────────────────────────────────────────────
//  MAIN LOOP  (Core 1 — renderer)
// ─────────────────────────────────────────────
// Core 1 is 100% dedicated to QOI decode.
// qoi_to_rgb565be() runs here: reads SRAM assembly, writes PSRAM tileFb directly.
// No MCU scatter-write phase, no bswap16 pass, no memcpy — one sequential decode.
//
//  Buffer ownership protocol (fixes the 3 sync bugs):
//
//  Render holds tileFb[writeSet] exclusively while readyMask is being built.
//  It acquires that ownership (bufFree[writeSet]) before the FIRST tile decode
//  of each new batch, and transfers ownership to displayTask by posting to
//  displayQueue.  displayTask gives bufFree[bufSet] after it finishes pushing.
//
//  writeSet ONLY flips on a SUCCESSFUL send to displayQueue.
//  On a failed send, bufFree[writeSet] is given back immediately so displayTask
//  (which is presumably backed up) can still finish and free the buffer.
//
//  bufTaken tracks whether we already hold bufFree[writeSet] for the current
//  batch, so frame aborts (readyMask reset mid-collection) don't double-take.
void loop() {
    static bool     streamStarted = false;
    static uint32_t decodeAcc     = 0;
    static uint32_t decodeCount   = 0;

    static uint8_t  pendingFrame  = 0xFF;
    static uint8_t  readyMask     = 0;
    static uint32_t frameStartMs  = 0;
    static bool     bufTaken      = false;   // true while we hold bufFree[writeSet]

    DecodeMsg msg;
    if (xQueueReceive(decodeQueue, &msg, pdMS_TO_TICKS(40)) != pdTRUE) return;

    // ── Stale frame guard ─────────────────────────────────────────────────────
    if (pendingFrame != 0xFF && frameStartMs > 0 && (millis() - frameStartMs) > 150) {
        // Abandon in-progress frame; release buffer ownership so displayTask
        // can unblock if it was waiting on a previous queue entry.
        if (bufTaken) {
            xSemaphoreGive(bufFree[writeSet]);
            bufTaken = false;
        }
        pendingFrame = 0xFF;
        readyMask    = 0;
        frameStartMs = 0;
    }

    // ── New frameId → start collecting tiles for this frame ───────────────────
    if (pendingFrame == 0xFF || msg.frameId != pendingFrame) {
        if (pendingFrame != 0xFF && readyMask != 0)
            g_abortedFrames++;
        // Do NOT release bufFree here: we keep exclusive ownership of writeSet
        // and continue writing the new frame into the same buffer.  The old
        // partial tiles will be overwritten before readyMask ever reaches 0x0F.
        pendingFrame = msg.frameId;
        readyMask    = 0;
        frameStartMs = millis();
    }

    // ── Acquire buffer ownership before the first tile write ──────────────────
    // Blocks if displayTask is still reading this buffer (prevents the race
    // where writeSet has cycled back to a buffer not yet fully pushed).
    if (!bufTaken) {
        xSemaphoreTake(bufFree[writeSet], portMAX_DELAY);
        bufTaken = true;
    }

    // ── Decode ────────────────────────────────────────────────────────────────
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
            statusLine(6, "Status:", "STREAMING QOI!", TFT_GREEN);
            Serial.printf("[RENDER] first tile=%u slot=%u len=%u dec=%luus\n",
                          msg.tId, msg.slotIdx, msg.len, decUs);
            delay(200);
        }

        if ((tiles[msg.tId].stat_decoded % 120) == 0 && g_avgDecodeUs > 0) {
            Serial.printf("[RENDER] avg QOI decode: %lu us  (%lu fps-equiv per tile)\n",
                          g_avgDecodeUs, 1000000ul / g_avgDecodeUs);
        }

        // ── All 4 tiles for this frame are decoded — attempt to display ───────
        if (readyMask == 0x0F) {
            DisplayMsg dmsg = { msg.frameId, writeSet };

            if (xQueueSend(displayQueue, &dmsg, pdMS_TO_TICKS(20)) == pdTRUE) {
                // Ownership of tileFb[writeSet] transferred to displayTask.
                // Flip to the other buffer for the next frame.
                writeSet ^= 1;
                bufTaken = false;   // will take bufFree[new writeSet] on next batch
            } else {
                // displayTask is backed up; discard this frame and release
                // the buffer so displayTask can eventually drain and unblock.
                xSemaphoreGive(bufFree[writeSet]);
                bufTaken = false;
                g_abortedFrames++;
            }

            readyMask    = 0;
            pendingFrame = 0xFF;
            frameStartMs = 0;
        }
    }

    // ── Jitter measurement ────────────────────────────────────────────────────
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