/*
 * ESP32-S3  |  ILI9341  |  8-bit Parallel i80  |  320x240
 *
 * TRANSPORT: USB CDC bulk (replaces WiFi + UDP — zero config, plug-and-play)
 * ───────────────────────────────────────────────────────────────────────────
 * Just plug the USB cable.  The Python host auto-detects the COM port by
 * Espressif VID (0x303A) and streams immediately — no WiFi, no IP, no AP.
 *
 * USB advantages over UDP:
 *   • No WiFi setup, no SSID/password, no AP needed
 *   • Reliable + ordered delivery → chunk reassembly eliminated entirely
 *   • Hardware flow-control → no pacing needed on the sender
 *   • ~800 KB/s effective throughput at USB Full Speed (more than enough)
 *   • ~168 KB PSRAM freed (chunkStorage gone)
 *
 * ── Packet format  PC → ESP32 ───────────────────────────────────────────────
 *
 *  TILE_PKT  (type 0x01):
 *    [0x55][0xAA]   2 B  magic sync
 *    [0x01]         1 B  type = TILE_PKT
 *    [frame_id]     1 B  0-255 rolling counter
 *    [tile_id]      1 B  0-3  (TL=0 TR=1 BL=2 BR=3)
 *    [enc_w]        1 B  encoded tile width  (0 = full TILE_W = 160)
 *    [enc_h]        1 B  encoded tile height (0 = full TILE_H = 120)
 *    [len_hi]       1 B  QOI data length, high byte
 *    [len_lo]       1 B  QOI data length, low byte
 *    [QOI data]     N B  raw QOI stream (N = len_hi<<8 | len_lo)
 *
 *  CMD_PKT  (type 0x02):
 *    [0x55][0xAA]   2 B  magic sync
 *    [0x02]         1 B  type = CMD_PKT
 *    [cmd]          1 B  0x01 = SET_DEBUG
 *    [param]        1 B  0 = off, 1 = on
 *
 * ── Stats  ESP32 → PC ────────────────────────────────────────────────────────
 *  Sent every second when debug enabled.  Python parser unchanged from UDP era.
 *    [0xAB][0xCD][stats text\n]
 *    FPS:X.X|TEMP:XX.X|JIT:X.X|DEC:XXXX|DROP:X|ABRT:X|SRAM:X/X|PSRAM:X/X
 *
 * ── Boot greeting  ESP32 → PC ────────────────────────────────────────────────
 *    "ESP32_READY\n"   Python waits for this before sending first frame
 *
 * ── Memory layout ────────────────────────────────────────────────────────────
 *  slot[0,1].assembly  SRAM  42 KB × 2  QOI stream buffer (fwd-sequential read)
 *  decodeTemp          SRAM  38 KB      upscale scratch (unused in full-size mode)
 *  tileFb[2][4]        PSRAM 38 KB × 8  double-buffered per-tile framebuffers
 *  chunkStorage        —     REMOVED    saved ~168 KB PSRAM vs UDP version
 *
 * ── Core assignment ──────────────────────────────────────────────────────────
 *  Core 0  networkTask (pri 3): USB Serial → decodeQueue
 *          displayTask  (pri 2): tileFb → ILI9341 pushImage
 *  Core 1  loop()      (pri 1): QOI decode → tileFb (PSRAM direct)
 */

#define LGFX_USE_V1
#include <LovyanGFX.hpp>
#include <Arduino.h>
#include <esp_attr.h>
#include <esp_heap_caps.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/queue.h"
#include <math.h>
#include <esp_task_wdt.h>
#include "qoi_dec.h"

// ─────────────────────────────────────────────
//  DISPLAY  (pin-out unchanged)
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
#define SCREEN_W     320
#define SCREEN_H     240
#define NUM_TILES    4
#define TILE_W       160
#define TILE_H       120
#define TILE_PIXELS  (TILE_W * TILE_H)    // 19 200

// Maximum QOI bytes per tile — must match Python MAX_TILE_QOI
#define MAX_TILE_QOI  42000u

// USB CDC protocol bytes
#define USB_SYNC_0       0x55u
#define USB_SYNC_1       0xAAu
#define USB_TYPE_TILE    0x01u
#define USB_TYPE_CMD     0x02u
#define USB_CMD_DEBUG    0x01u

// Tile screen origins: TL TR BL BR
static const int16_t TILE_X[NUM_TILES] = {  0, 160,   0, 160 };
static const int16_t TILE_Y[NUM_TILES] = {  0,   0, 120, 120 };

// ─────────────────────────────────────────────
//  PIPELINE BUFFERS
// ─────────────────────────────────────────────
struct PipeSlot {
    uint8_t* assembly;   // SRAM: QOI stream read (L1 hot, forward-sequential)
};
static PipeSlot slot[2];

// Double-buffered per-tile PSRAM framebuffers
// QOI decoder writes DIRECTLY here — no SRAM intermediate.
static uint16_t* tileFb[2][NUM_TILES] = {
    { nullptr, nullptr, nullptr, nullptr },
    { nullptr, nullptr, nullptr, nullptr }
};
static uint8_t writeSet = 0;   // Core 1 exclusive

struct TileMeta { uint8_t encW, encH; };   // 0 = full TILE_W/H
static TileMeta tileMeta[2][NUM_TILES] = {};

// ─────────────────────────────────────────────
//  PIPELINE MESSAGES
// ─────────────────────────────────────────────
struct DecodeMsg {
    uint8_t  frameId;
    uint8_t  tId;
    uint8_t  slotIdx;
    uint16_t len;
    uint8_t  encW;
    uint8_t  encH;
};

struct DisplayMsg {
    uint8_t frameId;
    uint8_t bufSet;
};

// ─────────────────────────────────────────────
//  PIPELINE SYNC
// ─────────────────────────────────────────────
static QueueHandle_t     decodeQueue;    // depth-1: netTask → renderer
static SemaphoreHandle_t slotFree[2];   // renderer gives when done with slot
static QueueHandle_t     displayQueue;  // depth-1: renderer → displayTask
static SemaphoreHandle_t bufFree[2];    // displayTask gives after push

// ─────────────────────────────────────────────
//  CROSS-CORE STATS  (volatile, no mutex — reads are best-effort)
// ─────────────────────────────────────────────
static volatile uint32_t g_avgDecodeUs     = 0;
static volatile uint32_t g_presentedFrames = 0;
static volatile uint32_t g_abortedFrames   = 0;
static volatile uint32_t g_stat_decoded[NUM_TILES] = {};
static volatile uint32_t g_stat_corrupt[NUM_TILES] = {};

// ─────────────────────────────────────────────
//  GLOBAL STATE
// ─────────────────────────────────────────────
static bool     debugEnabled = false;
static char     debugBuf[256];
static float    stat_jitter  = 0.0f;
static uint32_t stat_prevMs  = 0;
static uint16_t* decodeTemp  = nullptr;   // SRAM scratch for upscale path

// ─────────────────────────────────────────────
//  UPSCALE  (Core 1, IRAM) — unchanged from UDP version
// ─────────────────────────────────────────────
static uint16_t DRAM_ATTR s_xMap[TILE_W];
static uint16_t DRAM_ATTR s_yMap[TILE_H];
static uint16_t DRAM_ATTR s_lineBuf[TILE_W];
static int s_lutSrcW = -1, s_lutSrcH = -1;

static IRAM_ATTR void upscaleNN(
    const uint16_t* __restrict__ src,
    uint16_t*       __restrict__ dst,
    int srcW, int srcH)
{
    if (srcW != s_lutSrcW || srcH != s_lutSrcH) {
        for (int dx = 0; dx < TILE_W; dx++)
            s_xMap[dx] = (uint16_t)((dx * srcW) / TILE_W);
        for (int dy = 0; dy < TILE_H; dy++)
            s_yMap[dy] = (uint16_t)((dy * srcH) / TILE_H);
        s_lutSrcW = srcW; s_lutSrcH = srcH;
    }
    int prevSrcY = -1;
    for (int dy = 0; dy < TILE_H; dy++) {
        int sy = s_yMap[dy];
        if (sy != prevSrcY) {
            const uint16_t* srcRow = src + sy * srcW;
            int dx = 0;
            for (; dx <= TILE_W - 8; dx += 8) {
                s_lineBuf[dx+0] = srcRow[s_xMap[dx+0]]; s_lineBuf[dx+1] = srcRow[s_xMap[dx+1]];
                s_lineBuf[dx+2] = srcRow[s_xMap[dx+2]]; s_lineBuf[dx+3] = srcRow[s_xMap[dx+3]];
                s_lineBuf[dx+4] = srcRow[s_xMap[dx+4]]; s_lineBuf[dx+5] = srcRow[s_xMap[dx+5]];
                s_lineBuf[dx+6] = srcRow[s_xMap[dx+6]]; s_lineBuf[dx+7] = srcRow[s_xMap[dx+7]];
            }
            for (; dx < TILE_W; dx++) s_lineBuf[dx] = srcRow[s_xMap[dx]];
            prevSrcY = sy;
        }
        memcpy(dst + dy * TILE_W, s_lineBuf, TILE_W * sizeof(uint16_t));
    }
}

// ─────────────────────────────────────────────
//  QOI DECODE  (Core 1, IRAM) — unchanged from UDP version
// ─────────────────────────────────────────────
static IRAM_ATTR bool decodeSlot(const DecodeMsg& msg, uint32_t& decodeUs) {
    PipeSlot& s = slot[msg.slotIdx];
    if (msg.tId >= NUM_TILES || tileFb[writeSet][msg.tId] == nullptr) {
        decodeUs = 0; return false;
    }

    int srcW      = (msg.encW > 0) ? (int)msg.encW : TILE_W;
    int srcH      = (msg.encH > 0) ? (int)msg.encH : TILE_H;
    int srcPixels = srcW * srcH;
    uint32_t t0   = micros();

    if (srcW == TILE_W && srcH == TILE_H) {
        // Full-size: decode directly to PSRAM (zero copy, sequential writes)
        int rc = qoi_to_rgb565be(s.assembly, (int)msg.len,
                                 tileFb[writeSet][msg.tId], srcW, srcH);
        decodeUs = micros() - t0;
        if (rc != srcPixels) {
            Serial.printf("[DEC] slot%u QOI err: expected %d px got %d (len=%u)\n",
                          msg.slotIdx, srcPixels, rc, msg.len);
            return false;
        }
    } else {
        // Upscale: QOI → decodeTemp (SRAM) → upscaleNN → tileFb (PSRAM)
        if (!decodeTemp) {
            Serial.printf("[DEC] no decodeTemp for upscale srcW=%d srcH=%d\n", srcW, srcH);
            decodeUs = 0; return false;
        }
        int rc = qoi_to_rgb565be(s.assembly, (int)msg.len, decodeTemp, srcW, srcH);
        if (rc != srcPixels) {
            decodeUs = micros() - t0;
            Serial.printf("[DEC] slot%u QOI upscale err: expected %d got %d\n",
                          msg.slotIdx, srcPixels, rc);
            return false;
        }
        esp_task_wdt_reset();
        upscaleNN(decodeTemp, tileFb[writeSet][msg.tId], srcW, srcH);
        esp_task_wdt_reset();
        decodeUs = micros() - t0;
    }

    tileMeta[writeSet][msg.tId] = { 0, 0 };
    return true;
}

// ─────────────────────────────────────────────
//  NETWORK TASK  (Core 0, priority 3)
//
//  USB CDC Serial stream receiver.
//  Replaces the UDP socket + chunk reassembly from the WiFi version entirely.
//
//  Why no chunk reassembly?
//    UDP can drop/reorder packets → tiles had to be sent in 1400-byte chunks
//    and reassembled.  USB CDC is reliable and ordered at the hardware level,
//    so each tile is sent as one contiguous message and arrives intact.
//
//  Flow control:
//    If slotFree[back] is not available (renderer busy), this task blocks on
//    the semaphore.  The USB host receives a NAK at the hardware level and
//    simply pauses — zero data loss, zero intervention needed.
//
//  Throughput budget:
//    Tile avg ~8 KB × 4 tiles × 30 fps = ~960 KB/s
//    USB Full Speed effective: ~800–1000 KB/s  →  comfortable at 30 fps,
//    headroom at 20-25 fps for worst-case tiles.
// ─────────────────────────────────────────────
static void networkTask(void*) {
    // 500 ms inter-byte timeout for readBytes().
    // Normal tile transfer (~8 KB at ~500 KB/s) takes < 20 ms, so 500 ms is
    // generous and only triggers on genuine connection loss.
    Serial.setTimeout(500);

    uint8_t  back      = 0;
    uint32_t lastStatMs = 0;
    uint32_t pktCount  = 0;

    // Announce ready — Python waits for this string before streaming.
    delay(200);   // let USB CDC enumerate fully
    Serial.println("ESP32_READY");
    Serial.printf("[USB] Receiver up. MAX_TILE_QOI=%u B  slots=%d\n",
                  MAX_TILE_QOI, 2);

    while (true) {
        uint8_t b;

        // ── 1. Find sync byte 0 ───────────────────────────────────────────────
        if (Serial.readBytes(&b, 1) != 1) goto check_stats;
        if (b != USB_SYNC_0) goto check_stats;

        // ── 2. Find sync byte 1 ───────────────────────────────────────────────
        if (Serial.readBytes(&b, 1) != 1) goto check_stats;
        if (b != USB_SYNC_1) goto check_stats;

        // ── 3. Type byte ──────────────────────────────────────────────────────
        {
            uint8_t type;
            if (Serial.readBytes(&type, 1) != 1) goto check_stats;

            // ── TILE_PKT ───────────────────────────────────────────────────────
            if (type == USB_TYPE_TILE) {

                // 6-byte header: frame_id tile_id enc_w enc_h len_hi len_lo
                uint8_t hdr[6];
                if (Serial.readBytes((char*)hdr, 6) != 6) goto check_stats;

                const uint8_t  frameId = hdr[0];
                const uint8_t  tileId  = hdr[1];
                const uint8_t  encW    = hdr[2];
                const uint8_t  encH    = hdr[3];
                const uint16_t dataLen = ((uint16_t)hdr[4] << 8) | hdr[5];

                // Sanity-check header fields
                if (tileId >= NUM_TILES || dataLen == 0 || dataLen > MAX_TILE_QOI) {
                    Serial.printf("[NET] bad hdr: tile=%u len=%u\n", tileId, dataLen);
                    g_stat_corrupt[tileId < NUM_TILES ? tileId : 0]++;
                    goto check_stats;
                }

                // Wait for slot — renderer gives it back after decoding.
                // USB backs up gracefully if this blocks.
                if (xSemaphoreTake(slotFree[back], pdMS_TO_TICKS(400)) != pdTRUE) {
                    Serial.printf("[NET] slotFree[%u] timeout, skip tile %u\n",
                                  back, tileId);
                    g_stat_corrupt[tileId]++;
                    goto check_stats;
                }

                // Read entire QOI payload directly into SRAM assembly buffer.
                // Sequential SRAM write — L1 cache absorbs the stream cleanly.
                size_t got = Serial.readBytes((char*)slot[back].assembly, dataLen);
                if ((uint16_t)got != dataLen) {
                    xSemaphoreGive(slotFree[back]);   // release slot on partial read
                    Serial.printf("[NET] partial read got=%u of %u (tile %u)\n",
                                  (unsigned)got, dataLen, tileId);
                    g_stat_corrupt[tileId]++;
                    goto check_stats;
                }

                // Hand off to renderer
                DecodeMsg msg = { frameId, tileId, back, dataLen, encW, encH };
                if (xQueueSend(decodeQueue, &msg, pdMS_TO_TICKS(30)) == pdTRUE) {
                    back ^= 1;          // renderer owns this slot; switch
                } else {
                    xSemaphoreGive(slotFree[back]);   // renderer full, drop tile
                    g_abortedFrames++;
                }
                pktCount++;

            // ── CMD_PKT ────────────────────────────────────────────────────────
            } else if (type == USB_TYPE_CMD) {
                uint8_t payload[2];
                if (Serial.readBytes((char*)payload, 2) != 2) goto check_stats;
                if (payload[0] == USB_CMD_DEBUG)
                    debugEnabled = (payload[1] != 0);
            }
            // Unknown type: silently discard, re-sync on next bytes
        }

check_stats:
        // ── Send stats to PC once per second ─────────────────────────────────
        if (debugEnabled && (millis() - lastStatMs) >= 1000) {
            uint32_t el = millis() - lastStatMs;

            static uint32_t lastPresented = 0;
            const uint32_t nowPresented = g_presentedFrames;
            const float fps = (nowPresented - lastPresented) / (el / 1000.0f);
            lastPresented = nowPresented;

            uint32_t totalDrop = 0;
            for (int i = 0; i < NUM_TILES; i++) totalDrop += g_stat_corrupt[i];

            static uint32_t lastAborted = 0;
            const uint32_t nowAborted = g_abortedFrames;
            const uint32_t aborted = nowAborted - lastAborted;
            lastAborted = nowAborted;

            snprintf(debugBuf, sizeof(debugBuf),
                "FPS:%.1f|TEMP:%.1f|JIT:%.1f|DEC:%lu|DROP:%lu|ABRT:%lu"
                "|SRAM:%lu/%lu|PSRAM:%lu/%lu",
                fps,
                temperatureRead(),
                stat_jitter,
                (unsigned long)g_avgDecodeUs,
                (unsigned long)totalDrop,
                (unsigned long)aborted,
                heap_caps_get_free_size(MALLOC_CAP_INTERNAL)  / 1024,
                heap_caps_get_total_size(MALLOC_CAP_INTERNAL) / 1024,
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM)    / 1024,
                heap_caps_get_total_size(MALLOC_CAP_SPIRAM)   / 1024);

            // 0xAB 0xCD prefix — identical to UDP era; Python parser unchanged
            const uint8_t prefix[2] = { 0xAB, 0xCD };
            Serial.write(prefix, 2);
            Serial.print(debugBuf);
            Serial.write('\n');

            for (int i = 0; i < NUM_TILES; i++) g_stat_corrupt[i] = 0;
            pktCount   = 0;
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
    lcd.drawString("ESP32-S3 QOI USB", 8, 6);
    lcd.setTextSize(1); lcd.setTextColor(0x7BEF, 0x1082);
    lcd.drawString("ILI9341  320x240  USB bulk stream", 8, 34);
    lcd.drawFastHLine(0, 54, SCREEN_W, TFT_DARKGREY);
}

// ─────────────────────────────────────────────
//  DISPLAY TASK  (Core 0, priority 2) — unchanged
// ─────────────────────────────────────────────
static void displayTask(void*) {
    DisplayMsg dmsg;
    while (true) {
        if (xQueueReceive(displayQueue, &dmsg, portMAX_DELAY) != pdTRUE) continue;
        for (int t = 0; t < NUM_TILES; t++)
            lcd.pushImage(TILE_X[t], TILE_Y[t], TILE_W, TILE_H,
                          tileFb[dmsg.bufSet][t]);
        xSemaphoreGive(bufFree[dmsg.bufSet]);
        g_presentedFrames++;
    }
}

// ─────────────────────────────────────────────
//  SETUP
// ─────────────────────────────────────────────
void setup() {
    // Increase USB CDC RX ring-buffer before first I/O.
    // Default 256 B is fine for Serial terminal use but small for tile reads.
    // 8 KB gives comfortable headroom between timedRead() drains.
    Serial.setRxBufferSize(8192);
    Serial.begin(115200);   // baud rate ignored for USB CDC; set for tooling compat

    uint32_t t0 = millis();
    while (!Serial && (millis() - t0) < 3000) delay(10);   // wait for host

    Serial.println("\n[BOOT] QOI/USB pipeline — no WiFi, plug-and-play USB CDC");

    lcd.init(); lcd.setRotation(3); lcd.setColorDepth(16);
    lcd.setTextFont(2); lcd.setTextSize(1);
    drawBootHeader();
    statusLine(0, "Display:", "OK", TFT_GREEN);
    statusLine(1, "Transport:", "USB CDC bulk", TFT_CYAN);

    bool psramOk = psramFound();
    statusLine(2, "PSRAM:", psramOk ? "Found" : "MISSING!", psramOk ? TFT_GREEN : TFT_RED);
    if (!psramOk) { while (1) delay(1000); }

    // ── SRAM scratch for upscale decode path ─────────────────────────────────
    decodeTemp = (uint16_t*)heap_caps_aligned_alloc(
        16, TILE_PIXELS * 2, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!decodeTemp)
        Serial.println("[WARN] decodeTemp SRAM alloc failed — upscale unavailable");

    // ── SRAM assembly buffers (QOI stream: forward-sequential read → L1 hot) ─
    bool allocOk = true;
    for (int s = 0; s < 2; s++) {
        slot[s].assembly = (uint8_t*)heap_caps_malloc(
            MAX_TILE_QOI, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!slot[s].assembly) {
            Serial.printf("[ERROR] slot[%d].assembly alloc failed\n", s);
            allocOk = false; break;
        }
    }

    // ── Double-buffered per-tile PSRAM framebuffers ───────────────────────────
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
    // Note: no chunkStorage allocation — USB is reliable, no reassembly needed.

    if (!allocOk) {
        statusLine(3, "Buffers:", "ALLOC FAILED!", TFT_RED);
        while (1) delay(1000);
    }

    // ── Pipeline sync primitives ──────────────────────────────────────────────
    decodeQueue  = xQueueCreate(1, sizeof(DecodeMsg));
    displayQueue = xQueueCreate(1, sizeof(DisplayMsg));
    for (int s = 0; s < 2; s++) { slotFree[s] = xSemaphoreCreateBinary(); xSemaphoreGive(slotFree[s]); }
    for (int s = 0; s < 2; s++) { bufFree[s]  = xSemaphoreCreateBinary(); xSemaphoreGive(bufFree[s]);  }

    // ── Memory report ─────────────────────────────────────────────────────────
    Serial.printf("[MEM] slot[0+1].assembly : %u B × 2  SRAM\n", MAX_TILE_QOI);
    Serial.printf("[MEM] decodeTemp         : %u B  SRAM (upscale)\n",
                  decodeTemp ? (unsigned)(TILE_PIXELS * 2) : 0u);
    Serial.printf("[MEM] tileFb × 8         : %u B  PSRAM\n",
                  2 * NUM_TILES * TILE_PIXELS * 2);
    Serial.printf("[MEM] chunkStorage       : 0 B  (saved ~168 KB vs UDP)\n");
    Serial.printf("[MEM] free SRAM  : %lu / %lu KB\n",
        heap_caps_get_free_size(MALLOC_CAP_INTERNAL)  / 1024,
        heap_caps_get_total_size(MALLOC_CAP_INTERNAL) / 1024);
    Serial.printf("[MEM] free PSRAM : %lu / %lu KB\n",
        heap_caps_get_free_size(MALLOC_CAP_SPIRAM)    / 1024,
        heap_caps_get_total_size(MALLOC_CAP_SPIRAM)   / 1024);

    statusLine(3, "Buffers:", "QOI direct-PSRAM", TFT_GREEN);
    statusLine(4, "USB:", "Waiting for host...", TFT_YELLOW);
    statusLine(5, "Mode:", "QOI 4-tile USB bulk", TFT_CYAN);
    statusLine(6, "Status:", "Run captureQOI.py", TFT_YELLOW);

    Serial.println("[OK] Ready — run captureQOI.py on the host PC");

    xTaskCreatePinnedToCore(networkTask, "NetTask",  10240, NULL, 3, NULL, 0);
    xTaskCreatePinnedToCore(displayTask, "DispTask", 4096,  NULL, 2, NULL, 0);
}

// ─────────────────────────────────────────────
//  MAIN LOOP  (Core 1 — renderer) — unchanged from UDP version
// ─────────────────────────────────────────────
void loop() {
    static bool     streamStarted = false;
    static uint32_t decodeAcc     = 0;
    static uint32_t decodeCount   = 0;
    static uint8_t  pendingFrame  = 0xFF;
    static uint8_t  readyMask     = 0;
    static uint32_t frameStartMs  = 0;
    static bool     bufTaken      = false;

    DecodeMsg msg;
    if (xQueueReceive(decodeQueue, &msg, pdMS_TO_TICKS(40)) != pdTRUE) return;

    // Stale frame guard (> 150 ms with incomplete tiles)
    if (pendingFrame != 0xFF && frameStartMs > 0 && (millis() - frameStartMs) > 150) {
        if (bufTaken) { xSemaphoreGive(bufFree[writeSet]); bufTaken = false; }
        pendingFrame = 0xFF; readyMask = 0; frameStartMs = 0;
    }

    if (pendingFrame == 0xFF || msg.frameId != pendingFrame) {
        if (pendingFrame != 0xFF && readyMask != 0) g_abortedFrames++;
        pendingFrame = msg.frameId; readyMask = 0; frameStartMs = millis();
    }

    if (!bufTaken) {
        xSemaphoreTake(bufFree[writeSet], portMAX_DELAY);
        bufTaken = true;
    }

    uint32_t decUs = 0;
    bool ok = decodeSlot(msg, decUs);
    xSemaphoreGive(slotFree[msg.slotIdx]);

    if (ok) {
        g_stat_decoded[msg.tId]++;
        decodeAcc += decUs; decodeCount++;

        readyMask |= (uint8_t)(1u << msg.tId);

        if (decodeCount >= 16) {
            g_avgDecodeUs = decodeAcc / decodeCount;
            decodeAcc = 0; decodeCount = 0;
        }

        if (!streamStarted) {
            streamStarted = true;
            statusLine(4, "USB:", "Connected!", TFT_GREEN);
            statusLine(6, "Status:", "STREAMING QOI!", TFT_GREEN);
            Serial.printf("[RENDER] first tile=%u slot=%u len=%u dec=%luus\n",
                          msg.tId, msg.slotIdx, msg.len, decUs);
            delay(200);
        }

        if ((g_stat_decoded[msg.tId] % 120) == 0 && g_avgDecodeUs > 0)
            Serial.printf("[RENDER] avg QOI decode: %lu us  (%lu fps-equiv/tile)\n",
                          g_avgDecodeUs, 1000000ul / g_avgDecodeUs);

        if (readyMask == 0x0F) {
            DisplayMsg dmsg = { msg.frameId, writeSet };
            if (xQueueSend(displayQueue, &dmsg, pdMS_TO_TICKS(20)) == pdTRUE) {
                writeSet ^= 1; bufTaken = false;
            } else {
                xSemaphoreGive(bufFree[writeSet]); bufTaken = false; g_abortedFrames++;
            }
            readyMask = 0; pendingFrame = 0xFF; frameStartMs = 0;
        }
    }

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