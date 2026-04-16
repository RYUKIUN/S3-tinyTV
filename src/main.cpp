/*
 * ESP32-S3  |  ILI9341  |  8-bit Parallel i80  |  320x240
 *
 * PIPELINE ARCHITECTURE  —  QOI/YUV edition
 * ──────────────────────────────────────────
 *  Codec: PC sends a single packed frame per display frame.
 *  No tile splitting. No JPEGDEC. No bswap16_memcpy_simd.
 *
 *  Encode (PC, captureQoi.py):
 *    1. Capture RGB888 → BGR (OpenCV native)
 *    2. Convert BGR → YCbCr-4:2:0 (cv2.cvtColor COLOR_BGR2YCrCb + subsample)
 *    3. QOI-encode Y plane (320×240), Cb plane (160×120), Cr plane (160×120)
 *       using single-channel (grayscale) QOI
 *    4. Pack: [ySzHi ySzLo | Y-QOI | cbSzHi cbSzLo | Cb-QOI | crSzHi crSzLo | Cr-QOI]
 *    5. Fragment into CHUNK_DATA_SIZE UDP datagrams with 7-byte header
 *
 *  Decode (ESP32-S3):
 *    Core 0 (networkTask):
 *      recvfrom → reassemble chunks → slotAssembly (PSRAM) → post decodeQueue
 *    Core 1 (loop / renderTask):
 *      take decodeQueue
 *      qoi_frame_unpack()               → zero-copy pointers into slotAssembly
 *      qoi_decode_plane() × 3           → Y/Cb/Cr byte planes in decodeTemp (SRAM)
 *      yuv420_to_rgb565_simd()          → RGB565-BE directly into frameFb[writeSet] (PSRAM)
 *      post DisplayMsg → displayQueue
 *    Core 0 (displayTask, priority 2):
 *      lcd.pushImage(0,0,320,240, frameFb[bufSet])   ← one atomic DMA push/frame
 *
 *  Memory layout
 *  ─────────────
 *    slotAssembly[NUM_SLOTS]  PSRAM MAX_FRAME_QOI bytes each  (QOI packed frame; network writes, decoder reads)
 *    decodeTemp               SRAM  ~116 KB       Y(320×240) + Cb(160×120) + Cr(160×120) planes
 *    frameFb[0]               PSRAM 150 KB  ─┐ double-buffered RGB565-BE; DMA source
 *    frameFb[1]               PSRAM 150 KB  ─┘
 *
 *  SRAM budget
 *    decodeTemp               = 116 KB
 *    Total                    = 116 KB  — well within the ~512 KB SRAM on S3
 *
 *  PSRAM budget
 *    NUM_SLOTS × MAX_FRAME_QOI = 400 KB  (2 slots — dual-frame pipeline)
 *    frameFb x2               = 300 KB
 *    chunkStorage             = ~47 KB
 *    Total                    = 747 KB  — within 8MB PSRAM
 *
 *  Frame wire format  (magic 0xAA 0xDD replaces 0xAA 0xBB)
 *    [0xAA 0xDD frameId chunkId totalChunks frameSzHi frameSzLo] + payload
 *
 *  Stats packet (0xAB 0xCD prefix, every second when debugEnabled):
 *    FPS:X.X|TEMP:XX.X|JIT:X.X|DEC:XXXX|DROP:X|ABRT:X|SRAM:XXXX/XXXX|PSRAM:XXXX/XXXX
 *    DEC = avg full-frame decode time in µs (QOI×3 + YUV→RGB565, NOT pushImage)
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
#include "qoi_yuv_simd.h"

// ─────────────────────────────────────────────
//  DISPLAY
// ─────────────────────────────────────────────
class LGFX : public lgfx::LGFX_Device {
    lgfx::Bus_Parallel8  _bus;
    lgfx::Panel_ILI9341  _panel;
public:
    LGFX() {
        { auto cfg = _bus.config();
          cfg.freq_write = 30000000;
          cfg.pin_wr = 1; cfg.pin_rd = 40; cfg.pin_rs = 2;
          cfg.pin_d0 = 5; cfg.pin_d1 = 4;  cfg.pin_d2 = 10;
          cfg.pin_d3 = 9; cfg.pin_d4 = 3;  cfg.pin_d5 = 8;
          cfg.pin_d6 = 7; cfg.pin_d7 = 6;
          _bus.config(cfg); _panel.setBus(&_bus); }
        { auto cfg = _panel.config();
          cfg.pin_cs = 41; cfg.pin_rst = 39; cfg.pin_busy = -1;
          cfg.panel_width = 240; cfg.panel_height = 320;
          cfg.offset_x = 0; cfg.offset_y = 0; cfg.offset_rotation = 0;
          cfg.dummy_read_pixel = 8;
          cfg.readable = false; cfg.invert = false;
          cfg.rgb_order = false; cfg.dlen_16bit = false; cfg.bus_shared = false;
          _panel.config(cfg); }
        setPanel(&_panel);
    }
};
static LGFX lcd;

// ─────────────────────────────────────────────
//  CONFIG
// ─────────────────────────────────────────────
const char* WIFI_SSID  = "Endmin";
const char* WIFI_PASS  = "987654321";
const int   UDP_PORT   = 12345;

#define SCREEN_W         320
#define SCREEN_H         240

// Chroma plane dimensions (4:2:0)
#define CHROMA_W         (SCREEN_W / 2)   // 160
#define CHROMA_H         (SCREEN_H / 2)   // 120

// decodeTemp layout (all in SRAM):
//   [0 …  76799]  Y  plane  320×240  bytes
//   [76800 … 96959]  Cb plane  160×120  bytes
//   [96960 … 117119] Cr plane  160×120  bytes
#define Y_PLANE_BYTES    (SCREEN_W * SCREEN_H)           // 76 800
#define CHROMA_BYTES     (CHROMA_W * CHROMA_H)           // 19 200
#define DECODE_TEMP_SIZE (Y_PLANE_BYTES + 2 * CHROMA_BYTES)  // 115 200

// Slot size: max packed QOI frame.
// QOI worst case ≈ raw + overhead.  For 320×240 + 2×(160×120) ≈ 115 200 raw bytes.
// With QOI overhead headroom:  200 KB is conservative for typical screen content.
// Bump to 200 KB to be safe.  2 slots × 200 KB = 400 KB PSRAM.
#define MAX_FRAME_QOI 204800

#define CHUNK_DATA_SIZE  1400
#define MAX_FRAME_CHUNKS ((MAX_FRAME_QOI + CHUNK_DATA_SIZE - 1) / CHUNK_DATA_SIZE)  // 47
#define FRAME_TIMEOUT_MS 300

// ─────────────────────────────────────────────
//  PIPELINE SLOTS
// ─────────────────────────────────────────────
#define NUM_SLOTS 2
struct PipeSlot {
    uint8_t* assembly;   // SRAM — QOI packed frame; network writes, decoder reads
};
static PipeSlot slot[NUM_SLOTS];

// SRAM decode scratch — Y/Cb/Cr planes for one frame (Core 1 exclusive)
static uint8_t* decodeTemp = nullptr;

// Double-buffered full-frame RGB565-BE framebuffers in PSRAM
static uint16_t* frameFb[2] = { nullptr, nullptr };
static uint8_t   writeSet   = 0;   // Core 1 exclusive — no sync needed

// ─────────────────────────────────────────────
//  PIPELINE MESSAGES
// ─────────────────────────────────────────────
struct DecodeMsg {
    uint8_t  frameId;
    uint8_t  slotIdx;
    uint32_t len;       // packed QOI frame byte count
};

struct DisplayMsg {
    uint8_t frameId;
    uint8_t bufSet;
};

static QueueHandle_t     decodeQueue;
static SemaphoreHandle_t slotFree[NUM_SLOTS];
static QueueHandle_t     displayQueue;

// ─────────────────────────────────────────────
//  FRAME REASSEMBLY STATE  (single frame, not tiled)
// ─────────────────────────────────────────────
struct FrameState {
    bool     chunkGot[MAX_FRAME_CHUNKS];
    uint16_t chunkLen[MAX_FRAME_CHUNKS];
    uint8_t  frameId      = 0xFF;
    uint8_t  totalChunks  = 0;
    uint32_t frameSize    = 0;
    uint8_t  chunksGot    = 0;
    uint32_t firstChunkMs = 0;
    uint32_t stat_decoded = 0;
    uint32_t stat_corrupt = 0;
    uint32_t stat_timeout = 0;
};
static FrameState frameState;

// Chunk staging in PSRAM (network writes here; assembler copies to slot SRAM)
static uint8_t* chunkStorage = nullptr;   // MAX_FRAME_CHUNKS × CHUNK_DATA_SIZE bytes in PSRAM

// ─────────────────────────────────────────────
//  CROSS-CORE STATS
// ─────────────────────────────────────────────
static volatile uint32_t g_avgDecodeUs     = 0;
static volatile uint32_t g_presentedFrames = 0;
static volatile uint32_t g_abortedFrames   = 0;

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

// ─────────────────────────────────────────────
//  FRAME HELPERS
// ─────────────────────────────────────────────
static IRAM_ATTR void resetFrame() {
    memset(frameState.chunkGot, 0, sizeof(frameState.chunkGot));
    frameState.frameId      = 0xFF;
    frameState.totalChunks  = 0;
    frameState.frameSize    = 0;
    frameState.chunksGot    = 0;
    frameState.firstChunkMs = 0;
}

// Assemble complete QOI packed frame from PSRAM chunks into dst (SRAM slot).
// Returns assembled byte count, or 0 on error.
static IRAM_ATTR uint32_t assembleFrameInto(uint8_t* dst) {
    FrameState& fs = frameState;
    if (fs.totalChunks == 0) return 0;
    uint32_t offset = 0;
    for (uint8_t c = 0; c < fs.totalChunks; c++) {
        if (!fs.chunkGot[c]) return 0;
        uint8_t* src = chunkStorage + (uint32_t)c * CHUNK_DATA_SIZE;
        memcpy(dst + offset, src, fs.chunkLen[c]);
        offset += fs.chunkLen[c];
    }
    return offset;
}

// ─────────────────────────────────────────────
//  DECODE PIPELINE  (Core 1)
// ─────────────────────────────────────────────
//  1. qoi_frame_unpack()   — split packed buffer into Y/Cb/Cr QOI streams (zero-copy)
//  2. qoi_decode_plane() × 3 — decode each plane into decodeTemp (SRAM)
//  3. yuv420_to_rgb565_simd() — convert YUV→RGB565-BE, write into frameFb[writeSet] (PSRAM)
static IRAM_ATTR bool decodeSlot(const DecodeMsg& msg, uint32_t& decodeUs) {
    if (msg.slotIdx >= NUM_SLOTS || frameFb[writeSet] == nullptr) {
        decodeUs = 0;
        return false;
    }

    PipeSlot& s = slot[msg.slotIdx];
    const uint8_t* src = s.assembly;

    // Unpack: get pointers to the three QOI streams inside the assembly buffer
    const uint8_t *y_qoi, *cb_qoi, *cr_qoi;
    uint32_t       y_len,  cb_len,  cr_len;
    if (!qoi_frame_unpack(src, msg.len, &y_qoi, &y_len, &cb_qoi, &cb_len, &cr_qoi, &cr_len)) {
        decodeUs = 0;
        return false;
    }

    uint8_t* y_plane  = decodeTemp;
    uint8_t* cb_plane = decodeTemp + Y_PLANE_BYTES;
    uint8_t* cr_plane = decodeTemp + Y_PLANE_BYTES + CHROMA_BYTES;

    uint32_t t0 = micros();

    // Decode Y
    if (!qoi_decode_plane(y_qoi, y_len, y_plane, SCREEN_W, SCREEN_H)) {
        decodeUs = 0;
        return false;
    }
    // Decode Cb
    if (!qoi_decode_plane(cb_qoi, cb_len, cb_plane, CHROMA_W, CHROMA_H)) {
        decodeUs = 0;
        return false;
    }
    // Decode Cr
    if (!qoi_decode_plane(cr_qoi, cr_len, cr_plane, CHROMA_W, CHROMA_H)) {
        decodeUs = 0;
        return false;
    }

    // YUV → RGB565-BE → PSRAM framebuffer (single pass, SIMD on S3)
    yuv420_to_rgb565_simd(y_plane, cb_plane, cr_plane,
                          frameFb[writeSet],
                          SCREEN_W, SCREEN_H);

    decodeUs = micros() - t0;
    return true;
}

// ─────────────────────────────────────────────
//  DISPLAY HELPERS
// ─────────────────────────────────────────────
static void statusLine(uint8_t row, const char* label, const char* value,
                       uint32_t col = TFT_WHITE) {
    int y = 58 + row * 22;
    lcd.fillRect(0, y, SCREEN_W, 22, TFT_BLACK);
    lcd.setTextColor(0x7BEF, TFT_BLACK); lcd.drawString(label, 8,   y + 3);
    lcd.setTextColor(col,    TFT_BLACK); lcd.drawString(value, 138, y + 3);
}

static void drawBootHeader() {
    lcd.fillScreen(TFT_BLACK);
    lcd.setTextFont(2); lcd.setTextSize(1);
    lcd.fillRect(0, 0, SCREEN_W, 54, 0x1082);
    lcd.setTextColor(TFT_CYAN, 0x1082); lcd.setTextSize(2);
    lcd.drawString("ESP32-S3 STREAM", 8, 6);
    lcd.setTextSize(1); lcd.setTextColor(0x7BEF, 0x1082);
    lcd.drawString("ILI9341  320x240  QOI/YUV", 8, 34);
    lcd.drawFastHLine(0, 54, SCREEN_W, TFT_DARKGREY);
}

// ─────────────────────────────────────────────
//  DISPLAY TASK  (Core 0, priority 2)
// ─────────────────────────────────────────────
static void displayTask(void*) {
    DisplayMsg dmsg;
    while (true) {
        if (xQueueReceive(displayQueue, &dmsg, portMAX_DELAY) != pdTRUE) continue;
        lcd.pushImage(0, 0, SCREEN_W, SCREEN_H, frameFb[dmsg.bufSet]);
        g_presentedFrames++;
    }
}

// ─────────────────────────────────────────────
//  NETWORK TASK  (Core 0, priority 3)
// ─────────────────────────────────────────────
//  Packet format:
//    Data   : [0xAA 0xDD frameId chunkId totalChunks frameSzHi frameSzLo] + payload
//    Control: [0xAA 0xCC 0x01 debugState]
//
//  Pipeline flow when frame completes:
//    1. xSemaphoreTake(slotFree[back])          — wait for renderer to vacate slot
//    2. assembleFrameInto(slot[back].assembly)  — PSRAM chunks → SRAM slot
//    3. xQueueSend(decodeQueue, &msg)           — depth-NUM_SLOTS: returns immediately
//    4. back = (back + 1) % NUM_SLOTS
static IRAM_ATTR void networkTask(void*) {
    g_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (g_sock < 0) { vTaskDelete(NULL); return; }

    int rcvbuf = 65536;
    setsockopt(g_sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    struct sockaddr_in local = {};
    local.sin_family      = AF_INET;
    local.sin_port        = htons(UDP_PORT);
    local.sin_addr.s_addr = INADDR_ANY;
    if (bind(g_sock, (struct sockaddr*)&local, sizeof(local)) < 0) {
        close(g_sock); vTaskDelete(NULL); return;
    }
    fcntl(g_sock, F_SETFL, O_NONBLOCK);

    static uint8_t rxBuf[CHUNK_DATA_SIZE + 16];
    struct sockaddr_in sender;
    socklen_t slen = sizeof(sender);
    uint32_t  lastPktMs = millis(), lastBeaconMs = 0, lastStatMs = 0;
    uint8_t   back = 0;

    while (true) {
        int n = recvfrom(g_sock, rxBuf, sizeof(rxBuf), 0,
                         (struct sockaddr*)&sender, &slen);

        if (n < 0) {
            fd_set rfds; FD_ZERO(&rfds); FD_SET(g_sock, &rfds);
            struct timeval tv = { .tv_sec = 0, .tv_usec = 1000 };
            select(g_sock + 1, &rfds, NULL, NULL, &tv);

            if ((millis() - lastBeaconMs) > 2000 && (millis() - lastPktMs) > 2000) {
                struct sockaddr_in bc = {};
                bc.sin_family         = AF_INET;
                bc.sin_port           = htons(UDP_PORT);
                bc.sin_addr.s_addr    = htonl(INADDR_BROADCAST);
                int so = 1;
                setsockopt(g_sock, SOL_SOCKET, SO_BROADCAST, &so, sizeof(so));
                const char* b = "S3READY";
                sendto(g_sock, b, strlen(b), 0, (struct sockaddr*)&bc, sizeof(bc));
                lastBeaconMs = millis();
            }
            continue;
        }

        lastPktMs = millis();
        if (n < 4 || rxBuf[0] != 0xAA) { portYIELD(); continue; }
        memcpy(&g_remoteAddr, &sender, sizeof(sender));
        g_remoteAddrValid = true;

        // Control packet
        if (rxBuf[1] == 0xCC) {
            if (n >= 4 && rxBuf[2] == 0x01) debugEnabled = (rxBuf[3] == 1);
            portYIELD(); continue;
        }

        // Data packet: 0xAA 0xDD
        if (rxBuf[1] != 0xDD || n < 7) { portYIELD(); continue; }
        uint8_t  fId      = rxBuf[2];
        uint8_t  cId      = rxBuf[3];
        uint8_t  nChunks  = rxBuf[4];
        uint32_t fSize    = ((uint32_t)rxBuf[5] << 8) | rxBuf[6];
        int      dataLen  = n - 7;
        if (dataLen <= 0 || cId >= MAX_FRAME_CHUNKS) { portYIELD(); continue; }

        FrameState& fs = frameState;

        // Timeout stale partial frame
        if (fs.firstChunkMs > 0 && (millis() - fs.firstChunkMs) > FRAME_TIMEOUT_MS) {
            fs.stat_timeout++;
            resetFrame();
        }

        // New frame ID
        if (fId != fs.frameId) {
            if (fs.chunksGot > 0) g_abortedFrames++;
            resetFrame();
            fs.frameId      = fId;
            fs.totalChunks  = nChunks;
            fs.frameSize    = fSize;
            fs.firstChunkMs = millis();
        }

        // Store chunk in PSRAM staging
        if (!fs.chunkGot[cId]) {
            uint8_t* dst = chunkStorage + (uint32_t)cId * CHUNK_DATA_SIZE;
            memcpy(dst, &rxBuf[7], dataLen);
            fs.chunkLen[cId] = (uint16_t)dataLen;
            fs.chunkGot[cId] = true;
            fs.chunksGot++;
        }

        // Frame complete — assemble and post to decode queue
        if (fs.chunksGot >= fs.totalChunks) {
            xSemaphoreTake(slotFree[back], portMAX_DELAY);
            uint32_t len = assembleFrameInto(slot[back].assembly);

            if (len > 0) {
                DecodeMsg msg = { fId, back, len };
                xQueueSend(decodeQueue, &msg, portMAX_DELAY);
                back = (back + 1u) % NUM_SLOTS;
            } else {
                fs.stat_corrupt++;
                xSemaphoreGive(slotFree[back]);
            }
            resetFrame();
        }

        // Stats / debug
        if (debugEnabled && g_remoteAddrValid && (millis() - lastStatMs) > 500) {
            uint32_t el = millis() - lastStatMs;
            static uint32_t lastPresented = 0;
            uint32_t nowPresented = g_presentedFrames;
            uint32_t frames       = nowPresented - lastPresented;
            lastPresented         = nowPresented;
            float fps             = frames / (el / 1000.0f);

            uint32_t totalDrop = frameState.stat_corrupt + frameState.stat_timeout;

            static uint32_t lastAborted = 0;
            uint32_t nowAborted = g_abortedFrames;
            uint32_t aborted    = nowAborted - lastAborted;
            lastAborted         = nowAborted;

            uint32_t freeSRAM  = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
            uint32_t totalSRAM = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
            uint32_t freePSR   = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
            uint32_t totalPSR  = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
            float    tempC     = temperatureRead();

            snprintf(debugBuf, sizeof(debugBuf),
                "%c%cFPS:%.1f|TEMP:%.1f|JIT:%.1f|DEC:%lu|DROP:%lu|ABRT:%lu"
                "|SRAM:%lu/%lu|PSRAM:%lu/%lu",
                0xAB, 0xCD,
                fps, tempC, stat_jitter,
                (unsigned long)g_avgDecodeUs,
                (unsigned long)totalDrop,
                (unsigned long)aborted,
                (unsigned long)(freeSRAM  / 1024),
                (unsigned long)(totalSRAM / 1024),
                (unsigned long)(freePSR   / 1024),
                (unsigned long)(totalPSR  / 1024));

            sendto(g_sock, debugBuf, strlen(debugBuf), 0,
                   (struct sockaddr*)&g_remoteAddr, sizeof(g_remoteAddr));

            frameState.stat_decoded = frameState.stat_corrupt = frameState.stat_timeout = 0;
            lastStatMs = millis();
        }

        portYIELD();
    }
}

// ─────────────────────────────────────────────
//  SETUP
// ─────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    uint32_t t0 = millis();
    while (!Serial && (millis() - t0) < 2000) delay(10);
    Serial.println("\n[BOOT] QOI/YUV pipeline (SRAM decode + SIMD YUV->RGB565)");

    lcd.init(); lcd.setRotation(1); lcd.setColorDepth(16);
    lcd.setTextFont(2); lcd.setTextSize(1);
    drawBootHeader();
    statusLine(0, "Display:", "OK", TFT_GREEN);

    bool psramOk = psramFound();
    statusLine(1, "PSRAM:", psramOk ? "Found" : "MISSING!", psramOk ? TFT_GREEN : TFT_RED);
    if (!psramOk) { while (1) delay(1000); }

    // ── Allocate SRAM decode scratch (Core-1 exclusive) ──────────────────
    // Y(320×240) + Cb(160×120) + Cr(160×120) = 115 200 bytes, 16-byte aligned
    decodeTemp = (uint8_t*)heap_caps_aligned_alloc(
        16, DECODE_TEMP_SIZE, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!decodeTemp) {
        Serial.println("[ERROR] decodeTemp SRAM alloc failed");
        statusLine(2, "DecTemp:", "ALLOC FAILED!", TFT_RED);
        while (1) delay(1000);
    }

    // ── Allocate pipeline slots in PSRAM ──────────────────────────────────
    bool allocOk = true;
    for (int s = 0; s < NUM_SLOTS; s++) {
        slot[s].assembly = (uint8_t*)heap_caps_aligned_alloc(
            16, MAX_FRAME_QOI, MALLOC_CAP_SPIRAM);
        if (!slot[s].assembly) {
            Serial.printf("[ERROR] slot[%d].assembly PSRAM alloc failed\n", s);
            allocOk = false; break;
        }
    }

    // ── Allocate double-buffered full-frame framebuffers in PSRAM ────────
    for (int s = 0; s < 2 && allocOk; s++) {
        frameFb[s] = (uint16_t*)heap_caps_aligned_alloc(
            16, SCREEN_W * SCREEN_H * 2, MALLOC_CAP_SPIRAM);
        if (!frameFb[s]) {
            Serial.printf("[ERROR] frameFb[%d] PSRAM alloc failed\n", s);
            allocOk = false;
        }
    }

    // ── Allocate chunk staging in PSRAM ──────────────────────────────────
    if (allocOk) {
        chunkStorage = (uint8_t*)heap_caps_malloc(
            (size_t)MAX_FRAME_CHUNKS * CHUNK_DATA_SIZE, MALLOC_CAP_SPIRAM);
        if (!chunkStorage) {
            Serial.println("[ERROR] chunkStorage PSRAM alloc failed");
            allocOk = false;
        }
    }

    if (!allocOk) {
        statusLine(2, "Buffers:", "ALLOC FAILED!", TFT_RED);
        while (1) delay(1000);
    }

    // ── Pipeline sync primitives ──────────────────────────────────────────
    decodeQueue  = xQueueCreate(NUM_SLOTS, sizeof(DecodeMsg));
    displayQueue = xQueueCreate(2, sizeof(DisplayMsg));
    for (int s = 0; s < NUM_SLOTS; s++) {
        slotFree[s] = xSemaphoreCreateBinary();
        xSemaphoreGive(slotFree[s]);
    }

    Serial.printf("[MEM] decodeTemp       : %u B SRAM (16-byte aligned)\n", DECODE_TEMP_SIZE);
    for (int s = 0; s < NUM_SLOTS; s++)
        Serial.printf("[MEM] slot[%d].assembly : %u B PSRAM\n", s, MAX_FRAME_QOI);
    Serial.printf("[MEM] PSRAM slots total : %u B (%d x %u)\n",
                  NUM_SLOTS * MAX_FRAME_QOI, NUM_SLOTS, MAX_FRAME_QOI);
    Serial.printf("[MEM] frameFb x2       : %u B PSRAM (16-byte aligned, double-buffered)\n",
                  2 * SCREEN_W * SCREEN_H * 2);
    Serial.printf("[MEM] chunkStorage     : %u B PSRAM\n",
                  MAX_FRAME_CHUNKS * CHUNK_DATA_SIZE);
    Serial.printf("[MEM] free SRAM  : %lu KB / %lu KB\n",
        heap_caps_get_free_size(MALLOC_CAP_INTERNAL)  / 1024,
        heap_caps_get_total_size(MALLOC_CAP_INTERNAL) / 1024);
    Serial.printf("[MEM] free PSRAM : %lu KB / %lu KB\n",
        heap_caps_get_free_size(MALLOC_CAP_SPIRAM)  / 1024,
        heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024);

    statusLine(2, "Buffers:", "QOI/YUV PSRAM", TFT_GREEN);

    // ── WiFi ──────────────────────────────────────────────────────────────
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
    statusLine(4, "UDP:",    String(UDP_PORT).c_str(), TFT_CYAN);
    statusLine(5, "Mode:",   "QOI/YUV420",             TFT_CYAN);
    statusLine(6, "Status:", "Waiting for PC...",       TFT_YELLOW);
    Serial.printf("[OK] WiFi: %s\n", ip.c_str());

    // networkTask: Core 0, priority 3  — UDP receive + frame reassembly
    // displayTask: Core 0, priority 2  — lcd.pushImage (DMA, runs between UDP bursts)
    // loop()     : Core 1              — QOI decode + YUV→RGB565 SIMD
    xTaskCreatePinnedToCore(networkTask, "NetTask",  10240, NULL, 3, NULL, 0);
    xTaskCreatePinnedToCore(displayTask, "DispTask", 4096,  NULL, 2, NULL, 0);
    Serial.println("[OK] Ready.");
}

// ─────────────────────────────────────────────
//  MAIN LOOP  (Core 1 — renderer)
// ─────────────────────────────────────────────
void loop() {
    static bool     streamStarted = false;
    static uint32_t decodeAcc     = 0;
    static uint32_t decodeCount   = 0;

    DecodeMsg msg;
    if (xQueueReceive(decodeQueue, &msg, pdMS_TO_TICKS(40)) != pdTRUE) return;

    uint32_t decUs = 0;
    bool ok = decodeSlot(msg, decUs);

    xSemaphoreGive(slotFree[msg.slotIdx]);

    if (ok) {
        frameState.stat_decoded++;
        decodeAcc   += decUs;
        decodeCount++;

        if (decodeCount >= 16) {
            g_avgDecodeUs = decodeAcc / decodeCount;
            decodeAcc = 0; decodeCount = 0;
        }

        if (!streamStarted) {
            streamStarted = true;
            statusLine(6, "Status:", "STREAMING!", TFT_GREEN);
            delay(200);
        }

        DisplayMsg dmsg = { msg.frameId, writeSet };
        if (xQueueSend(displayQueue, &dmsg, pdMS_TO_TICKS(20)) != pdTRUE)
            g_abortedFrames++;
        writeSet ^= 1;
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