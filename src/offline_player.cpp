/*
 * offline_player.cpp
 * ──────────────────
 * Plays .mjpeg files from SPIFFS when no live stream is available.
 *
 * File format (matches Mjpegencoder.py output):
 *   Per frame:
 *     [4 B]  magic  0xAA 0xBB 0xCC 0xDD
 *     [4 B]  JPEG size N  (little-endian uint32)
 *     [N B]  raw JPEG data  (320×240, 4:2:0)
 *
 * Decode path (power-efficient):
 *   SPIFFS → jpegBuf (PSRAM)
 *     → JPEGDEC (RGB565_BIG_ENDIAN, no bswap needed)
 *     → frameFb[0] (PSRAM)
 *     → lcd.pushImage()
 *
 * CPU is reduced to 80 MHz and WiFi is off before this runs.
 * Frame pacing targets OFFLINE_FPS; vTaskDelay() keeps the RTOS healthy.
 */

#include "offline_player.h"
#include "display.h"
#include <SPIFFS.h>
#include <JPEGDEC.h>

// ── Tunables ─────────────────────────────────────────────────────────────────
#define OFFLINE_FPS        15                      // target playback frame rate
#define FRAME_INTERVAL_MS  (1000 / OFFLINE_FPS)    // 66 ms per frame budget
#define MAX_OFFLINE_FILES  16                      // max .mjpeg files to track
// Generous upper bound for a 320×240 Q40 4:2:0 JPEG — typically 8–20 KB.
// Raise if you encode at very high quality.
#define OFFLINE_JPEG_MAXSZ (60 * 1024)             // 60 KB

// ── File list populated by initOfflinePlayer() ────────────────────────────────
static char s_files[MAX_OFFLINE_FILES][36];
static int  s_fileCount = 0;

// ── JPEG decoder instance (not shared with streaming path) ────────────────────
static JPEGDEC s_dec;

// ── Frame magic ───────────────────────────────────────────────────────────────
static const uint8_t FRAME_MAGIC[4] = { 0xAA, 0xBB, 0xCC, 0xDD };

// ─────────────────────────────────────────────────────────────────────────────
//  MCU callback
//  Writes BE RGB565 pixels directly into frameFb[0] (PSRAM).
//  IRAM_ATTR keeps it out of flash cache — JPEGDEC may call it rapidly.
// ─────────────────────────────────────────────────────────────────────────────
static IRAM_ATTR int offlineMcuCb(JPEGDRAW* pDraw) {
    uint16_t*       dst = (uint16_t*)pDraw->pUser
                        + pDraw->y * SCREEN_W + pDraw->x;
    const uint16_t* src = (const uint16_t*)pDraw->pPixels;
    int w = pDraw->iWidth, h = pDraw->iHeight;
    for (int r = 0; r < h; r++)
        memcpy(dst + r * SCREEN_W,
               src + r * w,
               (size_t)w * 2);
    return 1;
}

// ─────────────────────────────────────────────────────────────────────────────
//  initOfflinePlayer()
//  Mounts SPIFFS and collects up to MAX_OFFLINE_FILES .mjpeg paths.
//  Returns true if at least one file exists.
// ─────────────────────────────────────────────────────────────────────────────
bool initOfflinePlayer() {
    s_fileCount = 0;

    if (!SPIFFS.begin(false)) {        // false = don't auto-format
        Serial.println("[OFFLINE] SPIFFS mount failed — no offline content");
        return false;
    }

    File root = SPIFFS.open("/");
    if (!root || !root.isDirectory()) {
        Serial.println("[OFFLINE] SPIFFS root open failed");
        return false;
    }

    File f = root.openNextFile();
    while (f && s_fileCount < MAX_OFFLINE_FILES) {
        if (!f.isDirectory()) {
            // f.name() returns "/filename.ext" on ESP32 SPIFFS
            const char* raw  = f.name();
            String       name = String(raw);
            name.toLowerCase();
            if (name.endsWith(".mjpeg")) {
                // Ensure path starts with '/'
                snprintf(s_files[s_fileCount], sizeof(s_files[0]),
                         "%s%s",
                         raw[0] == '/' ? "" : "/",
                         raw);
                Serial.printf("[OFFLINE] Found: %s  (%lu bytes)\n",
                              s_files[s_fileCount], (unsigned long)f.size());
                s_fileCount++;
            }
        }
        f = root.openNextFile();
    }

    Serial.printf("[OFFLINE] %d file(s) available\n", s_fileCount);
    return s_fileCount > 0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  resyncMagic()
//  After a bad header, scan byte-by-byte for the next FRAME_MAGIC.
//  Fills hdr[4..7] with the following 4 size bytes on success.
//  Returns true if resynced.
// ─────────────────────────────────────────────────────────────────────────────
static bool resyncMagic(File& f, uint8_t hdr[8]) {
    int   match = 0;
    uint8_t b;
    while (f.read(&b, 1) == 1) {
        if (b == FRAME_MAGIC[match]) {
            if (++match == 4) {
                if (f.read(hdr + 4, 4) == 4) {
                    memcpy(hdr, FRAME_MAGIC, 4);
                    return true;
                }
                return false;
            }
        } else {
            match = (b == FRAME_MAGIC[0]) ? 1 : 0;
        }
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
//  runOfflinePlayer()
//  Blocking infinite loop.  Call from Arduino loop() once in offline mode.
// ─────────────────────────────────────────────────────────────────────────────
void runOfflinePlayer() {
    // ── Allocate JPEG read buffer in PSRAM ───────────────────────────────────
    uint8_t* jpegBuf = (uint8_t*)heap_caps_malloc(OFFLINE_JPEG_MAXSZ,
                                                   MALLOC_CAP_SPIRAM);
    if (!jpegBuf) {
        Serial.println("[OFFLINE] PSRAM alloc for JPEG buffer failed!");
        statusLine(6, "Status:", "Buf alloc fail!", TFT_RED);
        for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    if (s_fileCount == 0) {
        statusLine(6, "Status:", "No .mjpeg files!", TFT_RED);
        for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    Serial.printf("[OFFLINE] Starting playback — %d file(s), %d fps\n",
                  s_fileCount, OFFLINE_FPS);

    int fileIdx = 0;

    // ── Outer loop: cycle through all files forever ───────────────────────────
    for (;;) {
        const char* path = s_files[fileIdx];

        // Show current filename (strip leading '/')
        const char* fname = (path[0] == '/') ? path + 1 : path;
        char dispBuf[30];
        snprintf(dispBuf, sizeof(dispBuf), "%.28s", fname);
        // statusLine(5, "File:", dispBuf, TFT_CYAN);
        // statusLine(6, "Status:", "Playing...", TFT_GREEN);

        File mf = SPIFFS.open(path, "r");
        if (!mf) {
            Serial.printf("[OFFLINE] Cannot open %s — skipping\n", path);
            fileIdx = (fileIdx + 1) % s_fileCount;
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        Serial.printf("[OFFLINE] Playing: %s (%lu bytes)\n",
                      path, (unsigned long)mf.size());

        uint8_t  hdr[8];
        uint32_t frameNum = 0;

        // ── Inner loop: decode frames until EOF / error ───────────────────────
        while (true) {
            uint32_t t0 = millis();

            // Read 8-byte frame header
            if (mf.read(hdr, 8) != 8) break;   // EOF

            // Validate magic; attempt resync on mismatch
            if (memcmp(hdr, FRAME_MAGIC, 4) != 0) {
                Serial.printf("[OFFLINE] Bad magic at frame %lu — resyncing\n",
                              (unsigned long)frameNum);
                if (!resyncMagic(mf, hdr)) break;
            }

            // Parse frame size (little-endian)
            uint32_t sz;
            memcpy(&sz, hdr + 4, 4);

            if (sz == 0 || sz > OFFLINE_JPEG_MAXSZ) {
                Serial.printf("[OFFLINE] Invalid frame size %lu — skip file\n",
                              (unsigned long)sz);
                break;
            }

            // Read JPEG payload into PSRAM buffer
            if ((uint32_t)mf.read(jpegBuf, sz) != sz) break;

            // ── Decode JPEG → frameFb[0] (BE RGB565, no bswap) ───────────────
            // RGB565_BIG_ENDIAN matches the byte order expected by lcd.pushImage
            // for this panel/bus combination, so no bswap16_memcpy_simd needed.
            if (s_dec.openRAM(jpegBuf, (int)sz, offlineMcuCb)) {
                s_dec.setPixelType(RGB565_BIG_ENDIAN);
                s_dec.setUserPointer(frameFb[0]);
                s_dec.decode(0, 0, 0);
                s_dec.close();

                lcd.pushImage(0, 0, SCREEN_W, SCREEN_H, frameFb[0]);
                g_presentedFrames++;
            }

            frameNum++;

            // ── Frame pacing — yield remainder of frame interval ──────────────
            uint32_t elapsed = millis() - t0;
            if (elapsed < FRAME_INTERVAL_MS) {
                vTaskDelay(pdMS_TO_TICKS(FRAME_INTERVAL_MS - elapsed));
            }
            // If decode+push took longer than one interval, just continue —
            // no frame-skip logic needed for offline playback.
        }

        mf.close();
        Serial.printf("[OFFLINE] Done: %s (%lu frames)\n",
                      path, (unsigned long)frameNum);

        // Advance to next file (wraps around)
        fileIdx = (fileIdx + 1) % s_fileCount;

        // Brief inter-file pause so the last frame is visible
        // statusLine(6, "Status:", "Next file...", TFT_YELLOW);
        // vTaskDelay(pdMS_TO_TICKS(500));
    }

    // unreachable — good practice
    heap_caps_free(jpegBuf);
}
