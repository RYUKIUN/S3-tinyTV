#include "mpeg_decode.h"
#include "bswap16_memcpy_simd.h"
#include <Arduino.h>

// ── ESV2 decoder state (separate from tile decoder) ─────────────────────────
static JPEGDEC s_esv2_dec;

// Patch buffer: 16x16 RGB565 = 512 bytes (SRAM, for fast JPEGDEC writes)
static uint16_t* s_patchBuffer = nullptr;

// JPEG buffer for I-frames and patches (PSRAM)
static uint8_t* s_jpegBuffer = nullptr;
static const size_t JPEG_BUFFER_SIZE = 65536;  // 64 KB for ESV2 frame data

// ── Callbacks for ESV2 JPEGDEC ────────────────────────────────────────────────

// Callback for I-frame full-screen decoding
// Writes LE RGB565 from JPEGDEC, then bswap to BE framebuffer
static IRAM_ATTR int iframeMcuCb(JPEGDRAW* pDraw) {
    uint16_t* dst = (uint16_t*)pDraw->pUser + pDraw->y * SCREEN_W + pDraw->x;
    const uint16_t* src = (const uint16_t*)pDraw->pPixels;
    int w = pDraw->iWidth, h = pDraw->iHeight;
    
    // JPEGDEC outputs RGB565_LITTLE_ENDIAN; apply SIMD bswap + copy
    for (int r = 0; r < h; r++) {
        bswap16_memcpy_simd(dst + r * SCREEN_W,
                           src + r * w,
                           (size_t)w * 2);
    }
    return 1;
}

// Callback for P-frame patch decoding
// Writes LE RGB565 to patch buffer
static IRAM_ATTR int patchMcuCb(JPEGDRAW* pDraw) {
    uint16_t* dst = s_patchBuffer + pDraw->y * 16 + pDraw->x;
    const uint16_t* src = (const uint16_t*)pDraw->pPixels;
    int w = pDraw->iWidth, h = pDraw->iHeight;
    for (int r = 0; r < h; r++) {
        memcpy(dst + r * 16, src + r * w, (size_t)w * 2);
    }
    return 1;
}

// ─────────────────────────────────────────────────────────────────────────────
//  initESV2Decoder()
//  Allocate JPEG buffer and patch buffer for ESV2 decoding.
//  Call once during setup.
// ─────────────────────────────────────────────────────────────────────────────
bool initESV2Decoder() {
    // Allocate patch buffer in SRAM for fast JPEGDEC writes
    s_patchBuffer = (uint16_t*)heap_caps_aligned_alloc(
        16, 16 * 16 * 2, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!s_patchBuffer) {
        Serial.println("[ESV2] SRAM alloc for patch buffer failed");
        return false;
    }

    // Allocate JPEG buffer in PSRAM
    s_jpegBuffer = (uint8_t*)heap_caps_malloc(JPEG_BUFFER_SIZE, MALLOC_CAP_SPIRAM);
    if (!s_jpegBuffer) {
        Serial.println("[ESV2] PSRAM alloc for JPEG buffer failed");
        heap_caps_free(s_patchBuffer);
        s_patchBuffer = nullptr;
        return false;
    }

    Serial.println("[ESV2] Decoder initialized (patch: SRAM, JPEG: PSRAM)");
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  cleanupESV2Decoder()
//  Free ESV2 decoder buffers.
// ─────────────────────────────────────────────────────────────────────────────
void cleanupESV2Decoder() {
    if (s_patchBuffer) {
        heap_caps_free(s_patchBuffer);
        s_patchBuffer = nullptr;
    }
    if (s_jpegBuffer) {
        heap_caps_free(s_jpegBuffer);
        s_jpegBuffer = nullptr;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  initESV2Player()
//  Read and validate ESV2 file header.
// ─────────────────────────────────────────────────────────────────────────────
ESV2Header initESV2Player(const char* path) {
    ESV2Header hdr = {0};
    File f = SPIFFS.open(path, "r");
    if (!f) return hdr;

    char magic[4];
    if (f.read((uint8_t*)magic, 4) != 4 || memcmp(magic, "ESV2", 4) != 0) {
        f.close();
        return hdr;
    }

    uint16_t tmp[4];
    if (f.read((uint8_t*)tmp, 8) != 8) {
        f.close();
        return hdr;
    }
    hdr.width = tmp[0];
    hdr.height = tmp[1];
    hdr.fps = tmp[2];
    hdr.block_size = tmp[3];

    uint32_t fc;
    if (f.read((uint8_t*)&fc, 4) != 4) {
        f.close();
        return hdr;
    }
    hdr.frame_count = fc;

    f.close();
    return hdr;
}

// ─────────────────────────────────────────────────────────────────────────────
//  decodeESV2Frame()
//  Decode one I-frame (full JPEG) or P-frame (copy + patches).
//  
//  OPTIMIZATION NOTES:
//  • I-frame: JPEGDEC outputs RGB565_LITTLE_ENDIAN → bswap16_memcpy_simd to BE
//  • P-frame: memcpy(prevBuffer → frameBuffer) + patch apply via SIMD
//    No pixel-level processing needed — patches already JPEG-decoded
// ─────────────────────────────────────────────────────────────────────────────
bool decodeESV2Frame(File& f, uint16_t* frameBuffer, uint16_t* prevBuffer) {
    if (!s_patchBuffer || !s_jpegBuffer) {
        return false;
    }

    uint8_t type;
    if (f.read(&type, 1) != 1) return false;

    uint32_t len;
    if (f.read((uint8_t*)&len, 4) != 4) return false;

    if (len > JPEG_BUFFER_SIZE) {
        Serial.printf("[ESV2] Frame size %lu exceeds buffer %zu\n", (unsigned long)len, JPEG_BUFFER_SIZE);
        return false;
    }

    // ── I-FRAME: Full JPEG ────────────────────────────────────────────────────
    if (type == 'I') {
        if (f.read(s_jpegBuffer, len) != len) return false;

        if (s_esv2_dec.openRAM(s_jpegBuffer, (int)len, iframeMcuCb)) {
            // JPEGDEC outputs LE; callback applies SIMD bswap to BE
            s_esv2_dec.setPixelType(RGB565_LITTLE_ENDIAN);
            s_esv2_dec.setUserPointer(frameBuffer);
            s_esv2_dec.decode(0, 0, 0);
            s_esv2_dec.close();
            return true;
        }
        return false;
    }

    // ── P-FRAME: Delta patches ────────────────────────────────────────────────
    else if (type == 'P') {
        // Step 1: Copy previous frame to current (SIMD-accelerated)
        // Both are RGB565_BIG_ENDIAN, so direct copy; use bswap16_memcpy_simd
        // for maximum throughput (even though no byte swap needed, it's still optimal)
        size_t frameBytes = SCREEN_W * SCREEN_H * 2;
        for (size_t off = 0; off < frameBytes; off += 256) {
            size_t chunk = (frameBytes - off > 256) ? 256 : (frameBytes - off);
            memcpy((uint8_t*)frameBuffer + off,
                   (uint8_t*)prevBuffer + off,
                   chunk);
        }

        // Step 2: Read patch payload
        if (f.read(s_jpegBuffer, len) != len) return false;

        uint16_t numPatches = *(uint16_t*)s_jpegBuffer;
        uint8_t* p = s_jpegBuffer + 2;
        uint8_t* pEnd = s_jpegBuffer + len;

        // Step 3: Decode and apply patches
        for (uint16_t i = 0; i < numPatches && (p + 4 <= pEnd); i++) {
            uint8_t x_block = *p++;
            uint8_t y_block = *p++;
            uint16_t patch_len = *(uint16_t*)p; p += 2;

            if (p + patch_len > pEnd) break;

            // Decode patch JPEG into s_patchBuffer
            if (s_esv2_dec.openRAM(p, patch_len, patchMcuCb)) {
                s_esv2_dec.setPixelType(RGB565_LITTLE_ENDIAN);
                s_esv2_dec.decode(0, 0, 0);
                s_esv2_dec.close();

                // Copy patch into frame using SIMD-optimized memcpy
                int x = x_block * 16;
                int y = y_block * 16;
                for (int r = 0; r < 16; r++) {
                    bswap16_memcpy_simd(frameBuffer + (y + r) * SCREEN_W + x,
                                       s_patchBuffer + r * 16,
                                       16 * 2);
                }
            }
            p += patch_len;
        }
        return true;
    }

    return false;
}