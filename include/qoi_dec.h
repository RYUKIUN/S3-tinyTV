#pragma once
/*
 * qoi_dec.h — Minimal QOI decoder for ESP32-S3
 * ─────────────────────────────────────────────
 * Decodes a QOI tile (RGB, no alpha) DIRECTLY into an RGB565 big-endian
 * framebuffer (ILI9341 native format).  No intermediate buffer needed.
 *
 * Key properties:
 *   • Reads  src (SRAM assembly buffer) → sequential forward reads, L1 hot
 *   • Writes dst (PSRAM tileFb)         → sequential forward writes, cache-line friendly
 *   • 64-entry hash table (256 B)       → stays permanently in L1 data cache
 *   • Inline RGB888→RGB565 + bswap16    → eliminates separate post-pass entirely
 *   • No external library, no heap alloc, no callbacks
 *   • __attribute__((optimize("O3"))) + restrict → GCC LX7 ILP maximized
 *
 * Run fills:  when a long run is detected (≥8 pixels) the inner fill is done
 * with 32-bit stores (2 pixels/store) to exploit LX7 store-buffer coalescing
 * into 32-byte PSRAM cache lines.
 *
 * QOI spec: https://qoiformat.org
 * Alpha channel: ignored (always treated as 255 — avoids RGBA decode overhead).
 *                Encoder must send channels=3.
 *
 * Usage:
 *   int n = qoi_to_rgb565be(src, src_len, dst_rgb565be, TILE_W, TILE_H);
 *   // returns TILE_W*TILE_H on success, 0 on error
 */

#include <stdint.h>
#include <string.h>
#include <esp_attr.h>

// ── Hash constant for alpha=255 (fixed):
//   255 * 11 = 2805  →  2805 & 63 = 53
#define QOI_ALPHA_HASH_BIAS  53u

// ── Inline RGB888 → RGB565 big-endian (ILI9341 wire format) ─────────────────
// R5[15:11] G6[10:5] B5[4:0] → byte-swap → [G2G1G0B4B3B2B1B0 | R4R3R2R1R0G5G4G3]
static inline __attribute__((always_inline))
uint16_t rgb_to_565be(uint8_t r, uint8_t g, uint8_t b) {
    uint16_t le = (uint16_t)(((uint16_t)(r & 0xF8u) << 8) |
                             ((uint16_t)(g & 0xFCu) << 3) |
                             ((uint16_t)(b          >> 3)));
    return (uint16_t)((le >> 8) | (le << 8));   // bswap16 inline
}

// ── Main decoder ─────────────────────────────────────────────────────────────
static IRAM_ATTR __attribute__((optimize("O3")))
int qoi_to_rgb565be(const uint8_t* __restrict__ src, int src_len,
                    uint16_t*      __restrict__ dst, int w, int h)
{
    // ── Header validation (14 bytes) ─────────────────────────────────────────
    if (src_len < 22) return 0;   // 14 header + 8 footer minimum
    if (src[0] != 'q' || src[1] != 'o' || src[2] != 'i' || src[3] != 'f') return 0;

    uint32_t iw = ((uint32_t)src[4]  << 24) | ((uint32_t)src[5]  << 16)
                | ((uint32_t)src[6]  <<  8) |  (uint32_t)src[7];
    uint32_t ih = ((uint32_t)src[8]  << 24) | ((uint32_t)src[9]  << 16)
                | ((uint32_t)src[10] <<  8) |  (uint32_t)src[11];
    if ((int)iw != w || (int)ih != h) return 0;
    // src[12]=channels(3), src[13]=colorspace(0) — not validated, encoder enforces

    // ── State ────────────────────────────────────────────────────────────────
    // 64 × uint32_t hash table: 0x00RRGGBB, alpha implicit 255
    // 256 bytes → fits in a single L1 cache set, never evicted during decode
    uint32_t idx[64];
    memset(idx, 0, sizeof(idx));

    const uint8_t* p   = src + 14;
    const uint8_t* end = src + src_len - 8;   // exclude 8-byte footer

    uint8_t pr = 0, pg = 0, pb = 0;   // previous pixel (start: 0,0,0,255)
    int     run   = 0;
    int     total = w * h;

    // ── Decode loop ──────────────────────────────────────────────────────────
    for (int i = 0; i < total; ) {
        uint8_t r, g, b;

        // ── Run-length continuation ──────────────────────────────────────────
        if (run > 0) {
            // Long run: fill with 32-bit stores (2 × RGB565 per store).
            // Amortises PSRAM write overhead for solid-color regions.
            uint16_t px = rgb_to_565be(pr, pg, pb);
            uint32_t px32 = ((uint32_t)px << 16) | px;

            // Align dst to 4-byte boundary if needed (PSRAM write performance)
            if ((uintptr_t)(dst + i) & 2u) {
                dst[i++] = px;
                if (--run == 0) continue;
            }

            uint32_t* dp32 = (uint32_t*)(dst + i);
            while (run >= 2 && i + 2 <= total) {
                *dp32++ = px32;
                i += 2; run -= 2;
            }
            if (run > 0 && i < total) {
                dst[i++] = px;
                --run;
            }
            continue;
        }

        // ── Decode next chunk ────────────────────────────────────────────────
        if (p >= end) return 0;
        uint8_t tag = *p++;

        if (tag == 0xFE) {                          // QOI_OP_RGB (4 bytes)
            r = *p++; g = *p++; b = *p++;

        } else if (tag == 0xFF) {                   // QOI_OP_RGBA (5 bytes, alpha discarded)
            r = *p++; g = *p++; b = *p++; ++p;

        } else {
            uint8_t op2 = tag >> 6;

            if (op2 == 0u) {                        // QOI_OP_INDEX (1 byte)
                uint32_t c = idx[tag & 0x3Fu];
                r = (uint8_t)(c >> 16);
                g = (uint8_t)(c >>  8);
                b = (uint8_t)(c);

            } else if (op2 == 1u) {                 // QOI_OP_DIFF (1 byte)
                r = (uint8_t)(pr + ((tag >> 4 & 3u) - 2u));
                g = (uint8_t)(pg + ((tag >> 2 & 3u) - 2u));
                b = (uint8_t)(pb + ((tag       & 3u) - 2u));

            } else if (op2 == 2u) {                 // QOI_OP_LUMA (2 bytes)
                uint8_t b2  = *p++;
                int     vg  = (int)(tag & 0x3Fu) - 32;
                r = (uint8_t)(pr + vg + (int)((b2 >> 4) & 0xFu) - 8);
                g = (uint8_t)(pg + vg);
                b = (uint8_t)(pb + vg + (int)(b2 & 0xFu) - 8);

            } else {                                // QOI_OP_RUN (1 byte)
                // run-1 bias: tag&0x3F=0 means this pixel + 0 more = run of 1
                // First pixel written below; run holds the additional copies.
                run = (int)(tag & 0x3Fu);
                r = pr; g = pg; b = pb;
            }
        }

        // ── Update hash table and previous pixel ─────────────────────────────
        uint8_t hi = (uint8_t)((r * 3u + g * 5u + b * 7u + QOI_ALPHA_HASH_BIAS) & 63u);
        idx[hi] = ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
        pr = r; pg = g; pb = b;

        // ── Write RGB565 BE to framebuffer ────────────────────────────────────
        dst[i++] = rgb_to_565be(r, g, b);
    }

    return total;
}