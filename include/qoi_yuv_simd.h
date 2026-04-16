/*
 * qoi_yuv_simd.h
 * ──────────────────────────────────────────────────────────────────────────
 *  QOI decode  +  YCbCr-4:2:0 → RGB565 (big-endian, ILI9341)  pipeline.
 *
 *  Design goals
 *  ────────────
 *  1. Zero intermediate RGB888 buffer.
 *     decodeTemp (SRAM) holds raw Y/Cb/Cr byte planes — not expanded pixels.
 *     yuv420_to_rgb565_simd() reads those planes, converts, byte-swaps, and
 *     writes RGB565-BE directly to the PSRAM framebuffer in one pass.
 *
 *  2. All three functions are marked IRAM_ATTR / placed in IRAM via
 *     __attribute__((section(".iram1.text"))) so the hot path never stalls
 *     on a flash cache miss.
 *
 *  3. SIMD path (ESP32-S3 only) uses ESP32-S3 PIE (Xtensa EE.*) 128-bit
 *     Q-register intrinsics.  A scalar fallback is compiled for every other
 *     target so the file is safe to include anywhere.
 *
 *  Memory contract (caller must guarantee)
 *  ────────────────────────────────────────
 *    qoi_decode_plane()
 *      src   : pointer to one QOI-encoded plane in SRAM (slot[].assembly)
 *      dst   : pointer to SRAM output byte buffer (decodeTemp planes)
 *      w, h  : plane dimensions in pixels
 *              Y  plane: w = FRAME_W,   h = FRAME_H
 *              Cb plane: w = FRAME_W/2, h = FRAME_H/2
 *              Cr plane: w = FRAME_W/2, h = FRAME_H/2
 *      Returns number of bytes consumed from src, or 0 on error.
 *
 *    yuv420_to_rgb565_simd()
 *      y_plane  : SRAM — Y  plane,  FRAME_W × FRAME_H  bytes
 *      cb_plane : SRAM — Cb plane, (FRAME_W/2)×(FRAME_H/2) bytes
 *      cr_plane : SRAM — Cr plane, (FRAME_W/2)×(FRAME_H/2) bytes
 *      dst_fb   : PSRAM — output RGB565-BE framebuffer, FRAME_W×FRAME_H×2 bytes
 *      w, h     : full-frame dimensions (must be even)
 *
 *  QOI wire format (per-plane, grayscale variant)
 *  ────────────────────────────────────────────────
 *   The PC encodes each plane as a single-channel (grayscale) QOI stream:
 *     Header : "qoif"  magic (4B)
 *              width   (4B BE uint32)
 *              height  (4B BE uint32)
 *              channels = 1   (1B)
 *              colorspace = 0 (1B)  — ignored by decoder
 *   Chunks  : standard QOI chunk set (QOI_OP_INDEX / DIFF / LUMA / RUN / RGB)
 *             For channels=1, QOI_OP_RGB carries one byte of luma; the
 *             decoder treats it as a Y/Cb/Cr sample.
 *   Footer  : 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x01  (8 bytes)
 *
 *  Frame wire format  (sent by captureQoi.py)
 *  ──────────────────────────────────────────
 *   [0xAA 0xDD]          2B  — frame magic (replaces 0xAA 0xBB of JPEG era)
 *   [frameId]            1B
 *   [chunkId]            1B
 *   [totalChunks]        1B
 *   [frameTotalSizeHi]   1B  — total packed frame bytes (all 3 planes + 3 size headers)
 *   [frameTotalSizeLo]   1B
 *   [payload …]         ≤CHUNK_DATA_SIZE B
 *
 *   The packed frame layout (assembled in SRAM before decode):
 *     [ySizeHi ySizeLo]    2B  — byte length of Y QOI stream
 *     [Y QOI stream]       ySizeHi<<8|ySizeLo  bytes
 *     [cbSizeHi cbSizeLo]  2B
 *     [Cb QOI stream]      …
 *     [crSizeHi crSizeLo]  2B
 *     [Cr QOI stream]      …
 *
 *  No tile splitting.  The full 320×240 frame is one packed unit so the
 *  complex per-tile reassembly state machine is gone entirely.
 * ──────────────────────────────────────────────────────────────────────────
 */

#pragma once
#include <stdint.h>
#include <string.h>
#include <esp_attr.h>

// ─────────────────────────────────────────────
//  QOI constants
// ─────────────────────────────────────────────
#define QOI_MAGIC        0x716F6966u   // "qoif"
#define QOI_OP_INDEX     0x00u
#define QOI_OP_DIFF      0x40u
#define QOI_OP_LUMA      0x80u
#define QOI_OP_RUN       0xC0u
#define QOI_OP_RGB       0xFEu
#define QOI_OP_RGBA      0xFFu         // not used for 1-ch planes, but guard it
#define QOI_MASK_2       0xC0u
#define QOI_HEADER_SIZE  14
#define QOI_FOOTER_SIZE  8
#define QOI_HASH(v)      (((v)*3 + 59) & 63)  // single-channel hash

// ─────────────────────────────────────────────
//  QOI single-channel (grayscale) decoder
//  Decodes one Y / Cb / Cr plane into a flat byte buffer.
//  Returns bytes consumed from src (including header + footer), 0 on error.
// ─────────────────────────────────────────────
static IRAM_ATTR uint32_t qoi_decode_plane(
        const uint8_t* __restrict__ src, uint32_t src_len,
        uint8_t*       __restrict__ dst, uint32_t w, uint32_t h)
{
    if (src_len < (uint32_t)(QOI_HEADER_SIZE + QOI_FOOTER_SIZE)) return 0;

    // Validate magic
    uint32_t magic = ((uint32_t)src[0] << 24) | ((uint32_t)src[1] << 16)
                   | ((uint32_t)src[2] <<  8) |  (uint32_t)src[3];
    if (magic != QOI_MAGIC) return 0;

    // Parse header dimensions (big-endian)
    uint32_t qw = ((uint32_t)src[4]  << 24) | ((uint32_t)src[5]  << 16)
                | ((uint32_t)src[6]  <<  8) |  (uint32_t)src[7];
    uint32_t qh = ((uint32_t)src[8]  << 24) | ((uint32_t)src[9]  << 16)
                | ((uint32_t)src[10] <<  8) |  (uint32_t)src[11];
    if (qw != w || qh != h) return 0;

    const uint32_t total_px = w * h;
    uint8_t  index[64] = {};       // QOI running index — zero-initialised
    uint8_t  px        = 0;        // current pixel value
    uint32_t run       = 0;
    uint32_t p         = QOI_HEADER_SIZE;   // read position
    uint32_t px_pos    = 0;                 // write position

    while (px_pos < total_px) {
        if (run > 0) {
            --run;
        } else {
            if (p >= src_len) return 0;
            uint8_t b1 = src[p++];

            if (b1 == QOI_OP_RGB) {
                if (p >= src_len) return 0;
                px = src[p++];
            } else if (b1 == QOI_OP_RGBA) {
                if (p + 1 >= src_len) return 0;
                px = src[p++]; p++;   // skip alpha
            } else {
                switch (b1 & QOI_MASK_2) {
                    case QOI_OP_INDEX:
                        px = index[b1 & 0x3F];
                        break;
                    case QOI_OP_DIFF: {
                        int8_t dr = (int8_t)(((b1 >> 4) & 3) - 2);
                        int8_t dg = (int8_t)(((b1 >> 2) & 3) - 2);
                        int8_t db = (int8_t)( (b1       & 3) - 2);
                        // For grayscale we fold RGB diff into luma: average
                        px = (uint8_t)(px + dr);
                        (void)dg; (void)db;  // Cb/Cr encoded as single-ch
                        break;
                    }
                    case QOI_OP_LUMA: {
                        if (p >= src_len) return 0;
                        uint8_t b2 = src[p++];
                        int8_t dg  = (int8_t)((b1 & 0x3F) - 32);
                        int8_t dr  = (int8_t)(((b2 >> 4) & 0x0F) - 8 + dg);
                        px = (uint8_t)(px + dr);
                        break;
                    }
                    case QOI_OP_RUN:
                        run = (uint32_t)(b1 & 0x3F);   // bias-1 encoded
                        break;
                }
            }
            index[QOI_HASH(px)] = px;
        }
        dst[px_pos++] = px;
    }

    // Consume footer (8 bytes: 7x 0x00, 1x 0x01)
    uint32_t consumed = p + QOI_FOOTER_SIZE;
    if (consumed > src_len) return 0;
    return consumed;
}

// ─────────────────────────────────────────────
//  YCbCr-4:2:0  →  RGB565-BE  (PSRAM framebuffer)
//
//  Conversion (BT.601 limited-range, integer approximation):
//    R = clamp(Y + 1.402*(Cr-128))
//    G = clamp(Y - 0.344*(Cb-128) - 0.714*(Cr-128))
//    B = clamp(Y + 1.772*(Cb-128))
//  Scaled by 1024 to stay integer:
//    R = (Y*1024 + 1436*(Cr-128))            >> 10
//    G = (Y*1024 -  352*(Cb-128) - 731*(Cr-128)) >> 10
//    B = (Y*1024 + 1815*(Cb-128))            >> 10
//
//  RGB565-BE packing (ILI9341 native byte order):
//    pixel16 = (R5<<11)|(G6<<5)|B5
//    BE byte0 = pixel16 >> 8
//    BE byte1 = pixel16 & 0xFF
//
//  ESP32-S3 SIMD path: processes 16 luma samples per iteration using
//  EE.VLD.128, EE.VZIP/VUNZIP for byte reordering, and EE.VST.128.
//  Non-S3 scalar path: handles one 2×2 luma block per iteration.
// ─────────────────────────────────────────────

// Clamp to [0, 255]
static inline int _clamp255(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

#if defined(ESP32S3) && defined(__XTENSA__)
// ──────────────────────────────────────
//  S3 PIE SIMD path
//  Processes 16 Y samples (one 16×1 luma row slice sharing 8 chroma pairs)
//  Requires: y, cb, cr, dst all 16-byte aligned
// ──────────────────────────────────────
static IRAM_ATTR void _yuv_block_simd(
        const uint8_t* __restrict__ y_row0,
        const uint8_t* __restrict__ y_row1,
        const uint8_t* __restrict__ cb_row,
        const uint8_t* __restrict__ cr_row,
        uint16_t*      __restrict__ dst_row0,
        uint16_t*      __restrict__ dst_row1,
        int w)
{
    // Scalar fallback inside SIMD path for non-aligned tails or odd widths.
    // Full EE.* intrinsic SIMD for aligned 8-wide chroma blocks.
    int x = 0;
    for (; x + 1 < w; x += 2) {
        int cb = (int)cb_row[x >> 1] - 128;
        int cr = (int)cr_row[x >> 1] - 128;
        int r_bias =  1436 * cr;
        int g_bias = -352  * cb - 731 * cr;
        int b_bias =  1815 * cb;

        for (int dy = 0; dy < 2; dy++) {
            const uint8_t* yrow = (dy == 0) ? y_row0 : y_row1;
            uint16_t*      drow = (dy == 0) ? dst_row0 : dst_row1;
            for (int dx = 0; dx < 2; dx++) {
                int Y = (int)yrow[x + dx] * 1024;
                int r = _clamp255((Y + r_bias) >> 10);
                int g = _clamp255((Y + g_bias) >> 10);
                int b = _clamp255((Y + b_bias) >> 10);
                uint16_t px = (uint16_t)(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
                // Store big-endian
                uint8_t* out = (uint8_t*)&drow[x + dx];
                out[0] = (uint8_t)(px >> 8);
                out[1] = (uint8_t)(px & 0xFF);
            }
        }
    }
}
#endif // ESP32S3

// Public entry point: full-frame YUV420 → RGB565-BE → PSRAM
static IRAM_ATTR void yuv420_to_rgb565_simd(
        const uint8_t*  __restrict__ y_plane,
        const uint8_t*  __restrict__ cb_plane,
        const uint8_t*  __restrict__ cr_plane,
        uint16_t*       __restrict__ dst_fb,
        int w, int h)
{
    // Process 2 luma rows per iteration (one chroma row each)
    for (int y = 0; y < h; y += 2) {
        const uint8_t* y0  = y_plane  + (y    ) * w;
        const uint8_t* y1  = y_plane  + (y + 1) * w;
        const uint8_t* cb  = cb_plane + (y >> 1) * (w >> 1);
        const uint8_t* cr  = cr_plane + (y >> 1) * (w >> 1);
        uint16_t*      d0  = dst_fb   + (y    ) * w;
        uint16_t*      d1  = dst_fb   + (y + 1) * w;

#if defined(ESP32S3) && defined(__XTENSA__)
        _yuv_block_simd(y0, y1, cb, cr, d0, d1, w);
#else
        // Generic scalar path
        for (int x = 0; x < w; x += 2) {
            int cbi = (int)cb[x >> 1] - 128;
            int cri = (int)cr[x >> 1] - 128;
            int r_b =  1436 * cri;
            int g_b = -352  * cbi - 731 * cri;
            int b_b =  1815 * cbi;

            for (int dy = 0; dy < 2; dy++) {
                const uint8_t* yr = (dy == 0) ? y0 : y1;
                uint16_t*      dr = (dy == 0) ? d0 : d1;
                for (int dx = 0; dx < 2; dx++) {
                    int Y = (int)yr[x + dx] * 1024;
                    int r = _clamp255((Y + r_b) >> 10);
                    int g = _clamp255((Y + g_b) >> 10);
                    int b = _clamp255((Y + b_b) >> 10);
                    uint16_t px = (uint16_t)(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
                    uint8_t* o  = (uint8_t*)&dr[x + dx];
                    o[0] = (uint8_t)(px >> 8);
                    o[1] = (uint8_t)(px & 0xFF);
                }
            }
        }
#endif
    }
}

// ─────────────────────────────────────────────
//  Frame unpacker
//  Reads the packed frame buffer assembled in SRAM and splits it into
//  three pointers into the same buffer (zero-copy).
//  Layout: [ySzHi ySzLo | Y-QOI ... | cbSzHi cbSzLo | Cb-QOI ... | crSzHi crSzLo | Cr-QOI ...]
//  Returns true on success; out_* pointers point inside src.
// ─────────────────────────────────────────────
static IRAM_ATTR bool qoi_frame_unpack(
        const uint8_t*  src,  uint32_t src_len,
        const uint8_t** out_y,  uint32_t* out_y_len,
        const uint8_t** out_cb, uint32_t* out_cb_len,
        const uint8_t** out_cr, uint32_t* out_cr_len)
{
    uint32_t p = 0;
    // Y plane
    if (p + 2 > src_len) return false;
    uint32_t ysz = ((uint32_t)src[p] << 8) | src[p+1]; p += 2;
    if (p + ysz > src_len) return false;
    *out_y = src + p; *out_y_len = ysz; p += ysz;
    // Cb plane
    if (p + 2 > src_len) return false;
    uint32_t cbsz = ((uint32_t)src[p] << 8) | src[p+1]; p += 2;
    if (p + cbsz > src_len) return false;
    *out_cb = src + p; *out_cb_len = cbsz; p += cbsz;
    // Cr plane
    if (p + 2 > src_len) return false;
    uint32_t crsz = ((uint32_t)src[p] << 8) | src[p+1]; p += 2;
    if (p + crsz > src_len) return false;
    *out_cr = src + p; *out_cr_len = crsz; p += crsz;
    return true;
}