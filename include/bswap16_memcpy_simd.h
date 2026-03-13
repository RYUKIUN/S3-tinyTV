#pragma once
/*
 * bswap16_memcpy_simd.h  —  copy uint16_t buffer while byte-swapping every element
 *
 * Purpose: convert JPEGDEC's RGB565 little-endian output (in SRAM decodeTemp)
 *          to the big-endian format expected by the ILI9341, writing directly
 *          into the PSRAM framebuffer in one pass.
 *
 *  Replaces the previous two-step approach:
 *    OLD: bswap16_simd(decodeTemp, N)               ← full 38 KB SRAM read+write
 *         memcpy(tileFb, decodeTemp, N*2)            ← full 38 KB SRAM read + PSRAM write
 *    NEW: bswap16_memcpy_simd(tileFb, decodeTemp, N) ← one 38 KB SRAM read + PSRAM write
 *
 *  Savings: eliminates one complete 38 KB read of decodeTemp.
 *
 * ── ESP32-S3 PIE path ────────────────────────────────────────────────────
 *  Processes 32 pixels (64 bytes) per iteration using four 128-bit Q-registers:
 *
 *    EE.VLD.128.IP  q0, src, 16    load pixels  0- 7  (LE), src += 16
 *    EE.VLD.128.IP  q1, src, 16    load pixels  8-15  (LE), src += 16
 *    EE.VLD.128.IP  q2, src, 16    load pixels 16-23  (LE), src += 16
 *    EE.VLD.128.IP  q3, src, 16    load pixels 24-31  (LE), src += 16
 *
 *    EE.VUNZIP.8    q0, q1         Q0 → low  bytes of px 0-15
 *                                  Q1 → high bytes of px 0-15
 *    EE.VUNZIP.8    q2, q3         Q2 → low  bytes of px 16-31
 *                                  Q3 → high bytes of px 16-31
 *
 *    EE.VZIP.8      q1, q0         Q1 → [Hi,Lo] BE px  0- 7
 *                                  Q0 → [Hi,Lo] BE px  8-15
 *    EE.VZIP.8      q3, q2         Q3 → [Hi,Lo] BE px 16-23
 *                                  Q2 → [Hi,Lo] BE px 24-31
 *
 *    EE.VST.128.IP  q1, dst, 16    store px  0- 7, dst += 16
 *    EE.VST.128.IP  q0, dst, 16    store px  8-15, dst += 16
 *    EE.VST.128.IP  q3, dst, 16    store px 16-23, dst += 16
 *    EE.VST.128.IP  q2, dst, 16    store px 24-31, dst += 16
 *
 *  Throughput: 32 pixels per ~10 cycles in L1 SRAM.
 *              TILE_PIXELS = 19 200 → 600 iterations, zero scalar tail.
 *
 *  Correctness proof for pixel pair [Lo Hi]:
 *    After VUNZIP.8: even-index bytes → Lo register, odd-index → Hi register
 *    After VZIP.8:   result = [Hi[0],Lo[0], Hi[1],Lo[1], …]  ← correct BE word
 *
 *  Requirements:
 *    • src and dst must be 16-byte aligned
 *      (guaranteed by heap_caps_aligned_alloc(16,…) for both decodeTemp and tileFb)
 *    • n must be a multiple of 32 for the SRAM-only fast path
 *      (TILE_PIXELS = 19 200 = 600 × 32 → tail never runs in normal operation)
 *    • CONFIG_IDF_TARGET_ESP32S3 defined (automatic via ESP-IDF / Arduino ESP32 for S3)
 *    • GCC target: xtensa-esp32s3-elf
 *
 * ── Non-S3 scalar fallback ────────────────────────────────────────────────
 *  Two uint16_t per 32-bit word.  GCC -O3 may auto-vectorize on x86 (SSE2).
 *  Included so the TU compiles cleanly on any target (CI, ESP32-P4, …).
 */

#include <stdint.h>
#include "esp_attr.h"   // IRAM_ATTR

// ─────────────────────────────────────────────────────────────────────────────
//  ESP32-S3 — explicit PIE inline assembly, 32 pixels per iteration
// ─────────────────────────────────────────────────────────────────────────────
#ifdef CONFIG_IDF_TARGET_ESP32S3

static IRAM_ATTR
void bswap16_memcpy_simd(uint16_t* __restrict__ dst,
                          const uint16_t* __restrict__ src, int n)
{
    /*
     * 32-pixel (64-byte) loop using 4 Q-registers.
     * src and dst are advanced by the EE.VLD/VST post-increment immediates (±16).
     * Declared "+r" so GCC tracks the pointer changes across asm boundaries.
     */
    uint8_t*       d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    int blocks = (unsigned)n >> 5;   // floor(n / 32)

    while (blocks--) {
        __asm__ __volatile__ (
            // ── Load 32 LE pixels from SRAM (src post-incremented) ─────────
            "EE.VLD.128.IP  q0, %[s], 16  \n"   // px  0- 7
            "EE.VLD.128.IP  q1, %[s], 16  \n"   // px  8-15
            "EE.VLD.128.IP  q2, %[s], 16  \n"   // px 16-23
            "EE.VLD.128.IP  q3, %[s], 16  \n"   // px 24-31

            // ── Separate even/odd bytes (Lo bytes → even reg, Hi → odd) ───
            "EE.VUNZIP.8    q0, q1         \n"   // q0=Lo[0-15], q1=Hi[0-15]
            "EE.VUNZIP.8    q2, q3         \n"   // q2=Lo[16-31], q3=Hi[16-31]

            // ── Re-interleave Hi-first → big-endian pairs ──────────────────
            "EE.VZIP.8      q1, q0         \n"   // q1=BE px 0-7, q0=BE px 8-15
            "EE.VZIP.8      q3, q2         \n"   // q3=BE px 16-23, q2=BE px 24-31

            // ── Store to PSRAM (dst post-incremented) ──────────────────────
            "EE.VST.128.IP  q1, %[d], 16  \n"
            "EE.VST.128.IP  q0, %[d], 16  \n"
            "EE.VST.128.IP  q3, %[d], 16  \n"
            "EE.VST.128.IP  q2, %[d], 16  \n"

            : [s] "+r"(s), [d] "+r"(d)
            :
            : "memory"
        );
    }

    // ── Scalar tail: handles n not a multiple of 32 ──────────────────────────
    // For TILE_PIXELS = 19 200 this never runs; present for correctness.
    int rem = n & 31;
    const uint32_t* ts = (const uint32_t*)s;
    uint32_t*       td = (uint32_t*)d;
    int words = rem >> 1;
    for (int i = 0; i < words; i++) {
        uint32_t v = ts[i];
        td[i] = ((v & 0xFF00FF00u) >> 8u) | ((v & 0x00FF00FFu) << 8u);
    }
    if (rem & 1) {
        const uint16_t* ls = (const uint16_t*)(ts + words);
        uint16_t*       ld = (uint16_t*)(td  + words);
        *ld = __builtin_bswap16(*ls);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Non-S3 scalar fallback  (x86 CI / ESP32 classic / ESP32-P4 …)
// ─────────────────────────────────────────────────────────────────────────────
#else

static inline __attribute__((optimize("O3")))
void bswap16_memcpy_simd(uint16_t* __restrict__ dst,
                          const uint16_t* __restrict__ src, int n)
{
    const uint32_t* s = (const uint32_t*)src;
    uint32_t*       d = (uint32_t*)dst;
    int w = n >> 1;
    for (int i = 0; i < w; i++) {
        uint32_t v = s[i];
        d[i] = ((v & 0xFF00FF00u) >> 8u) | ((v & 0x00FF00FFu) << 8u);
    }
    if (n & 1) dst[n - 1] = __builtin_bswap16(src[n - 1]);
}

#endif  // CONFIG_IDF_TARGET_ESP32S3