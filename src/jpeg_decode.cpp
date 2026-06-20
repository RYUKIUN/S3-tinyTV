#include "shared.h"
#include "jpeg_decode.h"

static JPEGDEC jpeg_dec;

// McuCtx now points straight at the tile's offset inside the PSRAM
// framebuffer. `stride` is SCREEN_W (destination rows are NOT contiguous —
// each tile is a sub-rectangle of the full frame), as opposed to the old
// decodeTemp scratch buffer where stride == TILE_W.
struct McuCtx {
    uint16_t* fb;     // base = frameFb[writeSet] + TILE_Y*SCREEN_W + TILE_X
    int       stride;
};
static McuCtx mcuCtx;

// Writes JPEGDEC's MCU output directly into the PSRAM framebuffer.
// No SRAM scratch, no separate byte-swap pass: JPEGDEC is told to emit
// RGB565_BIG_ENDIAN (see decodeSlot), which is exactly what the ILI9341
// wants over SPI, so this is a straight per-row memcpy.
//
// iWidth/iHeight can be a partial MCU at tile edges (subsampling-dependent),
// so this must NOT assume full TILE_W-wide, 32-byte-aligned rows the way the
// old bswap16_memcpy_simd routine did — plain memcpy has no such constraint.
static IRAM_ATTR int mcuCallback(JPEGDRAW* pDraw) {
    McuCtx*         ctx = (McuCtx*)pDraw->pUser;
    uint16_t*       dst = ctx->fb + pDraw->y * ctx->stride + pDraw->x;
    const uint16_t* src = (const uint16_t*)pDraw->pPixels;
    int w = pDraw->iWidth, h = pDraw->iHeight;
    for (int r = 0; r < h; r++)
        memcpy(dst + r * ctx->stride, src + r * w, (size_t)w * 2);
    return 1;
}

void initJpegDecoder() {
    // Currently no decoder-specific init required.
}

bool decodeSlot(const DecodeMsg& msg, uint32_t& decodeUs) {
    PipeSlot& s = slot[msg.slotIdx];

    if ((uintptr_t)s.assembly & 15) {
        decodeUs = 0;
        return false;
    }
    if (msg.tId >= NUM_TILES || frameFb[writeSet] == nullptr) {
        decodeUs = 0;
        return false;
    }

    // Destination is the tile's own offset inside the PSRAM frame buffer —
    // no SRAM scratch in between. writeSet does not change for the duration
    // of this call (Core-1 exclusive), so it's safe to read once up front.
    mcuCtx.fb     = frameFb[writeSet] + TILE_Y[msg.tId] * SCREEN_W + TILE_X[msg.tId];
    mcuCtx.stride = SCREEN_W;

    if (!jpeg_dec.openRAM(s.assembly, msg.len, mcuCallback)) {
        decodeUs = 0;
        return false;
    }

    // BIG_ENDIAN matches what the ILI9341 wants natively over SPI, so
    // JPEGDEC's own (SIMD-accelerated) color-conversion stage produces
    // bytes in final display order — no separate swap pass needed.
    jpeg_dec.setPixelType(RGB565_BIG_ENDIAN);
    jpeg_dec.setUserPointer(&mcuCtx);

    uint32_t t0 = micros();
    int rc = jpeg_dec.decode(0, 0, 0);
    jpeg_dec.close();

    if (!rc) {
        decodeUs = 0;
        return false;
    }

    decodeUs = micros() - t0;
    return true;
}