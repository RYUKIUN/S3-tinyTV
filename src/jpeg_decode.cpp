#include "shared.h"
#include "jpeg_decode.h"
#include "bswap16_memcpy_simd.h"

static JPEGDEC jpeg_dec;

struct McuCtx { uint16_t* fb; };
static McuCtx mcuCtx;

static IRAM_ATTR int mcuCallback(JPEGDRAW* pDraw) {
    uint16_t*       dst = ((McuCtx*)pDraw->pUser)->fb + pDraw->y * TILE_W + pDraw->x;
    const uint16_t* src = (const uint16_t*)pDraw->pPixels;
    int w = pDraw->iWidth, h = pDraw->iHeight;
    for (int r = 0; r < h; r++)
        memcpy(dst + r * TILE_W, src + r * w, (size_t)w * 2);
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

    mcuCtx.fb = decodeTemp;
    if (!jpeg_dec.openRAM(s.assembly, msg.len, mcuCallback)) {
        decodeUs = 0;
        return false;
    }

    jpeg_dec.setPixelType(RGB565_LITTLE_ENDIAN);
    jpeg_dec.setUserPointer(&mcuCtx);

    uint32_t t0 = micros();
    int rc = jpeg_dec.decode(0, 0, 0);
    jpeg_dec.close();

    if (!rc) {
        decodeUs = 0;
        return false;
    }

    uint16_t* fbBase = frameFb[writeSet]
                     + TILE_Y[msg.tId] * SCREEN_W
                     + TILE_X[msg.tId];
    for (int row = 0; row < TILE_H; row++) {
        bswap16_memcpy_simd(fbBase + row * SCREEN_W,
                            decodeTemp + row * TILE_W,
                            TILE_W);
    }

    decodeUs = micros() - t0;
    return true;
}
