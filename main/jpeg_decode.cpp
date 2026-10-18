#include "nexus.h"
#include "jpeg_decode.h"
#include "esp_jpeg_dec.h"
#include "esp_heap_caps.h"

/*
 * jpeg_decode.cpp — esp_new_jpeg replaces the old JPEGDEC fork.
 * ────────────────────────────────────────────────────────────
 * Target config (matches the component's own published benchmark):
 *   320x240 whole-frame, JPEG_SUBSAMPLE_420, quality 60,
 *   JPEG_PIXEL_FORMAT_RGB565_LE  ->  66.28 fps on an ESP32-S3, one core.
 * Our tiles are 160x120 — a quarter of that benchmark frame — so each
 * decodeSlot() call has roughly 4x the headroom that number implies,
 * which is exactly the margin this pipeline spends on network jitter and
 * the fallback-to-last-good-frame behaviour in main.cpp's decodeTask.
 *
 * API SHAPE DIFFERENCE FROM THE OLD JPEGDEC CODE (this is the one real
 * architectural change this swap required):
 * JPEGDEC let us register an MCU callback that wrote straight into
 * frameFb[writeSet] at the tile's own (x,y) offset, with an arbitrary
 * row stride (SCREEN_W) — so decode and "place the tile in the frame"
 * were the same memory write, zero extra copies.
 *
 * esp_new_jpeg's public API is open() -> parse_header() -> process() ->
 * close(): one call decodes the WHOLE image into ONE tightly-packed
 * output buffer (stride == image width, not SCREEN_W). There is no
 * per-row/per-MCU callback. So this tile's decode necessarily lands in a
 * packed TILE_W-wide scratch buffer first.
 *
 * Getting the tile into frameFb at the right offset still needs a copy —
 * but we were never going to avoid that stride mismatch with this
 * decoder's API, so instead of doing a pointless straight memcpy we use
 * that same pass to do the one other thing this swap owes us: LE->BE
 * byte swap. esp_new_jpeg's benchmarked fast path outputs
 * RGB565_LE; the ILI9341 wants RGB565_BE on the wire (DMA pushes frameFb
 * bytes to the panel completely raw, no swap anywhere else in the
 * pipeline — see display.cpp). Folding the swap into the unavoidable
 * copy costs nothing extra: one pass over the tile, same as before.
 */

static jpeg_dec_handle_t s_jpegDec = nullptr;

// Packed TILE_W x TILE_H RGB565_LE scratch output. 16-byte aligned per
// esp_new_jpeg's ESP32-S3 requirement (misaligned output buffer = visibly
// shifted columns, called out explicitly in the component's own FAQ).
static uint8_t* s_tileOut = nullptr;
static const size_t TILE_OUT_SIZE = (size_t)TILE_W * TILE_H * 2;

void initJpegDecoder() {
    jpeg_dec_config_t cfg = DEFAULT_JPEG_DEC_CONFIG();
    cfg.output_type  = JPEG_PIXEL_FORMAT_RGB565_LE;
    cfg.rotate       = JPEG_ROTATE_0D;
    cfg.block_enable = false;   // one-shot whole-tile decode, not block/streamed mode
    // scale / clipper left at {0,0} — disabled, decode at native 160x120.

    if (jpeg_dec_open(&cfg, &s_jpegDec) != JPEG_ERR_OK) {
        s_jpegDec = nullptr;
        return;
    }

    s_tileOut = (uint8_t*)heap_caps_aligned_alloc(
        16, TILE_OUT_SIZE, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
}

bool decodeSlot(const DecodeMsg& msg, uint32_t& decodeUs) {
    if (!s_jpegDec || !s_tileOut) {
        decodeUs = 0;
        return false;
    }

    PipeSlot& s = slot[msg.slotIdx];

    if ((uintptr_t)s.assembly & 15) {
        decodeUs = 0;
        return false;
    }
    if (msg.tId >= NUM_TILES || frameFb[writeSet] == nullptr) {
        decodeUs = 0;
        return false;
    }

    jpeg_dec_io_t io = {};
    io.inbuf     = s.assembly;
    io.inbuf_len = msg.len;
    io.outbuf    = s_tileOut;

    jpeg_dec_header_info_t info = {};

    uint32_t t0 = micros();

    if (jpeg_dec_parse_header(s_jpegDec, &io, &info) != JPEG_ERR_OK) {
        decodeUs = 0;
        return false;
    }
    // Sanity: every tile this project ever sends is TILE_W x TILE_H. A
    // mismatch here means a corrupt/foreign JPEG slipped past
    // assembleTileInto()'s SOI/EOI check in network.cpp — treat it the
    // same as any other decode failure rather than decoding into a
    // buffer sized for the wrong image.
    if (info.width != TILE_W || info.height != TILE_H) {
        decodeUs = 0;
        return false;
    }

    if (jpeg_dec_process(s_jpegDec, &io) != JPEG_ERR_OK) {
        decodeUs = 0;
        return false;
    }

    decodeUs = micros() - t0;

    // Destination is the tile's own offset inside the PSRAM frame buffer.
    // writeSet does not change for the duration of this call (Core-1
    // exclusive), so it's safe to read once up front — same assumption
    // the old mcuCallback-based code relied on.
    uint16_t*       dst0 = frameFb[writeSet] + TILE_Y[msg.tId] * SCREEN_W + TILE_X[msg.tId];
    const uint16_t* src  = (const uint16_t*)s_tileOut;

    for (int r = 0; r < TILE_H; r++) {
        uint16_t*       d = dst0 + (size_t)r * SCREEN_W;
        const uint16_t* srow = src + (size_t)r * TILE_W;
        for (int c = 0; c < TILE_W; c++) {
            uint16_t v = srow[c];
            d[c] = (uint16_t)((v << 8) | (v >> 8));   // RGB565 LE -> BE
        }
    }

    return true;
}
