#include "nexus.h"
#include "jpeg_decode.h"
#include "esp_jpeg_dec.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_cache.h"

static const char* TAG = "jpeg";
static jpeg_dec_handle_t s_jpegDec = nullptr;
static uint8_t* s_blockOut = nullptr;
static constexpr size_t BLOCK_OUT_CAPACITY = (size_t)SCREEN_W * 16 * 2;

bool initJpegDecoder() {
    jpeg_dec_config_t cfg = DEFAULT_JPEG_DEC_CONFIG();
    cfg.output_type = JPEG_PIXEL_FORMAT_RGB565_BE;
    cfg.rotate = JPEG_ROTATE_0D;
    cfg.block_enable = true;

    jpeg_error_t err = jpeg_dec_open(&cfg, &s_jpegDec);
    if (err != JPEG_ERR_OK || s_jpegDec == nullptr) {
        ESP_LOGE(TAG, "decoder open failed: %d", (int)err);
        s_jpegDec = nullptr;
        return false;
    }

    s_blockOut = (uint8_t*)heap_caps_aligned_alloc(
        16, BLOCK_OUT_CAPACITY, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (s_blockOut == nullptr) {
        ESP_LOGE(TAG, "cannot allocate %u-byte internal block buffer",
                 (unsigned)BLOCK_OUT_CAPACITY);
        jpeg_dec_close(s_jpegDec);
        s_jpegDec = nullptr;
        return false;
    }
    return true;
}

bool decodeJpegFrame(const DecodeMsg& msg, uint8_t framebufferIdx,
                     uint32_t& decodeUs) {
    decodeUs = 0;
    if (!s_jpegDec || !s_blockOut || msg.jpegSlot >= g_numJpegSlots ||
        framebufferIdx >= g_numDisplayBufs || frameFb[framebufferIdx] == nullptr) {
        return false;
    }

    JpegFrameSlot& frame = jpegFrames[msg.jpegSlot];
    if (frame.data == nullptr || (uintptr_t)frame.data % 16 != 0) return false;

    jpeg_dec_io_t io = {};
    io.inbuf = frame.data;
    io.inbuf_len = msg.len;
    io.outbuf = s_blockOut;
    jpeg_dec_header_info_t info = {};

    uint32_t t0 = micros();
    if (jpeg_dec_parse_header(s_jpegDec, &io, &info) != JPEG_ERR_OK ||
        info.width != SCREEN_W || info.height != SCREEN_H) {
        decodeUs = micros() - t0;
        return false;
    }

    int outputCapacity = 0;
    int processCount = 0;
    if (jpeg_dec_get_outbuf_len(s_jpegDec, &outputCapacity) != JPEG_ERR_OK ||
        jpeg_dec_get_process_count(s_jpegDec, &processCount) != JPEG_ERR_OK ||
        outputCapacity <= 0 || (size_t)outputCapacity > BLOCK_OUT_CAPACITY ||
        processCount <= 0) {
        decodeUs = micros() - t0;
        return false;
    }

    const size_t frameBytes = (size_t)SCREEN_W * SCREEN_H * 2;
    size_t outputOffset = 0;
    uint8_t* frameOut = (uint8_t*)frameFb[framebufferIdx];
    uint32_t decodeTotalUs = 0;

    for (int block = 0; block < processCount; ++block) {
        uint32_t decodeStart = micros();
        jpeg_error_t result = jpeg_dec_process(s_jpegDec, &io);
        decodeTotalUs += micros() - decodeStart;
        if (result != JPEG_ERR_OK || io.out_size <= 0 ||
            (size_t)io.out_size > frameBytes - outputOffset) {
            decodeUs = micros() - t0;
            return false;
        }

        memcpy(frameOut + outputOffset, s_blockOut, (size_t)io.out_size);
        outputOffset += (size_t)io.out_size;
    }

    // The memcpy above went through the S3's write-back data cache into PSRAM,
    // but the display DMA reads PSRAM directly and does NOT see dirty cache
    // lines. Without this, the newest rows of a frame can still be sitting in
    // the cache when the DMA starts, so it sends stale data from an older
    // frame -> torn/split picture whose position changes every frame. The
    // faster the SPI clock, the sooner the DMA reaches those rows, which is
    // why 80 MHz showed it far more than 40 MHz.
    // Address and size are 32-byte aligned (frameFb is 32-aligned, 153600 B).
    esp_err_t ce = esp_cache_msync(frameOut, frameBytes,
                                   ESP_CACHE_MSYNC_FLAG_DIR_C2M | ESP_CACHE_MSYNC_FLAG_TYPE_DATA);
    if (ce != ESP_OK) {
        static bool warned = false;
        if (!warned) { warned = true; ESP_LOGW(TAG, "cache writeback failed: %s", esp_err_to_name(ce)); }
    }

    decodeUs = decodeTotalUs;
    return outputOffset == frameBytes;
}