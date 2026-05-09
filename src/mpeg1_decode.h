#pragma once

#include "shared.h"

void initMpeg1Decoder();
bool decodeMpeg1Frame(const uint8_t* data, size_t len, uint16_t* outputBuffer, int width, int height, uint32_t& decodeUs);

// Forward declarations for hot-path functions (when library is integrated, mark with IRAM_ATTR):
// - decodeMcuBlock():      IDCT + entropy decode per 8x8 block
// - idctBlock():           IDCT computation (~80-160 calls/frame)
// - motionCompensate():    Motion vector prediction for P frames (~40-80 calls/frame)
// - yuvToRgb565Simd():     YUV→RGB565 color space conversion (pixel-intensive)
// All lookups (Huffman, quant, zigzag tables) must be in SRAM (DRAM_ATTR) for fast access.