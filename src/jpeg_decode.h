#pragma once

#include "nexus.h"

void initJpegDecoder();
// huffUs/idctUs are the Huffman-entropy-decode and IDCT portions of decodeUs,
// broken out for profiling (see JPEG_PROFILE in JPEGDEC.h). Both read back
// as 0 when JPEG_PROFILE is off — decodeUs is still valid either way.
bool decodeSlot(const DecodeMsg& msg, uint32_t& decodeUs, uint32_t& huffUs, uint32_t& idctUs);
