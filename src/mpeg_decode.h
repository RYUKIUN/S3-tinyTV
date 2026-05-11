#pragma once

#include <SPIFFS.h>
#include <JPEGDEC.h>
#include "shared.h"

// ESV2 file format header (little-endian)
struct ESV2Header {
    uint16_t width;
    uint16_t height;
    uint16_t fps;
    uint16_t block_size;
    uint32_t frame_count;
};

// Initialize ESV2 decoder (creates internal JPEGDEC, SRAM scratch buffers).
// Call once before decoding any frames.
// Returns true if initialization successful.
bool initESV2Decoder();

// Decode one frame from ESV2 file.
// frameBuffer: output RGB565 buffer (width*height*2 bytes)
// prevBuffer: previous frame for P-frames (must differ from frameBuffer)
// Returns true if frame decoded successfully
bool decodeESV2Frame(File& f, uint16_t* frameBuffer, uint16_t* prevBuffer);

// Initialize ESV2 playback from file.
// Returns header info, or empty if invalid.
ESV2Header initESV2Player(const char* path);

// Cleanup ESV2 decoder (frees internal buffers).
void cleanupESV2Decoder();