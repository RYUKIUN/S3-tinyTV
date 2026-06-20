#pragma once

#include "shared.h"

void initJpegDecoder();

// mcuCalls is optional diagnostic output: if non-null, receives the number of
// times JPEGDEC invoked mcuCallback for this tile (i.e. MCU-row granularity
// of the PSRAM writes). Pass nullptr (the default) to skip — existing call
// sites need no changes.
bool decodeSlot(const DecodeMsg& msg, uint32_t& decodeUs, uint16_t* mcuCalls = nullptr);