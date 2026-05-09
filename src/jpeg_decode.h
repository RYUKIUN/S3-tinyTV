#pragma once

#include "shared.h"

void initJpegDecoder();
bool decodeSlot(const DecodeMsg& msg, uint32_t& decodeUs);
