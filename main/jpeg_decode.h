#pragma once

#include "nexus.h"

bool initJpegDecoder();
bool decodeJpegFrame(const DecodeMsg& msg, uint8_t framebufferIdx,
                     uint32_t& decodeUs);
