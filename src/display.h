#pragma once

#include "shared.h"

void initDisplay();
void drawBootHeader();
void statusLine(uint8_t row, const char* label, const char* value, uint32_t col = TFT_WHITE);
void displayTask(void*);
