#pragma once

#include "nexus.h"
#include <LovyanGFX.hpp>

// ── Display layer: esp_lcd (panel IO + ILI9341) ─────────────────────────────
// Pixels reach the panel through esp_lcd's SPI (or i80) panel IO with a
// completion callback — displayTask SLEEPS on a semaphore while the DMA runs
// instead of polling. All pin/clock config still comes from nexus.h.
//
// LovyanGFX is kept ONLY as a software renderer for the boot/status/overlay/
// calibration text: `lcd` is a bus-less sprite that draws into a RAM canvas,
// and the canvas is pushed with displayFlush*(). Drawing into `lcd` shows
// nothing until you flush.
extern lgfx::LGFX_Sprite lcd;

bool initDisplay();          // esp_lcd bus + panel + canvas; false on failure
void displayTuneBus();       // pad drive strength; call after initDisplay()
void drawBootHeader();
void statusLine(uint8_t row, const char* label, const char* value, uint32_t col = TFT_WHITE);

// Canvas -> panel (blocking). Take the UI lock around draw+flush if several
// tasks may draw; the lock is recursive so a caller may hold it across calls.
void displayFlush();                          // whole canvas
void displayFlushRows(int y0, int rows);      // full-width band of the canvas
void displayUiLock();
void displayUiUnlock();

void displayTask(void*);
