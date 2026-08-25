#pragma once

#include "nexus.h"

// Loads calibration from NVS and, if none is stored (or the BOOT button was
// double-pressed), runs the interactive 4-corner calibration on screen.
// Call from setup() AFTER lcd.init() but BEFORE WiFi — the boot screen is
// already up at that point, and calibration is a one-time, blocking, purely
// local affair that has no business competing with the stream.
void touchBegin();

// Low-priority sampler. Sleeps on the PENIRQ interrupt when nobody is
// touching, so it costs literally zero CPU and zero SPI traffic at idle.
void touchTask(void*);
