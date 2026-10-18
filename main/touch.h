#pragma once

#include "nexus.h"

// Loads calibration from NVS and, if none is stored (or the BOOT button was
// double-pressed), runs the interactive grid calibration on screen.
// Call from app_main() AFTER lcd.init() but BEFORE WiFi — the boot screen is
// already up at that point, and calibration is a one-time, blocking, purely
// local affair that has no business competing with the stream.
#if TOUCH_FEATURE_ENABLED
void touchBegin();

// Low-priority sampler. Sleeps on the PENIRQ interrupt when nobody is
// touching (if wired), so it costs literally zero CPU and zero SPI traffic
// at idle.
void touchTask(void*);
#endif
