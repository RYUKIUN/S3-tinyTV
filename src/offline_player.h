#pragma once

#include "shared.h"

// Scan SPIFFS for .mjpeg files.
// Returns true if at least one file was found.
// Call once before runOfflinePlayer().
bool initOfflinePlayer();

// Blocking playback loop — never returns.
// Cycles through all .mjpeg files found by initOfflinePlayer().
// Must be called from Core-1 context (Arduino loop()).
void runOfflinePlayer();
