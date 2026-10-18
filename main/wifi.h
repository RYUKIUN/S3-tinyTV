#pragma once
/*
 * wifi.h — native esp_wifi/esp_netif station management.
 * ────────────────────────────────────────────────────────
 * New module. Previously the Arduino WiFi class (WiFi.begin/.status/
 * .localIP/.RSSI/.disconnect) handled all of this; pure ESP-IDF doesn't
 * have that wrapper, so this is the direct esp_wifi/esp_netif equivalent,
 * kept in its own file so main.cpp/network.cpp don't need to know the
 * difference — they still just read g_wifiOk / g_wifiConnectedMs from
 * nexus.h and call the functions below.
 */

#include <stdint.h>
#include <stdbool.h>

// Initializes NVS storage. Call before touchBegin() so the calibration can
// be loaded before WiFi starts.
void nvsInit();

// Brings up the netif + default event loop and WiFi driver, then starts
// association. NVS must already have been initialized with nvsInit().
// g_wifiOk / g_wifiConnectedMs (nexus.h) are updated asynchronously by
// the internal event handler from here on — call once, early in
// app_main(). Connecting itself is NOT blocking; app_main() polls
// g_wifiOk in a loop exactly like the old WiFi.status() polling did.
void wifiInit();

// Fire-and-forget reconnect attempt — used by wifiWatchdogTask in
// network.cpp when the link drops after streaming has already started,
// and by app_main()'s own initial-connect retry loop.
void wifiReconnect();

// "a.b.c.d" formatted into caller's buffer. Empty string if not connected.
void wifiGetIPString(char* buf, size_t len);

// Current RSSI in dBm, or 0 if not connected.
int32_t wifiGetRSSI();
