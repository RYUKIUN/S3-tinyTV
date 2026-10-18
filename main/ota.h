#pragma once
/*
 * ota.h — wireless firmware update, pure ESP-IDF style.
 * ────────────────────────────────────────────────────────
 * New module, replacing ArduinoOTA (which came bundled with the Arduino
 * framework and is not available here). This is a small esp_http_server
 * endpoint that streams a raw .bin straight into esp_ota_ops — the same
 * app0/app1 OTA partition pair in partitions_16MB_ota.csv is reused
 * untouched, only the transport changed.
 *
 * Usage from the PC/your own machine once the device is on WiFi:
 *   curl -X POST --data-binary @.pio/build/.../firmware.bin \
 *        http://esp32s3-display.local:3232/update
 * (or http://<device-ip>:3232/update — hostname needs mDNS/.local
 * resolution support on your OS; the IP always works.)
 *
 * This intentionally does NOT try to reproduce PlatformIO's `espota`
 * wire protocol — that protocol is part of arduino-esp32's OTA library,
 * which isn't part of this project anymore. A plain HTTP PUT/POST is the
 * standard pure-ESP-IDF equivalent (see esp-idf's own
 * examples/system/ota/native_ota_example for the same pattern).
 */

void otaStart();
