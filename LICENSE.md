# v4.0.0

A platform rebase. The firmware moves from Arduino/PlatformIO to **pure ESP-IDF**, the JPEG decoder moves from a JPEGDEC fork to Espressif's **`esp_new_jpeg`**, and the stream moves from four JPEG tiles to **one full-frame JPEG per frame**. Behaviour from the user's point of view is the same — a PC monitor mirrored to a 320×240 ILI9341 with touch-as-mouse — but the build, the OTA flow, the UDP protocol and the sender's controls have all changed, so v3 and v4 must not be mixed.

## Highlights

- **Pure ESP-IDF build.** No `arduino-esp32`, no PlatformIO. Build with `idf.py` or the VS Code ESP-IDF extension. The handful of things Arduino provided now have direct ESP-IDF equivalents, each isolated to one file:
  - `millis()` / `micros()` / `delay()` / `constrain()` → `main/port.h` (thin wrappers over `esp_timer` and FreeRTOS)
  - `WiFi.*` → `main/wifi.cpp` (native `esp_wifi` / `esp_netif` station, plus `nvsInit()`)
  - `ArduinoOTA` → `main/ota.cpp` (HTTP endpoint on top of `esp_ota_ops`)
  - `Preferences` → direct `nvs.h` calls in `main/touch.cpp`
  - `pinMode` / `attachInterrupt` → `driver/gpio.h`
  - `psramFound()` → `esp_psram_is_initialized()`
  - `temperatureRead()` → the native temperature-sensor driver
  - `platformio.ini` → `CMakeLists.txt` + `main/CMakeLists.txt` + `sdkconfig.defaults`
- **`esp_new_jpeg` block-mode decoding.** One JPEG per frame (capped at 35 KiB) is assembled in a JPEG slot, then decoded in 16-row blocks into a small aligned internal-RAM scratch buffer and copied directly into a free PSRAM framebuffer. The decoder checks that the image is exactly 320×240.
- **Full-frame transport replaces tiles.** The four-tile pipeline, per-tile ready masks and the "seed the buffer with the last good frame" workaround are gone. A frame that is incomplete, malformed or fails to decode is discarded whole, and the panel keeps showing the last good frame.
- **Explicit buffer ownership.** Two free queues (JPEG slots and display framebuffers) own every buffer. A framebuffer belongs to the display task until its asynchronous DMA finishes, so there is no whole-frame copy and no reuse-while-in-flight.
- **Sender rework.** Fixed 50 FPS send rate, a 25,000-byte frame budget with a quality ceiling of 80, immediate and escalating quality drops on overflow, an EMA-smoothed cooldown before quality climbs back, and 250 µs chunk pacing.
- **New debug overlays.** `Debug Mode` 1 shows ESP-side stats (FPS, temperature, jitter, decode time, drops, aborts, CPU0/CPU1, SRAM/PSRAM); `Debug Mode` 2 shows PC encoder stats (encode/send time, frame budget, pacing, overflow and cooldown streaks, quality fallback).

## Breaking changes

- **Build system.** `platformio.ini`, `src/`, `lib/JPEGDEC` and `.pio` outputs are gone. Sources live in `main/`; dependencies are fetched by the ESP-IDF component manager on the first build (`espressif/esp_new_jpeg` ^1.0.2 from the registry, LovyanGFX 1.2.33 from GitHub), so the first build needs internet and `git`.
- **UDP wire format.** The video header is now `AA BB <frameId> <chunkId> <nChunks> <flags=0> <sizeHi> <sizeLo>` — frame ID, chunk index/count, flags and total JPEG length. Tile-protocol senders are incompatible. **Update the PC sender and the ESP firmware together.**
- **OTA.** PlatformIO `espota` / ArduinoOTA is replaced by an HTTP endpoint: `curl -X POST --data-binary @build/s3_nextgen_jpeg.bin http://<device-ip>:3232/update` (port set by `OTA_HTTP_PORT`). The first v4.0 flash must be done over USB.
- **Sender controls.** `Priority` and `Show Stats` are removed. The controls are now `Sharpen` (0–20), `Debug Mode` (0/1/2), `Enable Touch` (0/1) and `Monitor`. `settings.json` stores `Sharpen`, `Debug Mode` and `Enable Touch`; the monitor choice is not persisted.
- **Configuration.** Everything still lives in `main/nexus.h`, but it now also carries `WIFI_TX_POWER`, `OTA_HTTP_PORT`, `TOUCH_FEATURE_ENABLED` and the full-frame sizing macros (`MAX_JPEG_FRAME`, `MAX_FRAME_CHUNKS`, `CFG_NUM_JPEG_SLOTS`). Board-level settings that used to be `board_build.*` entries are Kconfig symbols in `sdkconfig.defaults`.

## Changed

- Display SPI write clock is now 75 MHz (`LCD_WRITE_HZ`); the previous documentation listed 40 MHz.
- Wi‑Fi runs 802.11n at 20 MHz bandwidth (HT20); the previous documentation listed HT40. Power save is off and TX power is set from `WIFI_TX_POWER` in nexus.h
- `displayTask` priority is now 3 (was 2) and the OTA server runs as `esp_http_server`'s own task instead of a dedicated `otaTask`.
- JPEG slots: three full-frame slots, the first two preferring internal SRAM with PSRAM fallback, the rest in PSRAM. Display framebuffers: three, in PSRAM. Allocation falls back to fewer buffers (down to 1 JPEG slot / 2 framebuffers) and reports the result on the boot screen.
- Network receive path hardened: a packet is accepted only if chunk count, chunk index, chunk length and total size are mutually consistent and the flags are zero; completed frames must start with `FF D8` and end with `FF D9`; frames not completed within 80 ms are dropped.
- Wi‑Fi buffering tuned in `sdkconfig.defaults` (software coexistence, 16 static / 32 dynamic RX buffers, block-ack window 16, UDP receive mailbox of 32).
- The touchscreen is now optional at compile time (`TOUCH_FEATURE_ENABLED`); with it at `0` the touch driver, calibration, sampling task and touch networking are all excluded.
- The README, HARDWARE.md and PROJECT_STRUCTURE_AND_EXPLANATION.md were rewritten for the ESP-IDF layout and the full-frame protocol.

## Kept from v3

- Dual-core split: network, display DMA, Wi‑Fi watchdog and touch on Core 0, JPEG decoding on Core 1.
- XPT2046 touch with the 4×4 bilinear calibration, the shared-SPI mutex with priority inheritance, and the gestures (tap = click, slide = scroll, hold = drag) with cursor borrow-and-return on the PC.
- A touch calibration saved by the v3 firmware still loads (same NVS namespace, key and format).
- Auto-discovery via the `S3READY` beacon and one preview window per ESP.

## Removed

- PlatformIO project files, the Arduino framework, ArduinoOTA and the JPEGDEC library.
- The four-tile decode path and its partial-frame fallback logic.
- The `Priority` preset slider and `Show Stats` toggle in the sender.

## Notes

- Replace the Wi‑Fi credentials in `main/nexus.h` with your own before flashing, and avoid committing them.
- `scripts/upload_ota.ps1` is an empty placeholder; use the `curl` command above for OTA.
- No license file is included in the repository yet.

