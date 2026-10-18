# S3 Next-Gen JPEG Stream — pure ESP-IDF build

ESP32-S3 + ILI9341 wireless display stream. 320×240 full-frame JPEG over
chunked UDP, decoded in blocks and pushed to the panel via DMA. Touch
(XPT2046) reports back to the PC as mouse events.

This is a **rebase** of an earlier Arduino/PlatformIO version of the same
project onto **pure ESP-IDF** (native `idf.py`/VS Code ESP-IDF extension —
no PlatformIO, no arduino-esp32 component anywhere in the build). The JPEG
decoder was also swapped from a JPEGDEC fork to Espressif's `esp_new_jpeg`.
See "What changed in this rebase" below for the full list.

## Requirements

- **ESP-IDF v5.1+** (v5.3 recommended), installed via the
  [VS Code ESP-IDF extension](https://marketplace.visualstudio.com/items?itemName=espressif.esp-idf-extension)
  setup wizard, or manually per Espressif's Get Started guide.
- Target: `esp32s3`.
- Board assumption baked into `sdkconfig.defaults`: **N16R8** — 16 MB QIO
  flash + 8 MB Octal PSRAM. If your board differs, see "If your board
  isn't N16R8" below.
- Internet access on your FIRST build only — ESP-IDF's component manager
  needs to fetch `esp_new_jpeg` (from the official component registry) and
  `LovyanGFX` (from its GitHub repo) the first time you configure the
  project. After that everything is cached locally.

## Build & flash

Open this folder in VS Code with the ESP-IDF extension installed, select
`esp32s3` as the target (`ESP-IDF: Set Espressif Target`), then either use
the extension's build/flash/monitor buttons, or from a terminal with the
IDF environment sourced (`. $HOME/esp/esp-idf/export.sh` or similar):

```sh
idf.py set-target esp32s3
idf.py build
idf.py -p <PORT> flash monitor
```

The first `idf.py build` (or the extension's first "Build") is the one
that triggers the component-manager fetch mentioned above — expect it to
take noticeably longer than every build after it.

## Before you flash — edit `main/nexus.h`

Every setting you need to change for YOUR hardware/network is in
`main/nexus.h`, Zone 1, at the top of the file:

- `WIFI_SSID` / `WIFI_PASS` — your network.
- `LCD_PIN_*` / `TOUCH_PIN_*` — your panel's wiring, if it differs from the
  defaults documented there and in `HARDWARE.md`.
- `TOUCH_FEATURE_ENABLED` — set to `0` in `main/nexus.h` to disable touch
  setup, calibration, sampling, and touch event networking.
- `TOUCH_USE_IRQ` — leave at `0` unless PENIRQ is actually wired; see the
  comment there, this one is not just a performance knob.

Touch calibration is stored in NVS and reused after power cycles. The first
boot after erasing NVS (or changing the calibration format) requires a
calibration; double-press BOOT before streaming to recalibrate later.

Zone 2–3 (display geometry, pipeline depth, timing) have sane defaults for
a 320×240 panel and shouldn't need touching for the same hardware this was
built against.

## Firmware updates after the first flash (OTA)

`ArduinoOTA` doesn't exist outside the Arduino framework, so OTA here is a
small HTTP endpoint instead (`main/ota.cpp`) — same `app0`/`app1` OTA
partition pair, different transport:

```sh
idf.py build
curl -X POST --data-binary @build/s3_nextgen_jpeg.bin http://<device-ip>:3232/update
```

(`<device-ip>` — the IP printed on the panel's WiFi status line, or in the
serial monitor log. mDNS hostname resolution depends on your OS/network,
the plain IP always works.)

## What changed in this rebase

1. **Framework: Arduino → pure ESP-IDF.** No `arduino-esp32` component is
   compiled anywhere. The handful of things Arduino gave for free got
   direct ESP-IDF equivalents, each isolated to one file so the rest of
   the codebase didn't need to change line-by-line:
   - `millis()`/`micros()`/`delay()`/`constrain()` → `main/port.h` (tiny
     wrappers over `esp_timer`/FreeRTOS — not Arduino, just a few inline
     functions).
   - `WiFi.*` → `main/wifi.cpp` (native `esp_wifi`/`esp_netif` station).
   - `ArduinoOTA` → `main/ota.cpp` (HTTP + `esp_ota_ops`, see above).
   - `Preferences` → direct `nvs.h` calls in `main/touch.cpp` (same NVS
     blob format — a calibration saved by the old Arduino build still
     loads here).
   - `pinMode`/`attachInterrupt`/… → `driver/gpio.h` in `main/touch.cpp`.
   - `psramFound()` → `esp_psram_is_initialized()`.
   - Arduino-ESP32's `temperatureRead()` → the native temperature-sensor
     driver in `main/network.cpp`.
   - Project structure: `platformio.ini` → `CMakeLists.txt` (root) +
     `main/CMakeLists.txt` + `sdkconfig.defaults`; `board_build.*` INI
     settings → Kconfig symbols in `sdkconfig.defaults`.

2. **JPEG decoder: `esp_new_jpeg`, full-frame block mode.** The PC sends one
   JPEG per captured frame, capped at 30 KiB and sent with 1 ms gaps between
   chunks to reduce UDP bursts. The ESP assembles it in a PSRAM ring slot,
   then decodes 8/16-row blocks into a small aligned internal-RAM
   scratch buffer and copies each block directly into a free PSRAM framebuffer.

3. **Explicit buffer ownership.** Separate free queues manage JPEG slots and
   display framebuffers. A framebuffer remains owned by the display task
   until its asynchronous DMA transfer completes; there is no full-frame
   framebuffer copy or partial-tile ready-mask fallback.

4. **Wire format changed.** The 8-byte UDP header now carries frame ID,
   chunk index/count, flags, and total JPEG length. Restart both the PC sender
   and ESP firmware together; old tile-protocol senders are incompatible.

## If your board isn't N16R8

`sdkconfig.defaults` hardcodes 16 MB QIO flash + 8 MB Octal PSRAM. If
yours is a different flash/PSRAM size or mode (Quad PSRAM instead of
Octal, for instance), run `idf.py menuconfig` → `Serial flasher config`
and `Component config → ESP PSRAM` and adjust, then re-save to
`sdkconfig.defaults` if you want it to stick across a clean rebuild.

## Troubleshooting a fresh build

- **"could not resolve component 'X'"** — a Kconfig/component name shifted
  between IDF versions. `main/CMakeLists.txt` has a note on exactly this;
  add the missing name to its `REQUIRES` list.
- **"unknown kconfig symbol" warning** — same drift, in
  `sdkconfig.defaults` this time. Open `idf.py menuconfig`, search (`/`)
  for the setting by description, fix that one line.
- **LovyanGFX fails to fetch** — `main/idf_component.yml` pulls it via git
  from `https://github.com/lovyan03/LovyanGFX.git`; needs `git` on PATH
  and network access on first configure. Pin a different tag/branch there
  if you want a specific LovyanGFX version.

## Hardware & protocol docs

- `HARDWARE.md` — wiring and panel notes.
- `captureJpeg.py` + `requirements.txt` — the PC-side sender (unchanged by
  this rebase).
