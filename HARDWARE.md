# Hardware & Pin Wiring — S3-nextgen jpeg

Snapshot of the current hardware configuration as of the end of this
development session. Source of truth for all values below is the code
itself (`platformio.ini`, `src/display.h`, `src/shared.h`) — if these ever
diverge, trust the code.

## Board

| Item | Value |
|---|---|
| MCU | ESP32-S3 (Xtensa LX7, dual-core) |
| Board definition | `esp32-s3-devkitc-1` (PlatformIO) |
| Flash | 16 MB, QIO |
| PSRAM | 8 MB, OPI (`N16R8` variant) — **`board_build.arduino.memory_type = qio_opi` is required**; without it PSRAM is detected but not mapped into the heap |
| CPU frequency | 240 MHz (`board_build.f_cpu`) |
| Partition table | `partitions_16MB_ota.csv` (OTA-capable) |
| Filesystem | SPIFFS (`board_build.filesystem = spiffs`) — currently unused after the offline-playback feature was removed; only relevant if OTA filesystem updates are used |
| USB | Native USB (Full-Speed OTG, not High-Speed) used for `Serial` via USB CDC (`ARDUINO_USB_CDC_ON_BOOT=1`, `ARDUINO_USB_MODE=1`) — not currently used for data transfer, only console/programming |
| Upload | OTA (`espota`) at `esp32s3-display.local`; wired `esptool` upload available by toggling `upload_protocol` in `platformio.ini` |

## Display — ILI9341, 320×240, SPI

Driven via LovyanGFX (`LGFX` class in `src/display.h`), **not** Adafruit_GFX.

| Signal | GPIO | Notes |
|---|---|---|
| SCLK | 12 | shared SPI clock |
| MOSI | 13 | shared SPI data out |
| MISO | -1 (not connected) | display is write-only; nothing reads back from the panel |
| CS | 10 | ILI9341 chip select |
| DC | 4 | data/command |
| RST | 5 | reset |
| BUSY | -1 (not connected) | not applicable to this panel |

- SPI host: `SPI2_HOST` (HSPI)
- Write clock: 75 MHz (`cfg.freq_write`)
- Panel native size: 240×320 (`panel_width`/`panel_height`); software rotation 3 → logical 320×240 landscape
- `dummy_read_pixel = 8`, `readable = false`, `bus_shared = true` (bus left shareable — was set up anticipating a touch controller on the same bus; see below)
- Color depth: 16-bit (RGB565), `RGB565_BIG_ENDIAN` used throughout the decode pipeline to match the panel's native SPI byte order

No other GPIOs are used anywhere in the active `src/` codebase (network, decode, display, OTA) — confirmed by search at the time of writing.

## Touch — XPT2046 (stashed, not yet integrated)

Pin plan carried over from the separate `test unit-S3` reference project.
**Not wired into the main firmware yet** — this is the plan from the
stashed touch-integration task, recorded here for whenever that work
resumes.

| Signal | GPIO | Notes |
|---|---|---|
| TOUCH_CS | 9 | XPT2046 chip select |
| TOUCH_IRQ | 8 | touch interrupt (plan: use this to gate polling — only read the controller when it signals a touch, rather than blind-polling) |
| TOUCH_MISO | 11 | shared bus MISO — the display's own MISO is unused (-1), so this would need enabling on the shared `SPI2_HOST` bus config in `display.h`, or moving to an isolated bus |

Decision reached during planning: share the existing `SPI2_HOST` bus
(SCLK 12 / MOSI 13, display CS 10, touch CS 9) rather than isolate touch
onto a separate SPI host — bus lock (`use_lock = true`) already
serializes access safely, and touch polling is infrequent/small relative
to the display's periodic DMA pushes, so contention risk is low. No pin
conflicts either way; GPIO 8/9/11 are free.

Planned integration: touch read as its own low-priority FreeRTOS task on
Core 0, priority below `displayTask`/`networkTask` so display DMA always
wins over touch responsiveness.

## Networking

| Item | Value |
|---|---|
| Link | WiFi 802.11n, HT40 (40 MHz channel width), single spatial stream (no MIMO on S3) |
| Transport | UDP, port 12345, tile-chunked JPEG frames from a PC sender (`captureJpeg.py`) |
| TX power | `esp_wifi_set_max_tx_power(80)` (0.25 dBm units → 20 dBm requested; code comment says "~10 dBm — the community fix," this discrepancy hasn't been investigated) |
| Real-world throughput ceiling | Not measured on this hardware — estimated conservatively at ~10-20 Mbit/s sustained given CPU contention from decode/display/network sharing both cores; idle-chip iperf-style benchmarks for ESP32-S3 in this config are typically cited around 30-40 Mbit/s, but that number assumes no other workload running |

## Core / Task Split (current, post-refactor)

| Core | Tasks |
|---|---|
| Core 0 | `networkTask` (priority 3), `displayTask` (priority 2), `otaTask` (priority 1), `wifiWatchdogTask` (priority 1) |
| Core 1 | `decodeTask` (priority 2) — JPEG tile decode, formerly Arduino `loop()` |

## Memory Budget (approximate, at time of writing)

| Buffer | Location | Size |
|---|---|---|
| JPEG assembly slots (`slot[].assembly`) | SRAM (`MALLOC_CAP_INTERNAL`) | 4 × 33.6 KB = 134.4 KB |
| Display framebuffers (`frameFb[]`) | PSRAM | 3 × 150 KB = 450 KB (bumped from 2 to 3 buffers this session) |
| Tile chunk staging (`tileChunkStorage[]`) | PSRAM | 4 × 33.6 KB = 134.4 KB |
| Free SRAM headroom (approx, at last check) | — | ~72 KB |

Decoder reads raw JPEG bytes from SRAM specifically because it's on the
decode hot path and needs to be fast — this is why the JPEG slot count
can't easily grow without eating into that already-thin 72 KB margin.
