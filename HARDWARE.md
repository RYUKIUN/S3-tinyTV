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

Driven via LovyanGFX (`LGFX` class in `src/display.h`), **not** Adafruit_GFX. All pin numbers and SPI config are `#define`s in [`src/nexus.h`](src/nexus.h) (Zone 1) — `display.h` only references them, nothing is hardcoded there anymore.

| Signal | GPIO | `nexus.h` macro |
|---|---|---|
| SCLK | 12 | `LCD_PIN_SCLK` |
| MOSI | 13 | `LCD_PIN_MOSI` |
| MISO | -1 for the *panel* | `LCD_PIN_MISO` — the panel is write-only. The **bus** does carry MISO (GPIO 11) for the touch controller; see the note under Touch |
| CS | 10 | `LCD_PIN_CS` |
| DC | 4 | `LCD_PIN_DC` |
| RST | 5 | `LCD_PIN_RST` |
| BUSY | -1 (not connected) | `LCD_PIN_BUSY` — not applicable to this panel |

- SPI host: `SPI2_HOST` (HSPI) — `LCD_SPI_HOST`
- Write clock: 40 MHz — `LCD_WRITE_HZ` (dropped from 80 MHz after intermittent
  top/bottom frame tearing traced to signal integrity at that rate on
  jumper-wire wiring; raise in ~10 MHz steps if your wiring can carry it —
  see the comment in `nexus.h`)
- Panel native size: 240×320 (`LCD_PANEL_W`/`LCD_PANEL_H`); software rotation 3 → logical 320×240 landscape
- `dummy_read_pixel = 8`, `readable = false`, `bus_shared = true` (touch now actually shares this bus — see below)
- Color depth: 16-bit (RGB565), `RGB565_BIG_ENDIAN` used throughout the decode pipeline to match the panel's native SPI byte order

No other GPIOs are used anywhere in the active `src/` codebase (network, decode, display, OTA) — confirmed by search at the time of writing.

## Touch — XPT2046 (integrated)

Driven by LovyanGFX's own `Touch_XPT2046`, attached to the `LGFX` class in
[`src/display.h`](src/display.h) — **not** the `XPT2046_Touchscreen` library the
`test unit-S3` reference project uses. Sharing LovyanGFX's bus object avoids
mixing an Arduino `SPI` instance with LovyanGFX's `Bus_SPI` on the same host,
and brings 7-sample hardware median filtering and affine calibration for free.

| Signal | GPIO | `nexus.h` macro |
|---|---|---|
| CS | 9 | `TOUCH_PIN_CS` |
| IRQ (PENIRQ) | 8 | `TOUCH_PIN_IRQ` — **currently NOT connected**; see below |
| MISO | 11 | `TOUCH_PIN_MISO` |
| SCLK / MOSI | 12 / 13 | shared with the panel |
| BOOT button | 0 | `BOOT_PIN` — double-press before streaming starts to recalibrate |

- Read clock: 1 MHz (`TOUCH_SPI_HZ`). One read is 57 bytes ≈ **456 µs**.
- **MISO is declared on the panel's bus config**, not just the touch config.
  LovyanGFX's `spi::init()` guards `spi_bus_initialize()` behind "does this host
  already have a device handle", so the second call — the one `initTouch()`
  makes — cannot retrofit MISO into the IDF bus configuration. Declaring it on
  the first init is load-bearing.

### Bus arbitration — the display always wins

`Panel_Device::getTouchRaw()` calls `endTransaction()` if a write is open, which
from *another task*, mid-`pushPixelsDMA`, would tear the frame and desync
`displayTask`'s `startWrite`/`endWrite` pairing. LovyanGFX's own bus lock does
not cover an async DMA that spans several loop iterations.

So ownership is explicit: **`g_spiMutex`** (`nexus.h`). `displayTask` holds it
across the entire push (`startWrite` → DMA → `waitDMA` → `endWrite`);
`touchTask` takes it around each read, waiting up to `TOUCH_BUS_WAIT_MS` (25 ms)
and treating a timeout as "skip this sample", never as a release. Every other
path that draws — `statusLine()`, `drawBootHeader()` — goes through the same
mutex, because those are called from `decodeTask` (Core 1) and
`wifiWatchdogTask` (Core 0).

It is a real mutex rather than a binary semaphore specifically so **priority
inheritance** applies: when `displayTask` (prio 2) wants a bus held by
`touchTask` (prio 1), touch is boosted and hands it back within one read. That
456 µs is the hard upper bound on how much touch can ever delay a frame, and it
only applies while a finger is actually down.

### Cost

- **Idle (PENIRQ wired, `TOUCH_USE_IRQ` = 1): exactly zero.** `touchTask` blocks
  on a PENIRQ falling-edge interrupt — no polling, no SPI traffic, no CPU. The
  interrupt is detached for the duration of a press (PENIRQ stays low
  throughout, which would otherwise storm) and re-attached on release.
- **Idle (PENIRQ unwired, `TOUCH_USE_IRQ` = 0 — the current setting):** one full
  read per poll at `TOUCH_IDLE_POLL_HZ` (20 Hz) → ~0.9% bus occupancy, and up to
  50 ms of extra touch-down latency.
- **While touched:** one read per 33 ms (`TOUCH_REPORT_HZ` = 30) → ~1.4% bus
  occupancy. Identical in both modes.
- **Memory:** +704 B static RAM, +11.4 KB flash, +4 KB task stack from the
  internal heap. Measured against `7089fbe`. The stack is allocated *after* the
  JPEG slots, so it cannot push the slot count into its fallback path.

### PENIRQ is currently not connected

`TOUCH_USE_IRQ` in `nexus.h` is **0** because the IRQ line isn't wired yet.
Flipping it to **1** is the only change needed once it is.

This flag is not just an efficiency knob. `Touch_XPT2046::getTouchRaw()` opens
with:

```c
if (_cfg.pin_int >= 0 && gpio_in(_cfg.pin_int)) return 0;
```

A configured-but-floating PENIRQ reads high, so **every read would report "not
touched"** and touch would look completely dead. With the flag at 0 the driver
gets `pin_int = -1` and never consults the line, and the firmware leaves GPIO 8
entirely alone — no `pinMode`, no pull-up, no interrupt.

Touch detection itself does not depend on PENIRQ: the XPT2046 measures pressure,
and the driver already requires ≥3 valid X, Y and Z samples out of 7 plus a
non-zero Z before reporting a point. What is lost is only the cheap "is anything
touching?" test, so idle detection has to be polled.

If stray clicks ever show up in this mode, raise `TOUCH_PRESS_SAMPLES` to 2 —
it requires two corroborating reads before a press commits, at the cost of one
extra poll period.

### Calibration

**Not** LovyanGFX's `calibrateTouch()`. That samples 4 corners and fits a
strictly affine transform, which has two problems: with no redundancy every
tap's error lands straight in the transform, and affine cannot express the
gentle *twist* real resistive panels have (x drifting with y and vice versa) —
which is exactly why a 4-corner calibration feels sharp at the corners and
vague toward the middle of the edges.

Instead: a `TOUCH_CAL_GRID`² grid (default **4×4 = 16 targets**), median of
`TOUCH_CAL_SAMPLES` raw reads per target with the settling reads dropped,
least-squares fit to a **bilinear** model with one cross term per axis:

```
sx = a0 + a1*u + a2*v + a3*u*v        u,v = raw ADC / 4096
sy = b0 + b1*u + b2*v + b3*u*v
```

Raw values are normalised before fitting — the `u*v` term would otherwise span
~1.6 × 10⁷ next to a constant term of 1 and wreck the conditioning of the normal
equations in float. The 4×4 systems are solved in double by Gaussian elimination
with partial pivoting. The model degenerates to affine if the cross terms fit to
~0, so it can only match or beat the old behaviour.

Because the fit maps raw → *final screen* coordinates directly, panel rotation
is baked in and there is no LovyanGFX transform convention to get wrong. At
runtime `readPoint()` calls `getTouchRaw()` and applies the fit itself.

Simulated against a twisted panel with realistic tap and ADC noise:

| fit | mean err | p95 | worst |
|---|---|---|---|
| 4 corners, affine (old) | 1.93 px | 3.92 px | 6.05 px |
| 9-point, bilinear | 1.68 px | 3.25 px | 4.82 px |
| **16-point, bilinear (current)** | **1.26 px** | **2.67 px** | **4.43 px** |

Most of the residual is human tap scatter rather than model error, which is what
more points average away — hence 16 rather than 9.

The run reports its own **RMS residual** on screen when it finishes. If that
exceeds `TOUCH_CAL_MAX_RMS` (6 px) a tap almost certainly landed off its marker,
so the whole run repeats rather than persisting a calibration that is already
measurably wrong — bounded by `TOUCH_CAL_MAX_RETRY`.

Stored in NVS, namespace `touch`, key `cal`, magic `0x9341` + **version 2**.
Version 1 blobs (the old 4-corner `uint16_t[8]`) fail the check and trigger a
fresh run, which is the desired upgrade path. To redo it deliberately,
**double-press BOOT while the board is not yet streaming**.

`touchTask` gets a 6 KB stack rather than 4 KB because this fit runs on it when
triggered by BOOT — the sampling path itself needs almost nothing.

### Reporting to the PC

Raw events only — `DOWN` / `MOVE` / `UP` with panel coordinates. All gesture
interpretation (tap → click, slide → scroll) lives in `captureJpeg.py` so it can
be retuned without an OTA reflash.

Wire format, 9 bytes, sent on the **existing** UDP socket to the address the
video sender is already using (no second socket, no extra thread):

```
0xAA 0xDD <kind> <touchId> <seq> <xHi> <xLo> <yHi> <yLo>
```

`touchTask` never calls `sendto()` itself — it posts to `touchQueue` and
`networkTask` drains it, keeping the socket owned by exactly one task.

Events are sent only while a finger is down, capped at 30 Hz, and suppressed
entirely when the point hasn't moved by `TOUCH_MOVE_EPS` — apart from a
`TOUCH_KEEPALIVE_MS` (500 ms) heartbeat while pressed. The heartbeat exists
because otherwise "finger held perfectly still" and "the link died" are both
just silence, and the PC's stuck-button watchdog cannot tell them apart; without
it, pausing during a hold-drag would drop the window you were dragging.

That is ~1.5 KB/s and ~30 pps against the stream's ~420 KB/s and ~300 pps —
**~0.35% of bandwidth during a touch**, 2 pps while holding still, and nothing
at all the rest of the time.

Gestures (all decided PC-side, in `captureJpeg.py`): tap → left click, slide →
scroll wheel, **hold still past `TOUCH_HOLD_MS` → press-and-hold drag**. A press
commits to exactly one of the three. The drag has two independent safety
releases — a watchdog if the ESP goes quiet, and an unconditional release in the
Python teardown — because a left button left stuck down would strand the user's
desktop mid-drag.

### Cursor borrowing

Windows has exactly one system cursor — there is no second, independent pointer
to inject into, so a touch unavoidably moves the user's real cursor to the
mirrored monitor. Rather than leave it stolen, `TouchInjector` **borrows and
returns** it: `_borrow_cursor()` snapshots `GetCursorPos()` before the first
injected move, and `_do_restore()` puts the cursor back once the gesture ends.

Three details make this behave:

- The restore is **deferred** by `TOUCH_RESTORE_DELAY_MS` (40 ms) and executed
  from `tick()`, not inline. Apps frequently read the cursor position while
  handling the click or button-up they were just sent; snapping away in the same
  instant can land the click at the restored position instead.
- If the cursor is not within `TOUCH_RESTORE_TOLERANCE_PX` of where we last put
  it, the user has grabbed their physical mouse mid-gesture, so the restore is
  **abandoned**. The goal is to stop fighting the mouse, not to fight it in a
  new way.
- `_borrow_cursor()` **cancels a still-pending restore instead of re-snapshotting**.
  Otherwise back-to-back taps would capture the position we ourselves had just
  injected, and the restore target would walk across the screen.

`tick()` is therefore called unconditionally every frame, *not* gated on the
"Enable Touch" state — it is the only thing that hands the cursor back.

The PC can switch reporting off at the source via the existing control channel
(`0xAA 0xCC 0x02 <0|1>`), driven by the "Enable Touch" trackbar.

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
| Core 0 | `networkTask` (priority 3), `displayTask` (priority 2), `otaTask` (priority 1), `wifiWatchdogTask` (priority 1), `touchTask` (priority 1) |
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
