# Hardware & Pin Wiring — S3-nextgen jpeg (v4.0)

Snapshot of the hardware configuration for release 4.0. The source of truth for every value below is the code itself (`sdkconfig.defaults`, `main/nexus.h`, `main/display.h`, `main/CMakeLists.txt`) — if these ever diverge, trust the code.

## Board

| Item | Value |
|---|---|
| MCU | ESP32-S3 (Xtensa LX7, dual-core) |
| Board | Any ESP32-S3 devkit with the **N16R8** memory layout (e.g. ESP32-S3-DevKitC-1 N16R8) |
| Build system | ESP-IDF (`idf.py` / VS Code ESP-IDF extension); manifest requires IDF ≥ 5.1, developed against v6.1 |
| Flash | 16 MB, QIO, 80 MHz (`CONFIG_ESPTOOLPY_FLASHSIZE_16MB`, `CONFIG_ESPTOOLPY_FLASHMODE_QIO`) |
| PSRAM | 8 MB **Octal**, 80 MHz (`CONFIG_SPIRAM_MODE_OCT`, `CONFIG_SPIRAM_SPEED_80M`). `CONFIG_SPIRAM_USE_CAPS_ALLOC` maps it to the heap for `heap_caps_malloc(MALLOC_CAP_SPIRAM)`; `CONFIG_SPIRAM_USE_MALLOC` is deliberately **unset**, so plain `malloc()` never lands in PSRAM — PSRAM is explicit opt-in |
| CPU frequency | 240 MHz (`CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240`), dual-core, FreeRTOS tick 1000 Hz |
| Partition table | `partitions_16MB_ota.csv`: `nvs` 0x5000 @ 0x9000, `otadata` 0x2000, `app0`/`app1` 0x180000 (1.5 MB) each, `spiffs` 0xCF0000 |
| Filesystem | The `spiffs` partition is reserved but not used by the firmware |
| Console | UART0 at 115200 baud (`CONFIG_ESP_CONSOLE_UART_DEFAULT`). For boards that only expose native USB, switch to `CONFIG_ESP_CONSOLE_USB_CDC` |
| USB | Used for flashing/console only; not used as a data transport |
| Upload | First flash over USB (`idf.py -p <PORT> flash`); afterwards HTTP OTA: `curl -X POST --data-binary @build/s3_nextgen_jpeg.bin http://<device-ip>:3232/update` |
| Task watchdog | 5 s (`CONFIG_ESP_TASK_WDT_TIMEOUT_S`); `displayTask` registers itself |

For a board with different flash/PSRAM, adjust `sdkconfig.defaults` (see README → *If your board isn't N16R8*) and the partition table.

## Display — ILI9341, 320×240, SPI

Driven via LovyanGFX 1.2.33 (`LGFX` class in `main/display.h`), fetched by the ESP-IDF component manager. All pin numbers and SPI settings are `#define`s in [`main/nexus.h`](main/nexus.h) (Zone 1) — `display.h` only references them.

| Signal | GPIO | `nexus.h` macro |
|---|---|---|
| SCLK | 12 | `LCD_PIN_SCLK` |
| MOSI | 13 | `LCD_PIN_MOSI` |
| MISO | -1 for the *panel* | `LCD_PIN_MISO` — the panel is write-only. When touch is enabled the **bus** does carry MISO (GPIO 11) for the touch controller; see the note under Touch |
| CS | 10 | `LCD_PIN_CS` |
| DC | 4 | `LCD_PIN_DC` |
| RST | 5 | `LCD_PIN_RST` |
| BUSY | -1 (not connected) | `LCD_PIN_BUSY` — not applicable to this panel |

- SPI host: `SPI2_HOST` — `LCD_SPI_HOST`
- Write clock: **75 MHz** — `LCD_WRITE_HZ`. At 80 MHz, intermittent top/bottom frame tearing appeared on jumper-wire wiring (signal integrity); lower it in ~10 MHz steps if you see tearing, or raise it if your wiring (short leads, common ground, ideally soldered) can carry it. See the comment in `nexus.h`
- Panel native size: 240×320 (`LCD_PANEL_W`/`LCD_PANEL_H`); software rotation 3 → logical 320×240 landscape
- `dummy_read_pixel = 8`, `readable = false`, `bus_shared = true`
- Color depth: 16-bit RGB565. The JPEG decoder outputs **RGB565 big-endian** to match the panel's SPI byte order, so no byte swapping is needed before the DMA push

No other GPIOs are used by the firmware besides the display and touch pins listed here and the BOOT button.

## Touch — XPT2046 (optional)

Touch support is compiled in only when `TOUCH_FEATURE_ENABLED` is `1` in `main/nexus.h`. Set it to `0` to remove touch initialization, calibration, sampling and event networking entirely (the touch pins are then left unused, and the bus is configured without MISO).

Driven by LovyanGFX's own `Touch_XPT2046`, attached to the `LGFX` class in [`main/display.h`](main/display.h). Sharing LovyanGFX's bus object avoids a second SPI host and brings 7-sample hardware median filtering.

| Signal | GPIO | `nexus.h` macro |
|---|---|---|
| CS | 9 | `TOUCH_PIN_CS` |
| IRQ (PENIRQ) | 8 | `TOUCH_PIN_IRQ` — **not connected in the reference build**; see below |
| MISO | 11 | `TOUCH_PIN_MISO` |
| SCLK / MOSI | 12 / 13 | shared with the panel |
| BOOT button | 0 | `BOOT_PIN` — double-press before streaming starts to recalibrate |

- Read clock: 1 MHz (`TOUCH_SPI_HZ`). One read is 57 bytes ≈ **456 µs**.
- **MISO is declared on the panel's bus config**, not just the touch config. LovyanGFX's `spi::init()` guards `spi_bus_initialize()` behind "does this host already have a device handle", so the second call — the one made while initializing touch — cannot retrofit MISO into the IDF bus configuration. Declaring it on the first init is load-bearing.

### Bus arbitration — the display always wins

`Panel_Device::getTouchRaw()` calls `endTransaction()` if a write is open, which from *another task*, mid-`pushPixelsDMA`, would tear the frame and desync `displayTask`'s `startWrite`/`endWrite` pairing. LovyanGFX's own bus lock does not cover an async DMA that spans several loop iterations, nor the touch driver's direct `lgfx::spi::*` transfers.

So ownership is explicit: **`g_spiMutex`** (`nexus.h`). `displayTask` holds it across the entire push (`startWrite` → DMA → `waitDMA` → `endWrite`); `touchTask` takes it around each read, waiting up to `TOUCH_BUS_WAIT_MS` (25 ms) and treating a timeout as "skip this sample", never as a release. Every other path that draws — `statusLine()`, `drawBootHeader()` — goes through the same mutex, because those are called from `decodeTask` (Core 1) and `wifiWatchdogTask` (Core 0).

It is a real mutex rather than a binary semaphore specifically so **priority inheritance** applies: when `displayTask` (prio 3) wants the bus while `touchTask` (prio 1) holds it, touch is boosted and hands it back within one read. That 456 µs is the hard upper bound on how much touch can delay a frame, and it only applies while a finger is actually down.

### Cost

- **Idle (PENIRQ wired, `TOUCH_USE_IRQ` = 1): effectively zero.** `touchTask` blocks on a PENIRQ falling-edge interrupt — no polling, no SPI traffic. The interrupt is detached for the duration of a press (PENIRQ stays low throughout, which would otherwise storm) and re-attached on release.
- **Idle (PENIRQ unwired, `TOUCH_USE_IRQ` = 0 — the reference setting):** one full read per poll at `TOUCH_IDLE_POLL_HZ` (20 Hz) → ~0.9% bus occupancy, and up to 50 ms of extra touch-down latency.
- **While touched:** one read per 33 ms (`TOUCH_REPORT_HZ` = 30) → ~1.4% bus occupancy. Identical in both modes.
- **Memory:** a 6 KB task stack from internal heap, plus a few hundred bytes of static RAM. The stack is allocated after the JPEG slots, so it cannot push the slot count into its fallback path.

### PENIRQ is not connected in the reference build

`TOUCH_USE_IRQ` in `nexus.h` is **0** because the IRQ line isn't wired. Flipping it to **1** is the only change needed once it is.

This flag is not just an efficiency knob. `Touch_XPT2046::getTouchRaw()` opens with:

```c
if (_cfg.pin_int >= 0 && gpio_in(_cfg.pin_int)) return 0;
```

A configured-but-floating PENIRQ reads high, so **every read would report "not touched"** and touch would look completely dead. With the flag at 0 the driver gets `pin_int = -1` and never consults the line, and the firmware leaves GPIO 8 entirely alone — no pull-up, no interrupt.

Touch detection itself does not depend on PENIRQ: the XPT2046 measures pressure, and the driver requires ≥3 valid X, Y and Z samples out of 7 plus a non-zero Z before reporting a point. What is lost is only the cheap "is anything touching?" test, so idle detection has to be polled.

If stray clicks ever show up in this mode, raise `TOUCH_PRESS_SAMPLES` to 2 — it requires two corroborating reads before a press commits, at the cost of one extra poll period.

### Calibration

**Not** LovyanGFX's `calibrateTouch()`. That samples 4 corners and fits a strictly affine transform, which has two problems: with no redundancy every tap's error lands straight in the transform, and affine cannot express the gentle *twist* real resistive panels have (x drifting with y and vice versa) — which is why a 4-corner calibration feels sharp at the corners and vague toward the middle of the edges.

Instead: a `TOUCH_CAL_GRID`² grid (default **4×4 = 16 targets**, inset `TOUCH_CAL_INSET` = 26 px from the edges), median of `TOUCH_CAL_SAMPLES` (16) raw reads per target with the settling reads dropped, least-squares fit to a **bilinear** model with one cross term per axis:

```
sx = a0 + a1*u + a2*v + a3*u*v        u,v = raw ADC / 4096
sy = b0 + b1*u + b2*v + b3*u*v
```

Raw values are normalised before fitting — the `u*v` term would otherwise span ~1.6 × 10⁷ next to a constant term of 1 and wreck the conditioning of the normal equations in float. The 4×4 systems are solved in double by Gaussian elimination with partial pivoting. The model degenerates to affine if the cross terms fit to ~0, so it can only match or beat the old behaviour.

Because the fit maps raw → *final screen* coordinates directly, panel rotation is baked in and there is no LovyanGFX transform convention to get wrong. At runtime `readPoint()` calls `getTouchRaw()` and applies the fit itself.

Simulated against a twisted panel with realistic tap and ADC noise:

| fit | mean err | p95 | worst |
|---|---|---|---|
| 4 corners, affine (old) | 1.93 px | 3.92 px | 6.05 px |
| 9-point, bilinear | 1.68 px | 3.25 px | 4.82 px |
| **16-point, bilinear (current)** | **1.26 px** | **2.67 px** | **4.43 px** |

Most of the residual is human tap scatter rather than model error, which is what more points average away — hence 16 rather than 9.

The run reports its own **RMS residual** on screen when it finishes. If that exceeds `TOUCH_CAL_MAX_RMS` (6 px) a tap almost certainly landed off its marker, so the whole run repeats rather than persisting a calibration that is already measurably wrong — bounded by `TOUCH_CAL_MAX_RETRY` (2).

Stored in NVS, namespace `touch`, key `cal`, magic `0x9341` + **version 2**. A calibration saved by the v3 (Arduino) firmware uses the same namespace, key and format and still loads; older version-1 blobs (the original 4-corner format) fail the check and trigger a fresh run. To redo it deliberately, **double-press BOOT while the board is not yet streaming**.

`touchTask` gets a 6 KB stack because the fit runs on it when triggered by BOOT — the sampling path itself needs almost nothing.

### Reporting to the PC

Raw events only — `DOWN` / `MOVE` / `UP` with panel coordinates. All gesture interpretation (tap → click, slide → scroll, hold → drag) lives in `captureJpeg.py`, so it can be retuned without an OTA reflash.

Wire format, 9 bytes, sent on the **existing** UDP socket to the address the video sender is already using (no second socket, no extra thread):

```
0xAA 0xDD <kind> <touchId> <seq> <xHi> <xLo> <yHi> <yLo>
```

`touchTask` never calls `sendto()` itself — it posts to `touchQueue` and `networkTask` drains it, keeping the socket owned by exactly one task.

Events are sent only while a finger is down, capped at 30 Hz, and suppressed when the point hasn't moved by `TOUCH_MOVE_EPS` — apart from a `TOUCH_KEEPALIVE_MS` (500 ms) heartbeat while pressed. The heartbeat exists because otherwise "finger held perfectly still" and "the link died" are both just silence, and the PC's stuck-button watchdog cannot tell them apart; without it, pausing during a hold-drag would drop the window you were dragging.

That is ~1.5 KB/s and ~30 packets/s during a touch, 2 packets/s while holding still, and nothing at all the rest of the time. Against the sender's maximum video budget (25,000 bytes × 50 FPS ≈ 1.25 MB/s) that is roughly 0.1%.

Gestures (all decided PC-side): tap → left click, slide → scroll wheel (vertical or horizontal, axis locked when the slide commits), **hold still past `TOUCH_HOLD_MS` (400 ms) → press-and-hold drag**. A press commits to exactly one of the three. The drag has two independent safety releases — a 2 s watchdog if the ESP goes quiet, and an unconditional release in the Python teardown — because a left button left stuck down would strand the user's desktop mid-drag.

### Cursor borrowing

Windows has exactly one system cursor — there is no second, independent pointer to inject into, so a touch unavoidably moves the user's real cursor to the mirrored monitor. Rather than leave it stolen, `TouchInjector` **borrows and returns** it: `_borrow_cursor()` snapshots `GetCursorPos()` before the first injected move, and `_do_restore()` puts the cursor back once the gesture ends.

Three details make this behave:

- The restore is **deferred** by `TOUCH_RESTORE_DELAY_MS` (40 ms) and executed from `tick()`, not inline. Apps frequently read the cursor position while handling the click or button-up they were just sent; snapping away in the same instant can land the click at the restored position instead.
- If the cursor is not within `TOUCH_RESTORE_TOLERANCE_PX` (8 px) of where we last put it, the user has grabbed their physical mouse mid-gesture, so the restore is **abandoned**. The goal is to stop fighting the mouse, not to fight it in a new way.
- `_borrow_cursor()` **cancels a still-pending restore instead of re-snapshotting**. Otherwise back-to-back taps would capture the position we ourselves had just injected, and the restore target would walk across the screen.

`tick()` is therefore called unconditionally every frame, *not* gated on the "Enable Touch" state — it is the only thing that hands the cursor back.

The PC can switch reporting off at the source via the control channel (`0xAA 0xCC 0x02 <0|1>`), driven by the "Enable Touch" slider.

## Networking

| Item | Value |
|---|---|
| Link | Wi‑Fi 802.11n, **HT20** (`WIFI_BW20`), power save off (`WIFI_PS_NONE`), single spatial stream (no MIMO on S3) |
| Transport | UDP, port 12345. One JPEG per frame, split into indexed chunks of up to 1400 bytes, frame capped at 35 KiB (≤ 26 chunks) |
| Packet header | 8 bytes: `0xAA 0xBB <frameId> <chunkId> <nChunks> <flags=0> <sizeHi> <sizeLo>` |
| Frame validation | Chunk count, index, length and total size must agree; flags must be 0; the assembled frame must start `FF D8` and end `FF D9`; frames not complete within `FRAME_ASSEMBLY_TIMEOUT_MS` (80 ms) are dropped |
| Sender pacing | Fixed 50 FPS target, 250 µs gap between chunks |
| Receive buffering | `SO_RCVBUF` 64 KiB, `CONFIG_LWIP_UDP_RECVMBOX_SIZE` = 32, static Wi‑Fi RX buffers 16 / dynamic 32, RX block-ack window 16, software coexistence enabled |
| Discovery | ESP broadcasts `S3READY` every 2 s while no packets are arriving; the PC replies by simply streaming |
| TX power | `WIFI_TX_POWER` in `main/nexus.h`, in 0.25 dBm units (`80` requests 20 dBm; `20` requests 5 dBm), applied with `esp_wifi_set_max_tx_power()`. The chip and regional regulatory limits may clamp it |
| Wi‑Fi timeouts | Boot waits up to `WIFI_CONNECT_TIMEOUT_MS` (150 s) before restarting; "waiting for video" overlay after `PKT_TIMEOUT_MS` (3 s) of silence |
| Real-world throughput ceiling | Not measured on this hardware — estimated conservatively at ~10–20 Mbit/s sustained given CPU contention from decode/display/network sharing both cores; idle-chip iperf-style benchmarks for the ESP32-S3 are typically cited around 30–40 Mbit/s, but that assumes no other workload |

## Core / Task Split

| Core | Task | Priority | Stack |
|---|---|---|---|
| Core 0 | `networkTask` | 3 | 10 KB |
| Core 0 | `displayTask` | 3 | 4 KB |
| Core 0 | `wifiWatchdogTask` | 1 | 4 KB |
| Core 0 | `touchTask` (if enabled) | 1 | 6 KB |
| Core 0 | HTTP OTA server (`esp_http_server`'s own task) | IDF default | IDF default |
| Core 1 | `decodeTask` — full-frame JPEG block decode | 2 | 8 KB |

`app_main()` creates these tasks and returns.

## Memory Budget (approximate)

| Buffer | Location | Size |
|---|---|---|
| Full-frame JPEG slots (`jpegFrames[].data`) | First 2: internal SRAM (PSRAM fallback); remaining: PSRAM | 3 × up to 35 KiB (`CFG_NUM_JPEG_SLOTS` = 3, max 6) |
| Decoder block scratch (`s_blockOut`) | Internal SRAM, 16-byte aligned | 320 × 16 × 2 = 10 KiB |
| Display framebuffers (`frameFb[]`) | PSRAM, 16-byte aligned | 3 × 320 × 240 × 2 = 450 KiB (`CFG_NUM_DISPLAY_BUFS` = 3, max 6) |
| Chunk bitmap / per-slot metadata | Internal SRAM | 3 × 26 chunk flags plus slot state |
| Debug stats buffer | Internal SRAM | 320 B |

JPEG slots reserve up to 35 KiB each but only receive the bytes reported in each packet header. If the requested number of JPEG slots or framebuffers can't be allocated, `app_main()` retries with one fewer (down to 1 JPEG slot and 2 framebuffers) and shows the result on the boot screen (`J:<n> D:<n>`). `CFG_NUM_JPEG_SLOTS` can be raised to absorb more network/decode jitter, at a cost of up to 35 KiB per slot; the larger Wi‑Fi RX buffers and the UDP mailbox use additional RAM in exchange for more buffering.

Decoded blocks are copied into PSRAM; the only whole-frame RGB565 storage is the display framebuffer, and a framebuffer is never reused until its DMA transfer has completed. UDP remains best-effort: a missing chunk invalidates that JPEG frame, which is discarded after the 80 ms assembly timeout rather than partially shown.