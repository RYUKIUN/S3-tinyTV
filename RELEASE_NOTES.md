# v3.0.0

This release adds the missing product-level documentation and feature explanation that were not captured in v2.0.0. The real user-facing story is now documented clearly: this is a wireless screen-mirroring and touch-enabled display system for ESP32-S3, not just a JPEG decode test bed or generic media player.

## Highlights

- **Full project positioning rewrite in the README.** The project now opens with a clear statement of what it does, what it is for, and how it differs from similar projects such as tinytron, vPlayer, esp32-tv, Nomad, and esp_video (tuupola).
- **Explicit description of the real use case.** Documentation now explains that the project is for real-time PC-to-display mirroring and remote desktop-like interaction, especially on small TFT panels.
- **Touchscreen support is now called out as a major feature.** The README documents the XPT2046 touchscreen behavior: tap to click, scroll gestures, hold-drag behavior, and the fact that the display can act as a mouse input device for the mirrored PC.
- **Python sender UI is fully explained.** The README documents the live control window and explains each control in practical terms:
  - `Priority` — choose the streaming balance between FPS and image quality
  - `Sharpen` — sharpen live image edges or disable it
  - `Show Stats` — toggle live diagnostics overlay
  - `Enable Touch` — turn touchscreen input on or off
  - `Monitor` — select which screen to capture if multiple displays exist
- **Adaptive streaming is documented as a core feature.** The project now clearly explains the adaptive JPEG quality behavior, where the sender changes image quality based on throughput and decode headroom.
- **OTA and diagnostics are now part of the product story.** The docs explicitly note the OTA update flow and live diagnostic overlay to make the project easier to operate and maintain.

## Changed

- README rewritten in English with a much stronger product pitch for first-time readers.
- Added a clear “What it does” section near the top of the document.
- Added a “Main features” section summarizing the core capabilities.
- Added a “Why this is different from similar projects” comparison section.
- Added a dedicated “Python UI controls and how to use them” guide.
- Expanded the system overview to cover the live streaming pipeline and touch feedback loop.
- Improved the project narrative so it reads as a real wireless display solution rather than a loose collection of embedded experiments.

## Added / Documented but not previously emphasized in v2.0.0

- Wireless screen mirroring from PC to ESP32-S3 display
- Touch-enabled interaction for the mirrored PC
- Live Python control UI for priority/quality/sharpening/touch/stats
- Comparison against similar ESP32 display and media projects
- Broader end-user value proposition and practical use cases
- Clear explanation of the setup flow, display behavior, and expected operating model

## Impact

This release is mostly documentation and product communication, but it closes the gap between the code and the actual end-user story. It makes the project immediately understandable to new readers and explains the major features that were not meaningfully captured in v2.0.0, especially the touchscreen-driven control flow and Python UI settings.

---

# v2.0.0

A major pass on both stability and approachability. The headline fix closes out a
long-standing visual corruption bug under load; everything else in this release
is either cleaning up around that fix or making the project easier for someone
else (or future-you) to actually pick up and run.

## Highlights

- **Fixed the tile-shredding glitch.** Under sustained decode load, tiles 3/4
  would occasionally corrupt into torn, blocky garbage. Root cause: a tile's
  framebuffer region could be left holding partial/garbage MCU data from a
  failed or stalled decode, since nothing ever repainted it before the next
  frame reused the same buffer. Fixed by seeding each new write buffer with
  the last fully-decoded frame before decode starts writing into it — a
  stalled or failed tile now falls back to stale-but-coherent pixels instead
  of visible corruption. Confirmed on hardware: no more glitching or freezing,
  just an expected FPS dip under heavy decode load.
- **Refactored from Arduino's implicit `loop()` to a fully task-based
  architecture.** The decode path that used to run as `loop()` is now an
  explicit, named FreeRTOS task (`decodeTask`) pinned to Core 1, created
  alongside every other task in `setup()`. `setup()` now deletes its own task
  and reclaims its stack once every worker task is running, instead of
  falling through to an idle `loop()`.
- **Removed the offline/local-MJPEG-playback fallback entirely.** It was
  unused. This took `g_offlineMode`, `enterOfflineMode()`, the SPIFFS boot
  mount, and every task's offline-mode exit branch out of the firmware, along
  with the dead `mpeg_decode.cpp/.h` (an unused experimental I/P-frame codec),
  `offline_player.cpp/.h`, `bswap16_memcpy_simd.h`, and the PC-side
  `Mjpegencoder.py` tool that authored the now-unused file format.

## Breaking changes

- **`shared.h` → `src/nexus.h`.** Renamed and reorganized as the project's
  single central configuration hub, zoned by how likely you are to need to
  edit something: Zone 1 (WiFi credentials, OTA hostname, display pin wiring
  — edit before you flash), Zone 2 (resolution/tile/memory tunables), Zone 3
  (timing tunables), Zone 4 (internal structs/FreeRTOS handles you shouldn't
  need to touch). `shared.h` no longer exists — anything still including it
  needs to switch to `nexus.h`.
- **WiFi credentials and display pin wiring moved out of `main.cpp` /
  `display.h`.** They used to be hardcoded literals inside a class
  constructor and a couple of `const char*` definitions; both are now
  `#define`s in `nexus.h` Zone 1, the one file you need to edit before your
  first flash.
- **Sender UI controls changed shape** (`captureJpeg.py`). `Max FPS` and
  `Base Qual` sliders are gone, replaced by a `Mode (0=Auto 1=Manual)` toggle
  — Auto runs the existing adaptive-bitrate controller, Manual exposes a
  direct `Manual Quality` slider. `Bilateral Mix` and `Chroma sub` are gone
  entirely (chroma is now always 4:2:0; bilateral filtering was cut — it was
  the worst cost-to-benefit knob in the pipeline, expensive every frame for
  barely-visible benefit once re-encoded to JPEG anyway). Frame rate is now a
  fixed constant, not a slider.

## Fixed

- Tile-shredding/corruption glitch under decode load (see Highlights).
- Unclamped `nChunks` (an untrusted value read straight off the wire) could
  drive `assembleTileInto()`'s loop past the end of `TileState`'s fixed-size
  arrays. Never actually reachable given the sender's own size cap, but a
  real latent out-of-bounds read — now clamped explicitly.
- A dead code path in the WiFi watchdog that set a disconnect timestamp and
  then immediately overwrote it to zero on the very next line. The variable
  it touched (`g_wifiDisconnectedMs`) turned out to be fully vestigial —
  removed entirely rather than patched.
- Stale documentation comment claiming the WiFi TX power request was "~10
  dBm" when the actual value requests 20 dBm (the API's units are 0.25 dBm
  steps) — corrected.
- Mermaid architecture diagram in the README failed to render on GitHub
  ("Unable to render rich display") due to nested subgraphs and a cylinder
  node shape GitHub's pinned Mermaid version doesn't handle well — flattened
  and simplified.

## Changed

- Display framebuffer count bumped from 2 to 3 (`CFG_NUM_DISPLAY_BUFS`) —
  cheap PSRAM headroom that absorbs more Core 0 jitter before a frame gets
  dropped.
- `main.cpp`'s include list trimmed — it was re-including several headers
  `nexus.h` already pulls in.
- Hardcoded `"esp32s3-display"` hostname strings in log/status messages now
  reference the single `OTA_HOSTNAME` macro instead of silently risking
  drift from it.

## Added

- `README.md` — project pitch, feature highlights, architecture diagram,
  hardware summary, getting-started instructions, and an honest "known
  limitations" section.
- `HARDWARE.md` — full pin-level wiring, SPI bus configuration, and memory
  budget.
- `requirements.txt` for the Python sender (`pip install -r requirements.txt`).

## Investigated, not shipped

A few directions were explored in depth this cycle and deliberately not
pursued — noting them so the reasoning isn't lost:

- **JPEG decode SIMD (IDCT/dequant).** The vendored `s3_simd_idct.S_FUTURE`
  turned out to be genuinely unfinished (dequantizes twice, never actually
  performs the inverse-DCT transform, broken control flow between its two
  code paths) — not safe to enable. `s3_simd_dequant.S` is syntactically
  complete but has zero prior call sites anywhere and contains a suspicious
  register substitution in its shift-amount setup that couldn't be verified
  safe without ESP32-S3 PIE ISA documentation or hardware-validated pixel
  diffing. Left disabled.
- **USB as a transport.** Full-Speed USB tops out around half of the
  realistic WiFi throughput available here, and the current network
  environment has no packet loss to work around — not worth the added
  complexity.
- **Alternative codecs** (QOI/lossless, DXT-style block compression, other
  JPEG variants). Explored in detail; JPEG remains the best fit for this
  hardware and this content given the actual bandwidth/decode-cost tradeoffs
  involved.
- **Telemetry-based auto-quality regulation** (using the ESP's own
  live CPU/decode-time stats to drive the sender's quality controller,
  instead of just frame byte size). Deferred pending real-world results from
  this release's fixes.
