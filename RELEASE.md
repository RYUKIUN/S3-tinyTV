# v4.0.0

The firmware has moved to pure ESP-IDF (no more Arduino or PlatformIO), and each frame is now sent as one full JPEG instead of four tiles. It works the same from the outside: a PC monitor shown on the 320×240 screen, with touch acting as a mouse.

## What's new

- Builds with `idf.py` or the VS Code ESP-IDF extension
- Full-frame JPEG streaming with a new decoder (`esp_new_jpeg`)
- A frame that arrives broken is dropped, and the screen keeps the last good picture
- OTA updates are now one `curl` command
- Two debug overlays on the PC: ESP stats and PC encoder stats
- Touch support can be turned off at build time (`TOUCH_FEATURE_ENABLED`)
- Docs rewritten for the new layout

## Things that changed

- **v3 and v4 don't mix.** Update the PC sender and the ESP firmware together.
- **Flash over USB once.** Later updates can go over Wi‑Fi.
- **Sender controls:** `Priority` and `Show Stats` are gone. Now there are `Sharpen`, `Debug Mode`, `Enable Touch` and `Monitor`.
- **Settings** live in `main/nexus.h` and `sdkconfig.defaults` instead of `platformio.ini`.
- **First build needs internet** (it downloads two libraries).
- Replace the Wi‑Fi name and password in `main/nexus.h` with your own before flashing.

## Licensing

The project is now released under the MIT License (see `LICENSE`). Third-party components keep their own licenses.

