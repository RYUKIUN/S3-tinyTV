# S3 Next-Gen JPEG Stream

## 1. Project overview
This project is a WiFi-based screen mirroring system built around an ESP32-S3 and a small TFT display. A Windows PC captures a screen region, splits each frame into four tiles, JPEG-compresses them, and sends them over UDP to the ESP32-S3. The ESP32 reassembles each tile, decodes the JPEG data, and pushes the finished frame to the display using DMA.

The design is optimized for real-time display and uses:
- Dual-core ESP32-S3 operation
- PSRAM-backed frame buffers
- Tile-based parallel decode flow
- UDP packet reassembly
- Adaptive JPEG quality control on the PC side
- Optional touch input support
- OTA firmware updates

---

## 2. Top-level project structure

```text
S3-nextgen jpeg/
├── README.md
├── HARDWARE.md
├── platformio.ini
├── partitions_16MB_ota.csv
├── requirements.txt
├── captureJpeg.py
├── src/
│   ├── main.cpp
│   ├── network.cpp
│   ├── network.h
│   ├── jpeg_decode.cpp
│   ├── jpeg_decode.h
│   ├── display.cpp
│   ├── display.h
│   ├── touch.cpp
│   ├── touch.h
│   ├── nexus.h
│   └── ...
├── lib/
│   └── JPEGDEC/
│       ├── src/
│       ├── CMakeLists.txt
│       ├── README.md
│       └── ...
├── build/
│   └── generated PlatformIO/ESP-IDF build outputs
├── logs/
├── settings/
│   └── saved per-device configuration JSON files
└── PROJECT_STRUCTURE_AND_EXPLANATION.md
```

---

## 3. File-by-file explanation

### README.md
This is the main project overview and user guide. It explains:
- what the project does
- the streaming architecture
- supported hardware and hardware requirements
- how to flash and use the system
- how the PC sender and ESP receiver communicate
- the project structure and known limitations

It is the best starting point for understanding the intended behavior and system flow.

### HARDWARE.md
This explains the electronics side of the project:
- ESP32-S3 pin mappings
- display wiring to the ILI9341 panel
- SPI bus layout
- touch controller wiring
- memory allocation and PSRAM usage constraints
- timing and hardware-related tuning recommendations

This file is important when adapting the project to a different display or board layout.

### platformio.ini
This is the PlatformIO build configuration for the ESP32 firmware. It defines:
- target board: ESP32-S3 devkit
- framework: Arduino
- flash size and PSRAM configuration
- partition scheme for OTA support
- upload method (USB serial or OTA)
- library dependencies
- compiler flags and optimization settings

The project is configured specifically for the N16R8 memory layout, where 16 MB flash and 8 MB PSRAM are used.

### partitions_16MB_ota.csv
This file defines the ESP32 flash partition layout. It is used so the firmware can support OTA updates and reserve the appropriate regions for:
- app firmware
- OTA slot
- filesystem or SPIFFS

### requirements.txt
This lists the Python dependencies used by the PC-side capture application. It is used to install all required packages such as OpenCV, MSS, NumPy, and other support libraries.

### captureJpeg.py
This is the PC-side sender application. It is the main client that:
- captures screen frames using `mss`
- crops and splits the image into four tiles
- JPEG-compresses each tile
- sends the data to the ESP32 over UDP sockets
- monitors and adapts quality based on throughput
- exposes a UI control window for settings like quality, sharpen, stats, and monitor selection
- handles touchscreen events from the ESP by translating them into mouse actions on the PC

This file is the counterpart to the ESP firmware and is essential to the full project workflow.

---

## 4. Core firmware source files

### src/nexus.h
This is the central configuration file for the ESP firmware. It acts as the single source of truth for the device setup. It contains:
- WiFi SSID and password
- OTA hostname
- display SPI pins
- touch pins and touch configuration
- resolution constants and tile layout
- chunk size, pipeline buffering, and memory tuning
- timing values and watchdog thresholds
- core data structures used by other files

The project comments describe it as the main “nexus” file: most modules in the firmware depend on it. If a user is changing WiFi settings, panel pins, or timing values, this is the first place to look.

### src/main.cpp
This is the main firmware entry point and runtime coordinator. It contains:
- global frame and decode state
- memory allocation for JPEG assembly slots and PSRAM frame buffers
- FreeRTOS task setup and runtime scheduling
- startup initialization and boot display
- WiFi watchdog logic
- CPU utilization tracking
- OTA task setup
- cross-core communication between network, decode, and display tasks

This file is the orchestration hub. It initializes the system and wires together the major pieces of the pipeline.

### src/network.cpp and src/network.h
These handle the UDP receiving side of the project.

Responsibilities:
- create and bind the UDP socket
- receive JPEG tile chunks from the PC
- validate inbound packet structure
- track frame IDs and tile assembly state
- reassemble chunked tile data into complete JPEG images
- queue complete tiles to the decoder
- handle touch event packets coming back from the ESP to the PC
- manage debug stat packets and WiFi presence checks

The network layer is critical because the PC sends each frame in multiple chunks over UDP, and the device must reliably reconstruct them before decoding.

### src/jpeg_decode.cpp and src/jpeg_decode.h
These files are responsible for decode operations.

Responsibilities:
- open JPEG data that was reassembled from UDP chunks
- decode each tile directly into the correct region of the PSRAM framebuffer
- use JPEGDEC to convert image data into the native RGB565 format expected by the display
- write pixel data directly into the appropriate frame buffer offset with minimal memory copying

This is where the optimization is most visible: the JPEG decoder writes directly into the final display buffer, avoiding additional temporary buffers.

### src/display.cpp and src/display.h
These handle everything about the display driver and frame presentation.

Responsibilities:
- initialize the LovyanGFX display driver
- render the boot header and status lines
- manage the display DMA transfers
- hold the SPI bus while pushing frame buffer data
- show overlays for waiting states and WiFi status
- present completed frames from the decode pipeline

This layer makes sure the final output is drawn efficiently and without corrupting the SPI bus while touch input is active.

### src/touch.cpp and src/touch.h
These files implement the input side of the project.

Responsibilities:
- initialize the XPT2046 touch controller
- read touch samples from the panel
- calibrate the touch screen
- convert raw touch positions into events
- translate finger actions into PC mouse movement, clicks, drag, and scroll actions
- send touch events over the same UDP link to the PC-side sender

The touch system was designed to behave like a remote input device, allowing the display to interact with the PC without a separate USB mouse connection.

---

## 5. How the project actually works

### Data flow from PC to display
1. The PC app captures a frame using `mss` and OpenCV.
2. The image is divided into 4 tiles, matching the design of the ESP panel.
3. Each tile is JPEG-compressed and packetized into UDP chunks.
4. The ESP32 network code receives the chunks and reassembles the JPEG tile.
5. The decoder converts each tile directly into the target display frame area.
6. Once a complete frame is ready, the display task pushes it to the ILI9341 LCD using DMA.
7. The user sees live screen output on the ESP-connected TFT display.

### Data flow in reverse
- While the panel is touched, the ESP sends touch reports to the PC.
- The Python sender interprets those touch events and maps them to local mouse actions.
- The PC can be controlled remotely from the connected display.

---

## 6. Core architecture ideas

### Dual-core task structure
The ESP firmware separates work between tasks running on different cores:
- Core 0: network, display, OTA, WiFi watchdog
- Core 1: decoding work and related frame processing

This reduces contention and improves real-time performance.

### Tile-based pipeline
Each frame is broken into 4 tiles. This helps the system:
- parallelize network and decode work
- avoid blocking on a single large JPEG image
- allow incremental arrival and partial reassembly
- reduce the impact of packet loss or jitter

### PSRAM optimization
The project stores the fullscreen frame buffer in PSRAM instead of internal SRAM, which allows larger image data and better throughput. The decoder writes directly into the PSRAM frame buffer instead of decoding to a temporary buffer and then moving the result again.

### Adaptive encoding on the PC side
The Python sender measures its output size and adjusts JPEG quality in real time to keep the stream within target constraints. This helps maintain stable performance under network load or more complex visual scenes.

### OTA and live configuration
The system is designed to be updated wirelessly after initial flashing, and key settings are centralized so users do not need to rework the whole project to adapt it to their environment.

---

## 7. Critical design note: the “nexus” concept
The most important architectural idea in this project is that nearly all configuration and shared state are centralized in [src/nexus.h](src/nexus.h). This includes:
- WiFi and OTA settings
- display pin assignments
- screen size and tile geometry
- timing values
- global FreeRTOS objects
- shared structs for network/decode/display communication

This makes the firmware easier to maintain and makes the device easier to adapt to different hardware layouts without changing the logic in many source files.

---

## 8. Summary
This project is a full real-time streaming pipeline combining:
- a PC capture/send application
- a WiFi UDP transport
- a dual-core ESP32-S3 decode engine
- a DMA-driven display path
- optional touch interaction support

The overall idea is simple but technically rich: take screen content from a PC, efficiently compress and stream it over WiFi, decode it on an ESP32-S3, and display it on a small TFT panel with minimal latency and high throughput.

This is not just a demo; it is a performance-oriented embedded system designed around real-time constraints, buffering, low-latency rendering, and adaptive quality control.
