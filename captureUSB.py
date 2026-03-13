"""
captureQOI.py — USB bulk transport edition
════════════════════════════════════════════
Streams screen content to ESP32-S3 over USB CDC serial.
No WiFi, no IP address, no AP setup — just plug the USB cable and run.

Requirements:
    pip install opencv-python mss numpy psutil imagecodecs pyserial
    (or: pip install qoi   as fallback encoder)

Usage:
    python captureQOI.py          # auto-detects ESP32-S3 COM port
    python captureQOI.py COM7     # override port on Windows
    python captureQOI.py /dev/ttyACM0  # override port on Linux/macOS

Protocol (PC → ESP32):
    TILE_PKT: [0x55 0xAA 0x01 frame_id tile_id enc_w enc_h len_hi len_lo] + QOI bytes
    CMD_PKT:  [0x55 0xAA 0x02 0x01 debug_state]

Protocol (ESP32 → PC):
    STATS:    [0xAB 0xCD] + "FPS:X|TEMP:X|...\n"
    READY:    "ESP32_READY\n"   (sent once on boot)
"""

import sys
import time
import threading
import os
import ctypes
import cv2
import mss
import numpy as np
import psutil
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

# ─────────────────────────────────────────────
#  SERIAL IMPORT
# ─────────────────────────────────────────────
try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("[ERROR] pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

# ─────────────────────────────────────────────
#  QOI BACKEND DETECTION
#  Priority: imagecodecs (fastest, AVX2) > qoi C-ext > qoi pure-Python
# ─────────────────────────────────────────────
_QOI_BACKEND = None

try:
    from imagecodecs import qoi_encode as _ic_qoi_encode
    _QOI_BACKEND = "imagecodecs"
except ImportError:
    pass

if _QOI_BACKEND is None:
    try:
        import qoi as _qoi_lib
        import importlib.util as _ilu
        _spec = _ilu.find_spec("qoi")
        _is_native = _spec is not None and not str(_spec.origin).endswith(".py")
        _QOI_BACKEND = "qoi-native" if _is_native else "qoi-python"
        if not _is_native:
            print("[WARN] qoi is pure-Python (~15 ms/tile). "
                  "Run: pip install imagecodecs  for a fast C backend.")
    except ImportError:
        print("[ERROR] No QOI encoder. Run: pip install imagecodecs")

_QOI_AVAILABLE = _QOI_BACKEND is not None
print(f"[QOI] backend: {_QOI_BACKEND}")


def _qoi_encode_rgb(tile_rgb):
    """Encode HxWx3 uint8 RGB ndarray to QOI bytes."""
    if _QOI_BACKEND == "imagecodecs":
        return bytes(_ic_qoi_encode(tile_rgb))
    return bytes(_qoi_lib.encode(tile_rgb))


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
ESP_W, ESP_H = 320, 240
NUM_TILES    = 4
TILE_W, TILE_H = 160, 120
TILE_X = [  0, 160,   0, 160]
TILE_Y = [  0,   0, 120, 120]

MAX_TILE_QOI = 42000          # bytes — must match main.cpp MAX_TILE_QOI

# USB serial — baud rate is ignored by USB CDC but must be set for pyserial
SERIAL_BAUD        = 115200
SERIAL_TIMEOUT_R   = 0.05     # read timeout (non-blocking stat drain)
SERIAL_TIMEOUT_W   = 5.0      # write timeout (blocks on USB backpressure)
SERIAL_READY_WAIT  = 8.0      # seconds to wait for "ESP32_READY"

# ESP32-S3 native USB VID (Espressif Systems)
# PID list covers: native USB-CDC, JTAG/CDC composite, and boot ROM
ESP32_VID  = 0x303A
ESP32_PIDS = {0x1001, 0x0002, 0x8000, 0x1011}

WINDOW_NAME          = "QOI Stream Control (USB)"
UI_W, UI_H           = 480, 620
PREVIEW_W, PREVIEW_H = 480, 360
DEFAULT_FPS          = 30

UPSCALE_MODES = [
    {"label": "Full",  "enc_w": 320, "enc_h": 240, "tile_w": 160, "tile_h": 120},
    {"label": "2/3",   "enc_w": 214, "enc_h": 160, "tile_w": 107, "tile_h":  80},
    {"label": "1/2",   "enc_w": 160, "enc_h": 120, "tile_w":  80, "tile_h":  60},
]
DEFAULT_UPSCALE  = 0
UPSCALE_MODE_MAX = len(UPSCALE_MODES) - 1

DEFAULT_RGB565      = 1
DEFAULT_PALETTE_AGG = 0
PALETTE_AGG_MAX     = 8
PALETTE_SIZES       = [0, 128, 64, 48, 32, 24, 16, 8, 4]

KMEANS_ATTEMPTS  = 1
KMEANS_MAX_ITER  = 8
TIMING_ALPHA     = 0.1

DITHER_AMT          = 2
DITHER_VAR_THRESH   = 15
SHARP_AMT           = 0.15
SHARP_EDGE_THRESH   = 300

CURSOR_OUTER_R = 8
CURSOR_INNER_R = 5

DEBUG_OVERLAY_ALPHA   = 0.85
DEBUG_SEND_INTERVAL_S = 0.5

DIAG_FPS_WARN,  DIAG_FPS_ERR   =  15,    10
DIAG_TEMP_WARN, DIAG_TEMP_ERR  =  70,    85
DIAG_JIT_WARN,  DIAG_JIT_ERR   =  10,    30
DIAG_DEC_WARN,  DIAG_DEC_ERR   =  2000,  5000
DIAG_DROP_WARN, DIAG_DROP_ERR  =  1,     5
DIAG_SRAM_WARN, DIAG_SRAM_ERR  =  50,    20

UNIX_NICE_LEVEL = -10

# ─────────────────────────────────────────────
#  THREAD-SAFE QUEUES / EVENTS
# ─────────────────────────────────────────────
raw_queue     = Queue(maxsize=1)
preview_queue = Queue(maxsize=2)
stop_event    = threading.Event()
_frame_id     = 0
_frame_id_lock = threading.Lock()

# ─────────────────────────────────────────────
#  SHARED SETTINGS
# ─────────────────────────────────────────────
_settings = {
    "fps":          DEFAULT_FPS,
    "use_rgb565":   bool(DEFAULT_RGB565),
    "palette_agg":  DEFAULT_PALETTE_AGG,
    "upscale_mode": DEFAULT_UPSCALE,
    "debug":        1,
}
_settings_lock = threading.RLock()

def get_settings():
    with _settings_lock: return dict(_settings)

def update_settings(**kw):
    with _settings_lock: _settings.update(kw)

# ─────────────────────────────────────────────
#  TILE TIMING
# ─────────────────────────────────────────────
_tile_timing_lock = threading.Lock()
_tile_timing = {"ema": [0.0]*NUM_TILES, "codec": ["?"]*NUM_TILES}

def get_tile_timing():
    with _tile_timing_lock:
        return {"ema": list(_tile_timing["ema"]), "codec": list(_tile_timing["codec"])}

def update_tile_timing(idx, ms, codec):
    with _tile_timing_lock:
        prev = _tile_timing["ema"][idx]
        _tile_timing["ema"][idx]   = prev + TIMING_ALPHA * (ms - prev)
        _tile_timing["codec"][idx] = codec

# ─────────────────────────────────────────────
#  ESP STATS
# ─────────────────────────────────────────────
_esp_stats_lock = threading.Lock()
_esp_stats      = {}
_esp_raw        = "Waiting for ESP32..."

def set_esp_stats(raw, parsed):
    global _esp_raw
    with _esp_stats_lock:
        _esp_raw = raw
        _esp_stats.update(parsed)

def get_esp_stats():
    with _esp_stats_lock: return dict(_esp_stats), _esp_raw

# Accumulation buffer for incoming serial bytes (stats packets may span reads)
_rxbuf = b""
_rxbuf_lock = threading.Lock()

def _drain_serial_stats(ser):
    """
    Non-blocking read of pending serial data from ESP32.
    Parses stats packets: [0xAB 0xCD ... \\n]
    Called from encode_send_worker between frames — never blocks.
    """
    global _rxbuf
    try:
        n = ser.in_waiting
        if n == 0:
            return
        chunk = ser.read(n)
        with _rxbuf_lock:
            _rxbuf += chunk
            # Extract all complete stats packets
            while True:
                idx = _rxbuf.find(b"\xab\xcd")
                if idx == -1:
                    _rxbuf = b""          # discard non-stats data
                    break
                rest = _rxbuf[idx + 2:]
                nl = rest.find(b"\n")
                if nl == -1:
                    _rxbuf = _rxbuf[idx:]  # keep partial packet for next call
                    break
                raw = rest[:nl].decode("utf-8", errors="ignore").strip()
                if raw:
                    set_esp_stats(raw, parse_esp_stats(raw))
                _rxbuf = rest[nl + 1:]
    except Exception:
        pass

# ─────────────────────────────────────────────
#  MONITOR RECT
# ─────────────────────────────────────────────
_monitor_rect = [0, 0, 1920, 1080]

# ─────────────────────────────────────────────
#  SYSTEM HELPERS
# ─────────────────────────────────────────────
def set_high_resolution_timer():
    if os.name == "nt":
        try: ctypes.windll.winmm.timeBeginPeriod(1)
        except: pass

def reset_resolution_timer():
    if os.name == "nt":
        try: ctypes.windll.winmm.timeEndPeriod(1)
        except: pass

def set_high_priority():
    try:
        p = psutil.Process(os.getpid())
        if os.name == "nt": p.nice(psutil.NORMAL_PRIORITY_CLASS)
        else:                p.nice(UNIX_NICE_LEVEL)
    except: pass

def get_mouse_pos():
    if os.name == "nt":
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y
    return 0, 0

# ─────────────────────────────────────────────
#  CONTENT ANALYSIS
# ─────────────────────────────────────────────
STATIC_NOISE = np.zeros((ESP_H, ESP_W, 3), dtype=np.int8)
cv2.randn(STATIC_NOISE, 0, 2)

def get_scene_metrics(frame_small):
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    var  = cv2.meanStdDev(gray)[1][0][0]
    edge = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(var), float(edge)

# ─────────────────────────────────────────────
#  QUANTIZATION
# ─────────────────────────────────────────────
def quantize_rgb565(tile_rgb):
    r = tile_rgb[:, :, 0].astype(np.uint16)
    g = tile_rgb[:, :, 1].astype(np.uint16)
    b = tile_rgb[:, :, 2].astype(np.uint16)
    r5 = (r >> 3) & 0x1F; g6 = (g >> 2) & 0x3F; b5 = (b >> 3) & 0x1F
    r8 = ((r5 << 3) | (r5 >> 2)).astype(np.uint8)
    g8 = ((g6 << 2) | (g6 >> 4)).astype(np.uint8)
    b8 = ((b5 << 3) | (b5 >> 2)).astype(np.uint8)
    return np.stack([r8, g8, b8], axis=2)

def palette_quantize(tile_rgb, n_colors):
    n_colors = max(2, min(n_colors, 256))
    pixels   = tile_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, KMEANS_MAX_ITER, 1.0)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria,
                                    KMEANS_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS)
    return np.clip(centers[labels.flatten()].reshape(tile_rgb.shape), 0, 255).astype(np.uint8)

def encode_tile(tile_bgr, use_rgb565, palette_agg, tile_idx):
    """QOI-only per-tile encoder (unchanged from UDP version)."""
    if not _QOI_AVAILABLE:
        return None, "ERR"
    t0 = time.perf_counter()
    tile_rgb = tile_bgr[:, :, ::-1].copy()
    t1 = time.perf_counter()
    if use_rgb565:
        tile_rgb = quantize_rgb565(tile_rgb)
    t2 = time.perf_counter()
    if palette_agg > 0 and palette_agg < len(PALETTE_SIZES):
        n = PALETTE_SIZES[palette_agg]
        if n > 0: tile_rgb = palette_quantize(tile_rgb, n)
    t3 = time.perf_counter()
    result = _qoi_encode_rgb(tile_rgb)
    t4 = time.perf_counter()
    elapsed_ms = (t4 - t0) * 1000
    update_tile_timing(tile_idx, elapsed_ms, "QOI")
    if elapsed_ms > 20.0:
        print(f"[SLOW T{tile_idx}] copy:{(t1-t0)*1000:.1f} "
              f"rgb565:{(t2-t1)*1000:.1f} pal:{(t3-t2)*1000:.1f} "
              f"qoi:{(t4-t3)*1000:.1f}  total:{elapsed_ms:.1f}ms")
    return result, "QOI"

# ─────────────────────────────────────────────
#  USB TRANSPORT  — replaces _send_tile_chunks()
# ─────────────────────────────────────────────
# Packet: [0x55 0xAA 0x01 frame_id tile_id enc_w enc_h len_hi len_lo] + QOI data
# No chunking: USB is reliable, the whole tile is one contiguous write.
# USB hardware flow-control naturally throttles the sender if the ESP32
# is not reading fast enough — no explicit pacing needed.

def _send_tile_usb(ser, frame_id, tile_id, qoi_bytes, enc_tile_w, enc_tile_h):
    """
    Send one QOI-encoded tile over USB CDC serial.

    Returns number of QOI bytes sent, or negative on overflow/error.
    """
    total_len = len(qoi_bytes)
    if total_len > MAX_TILE_QOI:
        print(f"[WARN] tile {tile_id} QOI too large: {total_len} > {MAX_TILE_QOI} — skipping")
        return -total_len

    header = bytes([
        0x55, 0xAA,               # sync magic
        0x01,                     # type = TILE_PKT
        frame_id  & 0xFF,         # frame counter (0-255)
        tile_id   & 0xFF,         # tile index (0-3)
        enc_tile_w & 0xFF,        # encoded tile width  (0 = full TILE_W)
        enc_tile_h & 0xFF,        # encoded tile height (0 = full TILE_H)
        (total_len >> 8) & 0xFF,  # data length, high byte
        total_len & 0xFF,         # data length, low byte
    ])
    try:
        ser.write(header + bytes(qoi_bytes))
        return total_len
    except serial.SerialException as exc:
        print(f"[ERR] serial write failed (tile {tile_id}): {exc}")
        return 0


def _send_debug_cmd(ser, debug_state):
    """Send SET_DEBUG command packet: [0x55 0xAA 0x02 0x01 state]."""
    try:
        ser.write(bytes([0x55, 0xAA, 0x02, 0x01, int(bool(debug_state))]))
    except Exception:
        pass

# ─────────────────────────────────────────────
#  CAPTURE WORKER  (Thread 1) — unchanged
# ─────────────────────────────────────────────
def capture_worker(monitor_idx):
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_idx]
        while not stop_event.is_set():
            raw = sct.grab(monitor)
            frame_bgr = np.array(raw, dtype=np.uint8)[:, :, :3].copy()
            if raw_queue.full():
                try: raw_queue.get_nowait()
                except Empty: pass
            raw_queue.put(frame_bgr)

# ─────────────────────────────────────────────
#  ENCODE + SEND WORKER  (Thread 2)
#  Encodes 4 tiles in parallel, sends over USB.
#  USB flow-control replaces pacing; no CHUNK_DATA_SIZE splitting needed.
# ─────────────────────────────────────────────
def encode_send_worker(ser):
    global _frame_id
    last_debug_send = 0
    last_frame_t    = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_TILES) as pool:
        while not stop_event.is_set():
            try:
                frame_bgr = raw_queue.get(timeout=0.5)
            except Empty:
                continue

            cfg         = get_settings()
            fps         = max(1, cfg["fps"])
            use_rgb565  = cfg["use_rgb565"]
            palette_agg = cfg["palette_agg"]
            debug_state = cfg["debug"]

            # Drain incoming stats from ESP32 (non-blocking)
            _drain_serial_stats(ser)

            # Debug toggle — send command every 0.5 s
            if time.time() - last_debug_send > DEBUG_SEND_INTERVAL_S:
                _send_debug_cmd(ser, debug_state)
                last_debug_send = time.time()

            # Upscale mode
            upscale_mode = cfg.get("upscale_mode", 0)
            umode  = UPSCALE_MODES[upscale_mode]
            enc_w  = umode["enc_w"];  enc_h  = umode["enc_h"]
            tile_w = umode["tile_w"]; tile_h = umode["tile_h"]

            tile_x = [0, tile_w, 0,      tile_w]
            tile_y = [0, 0,      tile_h, tile_h]

            # Resize to encode resolution
            resized = cv2.resize(frame_bgr, (enc_w, enc_h),
                                 interpolation=cv2.INTER_AREA)

            # Cursor overlay
            mx, my = get_mouse_pos()
            rx = int((mx - _monitor_rect[0]) * enc_w / max(_monitor_rect[2], 1))
            ry = int((my - _monitor_rect[1]) * enc_h / max(_monitor_rect[3], 1))
            if 0 <= rx < enc_w and 0 <= ry < enc_h:
                cv2.circle(resized, (rx, ry), CURSOR_OUTER_R, (255, 255, 255), 2)
                cv2.circle(resized, (rx, ry), CURSOR_INNER_R, (0,   0, 255),  -1)

            # Content-adaptive pre-processing
            var, edges = get_scene_metrics(resized)
            if var < DITHER_VAR_THRESH:
                noise_scaled = cv2.resize(STATIC_NOISE, (enc_w, enc_h),
                                          interpolation=cv2.INTER_NEAREST)
                n = (noise_scaled.astype(np.float32) * (DITHER_AMT / 2.0)).astype(np.int8)
                resized = cv2.add(resized, n, dtype=cv2.CV_8U)
            if edges > SHARP_EDGE_THRESH:
                k = np.array([[0, -SHARP_AMT, 0],
                              [-SHARP_AMT, 1 + 4*SHARP_AMT, -SHARP_AMT],
                              [0, -SHARP_AMT, 0]])
                resized = cv2.filter2D(resized, -1, k)

            # Frame ID
            with _frame_id_lock:
                frame_id  = _frame_id & 0xFF
                _frame_id = (_frame_id + 1) & 0xFF

            # Extract tiles
            tiles_bgr = [
                resized[tile_y[t]:tile_y[t]+tile_h, tile_x[t]:tile_x[t]+tile_w].copy()
                for t in range(NUM_TILES)
            ]

            # Parallel encode
            futures = [
                pool.submit(encode_tile, tiles_bgr[t], use_rgb565, palette_agg, t)
                for t in range(NUM_TILES)
            ]
            encoded = [f.result() for f in futures]

            # Send all tiles over USB (sequential — serial port is one stream)
            total_bytes = 0
            tile_sizes  = []
            for t, (tile_bytes, _codec) in enumerate(encoded):
                if tile_bytes is None:
                    tile_sizes.append(0)
                    continue
                n = _send_tile_usb(ser, frame_id, t, tile_bytes, tile_w, tile_h)
                tile_sizes.append(n)
                if n > 0: total_bytes += n

            # FPS cap — sleep the remaining frame budget
            target_interval = 1.0 / fps
            now = time.perf_counter()
            elapsed = now - last_frame_t
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            last_frame_t = time.perf_counter()

            # Push preview to UI thread
            bw_kbps  = total_bytes * fps * 8 / 1000
            t_timing = get_tile_timing()
            pkt = {
                "enc_frame":     resized,
                "enc_w":         enc_w,    "enc_h":         enc_h,
                "tile_w":        tile_w,   "tile_h":        tile_h,
                "upscale_mode":  upscale_mode,
                "upscale_label": umode["label"],
                "tile_sizes":    tile_sizes,
                "total_bytes":   total_bytes,
                "bw_kbps":       bw_kbps,
                "use_rgb565":    use_rgb565,
                "palette_agg":   palette_agg,
                "tile_ema_ms":   t_timing["ema"],
                "tile_codec":    t_timing["codec"],
            }
            if preview_queue.full():
                try: preview_queue.get_nowait()
                except Empty: pass
            preview_queue.put(pkt)

# ─────────────────────────────────────────────
#  STATS PARSING + DISPLAY HELPERS — unchanged
# ─────────────────────────────────────────────
def parse_esp_stats(raw):
    result = {}
    for part in raw.split("|"):
        if ":" in part:
            k, v = part.split(":", 1)
            result[k.strip()] = v.strip()
    return result

def _diag_color(val_str, warn, err):
    try:
        v = float(val_str.split("/")[0])
        if v >= err:  return (0,   0, 255)
        if v >= warn: return (0, 165, 255)
    except: pass
    return (0, 255, 0)

def _sram_color(free_total_str):
    try:
        free = float(free_total_str.split("/")[0])
        if free < DIAG_SRAM_ERR:  return (0,   0, 255)
        if free < DIAG_SRAM_WARN: return (0, 165, 255)
    except: pass
    return (0, 255, 0)

# ─────────────────────────────────────────────
#  DISPLAY SELECTION — unchanged
# ─────────────────────────────────────────────
def select_display_mss():
    with mss.mss() as sct:
        monitors = sct.monitors[1:]
    if len(monitors) == 1:
        return 1
    areas = [m["width"] * m["height"] for m in monitors]
    primary_idx = areas.index(max(areas))
    for i, m in enumerate(monitors):
        if i != primary_idx:
            print(f"[*] Auto-selected monitor {i+1}: {m['width']}x{m['height']} "
                  f"at ({m['left']},{m['top']}) -- smaller than primary")
            return i + 1
    print("All monitors same size -- please select:")
    for i, m in enumerate(monitors, 1):
        print(f"  [{i}] {m['width']}x{m['height']} at ({m['left']},{m['top']})")
    try:   return int(input("Select monitor [1]: ") or "1")
    except ValueError: return 1

# ─────────────────────────────────────────────
#  USB PORT DISCOVERY  — replaces quick_find_esp()
# ─────────────────────────────────────────────
def find_esp32_port(timeout_s: float = 15.0) -> str | None:
    """
    Scan COM ports for an ESP32-S3 by Espressif USB VID (0x303A).
    Polls every 0.5 s for up to timeout_s seconds — handles the case where
    the device is still enumerating when the script starts.

    Returns the device path (e.g. "COM7" or "/dev/ttyACM0"), or None.
    """
    print(f"[USB] Scanning for ESP32-S3 (VID 0x{ESP32_VID:04X})  "
          f"— {timeout_s:.0f} s timeout...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for p in serial.tools.list_ports.comports():
            if p.vid == ESP32_VID:
                print(f"[USB] Found: {p.device}  "
                      f"VID:0x{p.vid:04X} PID:0x{p.pid:04X}  "
                      f"({p.description})")
                return p.device
        time.sleep(0.5)
    return None


def open_esp32_serial(port: str) -> serial.Serial:
    """
    Open the serial port and wait for the ESP32's "ESP32_READY" greeting.
    The greeting is sent by networkTask() ~200 ms after USB enumeration.
    If the greeting never arrives (e.g. ESP32 already running), we proceed
    anyway — streaming still works.
    """
    ser = serial.Serial(
        port,
        baudrate    = SERIAL_BAUD,   # ignored by USB CDC driver; needed by pyserial
        timeout     = SERIAL_TIMEOUT_R,
        write_timeout = SERIAL_TIMEOUT_W,
    )
    ser.reset_input_buffer()
    print(f"[USB] Opened {port}. Waiting for ESP32_READY ({SERIAL_READY_WAIT:.0f}s)...")

    deadline = time.time() + SERIAL_READY_WAIT
    while time.time() < deadline:
        if ser.in_waiting:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                print(f"[USB] ESP32 says: {line!r}")
                if "ESP32_READY" in line:
                    print("[USB] ESP32 ready — streaming starts now.")
                    return ser
            except Exception:
                pass
        time.sleep(0.05)

    print("[USB] No ESP32_READY received (device may already be running). "
          "Proceeding anyway.")
    return ser

# ─────────────────────────────────────────────
#  MAIN STREAM + UI  (main thread — UI only)
# ─────────────────────────────────────────────
def stream_mss_usb(serial_port: str, monitor_idx: int):
    global _monitor_rect

    ser = open_esp32_serial(serial_port)

    with mss.mss() as sct:
        m = sct.monitors[monitor_idx]
        _monitor_rect[:] = [m["left"], m["top"], m["width"], m["height"]]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, UI_W, UI_H)
    cv2.createTrackbar("Max FPS",      WINDOW_NAME, DEFAULT_FPS,          60,              lambda x: None)
    cv2.createTrackbar("RGB565 Quant", WINDOW_NAME, DEFAULT_RGB565,        1,               lambda x: None)
    cv2.createTrackbar("Palette Agg",  WINDOW_NAME, DEFAULT_PALETTE_AGG,   PALETTE_AGG_MAX, lambda x: None)
    cv2.createTrackbar("Upscale Mode", WINDOW_NAME, DEFAULT_UPSCALE,       UPSCALE_MODE_MAX,lambda x: None)
    cv2.createTrackbar("Debug Info",   WINDOW_NAME, 1,                     1,               lambda x: None)

    threading.Thread(target=capture_worker,     args=(monitor_idx,), daemon=True).start()
    threading.Thread(target=encode_send_worker, args=(ser,),         daemon=True).start()

    last_pkt = None

    try:
        while True:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            t_start = time.perf_counter()

            fps          = cv2.getTrackbarPos("Max FPS",      WINDOW_NAME)
            use_rgb565   = cv2.getTrackbarPos("RGB565 Quant", WINDOW_NAME) == 1
            palette_agg  = cv2.getTrackbarPos("Palette Agg",  WINDOW_NAME)
            upscale_mode = cv2.getTrackbarPos("Upscale Mode", WINDOW_NAME)
            debug_state  = cv2.getTrackbarPos("Debug Info",   WINDOW_NAME)
            update_settings(fps=fps, use_rgb565=use_rgb565, palette_agg=palette_agg,
                            upscale_mode=upscale_mode, debug=debug_state)

            try:
                last_pkt = preview_queue.get_nowait()
            except Empty:
                pass

            if last_pkt is None:
                blank = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
                cv2.putText(blank, f"Waiting for first frame...  [{serial_port}]",
                            (30, PREVIEW_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (180, 180, 180), 1)
                cv2.imshow(WINDOW_NAME, blank)
                if cv2.waitKey(30) & 0xFF == ord("q"): break
                continue

            pkt          = last_pkt
            enc_frame    = pkt["enc_frame"]
            pkt_enc_w    = pkt.get("enc_w",   ESP_W)
            pkt_enc_h    = pkt.get("enc_h",   ESP_H)
            pkt_up_label = pkt.get("upscale_label", "Full")
            tile_sizes   = pkt["tile_sizes"]
            total_bytes  = pkt["total_bytes"]
            bw_kbps      = pkt["bw_kbps"]
            pkt_rgb565   = pkt["use_rgb565"]
            pkt_pal_agg  = pkt["palette_agg"]
            tile_ema_ms  = pkt.get("tile_ema_ms", [0.0]*NUM_TILES)
            tile_codec   = pkt.get("tile_codec",  ["?"]*NUM_TILES)

            preview = cv2.resize(enc_frame, (PREVIEW_W, PREVIEW_H),
                                 interpolation=cv2.INTER_NEAREST)
            f, _ = get_esp_stats()

            p_label = ("off" if pkt_pal_agg == 0
                       else f"{PALETTE_SIZES[pkt_pal_agg]}col"
                       if pkt_pal_agg < len(PALETTE_SIZES) else "?")

            if debug_state == 1:
                overlay = preview.copy()
                cv2.rectangle(overlay, (0, 0), (PREVIEW_W, PREVIEW_H), (0, 0, 0), -1)
                preview = cv2.addWeighted(overlay, DEBUG_OVERLAY_ALPHA,
                                          preview, 1.0 - DEBUG_OVERLAY_ALPHA, 0)
                dashboard = [
                    (f"PORT : {serial_port}",                              (180, 220, 255)),
                    (f"FPS  : {f.get('FPS',  '?'):>8}",   _diag_color(f.get("FPS","0"),  DIAG_FPS_WARN,  DIAG_FPS_ERR)),
                    (f"TEMP : {f.get('TEMP', '?'):>7} C", _diag_color(f.get("TEMP","0"), DIAG_TEMP_WARN, DIAG_TEMP_ERR)),
                    (f"JIT  : {f.get('JIT',  '?'):>7} ms",_diag_color(f.get("JIT","0"),  DIAG_JIT_WARN,  DIAG_JIT_ERR)),
                    (f"DEC  : {f.get('DEC',  '?'):>7} us",_diag_color(f.get("DEC","0"),  DIAG_DEC_WARN,  DIAG_DEC_ERR)),
                    (f"DROP : {f.get('DROP', '?'):>8}",   _diag_color(f.get("DROP","0"), DIAG_DROP_WARN, DIAG_DROP_ERR)),
                    (f"ABRT : {f.get('ABRT', '?'):>8}",   _diag_color(f.get("ABRT","0"), DIAG_DROP_WARN, DIAG_DROP_ERR)),
                    (f"SRAM : {f.get('SRAM',  '?/?'):>11} KB", _sram_color(f.get("SRAM","999/1"))),
                    (f"PSRAM: {f.get('PSRAM', '?/?'):>11} KB", _sram_color(f.get("PSRAM","999/1"))),
                    (f"BW   : {bw_kbps:>7.0f} kbps (USB)",      (0, 255, 255)),
                    (f"TILE : {total_bytes//NUM_TILES if total_bytes else 0:>6}B avg", (0, 255, 255)),
                    (f"QUANT: RGB565={'ON' if pkt_rgb565 else 'off'}  Pal={p_label}", (200, 200, 200)),
                    (f"SCALE: {pkt_up_label}  enc={pkt_enc_w}x{pkt_enc_h}",          (200, 220, 255)),
                    (f"ENC  : {'  '.join(f'{tile_ema_ms[t]:.1f}ms' for t in range(NUM_TILES))}", (180, 255, 180)),
                ]
                y = 16
                for text, color in dashboard:
                    cv2.putText(preview, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)
                    y += 20

                bar_y = PREVIEW_H - 44
                bar_w = PREVIEW_W // NUM_TILES
                for t, tsz in enumerate(tile_sizes):
                    overflow  = tsz < 0
                    sz        = abs(tsz)
                    fill      = min(int(bar_w * sz / MAX_TILE_QOI), bar_w)
                    bar_color = (0, 0, 255) if overflow else (0, 200, 0)
                    cv2.rectangle(preview, (t*bar_w, bar_y),
                                  (t*bar_w+fill, bar_y+12), bar_color, -1)
                    size_lbl = f"{sz//1024}K" if sz >= 1024 else f"{sz}B"
                    cv2.putText(preview, size_lbl,
                                (t*bar_w+2, bar_y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255,255,255), 1)
                    ema_t = tile_ema_ms[t] if t < len(tile_ema_ms) else 0.0
                    cv2.putText(preview, f"QOI {ema_t:.1f}ms",
                                (t*bar_w+2, bar_y+26), cv2.FONT_HERSHEY_SIMPLEX, 0.34, bar_color, 1)
            else:
                per_tile = total_bytes // NUM_TILES if total_bytes else 0
                info = (f"QOI {total_bytes}B  PerTile:~{per_tile}B  "
                        f"BW:{bw_kbps:.0f}kbps(USB)  "
                        f"Scale:{pkt_up_label}({pkt_enc_w}x{pkt_enc_h})  "
                        f"RGB565={'ON' if pkt_rgb565 else 'off'}  Pal={p_label}  "
                        f"Port:{serial_port}")
                cv2.putText(preview, info, (10, PREVIEW_H-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,0), 1)

            cv2.imshow(WINDOW_NAME, preview)

            elapsed  = time.perf_counter() - t_start
            wait_ms  = max(1, int((1.0/30.0 - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        try: ser.close()
        except: pass

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if not _QOI_AVAILABLE:
        print("[ERROR] Install QOI encoder:  pip install imagecodecs")
        sys.exit(1)

    set_high_priority()
    set_high_resolution_timer()

    # Allow manual port override as first CLI argument
    manual_port = sys.argv[1] if len(sys.argv) > 1 else None

    if manual_port:
        port = manual_port
        print(f"[USB] Using manually specified port: {port}")
    else:
        port = find_esp32_port(timeout_s=15.0)

    if not port:
        print("\n[!] ESP32-S3 not found automatically.")
        print("    Tip: check Device Manager / lsusb for an Espressif USB device.")
        manual = input("    Enter port manually (e.g. COM7 or /dev/ttyACM0), or press Enter to exit: ").strip()
        if not manual:
            reset_resolution_timer()
            sys.exit(1)
        port = manual

    stream_mss_usb(port, select_display_mss())
    reset_resolution_timer()