import socket
import time
import cv2
import mss
import numpy as np
import os
import psutil
import ctypes
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

# ─────────────────────────────────────────────
#  QOI BACKEND DETECTION
#  Priority: imagecodecs (fastest C impl) > qoi C-ext > qoi pure-Python
#  pip install imagecodecs   (recommended — AVX2 SIMD, ~0.5ms/tile)
#  pip install qoi           (fallback — C ext ~2ms, pure-Python ~15ms)
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
        import importlib.util as _ilu, pathlib as _pl
        _spec = _ilu.find_spec("qoi")
        _is_native = _spec is not None and not str(_spec.origin).endswith(".py")
        _QOI_BACKEND = "qoi-native" if _is_native else "qoi-python"
        if not _is_native:
            print("[WARN] qoi is pure-Python (~15ms/tile). "
                  "Run: pip install imagecodecs  for a fast C backend.")
    except ImportError:
        print("[ERROR] No QOI encoder. Run: pip install imagecodecs")

_QOI_AVAILABLE = _QOI_BACKEND is not None
print(f"[QOI] backend: {_QOI_BACKEND}")

def _qoi_encode_rgb(tile_rgb):
    """Encode HxWx3 uint8 RGB ndarray to QOI bytes via best available backend."""
    if _QOI_BACKEND == "imagecodecs":
        return bytes(_ic_qoi_encode(tile_rgb))
    return bytes(_qoi_lib.encode(tile_rgb))

# ─────────────────────────────────────────────
#  GLOBAL SETTINGS
# ─────────────────────────────────────────────
PORT         = 12345
ESP_W, ESP_H = 320, 240

CHUNK_DATA_SIZE  = 1400
NUM_TILES        = 4
TILE_W, TILE_H   = 160, 120

TILE_X = [  0, 160,   0, 160]
TILE_Y = [  0,   0, 120, 120]

MAX_TILE_QOI = 30 * CHUNK_DATA_SIZE     # 42 000 B — must match main.cpp

WINDOW_NAME          = "QOI Stream Control"
UI_W, UI_H           = 480, 620
PREVIEW_W, PREVIEW_H = 480, 360
DEFAULT_FPS          = 30
DEFAULT_PACING_STEPS = 10
PACING_MAX_STEPS     = 20
PACING_STEP_S        = 0.0001

# Upscale mode — selects encode resolution, ESP side nearest-neighbour upscales to full tile.
#   0 = Full  320x240  tiles 160x120  (no upscale, highest quality, most bandwidth)
#   1 = 2/3   214x160  tiles 107x80   (ESP 3:2 upscale, ~55% pixels to encode)
#   2 = 1/2   160x120  tiles 80x60    (ESP 2x upscale, ~25% pixels to encode, lowest BW)
# Tile dims and encode resolution for each mode:
UPSCALE_MODES = [
    {"label": "Full",  "enc_w": 320, "enc_h": 240, "tile_w": 160, "tile_h": 120},
    {"label": "2/3",   "enc_w": 214, "enc_h": 160, "tile_w": 107, "tile_h":  80},
    {"label": "1/2",   "enc_w": 160, "enc_h": 120, "tile_w":  80, "tile_h":  60},
]
DEFAULT_UPSCALE  = 0
UPSCALE_MODE_MAX = len(UPSCALE_MODES) - 1

DEFAULT_RGB565       = 1
DEFAULT_PALETTE_AGG  = 0
PALETTE_AGG_MAX      = 8
PALETTE_SIZES        = [0, 128, 64, 48, 32, 24, 16, 8, 4]

KMEANS_ATTEMPTS      = 1
KMEANS_MAX_ITER      = 8

# Per-tile encode timing EMA (shown in debug overlay, ms)
TIMING_ALPHA         = 0.1

DITHER_AMT           = 2
DITHER_VAR_THRESH    = 15
SHARP_AMT            = 0.15
SHARP_EDGE_THRESH    = 300

CURSOR_OUTER_R       = 8
CURSOR_INNER_R       = 5

DEBUG_OVERLAY_ALPHA    = 0.85
DEBUG_SEND_INTERVAL_S  = 0.5

DIAG_FPS_WARN,  DIAG_FPS_ERR   =  15,    10
DIAG_TEMP_WARN, DIAG_TEMP_ERR  =  70,    85
DIAG_JIT_WARN,  DIAG_JIT_ERR   =  10,    30
DIAG_DEC_WARN,  DIAG_DEC_ERR   =  2000,  5000
DIAG_DROP_WARN, DIAG_DROP_ERR  =  1,     5
DIAG_SRAM_WARN, DIAG_SRAM_ERR  =  50,    20

ESP_BEACON_TIMEOUT_S = 5.0
SEND_RETRY_SLEEP_S   = 0.0005
UNIX_NICE_LEVEL      = -10

# ─────────────────────────────────────────────
#  THREAD-SAFE QUEUES
#
#  Thread 1 capture_worker  →  raw_queue (size 1, drop-on-full)
#                                   ↓
#  Thread 2 encode_worker   →  preview_queue (size 2) → UI thread
#                           →  UDP socket (direct send)
#
#  Main thread:  UI only — reads preview_queue, shows window, reads trackbars.
#                Never touches encoding, sending, or capturing.
# ─────────────────────────────────────────────
raw_queue      = Queue(maxsize=1)
preview_queue  = Queue(maxsize=2)
stop_event     = threading.Event()
_frame_id      = 0
_frame_id_lock = threading.Lock()

# ─────────────────────────────────────────────
#  SHARED SETTINGS  (main thread writes, encode thread reads)
# ─────────────────────────────────────────────
_settings = {
    "fps":         DEFAULT_FPS,
    "pacing_s":    DEFAULT_PACING_STEPS * PACING_STEP_S,
    "use_rgb565":  bool(DEFAULT_RGB565),
    "palette_agg":   DEFAULT_PALETTE_AGG,
    "upscale_mode":  DEFAULT_UPSCALE,
    "debug":         1,
}

# Per-tile encode timing (EMA, ms) — written by encode thread, read by UI thread.
# Index 0-3 = tiles; index 4 = whole-frame total.
# "codec" tracks which codec was used last frame per tile ("QOI" / "ERR").
_tile_timing_lock = threading.Lock()
_tile_timing = {
    "ema":   [0.0] * NUM_TILES,
    "codec": ["?"] * NUM_TILES,
}

def get_tile_timing():
    with _tile_timing_lock:
        return {
            "ema":   list(_tile_timing["ema"]),
            "codec": list(_tile_timing["codec"]),
        }

def update_tile_timing(idx, ms, codec):
    with _tile_timing_lock:
        prev = _tile_timing["ema"][idx]
        _tile_timing["ema"][idx]   = prev + TIMING_ALPHA * (ms - prev)
        _tile_timing["codec"][idx] = codec
_settings_lock = threading.RLock()

def get_settings():
    with _settings_lock:
        return dict(_settings)

def update_settings(**kwargs):
    with _settings_lock:
        _settings.update(kwargs)

# ─────────────────────────────────────────────
#  LATEST ESP STATS  (encode thread writes, UI thread reads)
# ─────────────────────────────────────────────
_esp_stats_lock = threading.Lock()
_esp_stats      = {}
_esp_raw        = "Waiting for ESP32-S3..."

def set_esp_stats(raw, parsed):
    global _esp_raw
    with _esp_stats_lock:
        _esp_raw = raw
        _esp_stats.update(parsed)

def get_esp_stats():
    with _esp_stats_lock:
        return dict(_esp_stats), _esp_raw

# ─────────────────────────────────────────────
#  MONITOR RECT  (set before threads start, read by encode thread for cursor)
# ─────────────────────────────────────────────
_monitor_rect = [0, 0, 1920, 1080]   # [left, top, width, height]

# ─────────────────────────────────────────────
#  SYSTEM HELPERS
# ─────────────────────────────────────────────
def set_high_resolution_timer():
    if os.name == 'nt':
        try: ctypes.windll.winmm.timeBeginPeriod(1)
        except: pass

def reset_resolution_timer():
    if os.name == 'nt':
        try: ctypes.windll.winmm.timeEndPeriod(1)
        except: pass

def set_high_priority():
    try:
        p = psutil.Process(os.getpid())
        if os.name == 'nt': p.nice(psutil.NORMAL_PRIORITY_CLASS)
        else:                p.nice(UNIX_NICE_LEVEL)
    except: pass

def get_mouse_pos():
    if os.name == 'nt':
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
    gray         = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    var          = cv2.meanStdDev(gray)[1][0][0]
    edge_density = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(var), float(edge_density)

# ─────────────────────────────────────────────
#  QUANTIZATION
# ─────────────────────────────────────────────
def quantize_rgb565(tile_rgb):
    r = tile_rgb[:, :, 0].astype(np.uint16)
    g = tile_rgb[:, :, 1].astype(np.uint16)
    b = tile_rgb[:, :, 2].astype(np.uint16)
    r5 = (r >> 3) & 0x1F
    g6 = (g >> 2) & 0x3F
    b5 = (b >> 3) & 0x1F
    r8 = ((r5 << 3) | (r5 >> 2)).astype(np.uint8)
    g8 = ((g6 << 2) | (g6 >> 4)).astype(np.uint8)
    b8 = ((b5 << 3) | (b5 >> 2)).astype(np.uint8)
    return np.stack([r8, g8, b8], axis=2)

def palette_quantize(tile_rgb, n_colors):
    n_colors = max(2, min(n_colors, 256))
    pixels   = tile_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                KMEANS_MAX_ITER, 1.0)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria,
                                    KMEANS_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS)
    return np.clip(centers[labels.flatten()].reshape(tile_rgb.shape),
                   0, 255).astype(np.uint8)

def encode_tile(tile_bgr, use_rgb565, palette_agg, tile_idx):
    """
    QOI-only per-tile encoder with per-stage timing.

    Pipeline:
      1. BGR -> RGB  (.copy() ensures contiguous — avoids cv2 crash on mss slices)
      2. RGB565 quantize  (optional — massively improves QOI run compression)
      3. Palette quantize (optional — k-means to N colors)
      4. QOI encode via best available backend (imagecodecs > qoi C-ext > qoi-python)

    Timing: wall time of the full pipeline fed into per-tile EMA.
    Returns (encoded_bytes, stage_label) or (None, "ERR") on failure.
    """
    if not _QOI_AVAILABLE:
        return None, "ERR"

    t0 = time.perf_counter()

    tile_rgb = tile_bgr[:, :, ::-1].copy()   # BGR->RGB, contiguous

    t1 = time.perf_counter()
    if use_rgb565:
        tile_rgb = quantize_rgb565(tile_rgb)

    t2 = time.perf_counter()
    if palette_agg > 0 and palette_agg < len(PALETTE_SIZES):
        n = PALETTE_SIZES[palette_agg]
        if n > 0:
            tile_rgb = palette_quantize(tile_rgb, n)

    t3 = time.perf_counter()
    result = _qoi_encode_rgb(tile_rgb)
    t4 = time.perf_counter()

    elapsed_ms = (t4 - t0) * 1000
    update_tile_timing(tile_idx, elapsed_ms, "QOI")

    # Stage breakdown stored for debug — only printed when timing is anomalous
    if elapsed_ms > 20.0:
        print(f"[SLOW T{tile_idx}] copy:{(t1-t0)*1000:.1f} "
              f"rgb565:{(t2-t1)*1000:.1f} "
              f"pal:{(t3-t2)*1000:.1f} "
              f"qoi:{(t4-t3)*1000:.1f}  total:{elapsed_ms:.1f}ms")

    return result, "QOI"

# ─────────────────────────────────────────────
#  TRANSMIT — chunked UDP
# ─────────────────────────────────────────────
def _send_tile_chunks(sock, dest, frame_id, tId, qoi_bytes,
                      pacing_s, enc_tile_w, enc_tile_h):
    """
    Send one QOI-encoded tile as chunked UDP packets.

    Header layout (10 bytes):
      [0] 0xAA          magic
      [1] 0xBB          magic
      [2] frame_id      0-255 rolling
      [3] tile_id       0-3
      [4] chunk_id      0-based
      [5] total_chunks
      [6] size_hi       encoded byte count high byte
      [7] size_lo       encoded byte count low byte
      [8] enc_tile_w    encoded tile width  (0 = use default TILE_W=160)
      [9] enc_tile_h    encoded tile height (0 = use default TILE_H=120)

    ESP side reads [8][9] and calls pushImage with those dimensions so
    LovyanGFX stretches the decoded pixels to the full 160x120 tile slot
    — nearest-neighbour hardware upscale, zero extra buffer.
    """
    total_len = len(qoi_bytes)
    if total_len > MAX_TILE_QOI:
        return -total_len
    num_chunks = (total_len + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE
    size_hi    = (total_len >> 8) & 0xFF
    size_lo    =  total_len       & 0xFF
    # Encode dims: 0 signals full-size (ESP uses its TILE_W/TILE_H defaults)
    hdr_w = enc_tile_w  & 0xFF
    hdr_h = enc_tile_h  & 0xFF
    buf = bytearray(10 + CHUNK_DATA_SIZE)
    buf[0]=0xAA; buf[1]=0xBB; buf[2]=frame_id; buf[3]=tId
    buf[6]=size_hi; buf[7]=size_lo; buf[8]=hdr_w; buf[9]=hdr_h
    for cId in range(num_chunks):
        offset = cId * CHUNK_DATA_SIZE
        clen   = min(CHUNK_DATA_SIZE, total_len - offset)
        buf[4]=cId; buf[5]=num_chunks
        buf[10:10+clen] = qoi_bytes[offset:offset+clen]
        while True:
            try:
                sock.sendto(bytes(buf[:10+clen]), dest)
                break
            except BlockingIOError:
                time.sleep(SEND_RETRY_SLEEP_S)
        if pacing_s > 0:
            time.sleep(pacing_s)
    return total_len

# ─────────────────────────────────────────────
#  CAPTURE WORKER  (Thread 1)
#  Grabs screen continuously, keeps only latest frame.
#  .copy() on the slice makes it contiguous — fixes the cv2 crash.
# ─────────────────────────────────────────────
def capture_worker(monitor_idx):
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_idx]
        while not stop_event.is_set():
            raw = sct.grab(monitor)
            # mss BGRA -> BGR, explicit .copy() = contiguous writable array
            frame_bgr = np.array(raw, dtype=np.uint8)[:, :, :3].copy()
            if raw_queue.full():
                try: raw_queue.get_nowait()
                except Empty: pass
            raw_queue.put(frame_bgr)

# ─────────────────────────────────────────────
#  ENCODE + SEND WORKER  (Thread 2)
#  Encodes all 4 tiles in PARALLEL with a 4-thread pool, then sends.
#  One ThreadPoolExecutor lives for the whole session — no spawn overhead.
# ─────────────────────────────────────────────
def encode_send_worker(target_ip, sock):
    global _frame_id
    dest = (target_ip, PORT)
    last_debug_send = 0

    with ThreadPoolExecutor(max_workers=NUM_TILES) as pool:
        while not stop_event.is_set():
            try:
                frame_bgr = raw_queue.get(timeout=0.5)
            except Empty:
                continue

            cfg = get_settings()
            fps         = max(1, cfg["fps"])
            use_rgb565  = cfg["use_rgb565"]
            palette_agg = cfg["palette_agg"]
            pacing_s    = cfg["pacing_s"]
            debug_state = cfg["debug"]
            # upscale_mode read inside loop body below

            # Drain any incoming ESP stat packets
            try:
                while True:
                    data, _ = sock.recvfrom(512)
                    if len(data) > 2 and data[0] == 0xAB:
                        raw = data[2:].decode('utf-8', errors='ignore')
                        set_esp_stats(raw, parse_esp_stats(raw))
            except Exception:
                pass

            # Debug toggle
            if time.time() - last_debug_send > DEBUG_SEND_INTERVAL_S:
                try:
                    sock.sendto(bytes([0xAA, 0xCC, 0x01, debug_state]), dest)
                except Exception:
                    pass
                last_debug_send = time.time()

            upscale_mode = cfg.get("upscale_mode", 0)
            umode        = UPSCALE_MODES[upscale_mode]
            enc_w        = umode["enc_w"]
            enc_h        = umode["enc_h"]
            tile_w       = umode["tile_w"]
            tile_h       = umode["tile_h"]

            # Tile layout for current encode resolution (always 2x2 grid)
            tile_x = [0, tile_w, 0,      tile_w]
            tile_y = [0, 0,      tile_h, tile_h]

            # Resize to encode resolution — INTER_AREA is best for downscale
            resized = cv2.resize(frame_bgr, (enc_w, enc_h),
                                 interpolation=cv2.INTER_AREA)

            # Cursor overlay — scale to encode resolution space
            mx, my = get_mouse_pos()
            rx = int((mx - _monitor_rect[0]) * enc_w / max(_monitor_rect[2], 1))
            ry = int((my - _monitor_rect[1]) * enc_h / max(_monitor_rect[3], 1))
            if 0 <= rx < enc_w and 0 <= ry < enc_h:
                cv2.circle(resized, (rx, ry), CURSOR_OUTER_R, (255, 255, 255), 2)
                cv2.circle(resized, (rx, ry), CURSOR_INNER_R, (0,   0, 255),  -1)

            # Content-adaptive pre-processing
            var, edges = get_scene_metrics(resized)
            if var < DITHER_VAR_THRESH:
                # Scale noise to encode resolution
                noise_scaled = cv2.resize(
                    STATIC_NOISE, (enc_w, enc_h), interpolation=cv2.INTER_NEAREST)
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

            # Extract tiles at current encode resolution
            tiles_bgr = [
                resized[tile_y[t]:tile_y[t]+tile_h,
                        tile_x[t]:tile_x[t]+tile_w].copy()
                for t in range(NUM_TILES)
            ]

            # Parallel encode: all 4 tiles simultaneously
            # encode_tile() returns (bytes, codec_str) — codec tracked for overlay
            futures = [
                pool.submit(encode_tile, tiles_bgr[t], use_rgb565, palette_agg, t)
                for t in range(NUM_TILES)
            ]
            encoded = [f.result() for f in futures]   # list of (bytes|None, codec_str)

            # Send all tiles
            total_bytes = 0
            tile_sizes  = []
            for t, (tile_bytes, codec) in enumerate(encoded):
                if tile_bytes is None:
                    tile_sizes.append(0)
                    continue
                n = _send_tile_chunks(sock, dest, frame_id, t, tile_bytes,
                                      pacing_s, tile_w, tile_h)
                tile_sizes.append(n)
                if n > 0:
                    total_bytes += n

            # Push preview + stats to UI thread
            bw_kbps  = total_bytes * fps * 8 / 1000
            t_timing = get_tile_timing()
            pkt = {
                "enc_frame":    resized,        # at encode resolution (not always 320x240)
                "enc_w":        enc_w,
                "enc_h":        enc_h,
                "tile_w":       tile_w,
                "tile_h":       tile_h,
                "upscale_mode": upscale_mode,
                "upscale_label":umode["label"],
                "tile_sizes":   tile_sizes,
                "total_bytes":  total_bytes,
                "bw_kbps":      bw_kbps,
                "use_rgb565":   use_rgb565,
                "palette_agg":  palette_agg,
                "tile_ema_ms":  t_timing["ema"],
                "tile_codec":   t_timing["codec"],
            }
            if preview_queue.full():
                try: preview_queue.get_nowait()
                except Empty: pass
            preview_queue.put(pkt)

            # FPS cap in encode thread
            # (only throttle if we're running faster than target)

# ─────────────────────────────────────────────
#  ESP STAT PARSING
# ─────────────────────────────────────────────
def parse_esp_stats(raw):
    result = {}
    for part in raw.split('|'):
        if ':' in part:
            k, v = part.split(':', 1)
            result[k.strip()] = v.strip()
    return result

def _diag_color(val_str, warn, err):
    try:
        v = float(val_str.split('/')[0])
        if v >= err:  return (0,   0, 255)
        if v >= warn: return (0, 165, 255)
    except: pass
    return (0, 255, 0)

def _sram_color(free_total_str):
    try:
        free = float(free_total_str.split('/')[0])
        if free < DIAG_SRAM_ERR:  return (0,   0, 255)
        if free < DIAG_SRAM_WARN: return (0, 165, 255)
    except: pass
    return (0, 255, 0)

# ─────────────────────────────────────────────
#  DISPLAY SELECTION  (auto-pick smallest monitor)
# ─────────────────────────────────────────────
def select_display_mss():
    with mss.mss() as sct:
        monitors = sct.monitors[1:]
    if len(monitors) == 1:
        return 1
    areas       = [m["width"] * m["height"] for m in monitors]
    primary_idx = areas.index(max(areas))
    for i, m in enumerate(monitors):
        if i != primary_idx:
            print(f"[*] Auto-selected monitor {i+1}: {m['width']}x{m['height']} "
                  f"at ({m['left']},{m['top']}) -- smaller than primary")
            return i + 1
    print("All monitors same size -- please select:")
    for i, m in enumerate(monitors, 1):
        print(f"  [{i}] {m['width']}x{m['height']} at ({m['left']},{m['top']})")
    try:
        return int(input("Select monitor [1]: ") or "1")
    except ValueError:
        return 1

# ─────────────────────────────────────────────
#  STREAM + UI  (Main thread — UI only)
# ─────────────────────────────────────────────
def stream_mss_udp(target_ip, monitor_idx):
    global _monitor_rect

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 0))
    sock.setblocking(False)

    with mss.mss() as sct:
        m = sct.monitors[monitor_idx]
        _monitor_rect[:] = [m["left"], m["top"], m["width"], m["height"]]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, UI_W, UI_H)
    cv2.createTrackbar("Max FPS",        WINDOW_NAME, DEFAULT_FPS,          60,              lambda x: None)
    cv2.createTrackbar("Pacing x0.1ms",  WINDOW_NAME, DEFAULT_PACING_STEPS, PACING_MAX_STEPS,lambda x: None)
    cv2.createTrackbar("RGB565 Quant",   WINDOW_NAME, DEFAULT_RGB565,        1,               lambda x: None)
    cv2.createTrackbar("Palette Agg",    WINDOW_NAME, DEFAULT_PALETTE_AGG,   PALETTE_AGG_MAX, lambda x: None)
    # Upscale: 0=Full 320x240, 1=2/3 214x160, 2=1/2 160x120 (ESP upscales to full tile)
    cv2.createTrackbar("Upscale Mode",   WINDOW_NAME, DEFAULT_UPSCALE,       UPSCALE_MODE_MAX,lambda x: None)
    cv2.createTrackbar("Debug Info",     WINDOW_NAME, 1,                     1,               lambda x: None)

    threading.Thread(target=capture_worker,     args=(monitor_idx,), daemon=True).start()
    threading.Thread(target=encode_send_worker, args=(target_ip, sock), daemon=True).start()

    last_pkt = None

    try:
        while True:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            t_start = time.perf_counter()

            fps          = cv2.getTrackbarPos("Max FPS",        WINDOW_NAME)
            pacing_steps = cv2.getTrackbarPos("Pacing x0.1ms",  WINDOW_NAME)
            use_rgb565   = cv2.getTrackbarPos("RGB565 Quant",   WINDOW_NAME) == 1
            palette_agg  = cv2.getTrackbarPos("Palette Agg",    WINDOW_NAME)
            upscale_mode = cv2.getTrackbarPos("Upscale Mode",   WINDOW_NAME)
            debug_state  = cv2.getTrackbarPos("Debug Info",      WINDOW_NAME)
            update_settings(
                fps          = fps,
                pacing_s     = pacing_steps * PACING_STEP_S,
                use_rgb565   = use_rgb565,
                palette_agg  = palette_agg,
                upscale_mode = upscale_mode,
                debug        = debug_state,
            )

            try:
                last_pkt = preview_queue.get_nowait()
            except Empty:
                pass

            if last_pkt is None:
                blank = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for first frame...", (60, PREVIEW_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
                cv2.imshow(WINDOW_NAME, blank)
                if cv2.waitKey(30) & 0xFF == ord('q'): break
                continue

            pkt            = last_pkt
            enc_frame      = pkt["enc_frame"]
            pkt_enc_w      = pkt.get("enc_w", ESP_W)
            pkt_enc_h      = pkt.get("enc_h", ESP_H)
            pkt_upscale    = pkt.get("upscale_mode", 0)
            pkt_up_label   = pkt.get("upscale_label", "Full")
            tile_sizes     = pkt["tile_sizes"]
            total_bytes    = pkt["total_bytes"]
            bw_kbps        = pkt["bw_kbps"]
            pkt_rgb565     = pkt["use_rgb565"]
            pkt_pal_agg    = pkt["palette_agg"]
            tile_ema_ms    = pkt.get("tile_ema_ms", [0.0]*NUM_TILES)
            tile_codec     = pkt.get("tile_codec",  ["?"]*NUM_TILES)

            # Always upscale enc_frame to full PREVIEW size for display
            # so the preview always shows a 480x360 image regardless of encode mode
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
                    (f"FPS  : {f.get('FPS',  '?'):>8}", _diag_color(f.get('FPS','0'),  DIAG_FPS_WARN,  DIAG_FPS_ERR)),
                    (f"TEMP : {f.get('TEMP', '?'):>7} C", _diag_color(f.get('TEMP','0'), DIAG_TEMP_WARN, DIAG_TEMP_ERR)),
                    (f"JIT  : {f.get('JIT',  '?'):>7} ms", _diag_color(f.get('JIT','0'),  DIAG_JIT_WARN,  DIAG_JIT_ERR)),
                    (f"DEC  : {f.get('DEC',  '?'):>7} us", _diag_color(f.get('DEC','0'),  DIAG_DEC_WARN,  DIAG_DEC_ERR)),
                    (f"DROP : {f.get('DROP', '?'):>8}", _diag_color(f.get('DROP','0'), DIAG_DROP_WARN, DIAG_DROP_ERR)),
                    (f"ABRT : {f.get('ABRT', '?'):>8}", _diag_color(f.get('ABRT','0'), DIAG_DROP_WARN, DIAG_DROP_ERR)),
                    (f"SRAM : {f.get('SRAM',  '?/?'):>11} KB", _sram_color(f.get('SRAM','999/1'))),
                    (f"PSRAM: {f.get('PSRAM', '?/?'):>11} KB", _sram_color(f.get('PSRAM','999/1'))),
                    (f"BW   : {bw_kbps:>7.0f} kbps", (0, 255, 255)),
                    (f"TILE : {total_bytes//NUM_TILES if total_bytes else 0:>6}B avg", (0, 255, 255)),
                    (f"QUANT: RGB565={'ON' if pkt_rgb565 else 'off'}  Pal={p_label}", (200, 200, 200)),
                    (f"SCALE: {pkt_up_label}  enc={pkt_enc_w}x{pkt_enc_h}", (200, 220, 255)),
                    (f"ENC  : {'  '.join(f'{tile_ema_ms[t]:.1f}ms' for t in range(NUM_TILES))}", (180, 255, 180)),
                ]
                y = 16
                for text, color in dashboard:
                    cv2.putText(preview, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.46, color, 1)
                    y += 20

                # Per-tile size bar + codec label + EMA timing
                bar_y = PREVIEW_H - 44
                bar_w = PREVIEW_W // NUM_TILES
                for t, tsz in enumerate(tile_sizes):
                    overflow = tsz < 0
                    sz   = abs(tsz)
                    fill = min(int(bar_w * sz / MAX_TILE_QOI), bar_w)
                    bar_color = (0, 0, 255) if overflow else (0, 200, 0)
                    cv2.rectangle(preview, (t*bar_w, bar_y),
                                  (t*bar_w+fill, bar_y+12), bar_color, -1)
                    size_lbl = f"{sz//1024}K" if sz >= 1024 else f"{sz}B"
                    cv2.putText(preview, size_lbl,
                                (t*bar_w+2, bar_y+10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.36, (255,255,255), 1)
                    # EMA timing below bar
                    ema_t = tile_ema_ms[t] if t < len(tile_ema_ms) else 0.0
                    cv2.putText(preview, f"QOI {ema_t:.1f}ms",
                                (t*bar_w+2, bar_y+26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.34, bar_color, 1)
            else:
                per_tile = total_bytes // NUM_TILES if total_bytes else 0
                pkts_per = (per_tile + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE if per_tile else 0
                info = (f"QOI {total_bytes}B  PerTile:~{per_tile}B/{pkts_per}pkts "
                        f"BW:{bw_kbps:.0f}kbps  "
                        f"Scale:{pkt_up_label}({pkt_enc_w}x{pkt_enc_h})  "
                        f"RGB565={'ON' if pkt_rgb565 else 'off'}  Pal={p_label}")
                cv2.putText(preview, info, (10, PREVIEW_H-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,255,0), 1)

            cv2.imshow(WINDOW_NAME, preview)

            elapsed = time.perf_counter() - t_start
            wait_ms = max(1, int((1.0/30.0 - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        sock.close()

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def quick_find_esp(timeout=5.0):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', PORT))
    s.settimeout(timeout)
    print(f"[*] Waiting {timeout}s for ESP32 beacon on port {PORT}...")
    try:
        data, addr = s.recvfrom(128)
        if b"ESP32_READY" in data:
            print(f"[*] Found ESP32 at {addr[0]}")
            return addr[0]
    except socket.timeout:
        print("[!] No beacon received. Hardcode the IP below if needed.")
    finally:
        s.close()
    return None


if __name__ == "__main__":
    if not _QOI_AVAILABLE:
        print("[ERROR] Install QOI encoder: pip install qoi")
        exit(1)

    set_high_priority()
    set_high_resolution_timer()

    ip = quick_find_esp(timeout=ESP_BEACON_TIMEOUT_S)
    # ip = "192.168.x.x"   # uncomment and set if auto-discovery fails

    if ip:
        stream_mss_udp(ip, select_display_mss())
    else:
        print("[!] Could not find ESP32. Stream aborted.")

    reset_resolution_timer()