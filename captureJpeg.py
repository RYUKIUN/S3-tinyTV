import socket
import time
import cv2
import mss
import numpy as np
import os
import psutil
import ctypes
import threading
from queue import Queue

# ─────────────────────────────────────────────
#  GLOBAL SETTINGS
# ─────────────────────────────────────────────
PORT         = 12345
ESP_W, ESP_H = 320, 240          # ILI9341 landscape

# Tile constants — MUST match main.cpp
CHUNK_DATA_SIZE  = 1400          # bytes of JPEG payload per UDP packet
NUM_TILES        = 4
TILE_W, TILE_H   = 160, 120

# Tile layout (x, y offsets in the 320x240 frame):
#   [0: TL]  [1: TR]
#   [2: BL]  [3: BR]
TILE_X = [  0, 160,   0, 160]
TILE_Y = [  0,   0, 120, 120]

# Per-tile JPEG size cap — matches MAX_TILE_JPEG in main.cpp
MAX_TILE_JPEG = 33600            # 24 x 1400

# UI / trackbar defaults
WINDOW_NAME          = "ESP32-S3 Stream [320x240]"
UI_W, UI_H           = 480, 620
PREVIEW_W, PREVIEW_H = 480, 360
DEFAULT_FPS          = 35
DEFAULT_QUAL         = 40
DEFAULT_PACING_STEPS = 20
PACING_MAX_STEPS     = 20
PACING_STEP_S        = 0.0001        # 0.1 ms per step

# Cursor overlay
CURSOR_OUTER_R = 8
CURSOR_INNER_R = 5

# Debug overlay
DEBUG_OVERLAY_ALPHA   = 0.85
DEBUG_SEND_INTERVAL_S = 0.5

# Diagnostic thresholds (warn, err)
DIAG_FPS_WARN,  DIAG_FPS_ERR   =  20,    15
DIAG_TEMP_WARN, DIAG_TEMP_ERR  =  70,    85
DIAG_JIT_WARN,  DIAG_JIT_ERR   =  10,    30
DIAG_DEC_WARN,  DIAG_DEC_ERR   =  8000,  15000
DIAG_DROP_WARN, DIAG_DROP_ERR  =  1,     5
DIAG_SRAM_WARN, DIAG_SRAM_ERR  =  50,    20
DIAG_ABRT_WARN, DIAG_ABRT_ERR  =  1,     5

# Networking / timing
ESP_BEACON_TIMEOUT_S = 15.0
SEND_RETRY_SLEEP_S   = 0.0005

# Process priority
UNIX_NICE_LEVEL = -10

# ─────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────
frame_queue = Queue(maxsize=1)
stop_event  = threading.Event()
_frame_id   = 0

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

def select_monitor():
    # Use the highest-index sub-FHD monitor; fall back to lowest resolution.
    # Returns (idx, monitor_dict) — caller gets geometry without a second mss open.
    with mss.mss() as sct:
        monitors = sct.monitors
        indices  = list(range(1, len(monitors)))
        candidates = [i for i in indices
                      if monitors[i]["width"] < 1920 or monitors[i]["height"] < 1080]
        if candidates:
            idx = candidates[-1]
            print(f"[Monitor] Using highest sub-FHD -> index {idx}  (candidates: {candidates})")
        else:
            idx = min(indices, key=lambda i: monitors[i]["width"] * monitors[i]["height"])
            print(f"[Monitor] No sub-FHD found. Falling back to lowest resolution: index {idx} "
                  f"({monitors[idx]['width']}x{monitors[idx]['height']})")
        return idx, dict(monitors[idx])

# ─────────────────────────────────────────────
#  TRANSMIT — tiled chunked UDP
# ─────────────────────────────────────────────
_send_buf  = bytearray(8 + CHUNK_DATA_SIZE)
_send_view = memoryview(_send_buf)

# Chroma subsampling modes — trackbar selects index 0/1/2
# 420: 2x2,1x1,1x1 — smallest JPEG, most chroma loss, best for motion
# 422: 2x1,1x1,1x1 — horizontal half only, good for video/UI content
# 444: 1x1,1x1,1x1 — no subsampling, sharpest, ~30% larger than 420
# All three decode transparently on ESP32_JPEG — no ESP change needed.
_JPEG_SUB_MODES = {
    0: (cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420, "4:2:0"),
    1: (cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422, "4:2:2"),
    2: (cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444, "4:4:4"),
}

def _send_udp(sock: socket.socket, data, dest):
    """Blocking-safe sendto for a non-blocking socket."""
    while True:
        try:
            sock.sendto(data, dest)
            return
        except BlockingIOError:
            time.sleep(SEND_RETRY_SLEEP_S)

def send_tiles(sock: socket.socket, target_ip: str, frame_bgr: np.ndarray,
               quality: int, pacing_s: float = 0.0, sub_flag: int = None) -> int:
    global _frame_id
    if sub_flag is None:
        sub_flag = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420

    frame_id  = _frame_id & 0xFF
    _frame_id = (_frame_id + 1) & 0xFF
    dest      = (target_ip, PORT)
    total_bytes = 0

    for tId in range(NUM_TILES):
        x, y = TILE_X[tId], TILE_Y[tId]
        tile = frame_bgr[y:y+TILE_H, x:x+TILE_W]

        _, enc = cv2.imencode('.jpg', tile,
                              [int(cv2.IMWRITE_JPEG_QUALITY), quality,
                               int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), sub_flag])
        total_len = len(enc)    # enc is uint8 ndarray — wrap in memoryview for bytearray assignment
        enc_view  = memoryview(enc)

        if total_len > MAX_TILE_JPEG:
            print(f"[WARN] Tile {tId} JPEG {total_len}B > MAX_TILE_JPEG — lower quality")
            continue

        total_bytes += total_len
        num_chunks   = (total_len + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE
        size_hi      = (total_len >> 8) & 0xFF
        size_lo      = total_len & 0xFF

        for cId in range(num_chunks):
            offset = cId * CHUNK_DATA_SIZE
            clen   = min(CHUNK_DATA_SIZE, total_len - offset)

            _send_buf[0] = 0xAA
            _send_buf[1] = 0xBB
            _send_buf[2] = frame_id
            _send_buf[3] = tId
            _send_buf[4] = cId
            _send_buf[5] = num_chunks
            _send_buf[6] = size_hi
            _send_buf[7] = size_lo
            _send_buf[8:8+clen] = enc_view[offset:offset+clen]

            _send_udp(sock, _send_view[:8+clen], dest)

        if pacing_s > 0:
            time.sleep(pacing_s)

    return total_bytes

# ─────────────────────────────────────────────
#  CAPTURE WORKER
# ─────────────────────────────────────────────
def capture_worker(monitor_idx):
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_idx]
        while not stop_event.is_set():
            sct_img = sct.grab(monitor)
            frame   = (np.frombuffer(sct_img.raw, dtype=np.uint8)
                       .reshape((monitor["height"], monitor["width"], 4))[:, :, :3]
                       .copy())
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except: pass
            frame_queue.put(frame)

# ─────────────────────────────────────────────
#  STAT PACKET PARSER
# ─────────────────────────────────────────────
def parse_esp_stats(raw: str) -> dict:
    fields = {}
    for token in raw.split('|'):
        if ':' in token:
            k, _, v = token.partition(':')
            fields[k.strip()] = v.strip()
    return fields

def _diag_color(val_str: str, warn: float, err: float, reverse: bool = False):
    try:
        v = float(''.join(c for c in val_str if c in '0123456789.-'))
        if not reverse:
            if v >= err:  return (0,   0, 255)
            if v >= warn: return (0, 165, 255)
        else:
            if v <= err:  return (0,   0, 255)
            if v <= warn: return (0, 165, 255)
    except: pass
    return (0, 255, 0)

def _sram_color(free_kb_str: str):
    try:
        free = float(free_kb_str.split('/')[0])
        if free < DIAG_SRAM_ERR:  return (0,   0, 255)
        if free < DIAG_SRAM_WARN: return (0, 165, 255)
    except: pass
    return (0, 255, 0)

# ─────────────────────────────────────────────
#  STREAM + UI
# ─────────────────────────────────────────────
def stream_mss_udp(target_ip: str, monitor_idx: int, monitor_info: dict):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 0))
    sock.setblocking(False)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, UI_W, UI_H)
    cv2.createTrackbar("Max FPS",       WINDOW_NAME, DEFAULT_FPS,          60,               lambda x: None)
    cv2.createTrackbar("Base Qual",     WINDOW_NAME, DEFAULT_QUAL,         95,               lambda x: None)
    cv2.createTrackbar("Pacing x0.1ms", WINDOW_NAME, DEFAULT_PACING_STEPS, PACING_MAX_STEPS, lambda x: None)
    cv2.createTrackbar("Sharpen x0.1",  WINDOW_NAME, 6,                    20,               lambda x: None)
    cv2.createTrackbar("Chroma sub",    WINDOW_NAME, 0,                    2,                lambda x: None)
    cv2.createTrackbar("Debug Info",    WINDOW_NAME, 1,                    1,                lambda x: None)

    threading.Thread(target=capture_worker, args=(monitor_idx,), daemon=True).start()

    m_left = monitor_info["left"]
    m_top  = monitor_info["top"]
    m_w    = monitor_info["width"]
    m_h    = monitor_info["height"]

    latest_esp_stats = {}
    last_debug_send  = 0
    last_frame_bytes = 0

    try:
        while True:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1: break
            t_start = time.perf_counter()

            fps           = cv2.getTrackbarPos("Max FPS",       WINDOW_NAME)
            user_qual     = cv2.getTrackbarPos("Base Qual",      WINDOW_NAME)
            pacing_steps  = cv2.getTrackbarPos("Pacing x0.1ms", WINDOW_NAME)
            sharpen_steps = cv2.getTrackbarPos("Sharpen x0.1",  WINDOW_NAME)
            sub_idx       = cv2.getTrackbarPos("Chroma sub",     WINDOW_NAME)
            debug_state   = cv2.getTrackbarPos("Debug Info",     WINDOW_NAME)
            pacing_s      = pacing_steps * PACING_STEP_S
            sharpen_amt   = sharpen_steps * 0.1
            sub_flag, sub_str = _JPEG_SUB_MODES.get(sub_idx, _JPEG_SUB_MODES[0])

            try:
                while True:
                    data, _ = sock.recvfrom(512)
                    if len(data) > 2 and data[0] == 0xAB:
                        raw = data[2:].decode('utf-8', errors='ignore')
                        latest_esp_stats = parse_esp_stats(raw)
            except: pass

            if time.time() - last_debug_send > DEBUG_SEND_INTERVAL_S:
                _send_udp(sock, bytes([0xAA, 0xCC, 0x01, debug_state]), (target_ip, PORT))
                last_debug_send = time.time()

            if frame_queue.empty(): continue
            frame = frame_queue.get()

            mx, my = get_mouse_pos()
            rx, ry = mx - m_left, my - m_top
            if 0 <= rx < m_w and 0 <= ry < m_h:
                cv2.circle(frame, (rx, ry), CURSOR_OUTER_R, (255, 255, 255), 2)
                cv2.circle(frame, (rx, ry), CURSOR_INNER_R, (0,   0, 255),  -1)

            resized = cv2.resize(frame, (ESP_W, ESP_H), interpolation=cv2.INTER_AREA)

            # Unsharp mask (trackbar-controlled) — sigma scales with amount
            if sharpen_amt > 0.0:
                _sigma  = 0.3 + sharpen_amt * 0.35   # 0.3 @ step 1 → 1.05 @ step 20
                blurred = cv2.GaussianBlur(resized, (0, 0), _sigma)
                resized = cv2.addWeighted(resized, 1.0 + sharpen_amt,
                                          blurred, -sharpen_amt, 0)

            last_frame_bytes = send_tiles(sock, target_ip, resized, user_qual,
                                          pacing_s=pacing_s, sub_flag=sub_flag)

            preview = cv2.resize(resized, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_NEAREST)
            f = latest_esp_stats

            if debug_state == 1:
                overlay = preview.copy()
                cv2.rectangle(overlay, (0, 0), (PREVIEW_W, PREVIEW_H), (0, 0, 0), -1)
                preview = cv2.addWeighted(overlay, DEBUG_OVERLAY_ALPHA,
                                          preview, 1.0 - DEBUG_OVERLAY_ALPHA, 0)
                dashboard = [
                    (f"FPS  : {f.get('FPS',  '?'):>8}",
                     _diag_color(f.get('FPS',  '0'), DIAG_FPS_WARN,  DIAG_FPS_ERR, reverse=True)),
                    (f"TEMP : {f.get('TEMP', '?'):>7} C",
                     _diag_color(f.get('TEMP', '0'), DIAG_TEMP_WARN, DIAG_TEMP_ERR)),
                    (f"JIT  : {f.get('JIT',  '?'):>7} ms",
                     _diag_color(f.get('JIT',  '0'), DIAG_JIT_WARN,  DIAG_JIT_ERR)),
                    (f"DEC  : {f.get('DEC',  '?'):>7} us",
                     _diag_color(f.get('DEC',  '0'), DIAG_DEC_WARN,  DIAG_DEC_ERR)),
                    (f"DROP : {f.get('DROP', '?'):>8}",
                     _diag_color(f.get('DROP', '0'), DIAG_DROP_WARN, DIAG_DROP_ERR)),
                    (f"ABRT : {f.get('ABRT', '?'):>8}",
                     _diag_color(f.get('ABRT', '0'), DIAG_ABRT_WARN, DIAG_ABRT_ERR)),
                    (f"SRAM : {f.get('SRAM',  '?/?'):>11} KB",
                     _sram_color(f.get('SRAM',  '999/1'))),
                    (f"PSRAM: {f.get('PSRAM', '?/?'):>11} KB",
                     _sram_color(f.get('PSRAM', '999/1'))),
                ]
                y = 16
                for text, color in dashboard:
                    cv2.putText(preview, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
                    y += 22
            else:
                per_tile = last_frame_bytes // NUM_TILES if last_frame_bytes else 0
                pkts_per = (per_tile + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE if per_tile else 0
                info = (f"Total:{last_frame_bytes}B  PerTile:~{per_tile}B/{pkts_per}pkts "
                        f"Q:{user_qual} Sub:{sub_str} Pace:{pacing_steps * PACING_STEP_S * 1000:.1f}ms")
                cv2.putText(preview, info, (10, PREVIEW_H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

            cv2.imshow(WINDOW_NAME, preview)

            elapsed = time.perf_counter() - t_start
            wait_ms = max(1, int(((1.0 / max(1, fps)) - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\n[ESP32-S3] Interrupted.")
    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        sock.close()

# ─────────────────────────────────────────────
#  ESP DISCOVERY
# ─────────────────────────────────────────────
def find_esp(timeout=15.0) -> str | None:
    print(f"[Discovery] Waiting for 'S3READY' ... (timeout={timeout}s)")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', PORT))
    s.settimeout(1.0)
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                data, addr = s.recvfrom(256)
                try:
                    if data.decode('utf-8', errors='ignore').strip() == "S3READY":
                        print(f"[Discovery] ESP32-S3 found at {addr[0]}")
                        return addr[0]
                except Exception:
                    pass
            except socket.timeout: pass
    finally:
        s.close()
    print("[Discovery] ESP32-S3 not found.")
    return None

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    set_high_priority()
    set_high_resolution_timer()

    ip = find_esp(timeout=ESP_BEACON_TIMEOUT_S)
    # ip = "192.168.x.x"   # <- uncomment and hardcode if discovery fails

    if ip:
        mon_idx, mon_info = select_monitor()
        stream_mss_udp(ip, mon_idx, mon_info)
    else:
        print("[ERROR] No ESP32-S3 found. Make sure it broadcasts 'S3READY'.")

    reset_resolution_timer()