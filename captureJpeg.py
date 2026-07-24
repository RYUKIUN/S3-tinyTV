import socket
import time
import cv2
import mss
import numpy as np
import os
import psutil
import ctypes
import threading
import json
from queue import Queue

# ─────────────────────────────────────────────
#  GLOBAL SETTINGS
# ─────────────────────────────────────────────
PORT         = 12345
ESP_W, ESP_H = 320, 240          

# Magic thresholds for auto-quality
MAGIC_420 = 17200
MAGIC_444 = 17000

# EMA Settings (Low-pass filter for bitrate)
EMA_ALPHA = 0.2  # ค่ายิ่งน้อย ยิ่งสมูทแต่ตอบสนองช้าลง (แนะนำ 0.1 - 0.2)

CHUNK_DATA_SIZE  = 1400          
NUM_TILES        = 4
TILE_W, TILE_H   = 160, 120
TILE_X = [  0, 160,   0, 160]
TILE_Y = [  0,   0, 120, 120]
MAX_TILE_JPEG  = 33600            

WINDOW_NAME          = "ESP32-S3 Stream [320x240]"
UI_W, UI_H           = 480, 660  
PREVIEW_W, PREVIEW_H = 480, 360
DEFAULT_FPS          = 35
DEFAULT_QUAL         = 40

CURSOR_OUTER_R = 8
CURSOR_INNER_R = 5
DEBUG_OVERLAY_ALPHA   = 0.85
DEBUG_SEND_INTERVAL_S = 0.5

# Diagnostic thresholds
DIAG_FPS_WARN,  DIAG_FPS_ERR   =  20,    15
DIAG_JIT_WARN, DIAG_JIT_ERR = 5.0, 10.0
DIAG_TEMP_WARN, DIAG_TEMP_ERR  =  70,    85
DIAG_DEC_WARN,  DIAG_DEC_ERR   =  8000,  15000
DIAG_DROP_WARN, DIAG_DROP_ERR  =  1,     5

ESP_BEACON_TIMEOUT_S = 150.0
SEND_RETRY_SLEEP_S   = 0.0005
UNIX_NICE_LEVEL      = -10

# Single JSON file for settings
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
SETTINGS_KEYS = ("Max FPS", "Base Qual", "Bilateral Mix", "Sharpen x0.1", "Chroma sub", "Debug Info")

# ─────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────
frame_queue = Queue(maxsize=1)
stop_event  = threading.Event()
_frame_id   = 0

# ─────────────────────────────────────────────
#  HELPERS & SYSTEM
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
    with mss.mss() as sct:
        monitors = sct.monitors
        indices  = list(range(1, len(monitors)))
        candidates = [i for i in indices if monitors[i]["width"] < 1920]
        idx = candidates[-1] if candidates else 1
        return idx, dict(monitors[idx])

# ─────────────────────────────────────────────
#  NETWORKING & DYNAMIC PACING
# ─────────────────────────────────────────────
_send_buf  = bytearray(8 + CHUNK_DATA_SIZE)
_send_view = memoryview(_send_buf)

_JPEG_SUB_MODES = {
    0: (cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420, "4:2:0", MAGIC_420),
    1: (cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444, "4:4:4", MAGIC_444),
}

def _send_udp(sock: socket.socket, data, dest):
    while True:
        try:
            sock.sendto(data, dest)
            return
        except BlockingIOError:
            time.sleep(SEND_RETRY_SLEEP_S)

def send_tiles(sock: socket.socket, target_ip: str, frame_bgr: np.ndarray,
               quality: int, sub_flag: int, t_start: float, target_fps: int) -> int:
    global _frame_id
    frame_id  = _frame_id & 0xFF
    _frame_id = (_frame_id + 1) & 0xFF
    dest      = (target_ip, PORT)
    total_bytes = 0

    # 1. Encode all tiles
    encoded_tiles = []
    for tId in range(NUM_TILES):
        x, y = TILE_X[tId], TILE_Y[tId]
        tile = frame_bgr[y:y+TILE_H, x:x+TILE_W]
        _, enc = cv2.imencode('.jpg', tile, [int(cv2.IMWRITE_JPEG_QUALITY), quality,
                                              int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), sub_flag])
        encoded_tiles.append(enc)
        total_bytes += len(enc)

    # 2. Calculate Pacing
    overhead_time = time.perf_counter() - t_start
    frame_budget  = 1.0 / max(1, target_fps)
    idle_time     = frame_budget - overhead_time
    auto_pacing_s = max(0.0, idle_time / NUM_TILES)

    # 3. Transmit
    for tId, enc in enumerate(encoded_tiles):
        total_len = len(enc)
        if total_len > MAX_TILE_JPEG: continue

        enc_view     = memoryview(enc)
        num_chunks   = (total_len + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE
        size_hi, size_lo = (total_len >> 8) & 0xFF, total_len & 0xFF

        for cId in range(num_chunks):
            offset = cId * CHUNK_DATA_SIZE
            clen   = min(CHUNK_DATA_SIZE, total_len - offset)
            _send_buf[0:8] = [0xAA, 0xBB, frame_id, tId, cId, num_chunks, size_hi, size_lo]
            _send_buf[8:8+clen] = enc_view[offset:offset+clen]
            _send_udp(sock, _send_view[:8+clen], dest)

        if auto_pacing_s > 0:
            time.sleep(auto_pacing_s)

    return total_bytes

# ─────────────────────────────────────────────
#  THREADS & STATS
# ─────────────────────────────────────────────
def capture_worker(monitor_idx):
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_idx]
        while not stop_event.is_set():
            sct_img = sct.grab(monitor)
            frame = np.frombuffer(sct_img.raw, dtype=np.uint8).reshape((monitor["height"], monitor["width"], 4))[:, :, :3].copy()
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except: pass
            frame_queue.put(frame)

def parse_esp_stats(raw: str) -> dict:
    return {k.strip(): v.strip() for token in raw.split('|') if ':' in token for k, _, v in [token.partition(':')]}

def _diag_color(val_str, warn, err, reverse=False):
    try:
        v = float(''.join(c for c in val_str if c in '0123456789.-'))
        if not reverse:
            if v >= err: return (0, 0, 255)
            if v >= warn: return (0, 165, 255)
        else:
            if v <= err: return (0, 0, 255)
            if v <= warn: return (0, 165, 255)
    except: pass
    return (0, 255, 0)

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def stream_mss_udp(target_ip: str, monitor_idx: int, monitor_info: dict):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 0))
    sock.setblocking(False)

    _tb_cfg = {
        "Max FPS": (DEFAULT_FPS, 60), "Base Qual": (DEFAULT_QUAL, 95),
        "Bilateral Mix": (0, 100), "Sharpen x0.1": (6, 20), 
        "Chroma sub": (0, 1), "Debug Info": (1, 1),
    }

    saved_data = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                saved_data = json.load(f)
        except: pass

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, UI_W, UI_H)
    for k, (v, m) in _tb_cfg.items():
        cv2.createTrackbar(k, WINDOW_NAME, saved_data.get(k, v), m, lambda x: None)

    threading.Thread(target=capture_worker, args=(monitor_idx,), daemon=True).start()

    m_left, m_top, m_w, m_h = monitor_info["left"], monitor_info["top"], monitor_info["width"], monitor_info["height"]
    latest_esp_stats, last_debug_send, last_frame_bytes = {}, 0, 0
    current_qual = saved_data.get("Base Qual", DEFAULT_QUAL)
    
    # EMA Accumulator
    ema_avg_bytes = None

    try:
        while True:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1: break
            t_start = time.perf_counter()

            fps = cv2.getTrackbarPos("Max FPS", WINDOW_NAME)
            user_max_qual = cv2.getTrackbarPos("Base Qual", WINDOW_NAME)
            bilateral_mix = cv2.getTrackbarPos("Bilateral Mix", WINDOW_NAME)
            sharpen_steps = cv2.getTrackbarPos("Sharpen x0.1", WINDOW_NAME)
            sub_idx = cv2.getTrackbarPos("Chroma sub", WINDOW_NAME)
            debug_state = cv2.getTrackbarPos("Debug Info", WINDOW_NAME)
            
            sub_flag, sub_str, magic_threshold = _JPEG_SUB_MODES.get(sub_idx, _JPEG_SUB_MODES[0])
            if current_qual > user_max_qual: current_qual = user_max_qual

            try:
                while True:
                    data, _ = sock.recvfrom(512)
                    if len(data) > 2 and data[0] == 0xAB:
                        latest_esp_stats = parse_esp_stats(data[2:].decode('utf-8', errors='ignore'))
            except: pass

            if time.time() - last_debug_send > DEBUG_SEND_INTERVAL_S:
                _send_udp(sock, bytes([0xAA, 0xCC, 0x01, debug_state]), (target_ip, PORT))
                last_debug_send = time.time()

            if frame_queue.empty(): continue
            frame = frame_queue.get()

            # Cursor
            mx, my = get_mouse_pos()
            rx, ry = mx - m_left, my - m_top
            if 0 <= rx < m_w and 0 <= ry < m_h:
                cv2.circle(frame, (rx, ry), CURSOR_OUTER_R, (255, 255, 255), 2)
                cv2.circle(frame, (rx, ry), CURSOR_INNER_R, (0, 0, 255), -1)

            resized = cv2.resize(frame, (ESP_W, ESP_H), interpolation=cv2.INTER_AREA)
            if bilateral_mix > 0:
                resized = cv2.bilateralFilter(resized, 5, bilateral_mix, bilateral_mix)
            if sharpen_steps > 0:
                s = sharpen_steps * 0.1
                resized = cv2.addWeighted(resized, 1.0 + s, cv2.GaussianBlur(resized, (0,0), 0.3 + s*0.35), -s, 0)

            last_frame_bytes = send_tiles(sock, target_ip, resized, current_qual, sub_flag, t_start, fps)

            # ─────────────────────────────────────────────
            #  AUTO QUALITY LOGIC (EMA SMOOTHED)
            # ─────────────────────────────────────────────
            if ema_avg_bytes is None:
                ema_avg_bytes = last_frame_bytes
            else:
                ema_avg_bytes = (EMA_ALPHA * last_frame_bytes) + ((1.0 - EMA_ALPHA) * ema_avg_bytes)

            upper_bound = magic_threshold * 1
            lower_bound = magic_threshold * 0.90

            if ema_avg_bytes > upper_bound:
                current_qual = max(5, current_qual - 2)
            elif ema_avg_bytes < lower_bound:
                current_qual = min(user_max_qual, current_qual + 1)
            # ─────────────────────────────────────────────

            # UI rendering
            preview = cv2.resize(resized, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_NEAREST)
            if debug_state == 1:
                overlay = preview.copy()
                cv2.rectangle(overlay, (0, 0), (PREVIEW_W, PREVIEW_H), (0, 0, 0), -1)
                preview = cv2.addWeighted(overlay, DEBUG_OVERLAY_ALPHA, preview, 1.0 - DEBUG_OVERLAY_ALPHA, 0)
                
                per_tile = last_frame_bytes // NUM_TILES
                pkts_per = (per_tile + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE
                size_col = (0, 0, 255) if last_frame_bytes > magic_threshold else (0, 255, 0)

                dashboard = [
                    (f"FPS  : {latest_esp_stats.get('FPS', '?'):>8}", _diag_color(latest_esp_stats.get('FPS', '0'), 20, 15, True)),
                    (f"TEMP : {latest_esp_stats.get('TEMP', '?'):>7} C", _diag_color(latest_esp_stats.get('TEMP', '0'), 70, 85)),
                    (f"JIT  : {latest_esp_stats.get('JIT', '?'):>7} ms", _diag_color(latest_esp_stats.get('JIT', '0'), DIAG_JIT_WARN, DIAG_JIT_ERR)),
                    (f"DEC  : {latest_esp_stats.get('DEC', '?'):>7} us", _diag_color(latest_esp_stats.get('DEC', '0'), 8000, 15000)),
                    (f"DROP : {latest_esp_stats.get('DROP', '?'):>8}", _diag_color(latest_esp_stats.get('DROP', '0'), 1, 5)),
                    (f"CPU0 : {latest_esp_stats.get('CPU0', '?'):>7} %", _diag_color(latest_esp_stats.get('CPU0', '0'), 85, 95)),
                    (f"CPU1 : {latest_esp_stats.get('CPU1', '?'):>7} %", _diag_color(latest_esp_stats.get('CPU1', '0'), 85, 95)),
                    (f"RAW  : {last_frame_bytes:>8} B", size_col),
                    (f"AVG  : {int(ema_avg_bytes):>8} B", (255, 200, 0)), # สีฟ้าอ่อนแสดงค่าเฉลี่ย
                    (f"QUAL : {current_qual:>8} (Max {user_max_qual})", (0, 255, 255)),
                    (f"SUB  : {sub_str:>8}", (200, 200, 200)),
                ]
                for i, (txt, col) in enumerate(dashboard):
                    cv2.putText(preview, txt, (10, 22 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            cv2.imshow(WINDOW_NAME, preview)
            elapsed = time.perf_counter() - t_start
            wait_ms = max(1, int(((1.0 / max(1, fps)) - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'): break

    except KeyboardInterrupt: pass
    finally:
        # Save JSON settings on exit
        final_settings = {}
        for k in SETTINGS_KEYS:
            val = cv2.getTrackbarPos(k, WINDOW_NAME)
            if val != -1: final_settings[k] = val
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(final_settings, f, indent=4)
        except: pass
        
        stop_event.set()
        cv2.destroyAllWindows()
        sock.close()

if __name__ == "__main__":
    set_high_priority(); set_high_resolution_timer()
    print("[Discovery] Searching for ESP32-S3")
    s_disc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s_disc.bind(('0.0.0.0', PORT))
    s_disc.settimeout(None)
    try:
        while True:
            data, addr = s_disc.recvfrom(256)
            if b"S3READY" in data:
                print(f"[Discovery] Found ESP32-S3 at {addr[0]}")
                stream_mss_udp(addr[0], *select_monitor())
                break
    except KeyboardInterrupt:
        print("[INFO] Discovery cancelled by user.")
    except Exception as e:
        print(f"[ERROR] Discovery failed: {e}")
    finally:
        s_disc.close(); reset_resolution_timer()