import socket
import time
import cv2
import mss
import numpy as np
import os
import psutil
import ctypes
import threading
import struct
from queue import Queue

# ─────────────────────────────────────────────
#  GLOBAL SETTINGS
# ─────────────────────────────────────────────
PORT         = 12345
ESP_W, ESP_H = 320, 240

CHUNK_DATA_SIZE  = 1400
MAX_FRAME_QOI    = 128 * 1024   # must match main.cpp (128KB for two tiles)

WINDOW_NAME          = "ESP32-S3 Stream [320x240] QOI/YUV"
UI_W, UI_H           = 480, 580
PREVIEW_W, PREVIEW_H = 480, 360
DEFAULT_FPS          = 35

CURSOR_OUTER_R = 8
CURSOR_INNER_R = 5

DEBUG_OVERLAY_ALPHA   = 0.85
DEBUG_SEND_INTERVAL_S = 0.5

DIAG_FPS_WARN,  DIAG_FPS_ERR   =  20,    15
DIAG_TEMP_WARN, DIAG_TEMP_ERR  =  70,    85
DIAG_JIT_WARN,  DIAG_JIT_ERR   =  10,    30
DIAG_DEC_WARN,  DIAG_DEC_ERR   =  8000,  15000
DIAG_DROP_WARN, DIAG_DROP_ERR  =  1,     5
DIAG_SRAM_WARN, DIAG_SRAM_ERR  =  50,    20
DIAG_ABRT_WARN, DIAG_ABRT_ERR  =  1,     5

ESP_BEACON_TIMEOUT_S = 15.0
SEND_RETRY_SLEEP_S   = 0.0005
UNIX_NICE_LEVEL      = -10

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
#  QOI ENCODER  (single-channel / grayscale)
#  Encodes a 2-D uint8 numpy array as a grayscale QOI stream.
#  Returns bytes object.
# ─────────────────────────────────────────────
_QOI_MAGIC  = b'qoif'
_QOI_FOOTER = bytes([0,0,0,0,0,0,0,1])

def qoi_encode_plane(plane: np.ndarray) -> bytes:
    """Encode a 2-D uint8 numpy array as single-channel QOI.  Returns bytes."""
    h, w   = plane.shape
    pixels = plane.ravel()
    n      = len(pixels)

    # Pre-allocate worst-case output: header + n*2 (every px as QOI_OP_RGB) + footer
    out = bytearray(14 + n * 2 + 8)
    p   = 0

    # Header
    out[p:p+4] = _QOI_MAGIC;  p += 4
    struct.pack_into('>I', out, p, w); p += 4
    struct.pack_into('>I', out, p, h); p += 4
    out[p] = 1;  p += 1   # channels = 1
    out[p] = 0;  p += 1   # colorspace = sRGB (ignored by decoder)

    index  = [0] * 64
    prev   = 0
    run    = 0

    def flush_run():
        nonlocal p, run
        if run > 0:
            out[p] = 0xC0 | (run - 1)
            p += 1
            run = 0

    for i in range(n):
        px = int(pixels[i])
        if px == prev:
            run += 1
            if run == 62:
                flush_run()
        else:
            flush_run()
            h_idx = (px * 3 + 59) & 63
            if index[h_idx] == px:
                out[p] = h_idx         # QOI_OP_INDEX
                p += 1
            else:
                index[h_idx] = px
                diff = px - prev
                if -2 <= diff <= 1:
                    out[p] = 0x40 | ((diff + 2) << 4) | ((diff + 2) << 2) | (diff + 2)
                    p += 1
                elif -32 <= diff <= 31:
                    dg = diff
                    dr = dg          # single channel: dr = dg = db
                    out[p]   = 0x80 | (dg + 32)
                    out[p+1] = ((dr - dg + 8) << 4) | (dr - dg + 8)
                    p += 2
                else:
                    out[p]   = 0xFE  # QOI_OP_RGB
                    out[p+1] = px
                    p += 2
            prev = px

    flush_run()

    # Footer
    out[p:p+8] = _QOI_FOOTER
    p += 8

    return bytes(out[:p])


# ─────────────────────────────────────────────
#  FRAME ENCODER  (RGB888 → YCbCr-4:2:0 → QOI × 3 per tile → packed tiles)
# ─────────────────────────────────────────────
def encode_frame(frame_bgr: np.ndarray) -> bytes:
    """
    frame_bgr : 320×240 BGR uint8 numpy array
    Returns packed bytes for two tiles (upper + lower):
        [tile0: ySzHi ySzLo | Y-QOI | cbSzHi cbSzLo | Cb-QOI | crSzHi crSzLo | Cr-QOI]
        [tile1: same for lower half]
    """
    # Split into upper and lower tiles
    upper = frame_bgr[:120, :]  # 320×120
    lower = frame_bgr[120:, :]  # 320×120

    def encode_tile(tile_bgr):
        # BGR → YCrCb
        ycrcb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2YCrCb)
        Y  = ycrcb[:, :, 0]          # 320×120
        Cr = ycrcb[:, :, 1]          # 320×120
        Cb = ycrcb[:, :, 2]          # 320×120

        # 4:2:0 downsample Cb and Cr (160×60)
        Cb2 = cv2.resize(Cb, (160, 60), interpolation=cv2.INTER_AREA)
        Cr2 = cv2.resize(Cr, (160, 60), interpolation=cv2.INTER_AREA)

        y_qoi  = qoi_encode_plane(Y)
        cb_qoi = qoi_encode_plane(Cb2)
        cr_qoi = qoi_encode_plane(Cr2)

        # Pack: [szHi szLo | data] × 3
        return (
            struct.pack('>H', len(y_qoi))  + y_qoi  +
            struct.pack('>H', len(cb_qoi)) + cb_qoi +
            struct.pack('>H', len(cr_qoi)) + cr_qoi
        )

    tile0_packed = encode_tile(upper)
    tile1_packed = encode_tile(lower)

    return tile0_packed + tile1_packed


# ─────────────────────────────────────────────
#  TRANSMIT  (chunked UDP, magic 0xAA 0xDD)
# ─────────────────────────────────────────────
_send_buf  = bytearray(7 + CHUNK_DATA_SIZE)
_send_view = memoryview(_send_buf)

def _send_udp(sock: socket.socket, data, dest):
    while True:
        try:
            sock.sendto(data, dest)
            return
        except BlockingIOError:
            time.sleep(SEND_RETRY_SLEEP_S)

def send_frame(sock: socket.socket, target_ip: str, packed: bytes,
               pacing_s: float = 0.0) -> int:
    global _frame_id
    frame_id  = _frame_id & 0xFF
    _frame_id = (_frame_id + 1) & 0xFF
    dest      = (target_ip, PORT)

    total_len  = len(packed)
    if total_len > MAX_FRAME_QOI:
        print(f"[WARN] Packed frame {total_len}B > MAX_FRAME_QOI ({MAX_FRAME_QOI}B) — skipping")
        return 0

    num_chunks = (total_len + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE
    frame_view = memoryview(packed)

    for cId in range(num_chunks):
        offset = cId * CHUNK_DATA_SIZE
        clen   = min(CHUNK_DATA_SIZE, total_len - offset)

        _send_buf[0] = 0xAA
        _send_buf[1] = 0xDD          # new magic — QOI/YUV codec
        _send_buf[2] = frame_id
        _send_buf[3] = cId
        _send_buf[4] = num_chunks
        _send_buf[5] = (total_len >> 8) & 0xFF
        _send_buf[6] = total_len & 0xFF
        _send_buf[7:7+clen] = frame_view[offset:offset+clen]

        _send_udp(sock, _send_view[:7+clen], dest)

        if pacing_s > 0:
            time.sleep(pacing_s)

    return total_len


# ─────────────────────────────────────────────
#  CAPTURE WORKER
# ─────────────────────────────────────────────
def capture_worker(monitor_idx):
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_idx]
        while not stop_event.is_set():
            sct_img = sct.grab(monitor)
            frame   = (np.frombuffer(sct_img.raw, dtype=np.uint8)
                       .reshape(sct_img.height, sct_img.width, 4)[:, :, :3]
                       .copy())
            if not frame_queue.full():
                try:    frame_queue.put_nowait(frame)
                except: pass


# ─────────────────────────────────────────────
#  STATS HELPERS
# ─────────────────────────────────────────────
def parse_esp_stats(raw: str) -> dict:
    result = {}
    for token in raw.split('|'):
        if ':' in token:
            k, v = token.split(':', 1)
            result[k.strip()] = v.strip()
    return result

def _diag_color(v_str, warn, err, reverse=False):
    try:
        v = float(v_str)
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
    cv2.createTrackbar("Sharpen x0.1",  WINDOW_NAME, 6,                    20,               lambda x: None)
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
            sharpen_steps = cv2.getTrackbarPos("Sharpen x0.1",  WINDOW_NAME)
            debug_state   = cv2.getTrackbarPos("Debug Info",    WINDOW_NAME)

            sharpen_amt = sharpen_steps * 0.1

            # Receive any pending ESP stats
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

            # Draw cursor overlay
            mx, my = get_mouse_pos()
            rx, ry = mx - m_left, my - m_top
            if 0 <= rx < m_w and 0 <= ry < m_h:
                cv2.circle(frame, (rx, ry), CURSOR_OUTER_R, (255, 255, 255), 2)
                cv2.circle(frame, (rx, ry), CURSOR_INNER_R, (0,   0, 255),  -1)

            resized = cv2.resize(frame, (ESP_W, ESP_H), interpolation=cv2.INTER_AREA)

            # Optional sharpening (unsharp mask)
            if sharpen_amt > 0.0:
                _sigma  = 0.3 + sharpen_amt * 0.35
                blurred = cv2.GaussianBlur(resized, (0, 0), _sigma)
                resized = cv2.addWeighted(resized, 1.0 + sharpen_amt,
                                          blurred, -sharpen_amt, 0)

            # ── ENCODE + SEND ─────────────────────────────────────────────
            encode_start = time.perf_counter()
            packed = encode_frame(resized)
            encode_time = time.perf_counter() - encode_start

            num_chunks = (len(packed) + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE
            frame_interval = 1.0 / max(1, fps)
            pacing_s = max(0, (frame_interval - encode_time) / num_chunks)

            last_frame_bytes = send_frame(sock, target_ip, packed, pacing_s=pacing_s)

            num_pkts = (last_frame_bytes + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE if last_frame_bytes else 0

            # Preview
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
                    (f"PKT  : {last_frame_bytes}B / {(last_frame_bytes + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE if last_frame_bytes else 0}pkts", (0, 200, 200)),
                ]
                y = 16
                for text, color in dashboard:
                    cv2.putText(preview, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
                    y += 22
                info = (f"QOI/Tiles  Frame:{last_frame_bytes}B  {num_pkts}pkts  "
                        f"Pace:{pacing_s * 1000:.1f}ms")

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
                if data.decode('utf-8', errors='ignore').strip() == "S3READY":
                    print(f"[Discovery] ESP32-S3 found at {addr[0]}")
                    return addr[0]
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