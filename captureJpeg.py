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
import multiprocessing as mp
from queue import Queue

# ─────────────────────────────────────────────
#  GLOBAL SETTINGS
# ─────────────────────────────────────────────
PORT         = 12345
ESP_W, ESP_H = 320, 240

# Priority presets: one trackbar step picks BOTH the per-frame byte budget
# the controller targets (4:2:0 chroma only — that's the only mode this
# sends now) AND the quality ceiling it's allowed to climb back up to.
# Index 0 = most FPS-protective, last index = most quality-protective.
# (label, magic_bytes_threshold, auto_max_quality)
PRIORITY_PRESETS = [
    ("MAX FPS",      8000, 45),
    ("FAST",        10500, 60),
    ("BALANCED",    13000, 70),
    ("QUALITY",      16500, 80),
    ("MAX QUALITY", 20000, 90),
]
PRIORITY_DEFAULT_IDX = 2  # BALANCED

# EMA Settings (Low-pass filter for bitrate)
EMA_ALPHA = 0.2  # ค่ายิ่งน้อย ยิ่งสมูทแต่ตอบสนองช้าลง (แนะนำ 0.1 - 0.2)

# After an overflow streak, how many consecutive under-threshold frames the
# EMA must hold before quality is allowed to climb back up. Prevents
# climbing right back into the same complex scene on one calmer frame.
COOLDOWN_FRAMES = 5

CHUNK_DATA_SIZE  = 1400
NUM_TILES        = 4
TILE_W, TILE_H   = 160, 120
TILE_X = [  0, 160,   0, 160]
TILE_Y = [  0,   0, 120, 120]
MAX_TILE_JPEG  = 33600

WINDOW_NAME_BASE     = "ESP32-S3 Stream [320x240]"
UI_W, UI_H           = 480, 600
PREVIEW_W, PREVIEW_H = 480, 360

# Send rate is fixed — not user-adjustable — so the pipeline behaves the
# same way every time instead of being a variable someone has to tune.
BASE_FPS = 35

CURSOR_OUTER_R = 8
CURSOR_INNER_R = 5
# Cursor ring colour while a hold-drag is active (BGR) — bright green.
CURSOR_DRAG_COLOR = (80, 240, 120)
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

# Settings dir: one JSON file per ESP IP so multiple instances don't collide
SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings")
SETTINGS_KEYS = ("Priority", "Sharpen", "Show Stats", "Enable Touch")

# Rediscovery: how often the main process re-checks for new ESPs after the
# first one has already been picked up (seconds).
REDISCOVERY_POLL_S = 2.0

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

# ─────────────────────────────────────────────
#  TOUCH → MOUSE INJECTION
# ─────────────────────────────────────────────
# The ESP reports raw DOWN / MOVE / UP with panel coordinates and nothing more.
# Every judgement about what a touch *means* is made here, on purpose: these
# thresholds are the part you actually want to tune, and tuning them here costs
# a script restart instead of an OTA reflash.
#
# Gesture vocabulary, three mutually-exclusive outcomes from one press:
#   tap        (press, release quickly)          -> left click at that point
#   slide      (press, move past the slop)       -> scroll wheel
#   hold-drag  (press, stay still past HOLD_MS)  -> button held down; the
#                                                   cursor then follows your
#                                                   finger until you lift
# There is no right click. A press commits to exactly one of the three, so
# nothing can be both a click and a scroll, and arming a drag cancels the click.

TOUCH_EV_DOWN, TOUCH_EV_MOVE, TOUCH_EV_UP = 0, 1, 2

# How far (in ESP panel pixels, 320x240) a finger may wander and still count as
# a tap. Above this the gesture commits to scrolling and can no longer click.
TOUCH_TAP_SLOP_PX = 6
# Stay still this long and the press becomes a DRAG: the left button goes down
# and stays down until you lift. This is the window-dragging gesture. Must be
# comfortably below TOUCH_TAP_MAX_MS so a drag always arms before the tap window
# closes — otherwise there'd be a dead band where a press does nothing at all.
TOUCH_HOLD_MS = 400
# Backstop only. Drags normally arm at TOUCH_HOLD_MS via tick(), so a press held
# longer than that is already a drag and never reaches this check. It only
# matters if ticking stalled (app busy), where it stops a very stale press from
# firing a click on release.
TOUCH_TAP_MAX_MS = 700
# If a drag is active and nothing has been heard from the ESP for this long,
# release the button. Without it, an ESP reboot or WiFi drop mid-drag would
# leave the left button stuck down on the desktop with no way to recover.
TOUCH_DRAG_TIMEOUT_S = 2.0

# ── Cursor restore ────────────────────────────────────────────────────────────
# Windows has exactly ONE system cursor; there is no such thing as a second,
# independent pointer. So touching the panel necessarily yanks the cursor over
# to the mirrored monitor, which is disruptive if you were using your real mouse
# somewhere else. The fix is to put it back: remember where the physical mouse
# was when the touch started, and return the cursor there once the gesture ends.
# The cursor still visibly travels during the gesture — unavoidable, since the
# click, the drag and the scroll all have to happen under it — but your mouse
# never *stays* stolen.
TOUCH_RESTORE_CURSOR = True
# Wait this long after the gesture ends before restoring. Apps often read the
# cursor position while handling the click or button-up they were just sent, so
# snapping away in the same instant can make a click land at the restored
# position instead. Deferred to a later frame rather than slept on, so the
# capture loop never blocks.
TOUCH_RESTORE_DELAY_MS = 40
# If the cursor is further than this from where we last put it, the user has
# grabbed their physical mouse mid-gesture. Restoring would then yank the cursor
# away from where they just deliberately moved it, so we stand down instead —
# the whole point is to stop fighting the mouse, not to fight it differently.
TOUCH_RESTORE_TOLERANCE_PX = 8
# Panel pixels of travel per wheel notch. Lower = faster scrolling.
# The panel is only 240 px tall, so a full-height swipe is roughly
# 240 / this many notches — 14 gives ~17 notches, about one long page.
TOUCH_SCROLL_PX_PER_NOTCH = 14
# Sideways slides emit horizontal wheel events. The axis is locked at the
# moment the slide commits, so a mostly-vertical swipe never leaks sideways
# scroll into a browser's back/forward gesture.
TOUCH_HSCROLL_ENABLED = True
# True = direct manipulation, like a phone: the content follows your finger.
# False = the finger acts like a scrollbar instead.
TOUCH_NATURAL_SCROLL = True

# Windows reports cursor position and virtual-screen metrics in *logical*
# pixels unless the process declares itself DPI-aware, while mss captures in
# *physical* pixels. On any display scaled above 100% those two disagree, which
# would put both the injected clicks and the cursor ring this script already
# draws in the wrong place. Declaring awareness puts everything in physical
# pixels and makes them agree. Set to False if your preview window comes up an
# odd size on a scaled display.
SET_DPI_AWARE = True

MOUSEEVENTF_MOVE       = 0x0001
MOUSEEVENTF_LEFTDOWN   = 0x0002
MOUSEEVENTF_LEFTUP     = 0x0004
MOUSEEVENTF_WHEEL      = 0x0800
MOUSEEVENTF_HWHEEL     = 0x1000
MOUSEEVENTF_ABSOLUTE   = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
WHEEL_DELTA = 120

SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79

if os.name == 'nt':
    _ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(_ULONG_PTR))]

    class _INPUT(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("mi", _MOUSEINPUT)]

    def _send_mouse(flags, dx=0, dy=0, mouse_data=0):
        inp = _INPUT(type=0, mi=_MOUSEINPUT(dx, dy, mouse_data, flags, 0, None))
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
else:
    def _send_mouse(flags, dx=0, dy=0, mouse_data=0):
        pass

def set_dpi_aware():
    if os.name != 'nt' or not SET_DPI_AWARE:
        return
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE (Win 8.1+)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

def _virtual_screen():
    if os.name != 'nt':
        return 0, 0, 1, 1
    gsm = ctypes.windll.user32.GetSystemMetrics
    return (gsm(SM_XVIRTUALSCREEN), gsm(SM_YVIRTUALSCREEN),
            max(1, gsm(SM_CXVIRTUALSCREEN)), max(1, gsm(SM_CYVIRTUALSCREEN)))

def move_cursor_abs(x, y):
    """Absolute move across the whole virtual desktop, so it works on any
    monitor including ones left of / above the primary (negative coords)."""
    if os.name != 'nt':
        return
    vx, vy, vw, vh = _virtual_screen()
    nx = int((x - vx) * 65535 / max(1, vw - 1))
    ny = int((y - vy) * 65535 / max(1, vh - 1))
    nx = max(0, min(65535, nx))
    ny = max(0, min(65535, ny))
    _send_mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, nx, ny)

def mouse_down():
    _send_mouse(MOUSEEVENTF_LEFTDOWN)

def mouse_up():
    _send_mouse(MOUSEEVENTF_LEFTUP)

def click_left():
    mouse_down()
    mouse_up()

def scroll_wheel(notches, horizontal=False):
    if not notches:
        return
    _send_mouse(MOUSEEVENTF_HWHEEL if horizontal else MOUSEEVENTF_WHEEL,
                mouse_data=int(notches * WHEEL_DELTA) & 0xFFFFFFFF)


class TouchInjector:
    """Turns the ESP's raw touch stream into mouse input.

    Holds one gesture's worth of state. Everything is driven by absolute panel
    coordinates, so a dropped MOVE packet is self-healing — the next one simply
    resumes from wherever the finger actually is.

    A gesture starts UNDECIDED and commits to exactly one outcome:

        lift early              -> CLICK   (left click where you touched)
        move past the slop      -> SCROLL  (wheel; axis locked at commit)
        stay still past HOLD_MS -> DRAG    (button held down until you lift)

    The three are mutually exclusive, so nothing can be both a click and a
    scroll, and arming a drag cancels the click.
    """

    IDLE, UNDECIDED, SCROLL, DRAG = 0, 1, 2, 3

    def __init__(self):
        self.mode      = self.IDLE
        self.touch_id  = None
        self.last_seq  = None
        self.start_xy  = (0, 0)
        self.start_ms  = 0.0
        self.last_ms   = 0.0     # arrival time of the most recent event
        self.axis      = None    # 'v' or 'h', locked when a slide commits
        self.anchor    = (0, 0)  # position the last wheel notch was measured from
        self.restore_xy    = None   # where the physical mouse was before we barged in
        self.last_injected = None   # last position WE moved the cursor to
        self.restore_at    = 0.0    # ms timestamp the deferred restore is due

    @property
    def dragging(self):
        return self.mode == self.DRAG

    # ── Cursor borrow / return ────────────────────────────────────────────────
    def _move(self, x, y):
        """Move the cursor and remember we were the one who moved it."""
        move_cursor_abs(x, y)
        self.last_injected = (x, y)

    def _borrow_cursor(self):
        """Called as a gesture starts. Snapshots where the real mouse was."""
        if not TOUCH_RESTORE_CURSOR:
            return
        if self.restore_at:
            # A restore from the previous gesture is still pending, which means
            # the cursor is still parked where WE left it, not where the user's
            # mouse is. Cancel that restore and keep the older snapshot — it is
            # the one that actually points at the physical mouse.
            self.restore_at = 0.0
            return
        self.restore_xy = get_mouse_pos()

    def _schedule_restore(self):
        if TOUCH_RESTORE_CURSOR and self.restore_xy is not None:
            self.restore_at = time.perf_counter() * 1000.0 + TOUCH_RESTORE_DELAY_MS

    def _do_restore(self):
        self.restore_at = 0.0
        target = self.restore_xy
        self.restore_xy = None
        if target is None:
            return
        if self.last_injected is not None:
            cur = get_mouse_pos()
            if (abs(cur[0] - self.last_injected[0]) > TOUCH_RESTORE_TOLERANCE_PX or
                    abs(cur[1] - self.last_injected[1]) > TOUCH_RESTORE_TOLERANCE_PX):
                # The cursor is not where we parked it, so the user has taken
                # hold of their physical mouse. Stand down.
                self.last_injected = None
                return
        move_cursor_abs(*target)
        self.last_injected = None

    def reset(self, immediate=False):
        """Abandon the current gesture. ALWAYS releases a held button.

        This is the safety valve: if a drag is in progress and the stream dies,
        touch gets switched off, or the window closes, a still-pressed left
        button would leave the user's desktop stuck mid-drag with no way to
        recover except clicking manually.

        `immediate` returns the cursor right now instead of on a later frame —
        for teardown paths where there will BE no later frame."""
        if self.mode == self.DRAG:
            mouse_up()
        self.mode     = self.IDLE
        self.touch_id = None
        self.axis     = None
        if immediate:
            self._do_restore()
        else:
            self._schedule_restore()

    @staticmethod
    def panel_to_screen(px, py, mon_left, mon_top, mon_w, mon_h):
        """Invert exactly what the capture path did to get here.

        Forward:  grab(monitor) -> [rotate 90 CCW if portrait] -> resize to 320x240
        For cv2.ROTATE_90_COUNTERCLOCKWISE the mapping is
            dst_x = src_y,  dst_y = (src_w - 1) - src_x
        so the inverse is
            src_x = (src_w - 1) - dst_y,  src_y = dst_x
        """
        rotated = mon_h > mon_w
        rot_w, rot_h = (mon_h, mon_w) if rotated else (mon_w, mon_h)

        # +0.5 samples the centre of the panel pixel rather than its corner.
        fx = (px + 0.5) / ESP_W * rot_w
        fy = (py + 0.5) / ESP_H * rot_h

        if rotated:
            sx = (mon_w - 1) - fy
            sy = fx
        else:
            sx, sy = fx, fy

        sx = max(0, min(mon_w - 1, int(round(sx))))
        sy = max(0, min(mon_h - 1, int(round(sy))))
        return mon_left + sx, mon_top + sy

    def tick(self, mon_left, mon_top, mon_w, mon_h):
        """Called once per frame from the main loop, independent of packets.

        Two things here cannot be driven by incoming events:

        1. Arming a drag. A finger held perfectly still generates NO packets at
           all — the ESP suppresses MOVEs below TOUCH_MOVE_EPS — so if we only
           acted on arrival, holding still would never fire the hold timer.

        2. The stuck-button watchdog. If the ESP reboots or WiFi drops mid-drag,
           the UP that would release the button never arrives.
        """
        now = time.perf_counter() * 1000.0

        # Deferred cursor return. Runs regardless of gesture state — this is the
        # only thing that hands the mouse back, so it must not be gated on a
        # gesture being in progress.
        if self.restore_at and now >= self.restore_at:
            self._do_restore()

        if self.mode == self.UNDECIDED and (now - self.start_ms) >= TOUCH_HOLD_MS:
            # Held still long enough: press and hold. The cursor is already
            # parked at the touch point from the DOWN, so the button goes down
            # exactly where the finger landed.
            self.mode = self.DRAG
            mouse_down()
            return

        if self.mode == self.DRAG and (now - self.last_ms) > TOUCH_DRAG_TIMEOUT_S * 1000.0:
            # Nothing heard for too long — assume the link died rather than
            # leave the button held down indefinitely.
            self.reset()

    def handle(self, kind, tid, seq, px, py, mon_left, mon_top, mon_w, mon_h):
        now = time.perf_counter() * 1000.0

        # A new touch id means a new gesture, even if its DOWN packet was lost.
        if tid != self.touch_id:
            if kind == TOUCH_EV_UP:
                return                      # tail of a gesture we never saw
            self.reset()                    # releases a held button if any
            self.touch_id = tid
            self.last_seq = None
            kind = TOUCH_EV_DOWN            # treat as the start regardless
        elif self.last_seq is not None:
            # Drop UDP-reordered stragglers: anything not strictly newer than
            # what we have already applied would rewind the gesture.
            delta = (seq - self.last_seq) & 0xFF
            if delta == 0 or delta > 128:
                return
        self.last_seq = seq
        self.last_ms  = now

        if kind == TOUCH_EV_DOWN:
            self.start_xy = (px, py)
            self.anchor   = (px, py)
            self.start_ms = now
            self.mode     = self.UNDECIDED
            self.axis     = None
            # Snapshot the real mouse position BEFORE we move anything, so the
            # gesture can hand the cursor back where it found it.
            self._borrow_cursor()
            # Park the cursor on the touched point. A tap then clicks right
            # here, a hold presses right here, and a slide sends its wheel
            # events to whatever window is under this point — which is what
            # makes "scroll the thing I put my finger on" work.
            self._move(*self.panel_to_screen(px, py, mon_left, mon_top, mon_w, mon_h))
            return

        if kind == TOUCH_EV_MOVE:
            if self.mode == self.DRAG:
                # Button is down; just keep the cursor under the finger.
                self._move(*self.panel_to_screen(px, py, mon_left, mon_top, mon_w, mon_h))
                return

            if self.mode == self.UNDECIDED:
                dx = px - self.start_xy[0]
                dy = py - self.start_xy[1]
                if (dx * dx + dy * dy) < (TOUCH_TAP_SLOP_PX * TOUCH_TAP_SLOP_PX):
                    return                  # still inside the tap/hold window
                # Commit to scrolling and lock the axis to the dominant
                # direction at the moment of commitment.
                self.mode = self.SCROLL
                self.axis = 'h' if (TOUCH_HSCROLL_ENABLED and abs(dx) > abs(dy)) else 'v'
                # Anchor at the ORIGINAL touch point, not here. A fast flick can
                # cross the slop and travel most of the panel inside a single
                # 30 Hz report; anchoring to the current point would throw all of
                # that away. Measuring from the start makes scroll distance equal
                # total finger travel. The discarded slop is 6 px against a 14 px
                # notch, so this still cannot fire a notch the instant it commits.
                self.anchor = self.start_xy

            if self.mode != self.SCROLL:
                return

            ax, ay = self.anchor
            travel = (px - ax) if self.axis == 'h' else (py - ay)
            notches = int(travel / TOUCH_SCROLL_PX_PER_NOTCH)
            if notches:
                # Consume only whole notches; the remainder stays in the anchor
                # so slow drags still accumulate instead of being lost.
                consumed = notches * TOUCH_SCROLL_PX_PER_NOTCH
                if self.axis == 'h':
                    self.anchor = (ax + consumed, ay)
                    # Natural: content follows the finger, so dragging right
                    # moves the view left.
                    scroll_wheel(-notches if TOUCH_NATURAL_SCROLL else notches, horizontal=True)
                else:
                    self.anchor = (ax, ay + consumed)
                    # Natural: dragging down moves content down = view goes up,
                    # and a positive wheel delta is "up" on Windows.
                    scroll_wheel(notches if TOUCH_NATURAL_SCROLL else -notches)
            return

        if kind == TOUCH_EV_UP:
            if self.mode == self.DRAG:
                # Land the drag where the finger actually left the panel, then
                # let go. reset() would also release, but doing it explicitly
                # here keeps the drop position exact.
                self._move(*self.panel_to_screen(px, py, mon_left, mon_top, mon_w, mon_h))
                mouse_up()
                self.mode = self.IDLE
            elif self.mode == self.UNDECIDED and (now - self.start_ms) <= TOUCH_TAP_MAX_MS:
                # The cursor is already parked at the DOWN point.
                click_left()
            self.mode     = self.IDLE
            self.touch_id = None
            self.axis     = None
            # Hand the cursor back. Deferred by TOUCH_RESTORE_DELAY_MS so the
            # click or button-up we just sent is fully processed first.
            self._schedule_restore()


def list_monitor_candidates():
    """Return [(idx, monitor_dict), ...] for all monitors (excluding the
    'all monitors combined' entry at index 0), same selection universe used
    by select_monitor()."""
    with mss.mss() as sct:
        monitors = sct.monitors
        return [(i, dict(monitors[i])) for i in range(1, len(monitors))]

def select_monitor(claimed_indices=()):
    """Pick a monitor the same way the original single-instance code did
    (prefer non-4K displays, take the last matching one), but skip any
    index already claimed by another running instance."""
    candidates_all = list_monitor_candidates()
    preferred = [i for i, m in candidates_all if m["width"] < 1920]
    ordered = (preferred[::-1] if preferred else [i for i, _ in candidates_all])
    # try preferred order first, skipping claimed
    for idx in ordered:
        if idx not in claimed_indices:
            mon = dict(next(m for i, m in candidates_all if i == idx))
            return idx, mon
    # fallback: everything is claimed, just take idx 1 anyway
    idx = candidates_all[0][0] if candidates_all else 1
    mon = dict(next((m for i, m in candidates_all if i == idx), {}))
    return idx, mon

# ─────────────────────────────────────────────
#  NETWORKING & DYNAMIC PACING
# ─────────────────────────────────────────────
# Chroma is always 4:2:0 — it's the cheapest mode for the ESP to decode and
# there's no user-facing reason to ever send anything else.
JPEG_SUB_FLAG = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420
JPEG_SUB_STR  = "4:2:0"

def _send_udp(sock: socket.socket, data, dest):
    while True:
        try:
            sock.sendto(data, dest)
            return
        except BlockingIOError:
            time.sleep(SEND_RETRY_SLEEP_S)

def send_tiles(sock: socket.socket, target_ip: str, frame_bgr: np.ndarray,
               quality: int, sub_flag: int, t_start: float, target_fps: int,
               send_buf: bytearray, send_view: memoryview, frame_id_box: list) -> int:
    frame_id = frame_id_box[0] & 0xFF
    frame_id_box[0] = (frame_id_box[0] + 1) & 0xFF
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
            send_buf[0:8] = [0xAA, 0xBB, frame_id, tId, cId, num_chunks, size_hi, size_lo]
            send_buf[8:8+clen] = enc_view[offset:offset+clen]
            _send_udp(sock, send_view[:8+clen], dest)

        if auto_pacing_s > 0:
            time.sleep(auto_pacing_s)

    return total_bytes

# ─────────────────────────────────────────────
#  THREADS & STATS
# ─────────────────────────────────────────────
def capture_worker(monitor_idx_box, frame_queue, stop_event, mss_lock):
    """monitor_idx_box is a 1-element list so the monitor can be changed
    live (when the user picks a different display from the UI) without
    restarting the thread."""
    with mss.mss() as sct:
        while not stop_event.is_set():
            idx = monitor_idx_box[0]
            try:
                monitor = sct.monitors[idx]
            except IndexError:
                monitor = sct.monitors[1]
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

def _mem_color(free_total_str):
    """Color code a 'free/total' KB string by free-memory percentage.
       Green ≥ 30 % free | Orange 15–30 % | Red < 15 %."""
    try:
        parts = free_total_str.split('/')
        free, total = int(parts[0]), int(parts[1])
        pct = free * 100 // total if total > 0 else 100
        if pct < 15:  return (0,   0, 255)   # red
        if pct < 30:  return (0, 165, 255)   # orange
    except: pass
    return (0, 255, 0)                        # green

# ─────────────────────────────────────────────
#  MODERN OVERLAY UI HELPERS
# ─────────────────────────────────────────────
# Dark "glass card" palette (BGR)
UI_BG_TOP     = (34, 30, 26)
UI_BG_BOTTOM  = (24, 20, 18)
UI_CARD_BG    = (46, 40, 36)
UI_CARD_EDGE  = (90, 80, 70)
UI_LABEL_COL  = (150, 150, 150)
UI_ACCENT     = (255, 200, 90)     # cyan-ish accent in BGR = orange highlight
UI_TITLE_COL  = (245, 245, 245)
UI_OK         = (110, 220, 130)
UI_WARN       = (70, 175, 245)
UI_ERR        = (80, 80, 240)
UI_FONT       = cv2.FONT_HERSHEY_SIMPLEX

def _lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def _draw_vertical_gradient(img, top_color, bottom_color):
    h, w = img.shape[:2]
    for y in range(h):
        t = y / max(1, h - 1)
        img[y, :] = _lerp_color(top_color, bottom_color, t)

def _round_rect(img, x, y, w, h, r, color, thickness=-1):
    """Draw a filled or outlined rounded rectangle."""
    x2, y2 = x + w, y + h
    r = max(0, min(r, w // 2, h // 2))
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x, y + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x + r, y + r), (x2 - r, y + r), (x + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.line(img, (x + r, y), (x2 - r, y), color, thickness)
        cv2.line(img, (x + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x, y + r), (x, y2 - r), color, thickness)
        cv2.line(img, (x2, y + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def _text_size(txt, scale, thickness):
    (w, h), base = cv2.getTextSize(txt, UI_FONT, scale, thickness)
    return w, h, base

def _draw_chip(img, x, y, w, h, label, value, value_color, ratio=None):
    """One metric card: small caps label on top, bold colored value below,
       with an optional slim usage bar under the value."""
    _round_rect(img, x, y, w, h, 6, UI_CARD_BG, -1)
    _round_rect(img, x, y, w, h, 6, UI_CARD_EDGE, 1)

    pad = 7
    lbl_scale = 0.34
    val_scale = 0.46
    cv2.putText(img, label.upper(), (x + pad, y + 15), UI_FONT, lbl_scale, UI_LABEL_COL, 1, cv2.LINE_AA)

    vw, vh, _ = _text_size(value, val_scale, 1)
    val_y = y + h - (10 if ratio is None else 16)
    cv2.putText(img, value, (x + pad, val_y), UI_FONT, val_scale, value_color, 1, cv2.LINE_AA)

    if ratio is not None:
        bar_x, bar_y = x + pad, y + h - 8
        bar_w, bar_h = w - pad * 2, 3
        _round_rect(img, bar_x, bar_y, bar_w, bar_h, 1, (60, 55, 50), -1)
        fill_w = int(bar_w * max(0.0, min(1.0, ratio)))
        if fill_w > 0:
            _round_rect(img, bar_x, bar_y, fill_w, bar_h, 1, value_color, -1)

def _pct_ratio(val_str, lo, hi, reverse=False):
    try:
        v = float(''.join(c for c in val_str if c in '0123456789.-'))
    except Exception:
        return 0.0
    span = max(1e-6, hi - lo)
    r = (v - lo) / span
    return (1.0 - r) if reverse else r

# ─────────────────────────────────────────────
#  PER-ESP INSTANCE (runs in its own process)
# ─────────────────────────────────────────────
def esp_instance_main(target_ip: str, claimed_monitors, instance_lock):
    """Entry point for a subprocess dedicated to one ESP. Owns its own
    window, socket, capture thread, and monitor selection."""
    set_high_priority()
    set_high_resolution_timer()
    set_dpi_aware()

    os.makedirs(SETTINGS_DIR, exist_ok=True)
    settings_file = os.path.join(SETTINGS_DIR, f"settings_{target_ip.replace('.', '_')}.json")
    window_name = f"{WINDOW_NAME_BASE} - {target_ip}"

    frame_queue = Queue(maxsize=1)
    stop_event  = threading.Event()
    frame_id_box = [0]
    send_buf  = bytearray(8 + CHUNK_DATA_SIZE)
    send_view = memoryview(send_buf)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 0))
    sock.setblocking(False)

    # ── Claim a monitor, avoiding ones already used by other instances ──
    with instance_lock:
        already_claimed = list(claimed_monitors)
        monitor_idx, monitor_info = select_monitor(already_claimed)
        claimed_monitors.append(monitor_idx)

    monitor_idx_box = [monitor_idx]
    all_monitors = list_monitor_candidates()   # [(idx, dict), ...] fixed for the session
    max_monitor_idx = max((i for i, _ in all_monitors), default=1)

    # Five controls, each self-explanatory from its own label (OpenCV's
    # trackbar UI has no separate space for tooltips, so the name itself
    # has to carry the explanation):
    #   Priority     — which PRIORITY_PRESETS step the byte-budget
    #                  controller targets; 0 = protect FPS hardest,
    #                  max = protect quality hardest. The controller
    #                  always runs — this only moves its goalposts.
    #   Sharpen      — 0 = off, higher = crisper edges
    #   Show Stats   — 0 = clean preview, 1 = performance overlay on top
    #   Enable Touch — 0 = ESP touchscreen ignored, 1 = it drives this mouse
    #   Monitor      — which screen to capture
    _tb_cfg = {
        "Priority":     (PRIORITY_DEFAULT_IDX, len(PRIORITY_PRESETS) - 1),
        "Sharpen":      (10, 20),
        "Show Stats":   (1, 1),
        "Enable Touch": (1, 1),
        "Monitor":      (monitor_idx, max_monitor_idx),
    }

    saved_data = {}
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r') as f:
                saved_data = json.load(f)
        except: pass

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, UI_W, UI_H)
    for k, (v, m) in _tb_cfg.items():
        if k == "Monitor":
            # Always start on the auto-selected monitor for this run;
            # we don't persist this choice across restarts.
            cv2.createTrackbar(k, window_name, monitor_idx, m, lambda x: None)
        else:
            cv2.createTrackbar(k, window_name, saved_data.get(k, v), m, lambda x: None)

    mss_lock = threading.Lock()
    threading.Thread(target=capture_worker, args=(monitor_idx_box, frame_queue, stop_event, mss_lock), daemon=True).start()

    m_left, m_top, m_w, m_h = monitor_info.get("left", 0), monitor_info.get("top", 0), monitor_info.get("width", ESP_W), monitor_info.get("height", ESP_H)
    latest_esp_stats, last_debug_send, last_frame_bytes = {}, 0, 0
    touch = TouchInjector()
    last_touch_send = 0
    _init_preset_idx = saved_data.get("Priority", PRIORITY_DEFAULT_IDX)
    if not (0 <= _init_preset_idx < len(PRIORITY_PRESETS)):
        _init_preset_idx = PRIORITY_DEFAULT_IDX
    current_qual = PRIORITY_PRESETS[_init_preset_idx][2]
    current_monitor_idx = monitor_idx

    # EMA Accumulator
    ema_avg_bytes = None
    # Streak tracking for escalating drop / cooldown before climb-back
    consecutive_overflows = 0
    under_threshold_streak = 0

    try:
        while True:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
            t_start = time.perf_counter()

            priority_idx   = cv2.getTrackbarPos("Priority", window_name)
            sharpen_steps  = cv2.getTrackbarPos("Sharpen", window_name)
            debug_state    = cv2.getTrackbarPos("Show Stats", window_name)
            touch_state    = cv2.getTrackbarPos("Enable Touch", window_name)
            selected_display = cv2.getTrackbarPos("Monitor", window_name)

            # ── Manual display override ──
            if selected_display != current_monitor_idx:
                target_mon = dict(next((m for i, m in all_monitors if i == selected_display), None) or {})
                if target_mon:
                    with instance_lock:
                        if current_monitor_idx in claimed_monitors:
                            claimed_monitors.remove(current_monitor_idx)
                        if selected_display not in claimed_monitors:
                            claimed_monitors.append(selected_display)
                    current_monitor_idx = selected_display
                    monitor_idx_box[0] = selected_display
                    monitor_info = target_mon
                    m_left, m_top, m_w, m_h = monitor_info["left"], monitor_info["top"], monitor_info["width"], monitor_info["height"]
                else:
                    # invalid index chosen, snap trackbar back
                    cv2.setTrackbarPos("Monitor", window_name, current_monitor_idx)

            sub_flag, sub_str = JPEG_SUB_FLAG, JPEG_SUB_STR
            priority_idx = max(0, min(len(PRIORITY_PRESETS) - 1, priority_idx))
            priority_label, magic_threshold, quality_ceiling = PRIORITY_PRESETS[priority_idx]

            if current_qual > quality_ceiling:
                # Enforce the ceiling immediately — matters right after the
                # user drags to a lower/FPS-favoring preset, so the change
                # takes effect on the very next frame instead of drifting
                # down one step at a time via the controller below.
                current_qual = quality_ceiling

            try:
                while True:
                    data, _ = sock.recvfrom(512)
                    if len(data) > 2 and data[0] == 0xAB:
                        latest_esp_stats = parse_esp_stats(data[2:].decode('utf-8', errors='ignore'))
                    elif len(data) == 9 and data[0] == 0xAA and data[1] == 0xDD:
                        # Touch event. Drained in order in the same pass as the
                        # stats packets, so no extra socket and no extra thread;
                        # worst-case added latency is one frame interval.
                        if touch_state == 1:
                            ev_x = (data[5] << 8) | data[6]
                            ev_y = (data[7] << 8) | data[8]
                            # Isolated: the enclosing bare except is the socket
                            # drain's "no more packets" exit, so letting an
                            # injection error escape here would silently stop
                            # draining stats packets too.
                            try:
                                touch.handle(data[2], data[3], data[4], ev_x, ev_y,
                                             m_left, m_top, m_w, m_h)
                            except Exception:
                                touch.reset()
            except: pass

            # Tell the ESP whether to bother sending touch at all. When this is
            # off it stops at the source, so a disabled touchscreen costs
            # nothing on the wire.
            if time.time() - last_touch_send > DEBUG_SEND_INTERVAL_S:
                _send_udp(sock, bytes([0xAA, 0xCC, 0x02, touch_state]), (target_ip, PORT))
                last_touch_send = time.time()
                if touch_state == 0:
                    # Immediate: with touch switched off, tick() would still run
                    # but there is no reason to make the user wait a frame to
                    # get their cursor back.
                    touch.reset(immediate=True)

            # Driven per-frame, not per-packet, and deliberately NOT gated on
            # touch_state: a finger held perfectly still emits no packets at all
            # (the ESP suppresses sub-threshold MOVEs), so the hold-to-drag
            # timer has to be checked from here or it would never fire. This
            # also runs the stuck-button watchdog and the deferred cursor
            # restore, and the restore in particular must never be skipped —
            # it is the thing that gives the mouse back.
            touch.tick(m_left, m_top, m_w, m_h)

            if time.time() - last_debug_send > DEBUG_SEND_INTERVAL_S:
                _send_udp(sock, bytes([0xAA, 0xCC, 0x01, debug_state]), (target_ip, PORT))
                last_debug_send = time.time()

            if frame_queue.empty(): continue
            frame = frame_queue.get()

            # Cursor
            mx, my = get_mouse_pos()
            rx, ry = mx - m_left, my - m_top
            if 0 <= rx < m_w and 0 <= ry < m_h:
                if touch.dragging:
                    # The only feedback channel that exists: this ring is drawn
                    # into the frame the ESP is about to display, so the panel
                    # itself shows when the drag has armed. Without it there is
                    # no way to tell a held finger from a dead one.
                    cv2.circle(frame, (rx, ry), CURSOR_OUTER_R + 3, CURSOR_DRAG_COLOR, 2)
                    cv2.circle(frame, (rx, ry), CURSOR_INNER_R, CURSOR_DRAG_COLOR, -1)
                else:
                    cv2.circle(frame, (rx, ry), CURSOR_OUTER_R, (255, 255, 255), 2)
                    cv2.circle(frame, (rx, ry), CURSOR_INNER_R, (0, 0, 255), -1)

            # Vertical monitor → rotate into the ESP's fixed landscape panel
            # (ESP side no longer rotates; it just decodes what it's given).
            if m_h > m_w:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            resized = cv2.resize(frame, (ESP_W, ESP_H), interpolation=cv2.INTER_AREA)
            if sharpen_steps > 0:
                s = sharpen_steps * 0.1
                resized = cv2.addWeighted(resized, 1.0 + s, cv2.GaussianBlur(resized, (0,0), 0.3 + s*0.35), -s, 0)

            last_frame_bytes = send_tiles(sock, target_ip, resized, current_qual, sub_flag, t_start, BASE_FPS,
                                           send_buf, send_view, frame_id_box)

            # ─────────────────────────────────────────────
            #  AUTO QUALITY LOGIC (EMA SMOOTHED, ESCALATING) — always runs.
            #  The Priority preset only changes magic_threshold /
            #  quality_ceiling above; this loop always closes the
            #  byte-budget control loop against whichever preset is active.
            # ─────────────────────────────────────────────
            if ema_avg_bytes is None:
                ema_avg_bytes = last_frame_bytes
            else:
                ema_avg_bytes = (EMA_ALPHA * last_frame_bytes) + ((1.0 - EMA_ALPHA) * ema_avg_bytes)

            lower_bound = magic_threshold * 0.90

            if last_frame_bytes > magic_threshold:
                # Hard/instant drop: THIS frame actually blew the threshold —
                # react now on the raw size, don't wait for the EMA to catch
                # up (by the time it does, the ESP has already stalled on it
                # and the backlog bleeds into the following frames too).
                #
                # The correction ESCALATES with how many frames in a row
                # have overflowed. A lone spike still gets the same gentle
                # -6 as before; a sustained complex scene gets hit harder
                # each additional frame instead of sawing down -6 at a time
                # while the ESP backlog keeps building the whole time it
                # takes to catch up.
                consecutive_overflows += 1
                drop = min(6 + (consecutive_overflows - 1) * 8, 40)
                current_qual = max(5, current_qual - drop)
                under_threshold_streak = 0
            else:
                consecutive_overflows = 0
                # Cooldown (only matters once quality has actually been
                # pushed down by a streak): require the EMA to stay under
                # lower_bound for several consecutive frames running before
                # climbing back, instead of climbing the instant a single
                # calmer frame drops the EMA below the line. Without this,
                # quality can climb right back into the same complex region
                # a moment later and oscillate.
                if ema_avg_bytes < lower_bound:
                    under_threshold_streak += 1
                    if under_threshold_streak >= COOLDOWN_FRAMES:
                        # Gentle climb back up once comfortably under
                        # threshold for a while, smoothed via EMA so
                        # quality doesn't flicker up and down.
                        current_qual = min(quality_ceiling, current_qual + 1)
                else:
                    under_threshold_streak = 0
            # ─────────────────────────────────────────────

            # UI rendering
            preview = cv2.resize(resized, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_NEAREST)
            if debug_state == 1:
                # Build the entire overlay (gradient + cards + text) on its own
                # canvas, then blend the WHOLE thing against the live video in
                # one shot -- this is what makes DEBUG_OVERLAY_ALPHA actually
                # control the transparency of the cards too, not just the gaps.
                overlay = np.empty_like(preview)
                _draw_vertical_gradient(overlay, UI_BG_TOP, UI_BG_BOTTOM)

                size_col = UI_ERR if last_frame_bytes > magic_threshold else UI_OK
                sram_str  = latest_esp_stats.get('SRAM',  '?/?')
                psram_str = latest_esp_stats.get('PSRAM', '?/?')

                # ── Header ──
                header_h = 26
                cv2.putText(overlay, target_ip, (10, 18), UI_FONT, 0.5, UI_TITLE_COL, 1, cv2.LINE_AA)
                disp_txt = f"Display {current_monitor_idx}"
                dw, _, _ = _text_size(disp_txt, 0.42, 1)
                cv2.putText(overlay, disp_txt, (PREVIEW_W - dw - 10, 18), UI_FONT, 0.42, UI_ACCENT, 1, cv2.LINE_AA)
                cv2.line(overlay, (10, header_h), (PREVIEW_W - 10, header_h), UI_CARD_EDGE, 1, cv2.LINE_AA)

                # ── Metric grid ──
                cols, rows = 4, 4
                margin, gap = 10, 6
                grid_top = header_h + 8
                grid_bottom = PREVIEW_H - 16
                cell_w = (PREVIEW_W - margin * 2 - gap * (cols - 1)) // cols
                cell_h = (grid_bottom - grid_top - gap * (rows - 1)) // rows

                metrics = [
                    ("FPS",   f"{latest_esp_stats.get('FPS', '-')}",              _diag_color(latest_esp_stats.get('FPS', '0'), 20, 15, True),
                     _pct_ratio(latest_esp_stats.get('FPS', '0'), 15, 35)),
                    ("TEMP",  f"{latest_esp_stats.get('TEMP', '-')}C",            _diag_color(latest_esp_stats.get('TEMP', '0'), 70, 85),
                     _pct_ratio(latest_esp_stats.get('TEMP', '0'), 40, 85)),
                    ("JITTER",f"{latest_esp_stats.get('JIT', '-')}ms",            _diag_color(latest_esp_stats.get('JIT', '0'), DIAG_JIT_WARN, DIAG_JIT_ERR),
                     _pct_ratio(latest_esp_stats.get('JIT', '0'), 0, 10)),
                    ("DECODE",f"{latest_esp_stats.get('DEC', '-')}us",            _diag_color(latest_esp_stats.get('DEC', '0'), 8000, 15000),
                     _pct_ratio(latest_esp_stats.get('DEC', '0'), 0, 15000)),
                    ("DROPS", f"{latest_esp_stats.get('DROP', '-')}",             _diag_color(latest_esp_stats.get('DROP', '0'), 1, 5),
                     _pct_ratio(latest_esp_stats.get('DROP', '0'), 0, 5)),
                    ("CPU 0", f"{latest_esp_stats.get('CPU0', '-')}%",            _diag_color(latest_esp_stats.get('CPU0', '0'), 85, 95),
                     _pct_ratio(latest_esp_stats.get('CPU0', '0'), 0, 100)),
                    ("CPU 1", f"{latest_esp_stats.get('CPU1', '-')}%",            _diag_color(latest_esp_stats.get('CPU1', '0'), 85, 95),
                     _pct_ratio(latest_esp_stats.get('CPU1', '0'), 0, 100)),
                    ("SRAM",  sram_str,                                          _mem_color(sram_str), None),
                    ("PSRAM", psram_str,                                        _mem_color(psram_str), None),
                    ("RAW",   f"{last_frame_bytes}B",                            size_col,
                     _pct_ratio(str(last_frame_bytes), 0, magic_threshold)),
                    ("AVG",   f"{int(ema_avg_bytes)}B",                          UI_ACCENT,
                     _pct_ratio(str(int(ema_avg_bytes)), 0, magic_threshold)),
                    ("QUALITY", f"{current_qual}",                               (255, 255, 255),
                     _pct_ratio(str(current_qual), 0, 95)),
                    ("PRIORITY", priority_label,
                     UI_ACCENT, None),
                ]

                for i, (label, value, col, ratio) in enumerate(metrics):
                    r, c = divmod(i, cols)
                    cx = margin + c * (cell_w + gap)
                    cy = grid_top + r * (cell_h + gap)
                    _draw_chip(overlay, cx, cy, cell_w, cell_h, label, value, col, ratio)

                # ── Footer: quick hint ──
                footer_txt = f"{sub_str} chroma  ·  press Q to quit"
                cv2.putText(overlay, footer_txt, (10, PREVIEW_H - 4), UI_FONT, 0.34, UI_LABEL_COL, 1, cv2.LINE_AA)

                # Single blend: this is the only place DEBUG_OVERLAY_ALPHA is used,
                # and now it governs the opacity of everything above at once.
                preview = cv2.addWeighted(overlay, DEBUG_OVERLAY_ALPHA, preview, 1.0 - DEBUG_OVERLAY_ALPHA, 0)

            cv2.imshow(window_name, preview)
            elapsed = time.perf_counter() - t_start
            wait_ms = max(1, int(((1.0 / BASE_FPS) - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'): break

    except KeyboardInterrupt: pass
    finally:
        # FIRST thing in teardown: if a drag was in flight when the window was
        # closed or the process interrupted, the left button is physically down.
        # Leaving it that way hands the user a desktop stuck mid-drag. Immediate,
        # because there is no next frame to run a deferred restore on.
        touch.reset(immediate=True)

        # Save JSON settings on exit (excluding transient display index)
        final_settings = {}
        for k in SETTINGS_KEYS:
            val = cv2.getTrackbarPos(k, window_name)
            if val != -1: final_settings[k] = val
        try:
            with open(settings_file, 'w') as f:
                json.dump(final_settings, f, indent=4)
        except: pass

        with instance_lock:
            if current_monitor_idx in claimed_monitors:
                claimed_monitors.remove(current_monitor_idx)

        stop_event.set()
        cv2.destroyAllWindows()
        sock.close()

# ─────────────────────────────────────────────
#  MAIN: CONTINUOUS DISCOVERY + INSTANCE SPAWNER
# ─────────────────────────────────────────────
def discovery_main():
    set_high_priority()
    set_high_resolution_timer()
    set_dpi_aware()

    manager = mp.Manager()
    claimed_monitors = manager.list()
    instance_lock = manager.Lock()

    known_ips = {}          # ip -> Process
    print("[Discovery] Searching for ESP32-S3 (continuous)")

    s_disc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s_disc.bind(('0.0.0.0', PORT))
    # After the first ESP is found we still keep listening, but at a
    # relaxed poll interval so we don't hog the port loop.
    s_disc.settimeout(REDISCOVERY_POLL_S)

    try:
        while True:
            # Reap any instance processes that have exited so their IP can
            # be rediscovered / reclaimed if the ESP reboots.
            for ip in list(known_ips.keys()):
                proc = known_ips[ip]
                if not proc.is_alive():
                    proc.join(timeout=0.1)
                    del known_ips[ip]

            try:
                data, addr = s_disc.recvfrom(256)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[ERROR] Discovery recv failed: {e}")
                continue

            if b"S3READY" not in data:
                continue

            ip = addr[0]
            if ip in known_ips and known_ips[ip].is_alive():
                # Already have a running instance for this ESP; ignore
                # repeated beacons.
                continue

            print(f"[Discovery] Found ESP32-S3 at {ip} -> spawning instance")
            proc = mp.Process(
                target=esp_instance_main,
                args=(ip, claimed_monitors, instance_lock),
                daemon=False,
            )
            proc.start()
            known_ips[ip] = proc

    except KeyboardInterrupt:
        print("[INFO] Discovery cancelled by user.")
    finally:
        s_disc.close()
        # Let running instances keep going; just wait briefly for cleanliness
        # if the user wants a hard stop, they can Ctrl+C again / close windows.
        for ip, proc in known_ips.items():
            if proc.is_alive():
                proc.join(timeout=0.1)
        reset_resolution_timer()

if __name__ == "__main__":
    mp.freeze_support()
    discovery_main()