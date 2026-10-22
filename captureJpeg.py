import socket
import select
import time
import cv2
import mss
import numpy as np
import os
import psutil
import ctypes
import json
import multiprocessing as mp
import subprocess

# ─────────────────────────────────────────────
#  GLOBAL SETTINGS
# ─────────────────────────────────────────────
PORT         = 12345
ESP_W, ESP_H = 320, 240
WIFI_SSID = "Endmin"
WIFI_PASS = "987654321"

# The encoder always targets the max-quality preset; quality still drops
# automatically when frames exceed the byte budget.
QUALITY_BYTE_THRESHOLD = 25000
MAX_QUALITY = 80

# EMA Settings (Low-pass filter for bitrate)
EMA_ALPHA = 0.2  # ค่ายิ่งน้อย ยิ่งสมูทแต่ตอบสนองช้าลง (แนะนำ 0.1 - 0.2)

# After an overflow streak, how many consecutive under-threshold frames the
# EMA must hold before quality is allowed to climb back up. Prevents
# climbing right back into the same complex scene on one calmer frame.
COOLDOWN_FRAMES = 3

CHUNK_DATA_SIZE  = 1400
MAX_FRAME_JPEG   = 35 * 1024

WINDOW_NAME_BASE     = "ESP32-S3 Stream [320x240]"
UI_W, UI_H           = 480, 600
PREVIEW_W, PREVIEW_H = 480, 360

# Send rate is fixed — not user-adjustable — so the pipeline behaves the
# same way every time instead of being a variable someone has to tune.
BASE_FPS = 45

# When the (post-processing) frame is bit-for-bit identical to the last one
# we sent, nothing is encoded or transmitted. To keep the ESP's stream alive
# (and to let it recover from a lost packet) the same frame is re-sent at
# this rate while the screen is static.
KEEPALIVE_FPS        = 2
KEEPALIVE_INTERVAL_S = 1.0 / KEEPALIVE_FPS

# Local preview window refresh rate. Independent of the send rate.
PREVIEW_FPS = 24
PREVIEW_INTERVAL_S = 1.0 / PREVIEW_FPS

# Debug mode 2 (PC encoder stats) overlay redraw period.
DEBUG_PC_REFRESH_S = 0.3

# Cursor ring is drawn on the 320x240 frame, with radii scaled so it keeps the
# apparent size it had when it was drawn at full resolution and then shrunk.
CURSOR_OUTER_R = 8
CURSOR_INNER_R = 5
CURSOR_MIN_R   = 0.6    # panel pixels; never let the ring vanish entirely
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

# All instances share one settings file beside this script; the latest save wins.
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
SETTINGS_KEYS = ("Sharpen", "Debug Mode", "Enable Touch")

# Rediscovery: how often the main process re-checks for new ESPs after the
# first one has already been picked up (seconds).
REDISCOVERY_POLL_S = 2.0

# ─────────────────────────────────────────────
#  HELPERS & SYSTEM
# ─────────────────────────────────────────────
def set_high_resolution_timer():
    if os.name == 'nt':
        try: ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception: pass

def reset_resolution_timer():
    if os.name == 'nt':
        try: ctypes.windll.winmm.timeEndPeriod(1)
        except Exception: pass

def set_high_priority():
    # Windows: stay at the default priority class. Setting NORMAL_PRIORITY_CLASS
    # was a no-op, and raising it would steal CPU from the user's other apps,
    # which is the opposite of what this script is trying to achieve.
    if os.name == 'nt':
        return
    try:
        psutil.Process(os.getpid()).nice(UNIX_NICE_LEVEL)
    except Exception:
        pass

if os.name == 'nt':
    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

def get_mouse_pos():
    if os.name == 'nt':
        pt = _POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y
    return 0, 0


def ensure_wifi_hotspot():
    """Support the original working flow: the user starts the hotspot manually.

    Windows' "Mobile hotspot" / "Hosted Network" feature is not the same as the
    low-level netsh hostednetwork API. Many laptops explicitly report that the
    hostednetwork API is unsupported even though the user can still create a
    hotspot manually in Settings. For this project, the reliable behavior is to
    leave hotspot creation to the user and simply proceed when the network is up.
    """
    if os.name != 'nt':
        return True

    try:
        driver_info = subprocess.run(
            ["netsh", "wlan", "show", "drivers"],
            capture_output=True,
            text=True,
            check=False,
        )
        driver_out = (driver_info.stdout or "") + (driver_info.stderr or "")
        if "Hosted network supported  : No" in driver_out or "Hosted network supported : No" in driver_out:
            print("[Hotspot] This adapter does not expose the netsh hostednetwork API.")
            print(f"[Hotspot] Use Windows 'Mobile hotspot' manually, then connect to SSID '{WIFI_SSID}' with password '{WIFI_PASS}'.")
            print("[Hotspot] Discovery will continue; the ESP joins the already-running hotspot.")
            return True
    except Exception:
        pass

    try:
        out = subprocess.run(
            ["netsh", "wlan", "show", "hostednetwork"],
            capture_output=True,
            text=True,
            check=False,
        )
        if "Status: Started" in (out.stdout or ""):
            print(f"[Hotspot] Already running: {WIFI_SSID}")
            return True
    except Exception:
        pass

    print(f"[Hotspot] Hotspot is expected to be started manually: {WIFI_SSID}")
    print("[Hotspot] Please enable Windows Mobile hotspot and keep the SSID/password matching the ESP config.")
    return True

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
#  NETWORKING
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

def send_frame(sock: socket.socket, target_ip: str, frame_bgr: np.ndarray,
               quality: int, sub_flag: int,
               send_buf: bytearray, send_view: memoryview, frame_id_box: list,
               prev_send_end: list = None) -> dict:
    """Encode and transmit one frame. Frame-rate pacing is NOT done here any
    more — the main loop owns the frame clock, because frames can now be
    skipped and the clock has to keep running through the skipped ones."""
    frame_id = frame_id_box[0] & 0xFF
    frame_id_box[0] = (frame_id_box[0] + 1) & 0xFF
    dest      = (target_ip, PORT)
    total_bytes = 0

    # Network jitter from sender side: gap between consecutive frame sends
    now_call = time.perf_counter()
    send_interval_s = 0.0
    if prev_send_end is not None and prev_send_end[0] > 0:
        send_interval_s = now_call - prev_send_end[0]

    # 1. Encode one complete image, lowering quality only if the protocol cap
    #    would otherwise be exceeded.
    encode_start = time.perf_counter()
    frame_quality = quality
    while True:
        ok, encoded = cv2.imencode(
            '.jpg', frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), frame_quality,
             int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), sub_flag])
        if not ok:
            raise RuntimeError("JPEG encode failed for full frame")
        total_bytes = len(encoded)
        if total_bytes <= MAX_FRAME_JPEG:
            break
        if frame_quality <= 5:
            raise RuntimeError(
                f"Full-frame JPEG is still {total_bytes} bytes at minimum quality "
                f"{frame_quality}; ESP limit is {MAX_FRAME_JPEG} bytes")
        frame_quality = max(5, frame_quality - 5)
    encode_time_s = time.perf_counter() - encode_start

    # 2. Transmit contiguous chunks of the complete JPEG back to back.
    send_time_s = 0.0
    total_len = len(encoded)
    enc_view = memoryview(encoded)
    num_chunks = (total_len + CHUNK_DATA_SIZE - 1) // CHUNK_DATA_SIZE
    size_hi, size_lo = (total_len >> 8) & 0xFF, total_len & 0xFF
    for cId in range(num_chunks):
        offset = cId * CHUNK_DATA_SIZE
        clen = min(CHUNK_DATA_SIZE, total_len - offset)
        send_buf[0:8] = [0xAA, 0xBB, frame_id, cId, num_chunks, 0, size_hi, size_lo]
        send_buf[8:8+clen] = enc_view[offset:offset+clen]
        send_start = time.perf_counter()
        _send_udp(sock, send_view[:8+clen], dest)
        send_time_s += time.perf_counter() - send_start

    call_end = time.perf_counter()
    if prev_send_end is not None:
        prev_send_end[0] = call_end

    return {
        "bytes": total_bytes,
        "encode_s": encode_time_s,
        "send_s": send_time_s,
        "auto_pacing_s": 0.0,
        "paced_frames": 0,
        "send_interval_s": send_interval_s,
        "quality_fallback": int(frame_quality < quality),
        "frame_quality": frame_quality,
    }

# ─────────────────────────────────────────────
#  STATS
# ─────────────────────────────────────────────
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
    except Exception: pass
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
    except Exception: pass
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

_gradient_cache = {}

def _get_gradient(h, w):
    """Vertical background gradient, built once per size with numpy and then
    reused (callers must .copy() before drawing on it)."""
    key = (h, w)
    g = _gradient_cache.get(key)
    if g is None:
        t = np.linspace(0.0, 1.0, max(1, h), dtype=np.float32)[:, None, None]
        top = np.array(UI_BG_TOP, dtype=np.float32)
        bot = np.array(UI_BG_BOTTOM, dtype=np.float32)
        col = (top + (bot - top) * t).astype(np.uint8)       # (h, 1, 3)
        g = np.ascontiguousarray(np.repeat(col, w, axis=1))  # (h, w, 3)
        _gradient_cache[key] = g
    return g

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
    window, socket and monitor selection.

    Frame flow, one iteration per BASE_FPS tick:
        grab (on demand, in this thread) -> resize straight from BGRA
        -> cursor ring on the small image -> [rotate if portrait]
        -> identical to last sent?  skip  (keepalive at KEEPALIVE_FPS)
        -> sharpen -> JPEG encode -> UDP send
    then sleep on the socket until the next frame deadline, so ESP packets
    (touch, stats) are handled the instant they arrive."""
    set_high_priority()
    set_high_resolution_timer()
    set_dpi_aware()

    settings_file = SETTINGS_FILE
    window_name = f"{WINDOW_NAME_BASE} - {target_ip}"

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

    all_monitors = list_monitor_candidates()   # [(idx, dict), ...] fixed for the session
    max_monitor_idx = max((i for i, _ in all_monitors), default=1)

    # Four controls, each self-explanatory from its own label (OpenCV's
    # trackbar UI has no separate space for tooltips, so the name itself
    # has to carry the explanation):
    #   Sharpen      — 0 = off, higher = crisper edges
    #   Debug Mode   — 0 = clean preview, 1 = ESP stats overlay, 2 = PC encoder stats overlay
    #   Enable Touch — 0 = ESP touchscreen ignored, 1 = it drives this mouse
    #   Monitor      — which screen to capture
    _tb_cfg = {
        "Sharpen":      (10, 20),
        "Debug Mode":   (1, 2),
        "Enable Touch": (1, 1),
        "Monitor":      (monitor_idx, max_monitor_idx),
    }

    saved_data = {}
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r') as f:
                saved_data = json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"[Settings] Could not load {settings_file}: {exc}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, UI_W, UI_H)
    for k, (v, m) in _tb_cfg.items():
        if k == "Monitor":
            # Always start on the auto-selected monitor for this run;
            # we don't persist this choice across restarts.
            cv2.createTrackbar(k, window_name, monitor_idx, m, lambda x: None)
        else:
            cv2.createTrackbar(k, window_name, saved_data.get(k, v), m, lambda x: None)

    # Capture happens on demand, in THIS thread (mss/GDI handles are
    # per-thread, so the instance must be created where it is used).
    sct = mss.mss()

    m_left, m_top, m_w, m_h = monitor_info.get("left", 0), monitor_info.get("top", 0), monitor_info.get("width", ESP_W), monitor_info.get("height", ESP_H)

    esp_state = {"stats": {}, "dirty": False}   # dirty = new ESP stats packet since last overlay draw
    last_debug_send = 0.0
    last_touch_send = 0.0
    last_frame_bytes = {"bytes": 0, "encode_s": 0.0, "send_s": 0.0, "auto_pacing_s": 0.0, "paced_frames": 0, "send_interval_s": 0.0, "quality_fallback": 0, "frame_quality": 0}
    touch = TouchInjector()
    touch_state = 0
    prev_send_end   = [0.0]
    ema_jitter_s    = None
    current_qual = MAX_QUALITY
    current_monitor_idx = monitor_idx

    # EMA Accumulator
    ema_avg_bytes = None
    # Streak tracking for escalating drop / cooldown before climb-back
    consecutive_overflows = 0
    under_threshold_streak = 0

    # Frame clock / counters
    frame_interval  = 1.0 / BASE_FPS
    prev_deadline   = time.perf_counter() - frame_interval
    loop_counter    = 0
    send_counter    = 0
    skip_counter    = 0
    fps_last_ts     = time.perf_counter()
    pc_fps_actual   = 0.0
    send_fps_actual = 0.0
    skipped_last_s  = 0
    work_s          = 0.0     # active time of the last iteration (excludes the wait)
    last_wait_s     = 0.0

    # Skip-identical state
    prev_cmp      = None      # last frame_small that was sent (pre-sharpen)
    prev_sharpen  = -1
    last_send_ts  = 0.0
    resized       = None      # last frame actually sent (post-sharpen), what the ESP shows
    frame_ver     = 0

    # Preview / overlay state
    preview_base   = None
    preview_ver    = -1
    last_preview_ts = 0.0
    preview_dirty  = True
    overlay_cache  = None
    overlay_mode   = None
    overlay_mon    = None
    overlay_ts     = 0.0
    last_debug_state = None
    last_cap_err_print = 0.0

    def drain_socket():
        """Read every pending packet. Called from the socket-wait, so it runs
        the moment something arrives rather than once per frame."""
        while True:
            try:
                data, _ = sock.recvfrom(512)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                # e.g. WSAECONNRESET from an ICMP unreachable; nothing to read.
                return
            if len(data) > 2 and data[0] == 0xAB:
                esp_state["stats"] = parse_esp_stats(data[2:].decode('utf-8', errors='ignore'))
                esp_state["dirty"] = True
            elif len(data) == 9 and data[0] == 0xAA and data[1] == 0xDD:
                if touch_state == 1:
                    ev_x = (data[5] << 8) | data[6]
                    ev_y = (data[7] << 8) | data[8]
                    # Isolated so an injection error can never abort the drain
                    # and leave stats packets sitting in the socket.
                    try:
                        touch.handle(data[2], data[3], data[4], ev_x, ev_y,
                                     m_left, m_top, m_w, m_h)
                    except Exception:
                        touch.reset()

    def wait_until(deadline):
        """Sleep until `deadline`, but wake instantly when a packet arrives.
        select() blocks in the kernel, so this costs no CPU while waiting."""
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0005:
                return
            try:
                readable, _, _ = select.select([sock], [], [], remaining)
            except (OSError, ValueError):
                time.sleep(min(remaining, 0.001))
                continue
            if readable:
                drain_socket()

    try:
        while True:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
            t_start = time.perf_counter()
            deadline = max(prev_deadline + frame_interval, t_start)
            prev_deadline = deadline

            sharpen_steps  = cv2.getTrackbarPos("Sharpen", window_name)
            debug_state    = cv2.getTrackbarPos("Debug Mode", window_name)
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
                    monitor_info = target_mon
                    m_left, m_top, m_w, m_h = monitor_info["left"], monitor_info["top"], monitor_info["width"], monitor_info["height"]
                    prev_cmp = None        # new screen: never treat it as "identical"
                else:
                    # invalid index chosen, snap trackbar back
                    cv2.setTrackbarPos("Monitor", window_name, current_monitor_idx)

            sub_flag, sub_str = JPEG_SUB_FLAG, JPEG_SUB_STR

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
                esp_debug_flag = 1 if debug_state == 1 else 0
                _send_udp(sock, bytes([0xAA, 0xCC, 0x01, esp_debug_flag]), (target_ip, PORT))
                last_debug_send = time.time()

            # ── Capture on demand + shrink immediately ──
            # Straight from the BGRA grab buffer: no full-size BGR copy, no
            # full-size rotate, no full-size cursor drawing. Portrait monitors
            # are resized to the *rotated* target shape (240x320) so one cheap
            # rotate of the small image finishes the job.
            frame_small = None
            try:
                sct_img = sct.grab(monitor_info)
                raw = np.frombuffer(sct_img.raw, dtype=np.uint8).reshape((sct_img.height, sct_img.width, 4))
                portrait = m_h > m_w
                small_w, small_h = (ESP_H, ESP_W) if portrait else (ESP_W, ESP_H)
                small = cv2.cvtColor(
                    cv2.resize(raw, (small_w, small_h), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGRA2BGR)

                # Cursor ring, drawn on the small image. Radii are scaled by the
                # same shrink factor the old full-res ring went through, so it
                # keeps the apparent size it had on the panel. Sub-pixel
                # coordinates (shift=4) + anti-aliasing stand in for the blur
                # the old downscale gave a thin ring.
                mx, my = get_mouse_pos()
                rx, ry = mx - m_left, my - m_top
                if 0 <= rx < m_w and 0 <= ry < m_h:
                    sx, sy = small_w / m_w, small_h / m_h
                    cs = (sx + sy) * 0.5
                    cx = int(round(((rx + 0.5) * sx - 0.5) * 16))
                    cy = int(round(((ry + 0.5) * sy - 0.5) * 16))
                    if touch.dragging:
                        # The only feedback channel that exists: this ring is
                        # drawn into the frame the ESP displays, so the panel
                        # itself shows when the drag has armed.
                        r_out = max(CURSOR_MIN_R, (CURSOR_OUTER_R + 3) * cs)
                        r_in  = max(CURSOR_MIN_R, CURSOR_INNER_R * cs)
                        cv2.circle(small, (cx, cy), int(r_out * 16), CURSOR_DRAG_COLOR, 1, cv2.LINE_AA, 4)
                        cv2.circle(small, (cx, cy), int(r_in * 16),  CURSOR_DRAG_COLOR, -1, cv2.LINE_AA, 4)
                    else:
                        r_out = max(CURSOR_MIN_R, CURSOR_OUTER_R * cs)
                        r_in  = max(CURSOR_MIN_R, CURSOR_INNER_R * cs)
                        cv2.circle(small, (cx, cy), int(r_out * 16), (255, 255, 255), 1, cv2.LINE_AA, 4)
                        cv2.circle(small, (cx, cy), int(r_in * 16),  (0, 0, 255), -1, cv2.LINE_AA, 4)

                # Vertical monitor → rotate into the ESP's fixed landscape panel
                # (ESP side no longer rotates; it just decodes what it's given).
                if portrait:
                    small = cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame_small = small
            except Exception as exc:
                # A failed grab (secure desktop, display change...) must not
                # kill the loop — the window still has to keep pumping events.
                if time.time() - last_cap_err_print > 5.0:
                    print(f"[Capture] {target_ip}: {exc}")
                    last_cap_err_print = time.time()
                time.sleep(0.005)

            if frame_small is not None:
                now_p = time.perf_counter()
                # Absolute equality only (max per-pixel difference == 0): any
                # change at all, including the cursor moving, counts as new.
                unchanged = (prev_cmp is not None
                             and sharpen_steps == prev_sharpen
                             and cv2.norm(frame_small, prev_cmp, cv2.NORM_INF) == 0.0)
                keepalive_due = (now_p - last_send_ts) >= KEEPALIVE_INTERVAL_S

                if unchanged and not keepalive_due:
                    skip_counter += 1
                else:
                    prev_cmp, prev_sharpen = frame_small, sharpen_steps
                    resized = frame_small
                    if sharpen_steps > 0:
                        s = sharpen_steps * 0.1
                        resized = cv2.addWeighted(resized, 1.0 + s, cv2.GaussianBlur(resized, (0,0), 0.3 + s*0.35), -s, 0)

                    last_frame_bytes = send_frame(sock, target_ip, resized, current_qual, sub_flag,
                                                  send_buf, send_view, frame_id_box, prev_send_end)
                    last_send_ts = time.perf_counter()
                    frame_ver += 1
                    send_counter += 1
                    if last_frame_bytes["frame_quality"] < current_qual:
                        current_qual = last_frame_bytes["frame_quality"]

                    # ── Sender-side jitter EMA ──
                    # Only between back-to-back sends; gaps made by skipped
                    # frames or the keepalive are not jitter.
                    ji = last_frame_bytes.get("send_interval_s", 0.0)
                    if 0 < ji < frame_interval * 3:
                        if ema_jitter_s is None:
                            ema_jitter_s = ji
                        else:
                            ema_jitter_s = EMA_ALPHA * ji + (1.0 - EMA_ALPHA) * ema_jitter_s

                    frame_byte_count = last_frame_bytes["bytes"]

                    # ─────────────────────────────────────────────
                    #  AUTO QUALITY LOGIC (EMA SMOOTHED, ESCALATING)
                    # ─────────────────────────────────────────────
                    if ema_avg_bytes is None:
                        ema_avg_bytes = frame_byte_count
                    else:
                        ema_avg_bytes = (EMA_ALPHA * frame_byte_count) + ((1.0 - EMA_ALPHA) * ema_avg_bytes)

                    lower_bound = QUALITY_BYTE_THRESHOLD * 0.90

                    if frame_byte_count > QUALITY_BYTE_THRESHOLD:
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
                                current_qual = min(MAX_QUALITY, current_qual + 1)
                                if current_qual >= MAX_QUALITY:
                                    under_threshold_streak = 0  # stop counting once at ceiling
                        else:
                            under_threshold_streak = 0
                    # ─────────────────────────────────────────────

            # ── Loop / send FPS (1-second window) ──
            loop_counter += 1
            now_fps = time.perf_counter()
            if now_fps - fps_last_ts >= 1.0:
                span = now_fps - fps_last_ts
                pc_fps_actual   = loop_counter / span
                send_fps_actual = send_counter / span
                skipped_last_s  = skip_counter
                loop_counter = send_counter = skip_counter = 0
                fps_last_ts = now_fps

            # ─────────────────────────────────────────────
            #  PREVIEW (PREVIEW_FPS) + DEBUG OVERLAYS
            # ─────────────────────────────────────────────
            now_pv = time.perf_counter()
            if debug_state != last_debug_state:
                last_debug_state = debug_state
                overlay_cache = None
                preview_dirty = True

            if resized is not None and (now_pv - last_preview_ts) >= PREVIEW_INTERVAL_S:
                last_preview_ts = now_pv

                if frame_ver != preview_ver:
                    preview_base = cv2.resize(resized, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_NEAREST)
                    preview_ver = frame_ver
                    preview_dirty = True

                # Does the overlay need rebuilding?
                need_overlay = False
                if debug_state == 1:
                    need_overlay = (overlay_cache is None or overlay_mode != 1
                                    or overlay_mon != current_monitor_idx
                                    or esp_state["dirty"])
                elif debug_state == 2:
                    need_overlay = (overlay_cache is None or overlay_mode != 2
                                    or overlay_mon != current_monitor_idx
                                    or (now_pv - overlay_ts) >= DEBUG_PC_REFRESH_S)
                else:
                    overlay_cache = None

                if need_overlay:
                    overlay = _get_gradient(PREVIEW_H, PREVIEW_W).copy()
                    frame_byte_count = last_frame_bytes["bytes"]
                    size_col = UI_ERR if frame_byte_count > QUALITY_BYTE_THRESHOLD else UI_OK
                    avg_bytes_i = int(ema_avg_bytes or 0)

                    # ── Header (shared) ──
                    header_h = 26
                    disp_txt = f"Display {current_monitor_idx}"
                    dw, _, _ = _text_size(disp_txt, 0.42, 1)

                    if debug_state == 1:
                        # ── Mode 1: ESP stats. Redrawn only when a new stats
                        # packet has arrived from the ESP. ──
                        latest_esp_stats = esp_state["stats"]
                        esp_state["dirty"] = False
                        sram_str  = latest_esp_stats.get('SRAM',  '?/?')
                        psram_str = latest_esp_stats.get('PSRAM', '?/?')

                        cv2.putText(overlay, target_ip, (10, 18), UI_FONT, 0.5, UI_TITLE_COL, 1, cv2.LINE_AA)
                        cv2.putText(overlay, disp_txt, (PREVIEW_W - dw - 10, 18), UI_FONT, 0.42, UI_ACCENT, 1, cv2.LINE_AA)
                        cv2.line(overlay, (10, header_h), (PREVIEW_W - 10, header_h), UI_CARD_EDGE, 1, cv2.LINE_AA)

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
                            ("RAW",   f"{frame_byte_count}B",                            size_col,
                             _pct_ratio(str(frame_byte_count), 0, QUALITY_BYTE_THRESHOLD)),
                            ("AVG",   f"{avg_bytes_i}B",                                 UI_ACCENT,
                             _pct_ratio(str(avg_bytes_i), 0, QUALITY_BYTE_THRESHOLD)),
                            ("QUALITY", f"{current_qual}",                               (255, 255, 255),
                             _pct_ratio(str(current_qual), 0, 95)),
                        ]
                        footer_txt = f"{sub_str} chroma  ·  press Q to quit"
                    else:
                        # ── Mode 2: PC-side encoder / pipeline debug overlay.
                        # Rebuilt every DEBUG_PC_REFRESH_S. ESP debug packets are
                        # suppressed (flag=0 above) because Mode 2 is PC-side only. ──
                        cv2.putText(overlay, "PC ENCODER DEBUG", (10, 18), UI_FONT, 0.5, UI_TITLE_COL, 1, cv2.LINE_AA)
                        cv2.putText(overlay, disp_txt, (PREVIEW_W - dw - 10, 18), UI_FONT, 0.42, UI_ACCENT, 1, cv2.LINE_AA)
                        cv2.line(overlay, (10, header_h), (PREVIEW_W - 10, header_h), UI_CARD_EDGE, 1, cv2.LINE_AA)

                        enc_us   = int(last_frame_bytes.get("encode_s", 0.0) * 1e6)
                        send_us  = int(last_frame_bytes.get("send_s",   0.0) * 1e6)
                        enc_ms   = enc_us  / 1000.0
                        send_ms  = send_us / 1000.0
                        frame_budget_s   = 1.0 / max(1, BASE_FPS)
                        budget_pct       = min(999.0, (work_s / frame_budget_s) * 100.0)
                        pacing_us        = int(last_wait_s * 1e6)
                        jitter_dev_ms    = abs((ema_jitter_s or frame_budget_s) - frame_budget_s) * 1000.0
                        ovf_strk         = consecutive_overflows
                        cool_strk        = under_threshold_streak

                        enc_col  = UI_OK if enc_ms < 5 else (UI_WARN if enc_ms < 10 else UI_ERR)
                        send_col = UI_OK if send_ms < 3 else (UI_WARN if send_ms < 7 else UI_ERR)
                        bgt_col  = UI_OK if budget_pct < 70 else (UI_WARN if budget_pct < 90 else UI_ERR)
                        fps_col  = UI_OK if pc_fps_actual >= 30 else (UI_WARN if pc_fps_actual >= 25 else UI_ERR)
                        jit_col  = UI_OK if jitter_dev_ms < 2 else (UI_WARN if jitter_dev_ms < 5 else UI_ERR)
                        ovf_col  = UI_OK if ovf_strk == 0 else (UI_WARN if ovf_strk < 3 else UI_ERR)
                        cool_frac = cool_strk / max(1, COOLDOWN_FRAMES)
                        cool_col = UI_LABEL_COL if cool_frac == 0 else (UI_OK if cool_frac >= 1.0 else UI_WARN)

                        metrics = [
                            ("ENC TIME",   f"{enc_us}us",                   enc_col,
                             _pct_ratio(str(enc_ms), 0, 20)),
                            ("SEND TIME",  f"{send_us}us",                  send_col,
                             _pct_ratio(str(send_ms), 0, 15)),
                            ("BUDGET %",   f"{budget_pct:.1f}%",            bgt_col,
                             _pct_ratio(str(budget_pct), 0, 100)),
                            ("LOOP FPS",   f"{pc_fps_actual:.1f}",          fps_col,
                             _pct_ratio(str(pc_fps_actual), 15, 35)),
                            ("SEND FPS",   f"{send_fps_actual:.1f}",        UI_ACCENT,
                             _pct_ratio(str(send_fps_actual), 0, BASE_FPS)),
                            ("SKIPPED/s",  f"{skipped_last_s}",             UI_ACCENT, None),
                            ("WAIT/T",     f"{pacing_us}us",                UI_ACCENT, None),
                            ("JITTER",     f"{jitter_dev_ms:.2f}ms",        jit_col,
                             _pct_ratio(str(jitter_dev_ms), 0, 10)),
                            ("OVF STRK",   f"{ovf_strk}",                   ovf_col,
                             _pct_ratio(str(ovf_strk), 0, 5)),
                            ("COOLDOWN",   f"{cool_strk}/{COOLDOWN_FRAMES}",cool_col,
                             cool_frac),
                            ("RAW",        f"{frame_byte_count}B",          size_col,
                             _pct_ratio(str(frame_byte_count), 0, QUALITY_BYTE_THRESHOLD)),
                            ("AVG EMA",    f"{avg_bytes_i}B",               UI_ACCENT,
                             _pct_ratio(str(avg_bytes_i), 0, QUALITY_BYTE_THRESHOLD)),
                            ("QUALITY",    f"{current_qual}",               (255, 255, 255),
                             _pct_ratio(str(current_qual), 0, 95)),
                            ("Q FALLBACK", "YES" if last_frame_bytes["quality_fallback"] else "NO",
                             UI_WARN if last_frame_bytes["quality_fallback"] else UI_OK, None),
                            ("FRAME Q", f"{last_frame_bytes['frame_quality']}", UI_ACCENT,
                             _pct_ratio(str(last_frame_bytes["frame_quality"]), 0, 95)),
                            ("TARGET FPS", f"{BASE_FPS}",                   UI_LABEL_COL, None),
                        ]
                        footer_txt = f"{sub_str} chroma  ·  PC encoder stats  ·  Q to quit"

                    # Fit the cards to the metrics shown in this overlay.
                    cols, rows = 4, (len(metrics) + 3) // 4
                    margin, gap = 10, 6
                    grid_top = header_h + 8
                    grid_bottom = PREVIEW_H - 16
                    cell_w = (PREVIEW_W - margin * 2 - gap * (cols - 1)) // cols
                    cell_h = (grid_bottom - grid_top - gap * (rows - 1)) // rows

                    for i, (label, value, col, ratio) in enumerate(metrics):
                        r, c = divmod(i, cols)
                        cx = margin + c * (cell_w + gap)
                        cy = grid_top + r * (cell_h + gap)
                        _draw_chip(overlay, cx, cy, cell_w, cell_h, label, value, col, ratio)

                    cv2.putText(overlay, footer_txt, (10, PREVIEW_H - 4), UI_FONT, 0.34, UI_LABEL_COL, 1, cv2.LINE_AA)

                    overlay_cache = overlay
                    overlay_mode  = debug_state
                    overlay_mon   = current_monitor_idx
                    overlay_ts    = now_pv
                    preview_dirty = True

                if preview_dirty and preview_base is not None:
                    if debug_state in (1, 2) and overlay_cache is not None:
                        # Blend the WHOLE overlay (cards + gaps) against the live
                        # video in one shot; this is what makes
                        # DEBUG_OVERLAY_ALPHA govern the cards' transparency too.
                        shown = cv2.addWeighted(overlay_cache, DEBUG_OVERLAY_ALPHA,
                                                preview_base, 1.0 - DEBUG_OVERLAY_ALPHA, 0)
                    else:
                        shown = preview_base
                    cv2.imshow(window_name, shown)
                    preview_dirty = False

            # waitKey must run every iteration or the window stops handling
            # events, even on iterations where nothing was redrawn.
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # ── Frame clock ──
            # Everything above is "work". Whatever is left of this frame's slot
            # is spent asleep on the socket, so touch/stat packets are handled
            # the instant they land. No sleep if we are already behind.
            t_work_end = time.perf_counter()
            work_s = t_work_end - t_start
            wait_until(deadline)
            last_wait_s = max(0.0, time.perf_counter() - t_work_end)

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
        temp_settings_file = f"{settings_file}.{os.getpid()}.tmp"
        try:
            with open(temp_settings_file, 'w') as f:
                json.dump(final_settings, f, indent=4)
            os.replace(temp_settings_file, settings_file)
        except (OSError, TypeError, ValueError) as exc:
            print(f"[Settings] Could not save {settings_file}: {exc}")
        finally:
            if os.path.exists(temp_settings_file):
                try:
                    os.remove(temp_settings_file)
                except OSError as exc:
                    print(f"[Settings] Could not remove temporary settings file: {exc}")

        with instance_lock:
            if current_monitor_idx in claimed_monitors:
                claimed_monitors.remove(current_monitor_idx)

        try: sct.close()
        except Exception: pass
        cv2.destroyAllWindows()
        sock.close()

# ─────────────────────────────────────────────
#  MAIN: CONTINUOUS DISCOVERY + INSTANCE SPAWNER
# ─────────────────────────────────────────────
def discovery_main():
    set_high_priority()
    set_high_resolution_timer()
    set_dpi_aware()
    # The original, working flow is: user turns on the Windows hotspot manually.
    # We do not force the unsupported hostednetwork API here.
    ensure_wifi_hotspot()

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