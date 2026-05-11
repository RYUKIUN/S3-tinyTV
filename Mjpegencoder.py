"""
MJPEG Encoder for ESP32-S3 Offline Playback
============================================

OUTPUT FORMAT  —  simple MJPEG container
-----------------------------------------
All frames are JPEG-encoded at 4:2:0 chroma subsampling.
The file is a flat sequence of frame records:

  Per frame:
    [4 bytes]  magic  0xAA 0xBB 0xCC 0xDD
    [4 bytes]  frame size  N  (little-endian uint32)
    [N bytes]  raw JPEG data

ESP32 decoder skeleton:
    while (fread(hdr, 1, 8, f) == 8) {
        if (memcmp(hdr, "\xAA\xBB\xCC\xDD", 4) != 0) { /* resync */ }
        uint32_t sz;
        memcpy(&sz, hdr+4, 4);          // LE
        fread(jpeg_buf, 1, sz, f);
        jpegdec_decode(jpeg_buf, sz);   // → RGB565 to display
    }

Settings
--------
  FPS        : output frame rate  (source frames are skipped to match)
  Quality    : JPEG quality  1–95
  Sharpen    : unsharp-mask strength  0.0–2.0
  Chroma     : forced 4:2:0  (always on — matches ESP streaming behaviour)
"""

import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import threading
import os
import struct
import json

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
ESP_W, ESP_H   = 320, 240
OUTPUT_DIR     = "d:/PlatformIO/S3-nextgen jpeg/data"
OUTPUT_FILE    = "video.mjpeg"
SETTINGS_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mjpeg_enc_settings.json")
THUMB_COUNT    = 120        # number of thumbnails in the timeline strip
THUMB_H        = 48         # thumbnail height in strip
THUMB_W        = int(THUMB_H * ESP_W / ESP_H)  # 64
STRIP_H        = THUMB_H + 24  # strip + label row

FRAME_MAGIC    = b'\xAA\xBB\xCC\xDD'

DEFAULT = {
    "fps":     15,
    "quality": 40,
    "sharpen": 0.6,
}

# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
class MJPEGEncoder:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MJPEG Encoder  //  ESP32-S3")
        self.root.geometry("1000x640")
        self.root.configure(bg="#111")
        self.root.resizable(False, False)

        # Video state
        self.cap          = None
        self.total_frames = 0
        self.src_fps      = 30.0
        self.in_frame     = 0
        self.out_frame    = 0

        # Drag state for handles
        self._drag        = None   # "in" | "out" | None
        self._strip_imgs  = []     # keep PhotoImage refs

        # Settings vars
        self.var_fps     = tk.IntVar(value=DEFAULT["fps"])
        self.var_quality = tk.IntVar(value=DEFAULT["quality"])
        self.var_sharpen = tk.DoubleVar(value=DEFAULT["sharpen"])

        # Cancel flag
        self._cancel = False

        self._load_settings()
        self._build_ui()

    # ─────────────────────────────────────
    #  SETTINGS PERSISTENCE
    # ─────────────────────────────────────
    def _load_settings(self):
        try:
            with open(SETTINGS_FILE) as f:
                d = json.load(f)
            self.var_fps.set(d.get("fps",     DEFAULT["fps"]))
            self.var_quality.set(d.get("quality", DEFAULT["quality"]))
            self.var_sharpen.set(d.get("sharpen", DEFAULT["sharpen"]))
        except Exception:
            pass

    def _save_settings(self):
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump({
                    "fps":     self.var_fps.get(),
                    "quality": self.var_quality.get(),
                    "sharpen": self.var_sharpen.get(),
                }, f, indent=2)
        except Exception:
            pass

    # ─────────────────────────────────────
    #  UI BUILD
    # ─────────────────────────────────────
    def _build_ui(self):
        S = self.S = {
            "bg":      "#111111",
            "panel":   "#1c1c1c",
            "border":  "#2e2e2e",
            "accent":  "#ffe033",
            "accent2": "#ff5500",
            "text":    "#dddddd",
            "dim":     "#666666",
            "handle":  "#ffe033",
            "handle_out": "#ff5500",
            "mono":    ("Courier New", 9),
            "mono_b":  ("Courier New", 10, "bold"),
            "hd":      ("Courier New", 13, "bold"),
        }

        style = ttk.Style()
        style.theme_use("clam")
        for widget in ("TFrame", "TLabel", "TLabelframe", "TLabelframe.Label",
                       "TButton", "TSpinbox", "TProgressbar"):
            style.configure(widget, background=S["bg"], foreground=S["text"],
                            font=S["mono"], borderwidth=0, relief="flat")
        style.configure("TLabelframe",       bordercolor=S["border"])
        style.configure("TLabelframe.Label", foreground=S["accent"], font=S["mono_b"])
        style.configure("TButton",
            background=S["border"], foreground=S["text"], font=S["mono"])
        style.map("TButton",
            background=[("active", S["accent"])],
            foreground=[("active", "#000")])
        style.configure("Enc.TButton",
            background=S["accent2"], foreground="#fff", font=S["hd"])
        style.map("Enc.TButton",
            background=[("active", S["accent"])], foreground=[("active","#000")])
        style.configure("TSpinbox",
            fieldbackground=S["panel"], foreground=S["text"],
            arrowcolor=S["accent"])
        style.configure("TProgressbar",
            troughcolor=S["border"], background=S["accent2"], thickness=8)

        # ── TITLE ─────────────────────────────
        tk.Label(self.root, text="MJPEG ENCODER  //  ESP32-S3 OFFLINE PLAYBACK",
                 bg=S["bg"], fg=S["accent"], font=S["hd"]).pack(
                 fill=tk.X, padx=14, pady=(12,0), anchor="w")
        tk.Frame(self.root, bg=S["accent"], height=1).pack(fill=tk.X, padx=14, pady=(4,8))

        # ── FILE ROW ──────────────────────────
        file_row = tk.Frame(self.root, bg=S["bg"])
        file_row.pack(fill=tk.X, padx=14, pady=(0,6))

        self.path_var = tk.StringVar(value="No file selected")
        tk.Entry(file_row, textvariable=self.path_var,
                 bg=S["panel"], fg=S["dim"], font=S["mono"],
                 insertbackground=S["accent"], relief="flat",
                 state="readonly", readonlybackground=S["panel"],
                 bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        ttk.Button(file_row, text="BROWSE MP4",
                   command=self._browse).pack(side=tk.RIGHT, padx=(6,0), ipadx=6, ipady=3)

        # ── TIMELINE STRIP ────────────────────
        strip_outer = ttk.LabelFrame(self.root, text="TIMELINE  —  drag handles to set IN / OUT")
        strip_outer.pack(fill=tk.X, padx=14, pady=(0,6))

        self.strip_canvas = tk.Canvas(strip_outer,
            height=STRIP_H + 20, bg=S["panel"],
            highlightthickness=0)
        self.strip_canvas.pack(fill=tk.X, padx=4, pady=4)
        self.strip_canvas.bind("<Configure>",   self._on_strip_resize)
        self.strip_canvas.bind("<ButtonPress-1>",   self._on_strip_press)
        self.strip_canvas.bind("<B1-Motion>",        self._on_strip_drag)
        self.strip_canvas.bind("<ButtonRelease-1>",  self._on_strip_release)

        # Frame counter row
        info_row = tk.Frame(self.root, bg=S["bg"])
        info_row.pack(fill=tk.X, padx=14, pady=(0,6))
        self.lbl_in  = tk.Label(info_row, text="IN  :   0", bg=S["bg"],
                                fg=S["handle"],     font=S["mono_b"])
        self.lbl_out = tk.Label(info_row, text="OUT :   0", bg=S["bg"],
                                fg=S["handle_out"], font=S["mono_b"])
        self.lbl_dur = tk.Label(info_row, text="DUR :   0 s", bg=S["bg"],
                                fg=S["dim"],        font=S["mono"])
        self.lbl_in.pack(side=tk.LEFT, padx=(0,20))
        self.lbl_out.pack(side=tk.LEFT, padx=(0,20))
        self.lbl_dur.pack(side=tk.LEFT)

        # ── SETTINGS ROW ──────────────────────
        settings_frame = ttk.LabelFrame(self.root, text="ENCODER SETTINGS")
        settings_frame.pack(fill=tk.X, padx=14, pady=(0,6))
        inner = tk.Frame(settings_frame, bg=S["bg"])
        inner.pack(fill=tk.X, padx=8, pady=6)

        def spinrow(parent, label, var, lo, hi, inc, tip):
            f = tk.Frame(parent, bg=S["bg"])
            f.pack(side=tk.LEFT, padx=(0, 30))
            tk.Label(f, text=label, bg=S["bg"], fg=S["dim"],
                     font=S["mono"]).pack(anchor="w")
            ttk.Spinbox(f, from_=lo, to=hi, increment=inc,
                        textvariable=var, width=7).pack(anchor="w", pady=(2,0))
            tk.Label(f, text=tip, bg=S["bg"], fg=S["border"],
                     font=("Courier New", 7)).pack(anchor="w")

        spinrow(inner, "OUTPUT FPS",  self.var_fps,     1,   60, 1,
                "frames/sec in output file")
        spinrow(inner, "JPEG QUALITY", self.var_quality, 1,  95, 1,
                "1=tiny  95=best  (JPEG q)")
        spinrow(inner, "SHARPEN",     self.var_sharpen, 0.0, 2.0, 0.1,
                "unsharp-mask  0=off  2=strong")

        tk.Label(inner, text="CHROMA\n4:2:0  ✓ forced",
                 bg=S["bg"], fg=S["accent"], font=S["mono"],
                 justify="center").pack(side=tk.LEFT, padx=(0,30))

        # Output path
        out_f = tk.Frame(inner, bg=S["bg"])
        out_f.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(out_f, text="OUTPUT", bg=S["bg"], fg=S["dim"],
                 font=S["mono"]).pack(anchor="w")
        self.out_path_var = tk.StringVar(
            value=os.path.join(OUTPUT_DIR, OUTPUT_FILE))
        tk.Entry(out_f, textvariable=self.out_path_var,
                 bg=S["panel"], fg=S["text"], font=S["mono"],
                 insertbackground=S["accent"], relief="flat",
                 bd=0).pack(fill=tk.X, ipady=4)

        # ── LOG ───────────────────────────────
        log_frame = ttk.LabelFrame(self.root, text="LOG")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0,6))
        self.log = tk.Text(log_frame, height=5, bg=S["panel"], fg=S["text"],
                           font=S["mono"], relief="flat", bd=4,
                           insertbackground=S["accent"])
        self.log.pack(fill=tk.BOTH, expand=True)
        self._log("Ready. Browse a video file to start.")

        # ── PROGRESS + BUTTONS ────────────────
        bot = tk.Frame(self.root, bg=S["bg"])
        bot.pack(fill=tk.X, padx=14, pady=(0,10))

        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(bot, variable=self.progress_var,
                        maximum=100, style="TProgressbar",
                        length=600).pack(side=tk.LEFT, fill=tk.X,
                                         expand=True, padx=(0,8), ipady=3)

        ttk.Button(bot, text="CANCEL",
                   command=self._do_cancel).pack(side=tk.RIGHT, padx=(4,0),
                                                  ipadx=6, ipady=3)
        self.enc_btn = ttk.Button(bot, text="ENCODE  [Enter]",
                                  style="Enc.TButton",
                                  command=self._start_encode)
        self.enc_btn.pack(side=tk.RIGHT, ipadx=10, ipady=3)

        self.root.bind("<Return>", lambda e: self._start_encode())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────
    #  LOG
    # ─────────────────────────────────────
    def _log(self, msg: str):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def _log_clear(self):
        self.log.delete("1.0", tk.END)

    # ─────────────────────────────────────
    #  FILE BROWSE
    # ─────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")])
        if not path:
            return
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file")
            return

        self.path_var.set(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.src_fps      = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        sw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        sh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.in_frame  = 0
        self.out_frame = self.total_frames - 1

        self._log_clear()
        self._log(f"Loaded : {os.path.basename(path)}")
        self._log(f"Source : {sw}×{sh}  {self.src_fps:.2f} fps  "
                  f"{self.total_frames} frames  "
                  f"({self.total_frames/self.src_fps:.1f}s)")
        self._log(f"Output : {ESP_W}×{ESP_H}  → {self.out_path_var.get()}")
        self._log("Building timeline strip…")

        self._update_labels()
        # Draw empty strip immediately so the UI is responsive
        self._strip_imgs = []
        self._redraw_strip()

        # Build thumbnails in background; a separate cap avoids racing with encode
        threading.Thread(target=self._build_strip_thread,
                         args=(path,), daemon=True).start()

    # ─────────────────────────────────────
    #  TIMELINE STRIP
    # ─────────────────────────────────────
    def _build_strip_thread(self, path: str):
        """
        Runs in background thread.
        Reads thumbnail frames sequentially (no random seek = much faster
        for most container formats), collects raw numpy arrays, then
        schedules PhotoImage creation on the main thread one-by-one so
        tkinter is never touched from a worker thread.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, THUMB_COUNT, dtype=int)
        index_set = set(indices.tolist())

        thumbs = []   # list of numpy arrays in order
        fn = 0
        idx_cursor = 0

        # Walk forward through the file — avoids expensive random seeks
        while idx_cursor < len(indices):
            target = int(indices[idx_cursor])
            if fn < target:
                # skip frames we don't need
                grabbed = cap.grab()
                if not grabbed:
                    break
                fn += 1
                continue
            # fn == target
            ret, frame = cap.retrieve() if fn == target else (False, None)
            if not ret:
                ret, frame = cap.read()
            if not ret or frame is None:
                thumbs.append(np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                thumbs.append(cv2.resize(frame, (THUMB_W, THUMB_H),
                                         interpolation=cv2.INTER_AREA))
            idx_cursor += 1
            fn += 1

        cap.release()

        # Schedule UI update on main thread
        self.root.after(0, self._on_strip_ready, thumbs)

    def _on_strip_ready(self, thumbs: list):
        """Called on main thread after background strip build completes."""
        self._strip_imgs = [
            ImageTk.PhotoImage(Image.fromarray(t)) for t in thumbs
        ]
        self._redraw_strip()
        self._log("Timeline ready.")

    def _on_strip_resize(self, event):
        self._redraw_strip()

    def _strip_width(self) -> int:
        return self.strip_canvas.winfo_width() or 900

    def _frame_to_x(self, frame: int) -> int:
        w = self._strip_width()
        return int(frame / max(1, self.total_frames - 1) * w)

    def _x_to_frame(self, x: int) -> int:
        w = self._strip_width()
        f = int(x / max(1, w) * self.total_frames)
        return max(0, min(self.total_frames - 1, f))

    def _redraw_strip(self):
        c = self.strip_canvas
        S = self.S
        c.delete("all")
        w = self._strip_width()
        if not self._strip_imgs or w < 2:
            return

        # ── thumbnail row ─────────────────────
        thumb_total = THUMB_W * THUMB_COUNT
        step = w / THUMB_COUNT
        for i, img in enumerate(self._strip_imgs):
            x = int(i * step)
            c.create_image(x, 0, anchor="nw", image=img)

        # ── dim overlay outside in/out ─────────
        in_x  = self._frame_to_x(self.in_frame)
        out_x = self._frame_to_x(self.out_frame)

        if in_x > 0:
            c.create_rectangle(0, 0, in_x, THUMB_H,
                               fill="#000", stipple="gray50", outline="")
        if out_x < w:
            c.create_rectangle(out_x, 0, w, THUMB_H,
                               fill="#000", stipple="gray50", outline="")

        # ── selected range highlight ───────────
        c.create_rectangle(in_x, 0, out_x, THUMB_H,
                           outline=S["accent"], width=2, fill="")

        # ── IN handle (yellow) ────────────────
        c.create_rectangle(in_x - 3, 0, in_x + 3, THUMB_H,
                           fill=S["handle"], outline="")
        c.create_polygon(in_x, THUMB_H,
                         in_x - 10, THUMB_H + 16,
                         in_x + 10, THUMB_H + 16,
                         fill=S["handle"], outline="")
        c.create_text(in_x, THUMB_H + 18, text=f"IN {self.in_frame}",
                      fill=S["handle"], font=("Courier New", 7, "bold"),
                      anchor="n")

        # ── OUT handle (orange) ───────────────
        c.create_rectangle(out_x - 3, 0, out_x + 3, THUMB_H,
                           fill=S["handle_out"], outline="")
        c.create_polygon(out_x, THUMB_H,
                         out_x - 10, THUMB_H + 16,
                         out_x + 10, THUMB_H + 16,
                         fill=S["handle_out"], outline="")
        c.create_text(out_x, THUMB_H + 18, text=f"OUT {self.out_frame}",
                      fill=S["handle_out"], font=("Courier New", 7, "bold"),
                      anchor="n")

    def _on_strip_press(self, event):
        if self.total_frames == 0:
            return
        in_x  = self._frame_to_x(self.in_frame)
        out_x = self._frame_to_x(self.out_frame)
        # Snap to whichever handle is closer
        if abs(event.x - in_x) <= abs(event.x - out_x):
            self._drag = "in"
        else:
            self._drag = "out"

    def _on_strip_drag(self, event):
        if not self._drag:
            return
        f = self._x_to_frame(event.x)
        if self._drag == "in":
            self.in_frame  = min(f, self.out_frame - 1)
        else:
            self.out_frame = max(f, self.in_frame + 1)
        self._redraw_strip()
        self._update_labels()

    def _on_strip_release(self, event):
        self._drag = None

    def _update_labels(self):
        dur = (self.out_frame - self.in_frame) / self.src_fps
        self.lbl_in.config(text=f"IN  : {self.in_frame:6d}")
        self.lbl_out.config(text=f"OUT : {self.out_frame:6d}")
        self.lbl_dur.config(text=f"DUR : {dur:6.2f} s"
                            f"  ({self.out_frame - self.in_frame} frames)")

    # ─────────────────────────────────────
    #  ENCODE
    # ─────────────────────────────────────
    def _do_cancel(self):
        self._cancel = True

    def _start_encode(self):
        if not self.cap:
            messagebox.showerror("Error", "No video loaded")
            return
        if self.in_frame >= self.out_frame:
            messagebox.showerror("Error", "IN must be before OUT")
            return
        self._cancel = False
        self.enc_btn.config(state="disabled")
        threading.Thread(target=self._encode_worker, daemon=True).start()

    def _encode_worker(self):
        out_fps   = self.var_fps.get()
        quality   = self.var_quality.get()
        sharpen   = self.var_sharpen.get()
        in_f      = self.in_frame
        out_f     = self.out_frame
        src_fps   = self.src_fps
        out_path  = self.out_path_var.get()

        # Which source frames to encode (stride to hit target fps)
        stride      = max(1, round(src_fps / out_fps))
        src_frames  = list(range(in_f, out_f + 1, stride))
        n_total     = len(src_frames)

        self.root.after(0, self._log_clear)
        self.root.after(0, self._log,
            f"Encoding {n_total} frames  "
            f"({in_f}→{out_f}  stride={stride})")
        self.root.after(0, self._log,
            f"FPS={out_fps}  Q={quality}  sharpen={sharpen:.1f}  "
            f"chroma=4:2:0")
        self.root.after(0, self._log, "─" * 52)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        # JPEG encode params — force 4:2:0
        encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY),          quality,
            int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),  cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
        ]

        total_bytes  = 0
        frames_done  = 0

        with open(out_path, "wb") as f:
            for fn in src_frames:
                if self._cancel:
                    self.root.after(0, self._log, "Cancelled.")
                    break

                self.cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # ── pre-process (same pipeline as captureJpeg.py) ─
                frame = cv2.resize(frame, (ESP_W, ESP_H),
                                   interpolation=cv2.INTER_AREA)

                if sharpen > 0:
                    frame = cv2.addWeighted(
                        frame, 1.0 + sharpen,
                        cv2.GaussianBlur(frame, (0, 0), 0.3 + sharpen * 0.35),
                        -sharpen, 0)

                # ── JPEG encode ──────────────────────────────────
                ret2, buf = cv2.imencode(".jpg", frame, encode_params)
                if not ret2:
                    continue

                data = buf.tobytes()
                sz   = len(data)

                # ── write frame record ───────────────────────────
                f.write(FRAME_MAGIC)                    # 4 B magic
                f.write(struct.pack("<I", sz))          # 4 B size LE
                f.write(data)                           # N B JPEG

                total_bytes  += 8 + sz
                frames_done  += 1

                # ── progress update every 15 frames ─────────────
                pct = frames_done / n_total * 100
                self.root.after(0, self.progress_var.set, pct)
                if frames_done % 15 == 0 or frames_done == n_total:
                    self.root.after(0, self._log,
                        f"[{frames_done:4d}/{n_total}]  "
                        f"fn={fn:5d}  sz={sz:5d} B  "
                        f"total={total_bytes/1024:.1f} kB")

        if not self._cancel:
            file_sz = os.path.getsize(out_path)
            self.root.after(0, self._log, "─" * 52)
            self.root.after(0, self._log,
                f"Done!  {frames_done} frames  "
                f"({file_sz/1024:.1f} kB  "
                f"/ {file_sz/1024/1024:.2f} MB)")
            self.root.after(0, self._log,
                f"Avg/frame : {file_sz//max(1,frames_done)} B")
            self.root.after(0, self._log, f"Saved → {out_path}")

        self.root.after(0, self.progress_var.set, 100 if not self._cancel else 0)
        self.root.after(0, self.enc_btn.config, {"state": "normal"})
        self._save_settings()

    # ─────────────────────────────────────
    #  CLOSE
    # ─────────────────────────────────────
    def _on_close(self):
        self._cancel = True
        self._save_settings()
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = MJPEGEncoder(root)
    root.mainloop()