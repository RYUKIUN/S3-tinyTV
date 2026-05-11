import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import threading
import os
import struct
import json

# ============================================================
# ESP32 HYBRID VIDEO ENCODER
# MJPEG + DELTA PATCH P-FRAMES
# ============================================================

ESP_W = 320
ESP_H = 240
BLOCK = 16
THUMB_COUNT = 120
THUMB_H = 48
THUMB_W = int(THUMB_H * ESP_W / ESP_H)
STRIP_H = THUMB_H + 24

OUTPUT_DIR = "output"
OUTPUT_FILE = "video.esv2"

SETTINGS_FILE = "esv2_settings.json"

DEFAULT = {
    "fps": 24,
    "iframe_interval": 24,
    "jpeg_quality": 50,
    "patch_quality": 35,
    "threshold": 12,
}


class HybridEncoder:
    def __init__(self, root):
        self.root = root
        self.root.title("ESP32 Hybrid Video Encoder")
        self.root.geometry("1000x700")
        self.root.configure(bg="#111")

        self.cap = None
        self.total_frames = 0
        self.src_fps = 30

        self.in_frame = 0
        self.out_frame = 0

        self._drag = None
        self._strip_imgs = []
        self._cancel = False

        self.var_fps = tk.IntVar(value=DEFAULT["fps"])
        self.var_iframe = tk.IntVar(value=DEFAULT["iframe_interval"])
        self.var_jpeg = tk.IntVar(value=DEFAULT["jpeg_quality"])
        self.var_patch = tk.IntVar(value=DEFAULT["patch_quality"])
        self.var_thresh = tk.IntVar(value=DEFAULT["threshold"])

        self._load_settings()
        self._build_ui()

    # ============================================================
    # SETTINGS
    # ============================================================

    def _load_settings(self):
        try:
            with open(SETTINGS_FILE, "r") as f:
                d = json.load(f)

            self.var_fps.set(d.get("fps", DEFAULT["fps"]))
            self.var_iframe.set(d.get("iframe_interval", DEFAULT["iframe_interval"]))
            self.var_jpeg.set(d.get("jpeg_quality", DEFAULT["jpeg_quality"]))
            self.var_patch.set(d.get("patch_quality", DEFAULT["patch_quality"]))
            self.var_thresh.set(d.get("threshold", DEFAULT["threshold"]))
        except:
            pass

    def _save_settings(self):
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump({
                    "fps": self.var_fps.get(),
                    "iframe_interval": self.var_iframe.get(),
                    "jpeg_quality": self.var_jpeg.get(),
                    "patch_quality": self.var_patch.get(),
                    "threshold": self.var_thresh.get(),
                }, f, indent=2)
        except:
            pass

    # ============================================================
    # UI
    # ============================================================

    def _build_ui(self):

        top = tk.Frame(self.root, bg="#111")
        top.pack(fill=tk.X, padx=10, pady=10)

        self.path_var = tk.StringVar(value="No file selected")

        tk.Entry(
            top,
            textvariable=self.path_var,
            bg="#222",
            fg="#fff"
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(
            top,
            text="Browse",
            command=self._browse
        ).pack(side=tk.LEFT, padx=5)

        # timeline
        self.strip_canvas = tk.Canvas(
            self.root,
            height=STRIP_H + 20,
            bg="#222",
            highlightthickness=0
        )
        self.strip_canvas.pack(fill=tk.X, padx=10)

        self.strip_canvas.bind("<Configure>", self._on_strip_resize)
        self.strip_canvas.bind("<ButtonPress-1>", self._on_strip_press)
        self.strip_canvas.bind("<B1-Motion>", self._on_strip_drag)
        self.strip_canvas.bind("<ButtonRelease-1>", self._on_strip_release)

        # settings
        sf = tk.Frame(self.root, bg="#111")
        sf.pack(fill=tk.X, padx=10, pady=10)

        def add_spin(label, var, lo, hi):
            f = tk.Frame(sf, bg="#111")
            f.pack(side=tk.LEFT, padx=8)

            tk.Label(f, text=label, bg="#111", fg="#fff").pack()

            tk.Spinbox(
                f,
                from_=lo,
                to=hi,
                textvariable=var,
                width=8,
                bg="#222",
                fg="#fff"
            ).pack()

        add_spin("FPS", self.var_fps, 1, 60)
        add_spin("I-FRAME", self.var_iframe, 1, 300)
        add_spin("JPEG Q", self.var_jpeg, 1, 100)
        add_spin("PATCH Q", self.var_patch, 1, 100)
        add_spin("THRESH", self.var_thresh, 1, 255)

        # output
        out_f = tk.Frame(self.root, bg="#111")
        out_f.pack(fill=tk.X, padx=10)

        self.out_path_var = tk.StringVar(
            value=os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        )

        tk.Entry(
            out_f,
            textvariable=self.out_path_var,
            bg="#222",
            fg="#fff"
        ).pack(fill=tk.X)

        # log
        self.log = tk.Text(
            self.root,
            bg="#1a1a1a",
            fg="#fff",
            height=14
        )
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # progress
        bot = tk.Frame(self.root, bg="#111")
        bot.pack(fill=tk.X, padx=10, pady=10)

        self.progress_var = tk.DoubleVar()

        ttk.Progressbar(
            bot,
            variable=self.progress_var,
            maximum=100
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.enc_btn = tk.Button(
            bot,
            text="ENCODE",
            bg="#ff5500",
            fg="#fff",
            command=self._start_encode
        )
        self.enc_btn.pack(side=tk.RIGHT, padx=5)

        tk.Button(
            bot,
            text="Cancel",
            command=self._do_cancel
        ).pack(side=tk.RIGHT)

    # ============================================================
    # LOG
    # ============================================================

    def _log(self, txt):
        self.log.insert(tk.END, txt + "\n")
        self.log.see(tk.END)

    # ============================================================
    # VIDEO LOAD
    # ============================================================

    def _browse(self):

        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov")]
        )

        if not path:
            return

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open file")
            return

        self.path_var.set(path)

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

        self.in_frame = 0
        self.out_frame = self.total_frames - 1

        self._log(f"Loaded: {os.path.basename(path)}")
        self._log(f"Frames: {self.total_frames}")
        self._log(f"FPS: {self.src_fps:.2f}")

        self._strip_imgs = []
        self._redraw_strip()

        threading.Thread(
            target=self._build_strip_thread,
            args=(path,),
            daemon=True
        ).start()

    # ============================================================
    # TIMELINE
    # ============================================================

    def _build_strip_thread(self, path):

        cap = cv2.VideoCapture(path)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        indices = np.linspace(0, total - 1, THUMB_COUNT, dtype=int)

        thumbs = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (THUMB_W, THUMB_H))

            thumbs.append(frame)

        cap.release()

        self.root.after(0, self._on_strip_ready, thumbs)

    def _on_strip_ready(self, thumbs):

        self._strip_imgs = [
            ImageTk.PhotoImage(Image.fromarray(t))
            for t in thumbs
        ]

        self._redraw_strip()

    def _strip_width(self):
        return self.strip_canvas.winfo_width() or 800

    def _frame_to_x(self, frame):
        return int(frame / max(1, self.total_frames - 1) * self._strip_width())

    def _x_to_frame(self, x):
        return max(0, min(self.total_frames - 1,
                          int(x / self._strip_width() * self.total_frames)))

    def _redraw_strip(self):

        c = self.strip_canvas
        c.delete("all")

        if not self._strip_imgs:
            return

        w = self._strip_width()
        step = w / THUMB_COUNT

        for i, img in enumerate(self._strip_imgs):
            c.create_image(int(i * step), 0, anchor="nw", image=img)

        in_x = self._frame_to_x(self.in_frame)
        out_x = self._frame_to_x(self.out_frame)

        c.create_rectangle(in_x, 0, out_x, THUMB_H,
                           outline="#ffe033", width=2)

    def _on_strip_resize(self, e):
        self._redraw_strip()

    def _on_strip_press(self, e):

        in_x = self._frame_to_x(self.in_frame)
        out_x = self._frame_to_x(self.out_frame)

        if abs(e.x - in_x) < abs(e.x - out_x):
            self._drag = "in"
        else:
            self._drag = "out"

    def _on_strip_drag(self, e):

        if not self._drag:
            return

        f = self._x_to_frame(e.x)

        if self._drag == "in":
            self.in_frame = min(f, self.out_frame - 1)
        else:
            self.out_frame = max(f, self.in_frame + 1)

        self._redraw_strip()

    def _on_strip_release(self, e):
        self._drag = None

    # ============================================================
    # ENCODER
    # ============================================================

    def _do_cancel(self):
        self._cancel = True

    def _start_encode(self):

        if not self.cap:
            return

        self._cancel = False

        self.enc_btn.config(state="disabled")

        threading.Thread(
            target=self._encode_worker,
            daemon=True
        ).start()

    def _write_header(self, f, frame_count):

        f.write(b'ESV2')
        f.write(struct.pack('<H', ESP_W))
        f.write(struct.pack('<H', ESP_H))
        f.write(struct.pack('<H', self.var_fps.get()))
        f.write(struct.pack('<H', BLOCK))
        f.write(struct.pack('<I', frame_count))

    def _encode_iframe(self, f, frame):

        ok, enc = cv2.imencode(
            '.jpg',
            frame,
            [cv2.IMWRITE_JPEG_QUALITY,
             self.var_jpeg.get()]
        )

        if not ok:
            return

        data = enc.tobytes()

        f.write(b'I')
        f.write(struct.pack('<I', len(data)))
        f.write(data)

        return len(data)

    def _encode_pframe(self, f, cur, prev):

        threshold = self.var_thresh.get()

        patches = []

        for y in range(0, ESP_H, BLOCK):
            for x in range(0, ESP_W, BLOCK):

                c = cur[y:y+BLOCK, x:x+BLOCK]
                p = prev[y:y+BLOCK, x:x+BLOCK]

                sad = np.mean(np.abs(
                    c.astype(np.int16) - p.astype(np.int16)
                ))

                if sad < threshold:
                    continue

                ok, enc = cv2.imencode(
                    '.jpg',
                    c,
                    [cv2.IMWRITE_JPEG_QUALITY,
                     self.var_patch.get()]
                )

                if not ok:
                    continue

                patches.append((x, y, enc.tobytes()))

        payload = bytearray()

        payload += struct.pack('<H', len(patches))

        for x, y, data in patches:
            payload += struct.pack('<BBH', x // BLOCK, y // BLOCK, len(data))
            payload += data

        f.write(b'P')
        f.write(struct.pack('<I', len(payload)))
        f.write(payload)

        return len(payload), len(patches)

    def _encode_worker(self):

        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            out_path = self.out_path_var.get()

            cap = cv2.VideoCapture(self.path_var.get())

            out_fps = self.var_fps.get()
            stride = max(1, round(self.src_fps / out_fps))

            frames = list(range(self.in_frame,
                                self.out_frame + 1,
                                stride))

            total = len(frames)

            self.root.after(0, self._log,
                            f"Encoding {total} frames...")

            with open(out_path, 'wb') as f:

                self._write_header(f, total)

                prev = None

                iframe_interval = self.var_iframe.get()

                total_bytes = 0

                for idx, fn in enumerate(frames):

                    if self._cancel:
                        break

                    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)

                    ret, frame = cap.read()

                    if not ret:
                        continue

                    frame = cv2.resize(frame, (ESP_W, ESP_H))

                    force_i = (
                        prev is None or
                        idx % iframe_interval == 0
                    )

                    if force_i:

                        sz = self._encode_iframe(f, frame)

                        total_bytes += sz

                        self.root.after(
                            0,
                            self._log,
                            f"[{idx+1}/{total}] I-frame {sz/1024:.1f} KB"
                        )

                    else:

                        sz, patches = self._encode_pframe(f, frame, prev)

                        total_bytes += sz

                        self.root.after(
                            0,
                            self._log,
                            f"[{idx+1}/{total}] P-frame {patches} patches {sz/1024:.1f} KB"
                        )

                    prev = frame.copy()

                    if idx % 2 == 0:
                        pct = (idx + 1) / total * 100
                        self.root.after(0, self.progress_var.set, pct)

            cap.release()

            final_size = os.path.getsize(out_path)

            self.root.after(0, self._log,
                            "=" * 40)

            self.root.after(0, self._log,
                            f"Done: {final_size/1024/1024:.2f} MB")

            self.root.after(0, self.progress_var.set, 100)

        except Exception as e:
            self.root.after(0, self._log,
                            f"ERROR: {e}")

        self.root.after(0,
                        lambda: self.enc_btn.config(state="normal"))

        self._save_settings()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = HybridEncoder(root)

    root.mainloop()
