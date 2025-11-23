# gui.py -- simple Tkinter GUI for your signature verifier (dynamic loading, simple UI)
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Ensure src folder is importable (works when gui.py is in project root)
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import your pipeline functions
try:
    from preprocess import preprocess_image
    from features import contour_to_points
    from greedy import greedy_score
    from dtw_dp import dtw_distance
except Exception as e:
    raise ImportError("Failed to import pipeline modules from src/. Make sure src/ contains preprocess.py, features.py, greedy.py, dtw_dp.py") from e

import cv2
from PIL import Image, ImageTk
import numpy as np

DEBUG_DIR = os.path.join(ROOT, "data", "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

# GUI constants
PREVIEW_W = 460
PREVIEW_H = 160

class SignatureGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Signature Verification — GUI")
        self.geometry("980x520")
        self.resizable(False, False)

        self.ref_path = None
        self.qry_path = None

        self.create_widgets()

    def create_widgets(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side="top", fill="x")

        btn_ref = ttk.Button(top, text="Select Reference Signature", command=self.select_ref)
        btn_ref.grid(row=0, column=0, padx=6, pady=6)

        self.lbl_ref_file = ttk.Label(top, text="No file selected", width=60)
        self.lbl_ref_file.grid(row=0, column=1, padx=6, pady=6)

        btn_qry = ttk.Button(top, text="Select Query Signature", command=self.select_qry)
        btn_qry.grid(row=1, column=0, padx=6, pady=6)

        self.lbl_qry_file = ttk.Label(top, text="No file selected", width=60)
        self.lbl_qry_file.grid(row=1, column=1, padx=6, pady=6)

        btn_verify = ttk.Button(top, text="Verify", command=self.on_verify, width=20)
        btn_verify.grid(row=0, column=2, rowspan=2, padx=12, pady=6)

        # Image previews
        frame = ttk.Frame(self, padding=8)
        frame.pack(side="top", fill="both", expand=False)

        # Reference preview
        ref_frame = ttk.LabelFrame(frame, text="Reference", width=PREVIEW_W, height=PREVIEW_H)
        ref_frame.grid(row=0, column=0, padx=8, pady=8)
        self.ref_canvas = tk.Canvas(ref_frame, width=PREVIEW_W, height=PREVIEW_H, bg="#222")
        self.ref_canvas.pack()
        # Query preview
        qry_frame = ttk.LabelFrame(frame, text="Query", width=PREVIEW_W, height=PREVIEW_H)
        qry_frame.grid(row=0, column=1, padx=8, pady=8)
        self.qry_canvas = tk.Canvas(qry_frame, width=PREVIEW_W, height=PREVIEW_H, bg="#222")
        self.qry_canvas.pack()

        # Results area
        bottom = ttk.Frame(self, padding=8)
        bottom.pack(side="top", fill="x")

        self.lbl_greedy = ttk.Label(bottom, text="Greedy score: -", font=("Segoe UI", 11))
        self.lbl_greedy.grid(row=0, column=0, padx=6, pady=6, sticky="w")

        self.lbl_dtw = ttk.Label(bottom, text="DTW distance: -", font=("Segoe UI", 11))
        self.lbl_dtw.grid(row=1, column=0, padx=6, pady=6, sticky="w")

        self.lbl_decision = ttk.Label(bottom, text="Decision: -", font=("Segoe UI", 14, "bold"))
        self.lbl_decision.grid(row=0, column=1, rowspan=2, padx=20, pady=6)

        # Status bar
        self.status = ttk.Label(self, text="Ready", relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

    def select_ref(self):
        p = filedialog.askopenfilename(
            title="Select reference signature image",
            filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp"), ("All files","*.*")]
        )
        if p:
            self.ref_path = p
            self.lbl_ref_file.config(text=os.path.basename(p))
            self.display_image_on_canvas(p, self.ref_canvas, is_ref=True)

    def select_qry(self):
        p = filedialog.askopenfilename(
            title="Select query signature image",
            filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp"), ("All files","*.*")]
        )
        if p:
            self.qry_path = p
            self.lbl_qry_file.config(text=os.path.basename(p))
            self.display_image_on_canvas(p, self.qry_canvas, is_ref=False)

    def display_image_on_canvas(self, img_path, canvas, is_ref=False):
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((PREVIEW_W, PREVIEW_H), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)

        # keep reference alive
            if is_ref:
                self.ref_tkimg = photo
            else:
                self.qry_tkimg = photo

            canvas.delete("all")
            canvas.create_image(PREVIEW_W//2, PREVIEW_H//2, image=photo)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image: {e}")

    def on_verify(self):
        if not self.ref_path or not self.qry_path:
            messagebox.showwarning("Missing files", "Please select both reference and query images.")
            return
        # run in a thread to keep UI responsive
        threading.Thread(target=self.run_verification, daemon=True).start()

    def run_verification(self):
        try:
            self.status.config(text="Preprocessing images...")
            # preprocess and save debug images (same behavior as demo)
            ref_pre = preprocess_image(self.ref_path)
            qry_pre = preprocess_image(self.qry_path)
            cv2.imwrite(os.path.join(DEBUG_DIR, "gui_ref_pre.jpg"), ref_pre)
            cv2.imwrite(os.path.join(DEBUG_DIR, "gui_query_pre.jpg"), qry_pre)

            self.status.config(text="Extracting features...")
            A = contour_to_points(ref_pre)
            B = contour_to_points(qry_pre)

            if A is None or B is None:
                self.lbl_greedy.config(text="Greedy score: -")
                self.lbl_dtw.config(text="DTW distance: -")
                self.lbl_decision.config(text="Decision: Cannot compute")
                self.status.config(text="Failed to extract features; check data/debug for preprocessed images.")
                return

            self.status.config(text="Computing greedy score...")
            gscore = greedy_score(A, B)
            self.lbl_greedy.config(text=f"Greedy score: {gscore:.4f}")

            self.status.config(text="Computing DTW distance...")
            d = dtw_distance(A, B, band=None)
            self.lbl_dtw.config(text=f"DTW distance: {d:.4f}")

            # threshold — matches what we used in demo (tune if needed)
            decision = "Genuine" if d < 0.45 else "Forged"
            self.lbl_decision.config(text=f"Decision: {decision}")

            # Save visualizations (points)
            self.status.config(text="Saving debug visualizations...")
            self.save_feature_visualizations(ref_pre, A, "gui_ref_points.jpg")
            self.save_feature_visualizations(qry_pre, B, "gui_query_points.jpg")

            self.status.config(text=f"Done — decision: {decision}. Debug saved to data/debug")
        except Exception as e:
            messagebox.showerror("Error", f"Verification failed: {e}")
            self.status.config(text="Error during verification")

    def save_feature_visualizations(self, img, pts, fname):
        try:
            out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            h, w = out.shape[:2]
            for (x,y,*_) in pts:
                cx = int(x * w)
                cy = int(y * h)
                cv2.circle(out, (cx, cy), 2, (0,255,0), -1)
            cv2.imwrite(os.path.join(DEBUG_DIR, fname), out)
        except Exception:
            pass

if __name__ == "__main__":
    app = SignatureGUI()
    app.mainloop()
