#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_label_eeg_images.py
----------------------
Label EEG samples and camera frames with ground-truth driving actions.

Inputs  : Extracted CSVs and images 
Outputs : Labeled CSVs and overlaid images 

Author : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University 
"""

import os
import csv
import math
import cv2
import numpy as np
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Configuration: input/output paths
# ---------------------------------------------------------------------------
input_root  = "data/extracted"
output_root = "data/labeled"

# ---------------------------------------------------------------------------
# File naming conventions
# ---------------------------------------------------------------------------
EEG_SUFFIX   = "_eeg.csv"
GPS_SUFFIX   = "_gps.csv"
ODOM_SUFFIX  = "_odom.csv"
IMGIDX_SUFFIX = "_image_index.csv"
IMAGES_DIRNAME = "images"

OUTPUT_CSV_NAME     = "{prefix}_eeg_labeled.csv"
OUTPUT_IMG_LABELS   = "{prefix}_image_labels.csv"
LABELED_IMAGE_DIR   = "eeg_labeled_images"

# ---------------------------------------------------------------------------
# Motion labelling thresholds
# ---------------------------------------------------------------------------
VEL_THRESHOLD = 0.1    # m/s magnitude threshold ~ stop
YAW_THRESHOLD = 0.05   # rad/s threshold ~ turning

# ---------------------------------------------------------------------------
# Prediction horizons (ms)
# ---------------------------------------------------------------------------
DELTAS_MS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]  # in ms

# ---------------------------------------------------------------------------
# Label overlay colours for validation video (BGR format)
# ---------------------------------------------------------------------------
LABEL_COLORS = {
    "Stop":          (0, 0, 255),
    "Forward":       (0, 255, 0),
    "Reverse":       (255, 0, 0),
    "Turning Left":  (0, 255, 255),
    "Turning Right": (0, 140, 255),
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def safe_float(x):
    """Convert string to float robustly."""
    try:
        return float(str(x).strip())
    except Exception:
        # Fallback: remove stray characters (tabs, commas in header, etc.)
        try:
            return float(str(x).strip().replace('\t', '').replace(' ', ''))
        except Exception:
            return np.nan

def load_csv_rows(path):
    """
    Load CSV as list of rows (list[str]), skipping header.
    Handles commas/tabs and stray spaces.
    Falls back to manual line splitting if standard CSV parsing fails.
    """
    rows = []
    if not os.path.exists(path):
        return rows
    
    # Try standard CSV parsing first
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            rows = [r for r in reader if r]
        if rows:
            return rows
    except Exception:
        pass

    # Fallback: manual split on comma or tab
    with open(path, "r") as f:
        lines = f.read().splitlines()
    if not lines:
        return rows
    for ln in lines[1:]:  # skip header
        parts_comma = [p.strip() for p in ln.split(",")]
        parts_tab   = [p.strip() for p in ln.split("\t")]
        parts = parts_tab if len(parts_tab) > len(parts_comma) else parts_comma
        if len(parts) == 1 and parts[0] == "":
            continue
        rows.append(parts)
    return rows

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def build_interp(x, y):
    """
    Build a linear interpolator with safe defaults.
    Requires at least 2 points; if only 1 point, return constant function.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if len(x) == 0 or len(y) == 0 or np.isnan(x).any() or np.isnan(y).any():
        return None
    
    if len(x) == 1:
        v = float(y[0])
        return lambda t: v
    
    # Enforce sorted x
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    
    return interp1d(x_sorted, y_sorted, kind="linear", bounds_error=False, fill_value="extrapolate")

def movement_label(vx, vy, yaw, yaw_rate):
    """Compute movement direction label from velocities and yaw/yaw_rate."""
    vx = float(vx); vy = float(vy); yaw = float(yaw); yaw_rate = float(yaw_rate)
    speed_mag = math.hypot(vx, vy)

    # Stop first
    if (speed_mag < VEL_THRESHOLD) and (abs(yaw_rate) < YAW_THRESHOLD):
        return "Stop"

    velocity_angle = math.atan2(vy, vx)  # world-frame vel
    angle_diff = (yaw - velocity_angle + math.pi) % (2 * math.pi) - math.pi  # [-pi, pi]

    # Reverse if heading opposite to velocity (> 90 deg)
    if speed_mag >= VEL_THRESHOLD and abs(angle_diff) > (math.pi / 2):
        return "Reverse"

    # Turning (use yaw rate)
    if yaw_rate > YAW_THRESHOLD:
        return "Turning Left"
    if yaw_rate < -YAW_THRESHOLD:
        return "Turning Right"

    return "Forward"

def read_session_files(session_dir, session_prefix):
    """
    Load EEG, GPS, ODOM, and image index CSVs for this session.
    Returns dict or None if required files missing.
    """
    eeg_csv   = os.path.join(session_dir, f"{session_prefix}{EEG_SUFFIX}")
    gps_csv   = os.path.join(session_dir, f"{session_prefix}{GPS_SUFFIX}")
    odom_csv  = os.path.join(session_dir, f"{session_prefix}{ODOM_SUFFIX}")
    imgidx_csv= os.path.join(session_dir, f"{session_prefix}{IMGIDX_SUFFIX}")
    images_dir= os.path.join(session_dir, IMAGES_DIRNAME)

    missing = [p for p in [eeg_csv, gps_csv, odom_csv, imgidx_csv] if not os.path.exists(p)]
    if missing:
        print(f"Missing files in {session_dir}:")
        for m in missing: print("   -", m)
        return None

    # EEG
    eeg_rows = load_csv_rows(eeg_csv)
    if not eeg_rows:
        print(f"Empty EEG file: {eeg_csv}")
        return None
    
    # First column timestamp, rest channels
    eeg_t = np.array([safe_float(r[0]) for r in eeg_rows], dtype=float)
    eeg_data = [r[1:] for r in eeg_rows]  # keep strings to preserve formatting
    
    # Header for EEG (read from raw to preserve channel names)
    with open(eeg_csv, "r") as f:
        eeg_header_line = f.readline().strip()
    
    # Header split
    eeg_header = [h.strip() for h in (eeg_header_line.split(",") if "," in eeg_header_line else eeg_header_line.split("\t"))]
    if len(eeg_header) <= 1:
        eeg_header = ["timestamp","Fp1","Fp2","C3","C4","T3","T4","O1","O2","F7","F8","F3","F4","T5","T6","P3","P4"]

    # GPS
    gps_rows = load_csv_rows(gps_csv)
    gps_t = np.array([safe_float(r[0]) for r in gps_rows], dtype=float)
    gps_vx = np.array([safe_float(r[1]) for r in gps_rows], dtype=float)
    gps_vy = np.array([safe_float(r[2]) for r in gps_rows], dtype=float)

    # ODOM
    odom_rows = load_csv_rows(odom_csv)
    odom_t  = np.array([safe_float(r[0]) for r in odom_rows], dtype=float)
    odom_yaw = np.array([safe_float(r[1]) for r in odom_rows], dtype=float)
    odom_yaw_rate = np.array([safe_float(r[2]) for r in odom_rows], dtype=float)

    # Image index
    imgidx_rows = load_csv_rows(imgidx_csv)
    img_ts   = []
    img_paths= []
    for r in imgidx_rows:
        ts = safe_float(r[0])
        pth = r[1].strip() if len(r) > 1 else ""
        pth = pth.replace("/media/ghada/Expansion2/Twizy Recordings/1_Extracted_Data/",
                  "/media/nova/hdd8tb/Twizy/1_Extracted_Data/")

        if not os.path.isabs(pth):
            pth = os.path.join(images_dir, pth)
        img_ts.append(ts)
        img_paths.append(pth)
    img_ts   = np.array(img_ts, dtype=float)
    img_paths= np.array(img_paths, dtype=object)

    # Drop NaNs, if any
    def drop_nans_pair(t, *cols):
        mask = ~np.isnan(t)
        outs = [t[mask]]
        for c in cols:
            c = np.asarray(c)
            outs.append(c[mask])
        return outs

    eeg_t, = drop_nans_pair(eeg_t)
    gps_t, gps_vx, gps_vy = drop_nans_pair(gps_t, gps_vx, gps_vy)
    odom_t, odom_yaw, odom_yaw_rate = drop_nans_pair(odom_t, odom_yaw, odom_yaw_rate)
    img_ts, img_paths = drop_nans_pair(img_ts, img_paths)

    if len(eeg_t) == 0 or len(gps_t) == 0 or len(odom_t) == 0 or len(img_ts) == 0:
        print(f"⚠️ Insufficient data in {session_dir}")
        return None

    # Sort by time
    eeg_idx  = np.argsort(eeg_t);  eeg_t = eeg_t[eeg_idx];  eeg_data = [eeg_data[i] for i in eeg_idx]
    gps_idx  = np.argsort(gps_t);  gps_t = gps_t[gps_idx];  gps_vx = gps_vx[gps_idx]; gps_vy = gps_vy[gps_idx]
    odom_idx = np.argsort(odom_t); odom_t = odom_t[odom_idx]; odom_yaw = odom_yaw[odom_idx]; odom_yaw_rate = odom_yaw_rate[odom_idx]
    img_idx = np.argsort(img_ts)
    img_ts = img_ts[img_idx]
    img_paths = img_paths[img_idx]
    
    # Return all data
    return {
        "eeg_t": eeg_t,
        "eeg_data": eeg_data,
        "eeg_header": eeg_header,
        "gps_t": gps_t,
        "gps_vx": gps_vx,
        "gps_vy": gps_vy,
        "odom_t": odom_t,
        "odom_yaw": odom_yaw,
        "odom_yaw_rate": odom_yaw_rate,
        "img_ts": img_ts,
        "img_paths": img_paths,
        "images_dir": images_dir
    }

def nearest_indices(sorted_ref_ts, query_ts):
    """
    For each query_ts, return index of closest value in sorted_ref_ts.
    Both must be 1D numpy arrays sorted ascending.
    """
    # positions where elements should be inserted to maintain order
    pos = np.searchsorted(sorted_ref_ts, query_ts)
    pos0 = np.clip(pos - 1, 0, len(sorted_ref_ts) - 1)
    pos1 = np.clip(pos,     0, len(sorted_ref_ts) - 1)
    d0 = np.abs(query_ts - sorted_ref_ts[pos0])
    d1 = np.abs(sorted_ref_ts[pos1] - query_ts)
    choose_pos1 = d1 < d0
    out = np.where(choose_pos1, pos1, pos0)
    return out

def label_and_save_images(session_data, eeg_labels, out_images_dir, img_labels_csv_path):
    """Label each image once using the nearest EEG label (ground truth derived from GPS+Odom)."""
    ensure_dir(out_images_dir)
    
    img_ts   = session_data["img_ts"]
    img_paths= session_data["img_paths"]
    eeg_t    = session_data["eeg_t"]
    
    # Compute nearest EEG index for each image timestamp
    nn_index = nearest_indices(eeg_t, img_ts)           

    img_label_rows = [ ["timestamp", "label", "image_path"] ]
    
    for img_i, (ts, path) in enumerate(zip(img_ts, img_paths)):
        # Nearest EEG label
        eeg_label = eeg_labels[nn_index[img_i]]
        
        # Load image
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ imread failed for: {path}")
            img_label_rows.append([f"{ts:.9f}", eeg_label, path])
            continue

        # Draw label text
        text_pos = (50, 50)
        color = LABEL_COLORS.get(eeg_label, (255, 255, 255))
        cv2.rectangle(
            img,
            (text_pos[0] - 6, text_pos[1] - 36),
            (text_pos[0] + 260, text_pos[1] + 8),
            (0, 0, 0),
            -1
        )
        cv2.putText(img, eeg_label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        out_path = os.path.join(out_images_dir, f"{ts:.9f}.png")
        cv2.imwrite(out_path, img)
        img_label_rows.append([f"{ts:.9f}", eeg_label, out_path])

    # Save image-label audit CSV
    with open(img_labels_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(img_label_rows)

def write_eeg_labeled_csv(session_data, out_csv_path):
    """
    Label each EEG sample using interpolated GPS+Odom motion features.

    - Label at t      → Label_action  (ground-truth movement now)
    - Labels at t+Δt  → Label_plus_XXXms (future intention labels)

    Returns:
        eeg_labels: list of Label_action (used later to label images).
    """
    eeg_t = session_data["eeg_t"]
    eeg_data = session_data["eeg_data"]
    eeg_header = session_data["eeg_header"]

    # Build interpolators
    f_vx  = build_interp(session_data["gps_t"], session_data["gps_vx"])
    f_vy  = build_interp(session_data["gps_t"], session_data["gps_vy"])
    f_yaw = build_interp(session_data["odom_t"], session_data["odom_yaw"])
    f_yaw_rate = build_interp(session_data["odom_t"], session_data["odom_yaw_rate"])

    # Header: Timestamp + action label + future labels + EEG channels
    full_header = (
        ["Timestamp", "Label_action"]
        + [f"Label_plus_{ms}ms" for ms in DELTAS_MS]
        + [h for h in eeg_header[1:]]   # skip original timestamp col
    )

    rows = []
    eeg_labels = [] # only Label_action; this is what images will inherit
    
    for t, eeg_row in zip(eeg_t, eeg_data):
        # Label at time t
        vx_now, vy_now, yaw_now, yaw_rate_now = f_vx(t), f_vy(t), f_yaw(t), f_yaw_rate(t)
        label_now = movement_label(vx_now, vy_now, yaw_now, yaw_rate_now)
        
        # Labels at future times t + Δt
        future_labels = []
        for ms in DELTAS_MS:
            t_future = t + ms / 1000.0
            vx_f  = f_vx(t_future)
            vy_f  = f_vy(t_future)
            yaw_f = f_yaw(t_future)
            yr_f  = f_yaw_rate(t_future)
            label_f = movement_label(vx_f, vy_f, yaw_f, yr_f)
            future_labels.append(label_f)
        
        # Save baseline and future labels
        eeg_labels.append(label_now)
        rows.append([f"{t:.9f}", label_now] + future_labels + eeg_row)

    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(full_header)
        w.writerows(rows)

    return eeg_labels

def process_session(subject_dir, session_name):
    session_dir = os.path.join(subject_dir, session_name)
    prefix = session_name
    print(f"▶ Processing {os.path.basename(subject_dir)} / {session_name}")

    data = read_session_files(session_dir, prefix)
    if data is None:
        print("   Skipped (missing/invalid data).")
        return

    # Prepare outputs
    out_subject_dir = ensure_dir(os.path.join(output_root, os.path.basename(subject_dir)))
    out_session_dir = ensure_dir(os.path.join(out_subject_dir, session_name))
    out_images_dir  = ensure_dir(os.path.join(out_session_dir, LABELED_IMAGE_DIR))

    out_eeg_csv     = os.path.join(out_session_dir, OUTPUT_CSV_NAME.format(prefix=prefix))
    out_img_labels  = os.path.join(out_session_dir, OUTPUT_IMG_LABELS.format(prefix=prefix))

    # 1) Label EEG (GPS + Odom ground truth)
    eeg_labels = write_eeg_labeled_csv(data, out_eeg_csv)
    
    # 2) Label images using nearest EEG label (V5)
    label_and_save_images(data, eeg_labels, out_images_dir, out_img_labels)

    print(f"   ✓ Labeled images: {out_images_dir}")
    print(f"   ✓ EEG labeled CSV: {out_eeg_csv}")
    print(f"   ✓ Image labels CSV: {out_img_labels}")

def main():
    # Walk subjects (e.g., ST01, ST02, ...)
    for subject in sorted(os.listdir(input_root)):
        subject_dir = os.path.join(input_root, subject)
        if not (os.path.isdir(subject_dir) and subject.startswith("ST")):
            continue

        # Sessions are subfolders like ST04S02 with CSVs inside
        for entry in sorted(os.listdir(subject_dir)):
            sess_path = os.path.join(subject_dir, entry)
            if os.path.isdir(sess_path) and entry.startswith(subject):  # e.g., ST01S01 under ST01/
                # ensure at least EEG csv exists to count as a valid session
                if os.path.exists(os.path.join(sess_path, f"{entry}{EEG_SUFFIX}")):
                    print(f"Processing session: {entry}")
                    process_session(subject_dir, entry)

    print("All sessions labeled and saved.")

if __name__ == "__main__":
    main()
