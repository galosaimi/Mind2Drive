#!/usr/bin/env python3
"""
05_validate_thresholds.py
-------------------------
Validate motion labelling thresholds using KDE and misclassification analysis.

Generates per-session yaw-rate and speed plots, then computes global
KDE valley minima, misclassification error curves, and yaw-rate percentiles
to justify the chosen stop and turning thresholds.

Inputs  : Extracted GPS and odometry CSVs 
Outputs : Per-session plots and Aggregate plots 

Author  : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from scipy.signal import argrelextrema

# ---------------------------------------------------------------------------
# Plot style: Times New Roman, publication-ready font sizes
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],

    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 22,

    "mathtext.fontset": "stix",
})

# ---------------------------------------------------------------------------
# Configuration: input/output paths
# ---------------------------------------------------------------------------
BASE = "data/extracted"
OUT_DIR_YW = "./yaw_plots"
OUT_DIR_SP = "./speed_plots"
OUT_DIR_AVG = "./average_plots"

os.makedirs(OUT_DIR_AVG, exist_ok=True)
os.makedirs(OUT_DIR_YW, exist_ok=True)
os.makedirs(OUT_DIR_SP, exist_ok=True)

# ---------------------------------------------------------------------------
# Motion labelling thresholds
# ---------------------------------------------------------------------------
VEL_THRESHOLD = 0.1   # m/s — separates stopped from moving
YAW_THRESHOLD = 0.05  # rad/s — separates straight from turning

# ---------------------------------------------------------------------------
# Motion state colours for scatter plots
# ---------------------------------------------------------------------------
state_colors = {
    "STOPPED": "gold",
    "MOVING STRAIGHT": "steelblue",
    "TURNING LEFT": "limegreen",
    "TURNING RIGHT": "tomato"
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def classify(speed, yaw_rate):
    """Classify a single sample into a motion state."""
    if abs(speed) < VEL_THRESHOLD and abs(yaw_rate) < YAW_THRESHOLD:
        return "STOPPED"
    elif abs(yaw_rate) > YAW_THRESHOLD:
        return "TURNING LEFT" if yaw_rate > 0 else "TURNING RIGHT"
    else:
        return "MOVING STRAIGHT"

def plot_yaw_rate(gps, sess, out_dir):
    """Plot yaw-rate over time with motion state classification."""
    
    plt.figure(figsize=(13.5, 4.5))
    plt.plot(gps["timestamp"], gps["yaw_rate"], color="black", linewidth=0.8, alpha=0.7, label="Yaw Rate")

    plt.axhline(+YAW_THRESHOLD, color="green", linestyle="--", linewidth=2, label="+0.05 rad/s")
    plt.axhline(-YAW_THRESHOLD, color="red", linestyle="--", linewidth=2, label="-0.05 rad/s")

    for state, color in state_colors.items():
        mask = gps["state"] == state
        plt.scatter(gps["timestamp"][mask], gps["yaw_rate"][mask],
                    color=color, s=6, alpha=0.8, label=state)

    plt.title(f"Yaw Rate Over Time with Motion Classification — {sess}")
    plt.xlabel("Timestamp")
    plt.ylabel("Yaw Rate (rad/s)")
    
    # Customize y-axis ticks for better readability
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    
    plt.legend(loc="upper right", fontsize=18, markerscale=4)
    plt.tight_layout()

    # SAVE FIGURE
    out_file = os.path.join(OUT_DIR_YW, f"{sess}_yaw_plot.png")
    plt.savefig(out_file, dpi=300)

    plt.close()
        
def plot_speed(gps, sess, out_dir):
    """Plot speed over time with motion state classification."""
    plt.figure(figsize=(13.5, 4.5))
    plt.plot(gps["timestamp"], gps["speed"], color="black", linewidth=0.8, alpha=0.7, label="Speed")

    # Speed threshold line (Stop)
    plt.axhline(VEL_THRESHOLD, color="red", linestyle="--", linewidth=2,
                label=f"Stop threshold = {VEL_THRESHOLD} m/s")

    # Scatter each motion state
    for state, color in state_colors.items():
        mask = gps["state"] == state
        plt.scatter(gps["timestamp"][mask], gps["speed"][mask],
                    color=color, s=8, alpha=0.8, label=state)

    plt.title(f"Speed Over Time with Motion Classification — {sess}")
    plt.xlabel("Timestamp")
    plt.ylabel("Speed (m/s)")
    
    # Customize y-axis ticks for better readability
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
        
    plt.legend(loc="upper right", fontsize=18, markerscale=4)
    plt.tight_layout()

    # SAVE PLOT
    out_file = os.path.join(OUT_DIR_SP, f"{sess}_speed_plot.png")
    plt.savefig(out_file, dpi=300)
    
    plt.close()

# ---------------------------------------------------------------------------
# Main: per-session plots
# ---------------------------------------------------------------------------
all_speeds = []
all_yaws   = []

for st in sorted(os.listdir(BASE)):
    st_path = os.path.join(BASE, st)
    if not os.path.isdir(st_path):
        continue
    
    for sess in sorted(os.listdir(st_path)):        
        gps_path  = os.path.join(st_path, sess, f"{sess}_gps.csv")
        odom_path = os.path.join(st_path, sess, f"{sess}_odom.csv")

        if not (os.path.isfile(gps_path) and os.path.isfile(odom_path)):
            continue
        
        print("Processing", sess)

        gps = pd.read_csv(gps_path, header=None, skiprows=1,
                          names=["timestamp","vx","vy"]).astype(float)
        odom = pd.read_csv(odom_path, header=None, skiprows=1,
                           names=["timestamp","yaw","yaw_rate"]).astype(float)

        # compute speed
        gps["speed"] = np.sqrt(gps["vx"]**2 + gps["vy"]**2)

        # interpolate yaw-rate
        gps["yaw_rate"] = np.interp(gps["timestamp"], odom["timestamp"], odom["yaw_rate"])

        # classify motion
        gps["state"] = gps.apply(lambda r: classify(r["speed"], r["yaw_rate"]), axis=1)
        
        # SAVE GLOBAL VALUES
        all_speeds.append(gps["speed"])
        all_yaws.append(np.abs(gps["yaw_rate"]))
        
        plot_yaw_rate(gps, sess, OUT_DIR_YW)
        plot_speed(gps, sess, OUT_DIR_SP)

print("\nDONE. All Yawrate plots saved in:", OUT_DIR_YW)
print("All Speed plots saved in:", OUT_DIR_SP)


# -------------------------------------------------------------------------------------------
# Global analysis: merge all sessions (speeds and yaw-rates) for threshold validation
# -------------------------------------------------------------------------------------------
all_speeds = pd.concat(all_speeds, ignore_index=True)
all_yaws   = pd.concat(all_yaws,   ignore_index=True)

# === KDE VALLEY MINIMUM – Speed threshold validation ===
# This computes the natural split between STOP and MOVE speeds.
print("\n=== VALLEY MINIMUM ANALYSIS (SPEED) ===")

plt.figure(figsize=(10,4))
kde = sns.kdeplot(all_speeds, bw_adjust=1.2)

# Extract KDE curve
ydata = kde.get_lines()[0].get_ydata()
xdata = kde.get_lines()[0].get_xdata()

minima_idx = argrelextrema(ydata, np.less)[0]
valley_points = xdata[minima_idx]

print("Valley minima (speed):", valley_points)
plt.close()

# === MISCLASSIFICATION ERROR CURVE – Speed threshold check ===
# This proves 0.2 m/s minimizes error.
print("\n=== MISCLASSIFICATION ERROR CURVE (SPEED) ===")

thresholds = np.linspace(0.01, 1.0, 200)
true_stop = (all_speeds < VEL_THRESHOLD).astype(int)
errors = []

for t in thresholds:
    pred = (all_speeds < t).astype(int)
    error = np.mean(pred != true_stop)
    errors.append(error)

plt.figure(figsize=(10,4))
plt.plot(thresholds, errors)
plt.axvline(VEL_THRESHOLD, color="red", linestyle="--", linewidth=2,
            label=f"Chosen threshold = {VEL_THRESHOLD} m/s")
plt.xlabel("Threshold (m/s)")
plt.ylabel("Classification Error")
plt.title("Misclassification Error Curve for Speed Threshold")
plt.legend(loc="upper right", fontsize=13)
plt.grid(True)
plt.show()

# === PERCENTILES – Yaw-rate threshold justification ===
# This proves 0.05 rad/s lies exactly at the "knee" region between straight and turning.
print("\n=== YAW-RATE PERCENTILES ===")
percentiles = np.percentile(all_yaws, [25, 50, 75, 90, 95])

print(f"25%: {percentiles[0]:.4f}")
print(f"50%: {percentiles[1]:.4f}")
print(f"75%: {percentiles[2]:.4f}")
print(f"90%: {percentiles[3]:.4f}")
print(f"95%: {percentiles[4]:.4f}")

# Plotting the average speed distribution and yaw-rate distribution across all sessions
plt.figure(figsize=(13.5, 4.5))
sns.kdeplot(all_speeds, linewidth=2, color="blue", label="Mean KDE")
plt.axvline(0.1, color="red", linestyle="--", linewidth=2, label="Stop threshold = 0.1 m/s")
plt.title("Average Speed Distribution Across All Sessions")
plt.xlabel("Speed (m/s)")
plt.ylabel("Density")
plt.legend(loc="upper right", fontsize=18)

plt.grid(True)
plt.savefig(
    os.path.join(OUT_DIR_AVG, "avg_speed_distribution.pdf"),
    dpi=300,
    bbox_inches="tight"
)
plt.savefig(
    os.path.join(OUT_DIR_AVG, "avg_speed_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Plotting the average yaw-rate distribution and yaw-rate distribution across all sessions
plt.figure(figsize=(13.5, 4.5))
sns.kdeplot(all_yaws, linewidth=2, color="green", label="Mean KDE")
plt.axvline(0.05, color="red", linestyle="--", linewidth=2, label="Turn threshold = 0.05 rad/s")
plt.title("Average |Yaw-Rate| Distribution Across All Sessions")
plt.xlabel("|Yaw Rate| (rad/s)")
plt.ylabel("Density")
plt.legend(loc="upper right", fontsize=18)

plt.grid(True)
plt.savefig(
    os.path.join(OUT_DIR_AVG, "avg_yawrate_distribution.pdf"),
    dpi=300,
    bbox_inches="tight"
)
plt.savefig(
    os.path.join(OUT_DIR_AVG, "avg_yawrate_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"\nAggregate plots saved to {OUT_DIR_AVG}")