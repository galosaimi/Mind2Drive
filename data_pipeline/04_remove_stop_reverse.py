#!/usr/bin/env python3
"""
04_remove_stop_reverse.py
-------------------------
Remove Stop and Reverse class samples from split EEG CSVs.

Iterates over all label folders and subjects, filters out rows
with Stop or Reverse labels, and logs class counts before and after.

Inputs  : Split CSVs 
Outputs : Filtered CSVs

Author  : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""

import os
import csv
from collections import Counter

# ---------------------------------------------------------------------------
# Configuration: input/output paths
# ---------------------------------------------------------------------------
input_root = "data/splits"
output_root = "data/processed"
log_file = os.path.join(output_root, "label_distribution.txt")

# ---------------------------------------------------------------------------
# Labels to remove
# ---------------------------------------------------------------------------
REMOVE_LABELS = {"Stop", "Reverse"}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def filter_file(input_csv, output_csv, label_name="Label"):
    """Filter out Stop and Reverse labels from one CSV file."""
    with open(input_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]

    # Locate label column by name, fallback to column index 1
    try:
        label_idx = header.index(label_name)
    except ValueError:
        label_idx = 1
   
    # EEG channels start after Timestamp and Label columns
    eeg_start = 2  
    eeg_channels = header[eeg_start:]

    before_counter = Counter()
    after_counter = Counter()

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["Timestamp", label_name] + eeg_channels)

        for row in rows:
            timestamp = row[0]
            label_val = row[label_idx].strip()
            before_counter[label_val] += 1
            
            if label_val in REMOVE_LABELS:
                continue  # skip filtered labels

            eeg_vals = row[eeg_start:]
            writer.writerow([timestamp, label_val] + eeg_vals)
            after_counter[label_val] += 1

    return before_counter, after_counter


# ---------------------------------------------------------------------------
# Main: traverse all label folders, subjects, and CSV files
# ---------------------------------------------------------------------------
log_lines = []

for label_folder in sorted(os.listdir(input_root)):
    label_path = os.path.join(input_root, label_folder)
    if not os.path.isdir(label_path):
        continue

    # Traverse subjects
    for subject in sorted(os.listdir(label_path)):
        subj_path = os.path.join(label_path, subject)
        if not os.path.isdir(subj_path):
            continue

        # Traverse all CSVs
        for file in sorted(os.listdir(subj_path)):
            if not file.endswith(".csv"):
                continue

            input_csv = os.path.join(subj_path, file)
            out_dir = os.path.join(output_root, label_folder, subject)
            output_csv = os.path.join(out_dir, file)

            before_counter, after_counter = filter_file(input_csv, output_csv, label_name="Label")

            # Logging
            log_lines.append(f"[{label_folder}] {subject}/{file}")
            log_lines.append("  --- Before filtering ---")
            for lab, count in before_counter.items():
                log_lines.append(f"    {lab}: {count}")
            log_lines.append(f"    Total_before: {sum(before_counter.values())}")

            log_lines.append("  --- After filtering (Stop/Reverse removed) ---")
            for lab, count in after_counter.items():
                log_lines.append(f"    {lab}: {count}")
            log_lines.append(f"    Total_after: {sum(after_counter.values())}")
            log_lines.append("")  # spacing line

            print(f"Processed: {label_folder}/{subject}/{file}")

# Write log file
os.makedirs(output_root, exist_ok=True)
with open(log_file, "w") as f_log:
    f_log.write("\n".join(log_lines))

print(f"\nDone! Detailed label counts written to:\n{log_file}")