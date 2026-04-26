#!/usr/bin/env python3
"""
03_split_by_label.py
--------------------
Split labeled EEG CSVs by prediction horizon label into separate files.

For each session, produces one CSV per label column (Label_action,
Label_plus_100ms, ..., Label_plus_1000ms), and writes a combined
label distribution summary across all sessions.

Inputs  : Labeled CSVs 
Outputs : Split CSVs 

Author  : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""

import os
import csv

# ---------------------------------------------------------------------------
# Configuration: input/output paths
# ---------------------------------------------------------------------------
input_root = "data/labeled"
output_root = "data/splits"

# ---------------------------------------------------------------------------
# Prediction horizons and label columns
# ---------------------------------------------------------------------------
DELTAS_MS = [100,200,300,400,500,600,700,800,900,1000]
LABEL_COLS = ["Label_action"] + [f"Label_plus_{ms}ms" for ms in DELTAS_MS]

# ---------------------------------------------------------------------------
# Summary file
# ---------------------------------------------------------------------------
summary_txt = os.path.join(output_root, "Label_Distribution_Summary.txt")
summary_lines = []

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def split_by_label(input_csv, subject, session):
    """Split one labeled EEG CSV into per-label-column output files."""
    with open(input_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]

    # EEG channels start after label columns
    eeg_start = header.index("Fp1")  # first EEG channel
    eeg_channels = header[eeg_start:]

    for label_name in LABEL_COLS:
        label_idx = header.index(label_name)

        # output folder: /output_root/Label_xxx/STxx/
        subject_dir = os.path.join(output_root, label_name, subject)
        os.makedirs(subject_dir, exist_ok=True)

        # output file: STxxSyy_eeg_Label_xxx.csv
        out_csv = os.path.join(subject_dir, f"{session}_eeg_{label_name}.csv")

        with open(out_csv, "w", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["Timestamp", label_name] + eeg_channels)

            for row in rows:
                timestamp = row[0]
                label_val = row[label_idx]
                eeg_vals = row[eeg_start:]
                writer.writerow([timestamp, label_val] + eeg_vals)

        print(f"Wrote: {out_csv}")

def compute_label_distribution(input_csv):
    """Count occurrences of each label value per label column."""
    with open(input_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Find label column indexes
        label_idx_map = {label: header.index(label) for label in LABEL_COLS}

        # Counters
        dist = {label: {} for label in LABEL_COLS}

        for row in reader:
            for label, idx in label_idx_map.items():
                val = row[idx]
                dist[label][val] = dist[label].get(val, 0) + 1

    return dist

# ---------------------------------------------------------------------------
# Main: traverse all subjects and sessions
# ---------------------------------------------------------------------------
for subject in sorted(os.listdir(input_root)):
    subj_path = os.path.join(input_root, subject)
    
    print(f"Processing subject: {subject}")
    # print(f"Subject path: {subj_path}")
    
    if not os.path.isdir(subj_path) or not subject.startswith("ST"):
        continue

    for session in sorted(os.listdir(subj_path)):
        session_path = os.path.join(subj_path, session)
        
        if not os.path.isdir(session_path) or not session.startswith(subject):
            continue
        
        print(f" Processing session: {session}")
        # print(f"Session path: {session_path}")
        
        # Expected csv name
        csv_name = f"{session}_eeg_labeled.csv"
        csv_path = os.path.join(session_path, csv_name)
                
        if os.path.isfile(csv_path):
            print(f"     Processing CSV: {csv_name}")
            split_by_label(csv_path, subject, session)
            
            # Compute and store label distribution for this session
            dist = compute_label_distribution(csv_path)

            session_id = f"{subject}/{session}"

            for label in LABEL_COLS:
                # Format line for the summary txt
                counts_str = ", ".join([f"{val}:{cnt}" for val, cnt in dist[label].items()])
                summary_lines.append(f"{label}  --  {session_id}  --  {counts_str}")
        else:
            print(f"     WARNING: No EEG CSV found in {session_path}")

# Print a summary of the label distribution for all sessions 
print("Done splitting all sessions into label folders.")

# Write combined label distribution summary 
with open(summary_txt, "w") as f:
    current_label = None
    for line in summary_lines:
        label, session_path, counts = line.split("  --  ")
        if label != current_label:
            f.write(f"\n========================\n{label}\n========================\n")
            current_label = label
        f.write(f"{session_path}: {counts}\n")

print(f"\nSaved combined label summary to:\n{summary_txt}")