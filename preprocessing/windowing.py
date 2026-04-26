#!/usr/bin/env python3
"""
windowing.py
------------
Apply sliding window segmentation with rejection rule to EEG CSVs.

Each window is labelled using the rejection rule: windows spanning more
than one action class are discarded to ensure label purity. Windows are
extracted with 50% overlap by default.

Inputs  : Train/test CSVs 
Outputs : Windowed CSVs and per-subject windowing log files

Author  : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration: input/output paths
# ---------------------------------------------------------------------------
input_dir = "data/train_test_splits"
output_dir = "data/windowed"

# ---------------------------------------------------------------------------
# Windowing parameters
# ---------------------------------------------------------------------------
SFREQ       = 125               # EEG sampling frequency (Hz)
WINDOW_SIZE = SFREQ * 1         # 1-second window = 125 samples
OVERLAP     = SFREQ // 2        # 50% overlap = 62 samples

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def window_data(df, window_size, overlap_size):
    """
    Segment EEG DataFrame into fixed-length windows with rejection rule.

    Windows containing more than one unique label are rejected to ensure
    manoeuvre purity. Returns windowed data and rejection statistics.

    Args:
        df          : Input DataFrame with Timestamp, Label, EEG columns
        window_size : Number of samples per window
        overlap     : Number of overlapping samples between windows

    Returns:
        df_windowed                 : Windowed EEG DataFrame
        total_windows               : Number of accepted windows
        rejected_windows            : Number of rejected windows
        total_windows_before_reject : Total windows before rejection
    """
    num_samples = len(df)
    step_size = window_size - overlap_size
    
    windowed_data = []
    windowed_labels = []
    windowed_timestamps = []
    rejected_windows = 0
    total_windows_before_rejection = 0

    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        window = df.iloc[start:end, 2:].values      # Extract EEG window
        window_flattened = window.flatten()         # Flatten window into 1D
        
        # Determine Labeling Strategy
        window_labels = df.iloc[start:end, 1]       # Extract corresponding labels
        total_windows_before_rejection += 1         # Track total number of windows

        # Reject windows with multiple labels
        if len(window_labels.unique()) > 1:
                rejected_windows += 1
                continue  # Skip this window if mixed labels
        
        # Assign first label if not rejected
        assigned_label = window_labels.iloc[0]  
        
        # Append Processed Data
        windowed_data.append(window_flattened)
        windowed_labels.append(assigned_label)
        windowed_timestamps.append(df.iloc[start, 0])

    # Create DataFrame with flattened channel-time columns
    num_channels = df.shape[1] - 2  # Exclude Timestamp and Label columns
    flattened_columns = [f"Ch{ch+1}_T{t}" for t in range(window_size) for ch in range(num_channels)]
    
    df_windowed = pd.DataFrame(windowed_data, columns=flattened_columns)
    df_windowed.insert(0, "Label", windowed_labels)
    df_windowed.insert(0, "Timestamp", windowed_timestamps)

    total_windows = len(df_windowed)
    return df_windowed, total_windows, rejected_windows, total_windows_before_rejection

# ---------------------------------------------------------------------------
# Main: traverse all label folders and subjects
# ---------------------------------------------------------------------------

for subject_folder in os.listdir(input_dir):
    subject_folder_path = os.path.join(input_dir, subject_folder)
    if os.path.isdir(subject_folder_path):
        # Create the corresponding output folder for the subject
        output_subject_folder = os.path.join(output_dir, subject_folder)
        os.makedirs(output_subject_folder, exist_ok=True)

        # Create a log file for the subject folder
        log_file = os.path.join(output_subject_folder, f"{subject_folder}_windowing_log.txt")
        summary_lines = []

        # Loop through each CSV file in the subject folder
        for file_name in os.listdir(subject_folder_path):
            if file_name.endswith("_train.csv") or file_name.endswith("_test.csv"):  # Check for train or test files
                input_csv_path = os.path.join(subject_folder_path, file_name)
                prefix = file_name[:6]  # First six characters of the file name
                
                # Perform windowing
                df_overlap, total_overlap, rejected_overlap, total_before_rejection = window_data(pd.read_csv(input_csv_path), WINDOW_SIZE, OVERLAP)
                
                # Check if it's a train or test file and create the appropriate output file name
                if "_train" in file_name:
                    output_csv_overlap = os.path.join(output_subject_folder, f"{prefix}_train.csv")
                else:
                    output_csv_overlap = os.path.join(output_subject_folder, f"{prefix}_test.csv")
                    
                df_overlap.to_csv(output_csv_overlap, index=False)
                
                # Get label distribution
                label_counts = df_overlap['Label'].value_counts().to_dict()

                # Format text summary
                summary_lines.append(f"Overlapping File: {output_csv_overlap}")
                summary_lines.append(f"  -> Total Windows before rejection: {total_before_rejection}")
                summary_lines.append(f"  -> Rejected Windows: {rejected_overlap}")
                summary_lines.append(f"  -> Total Windows after rejection: {total_overlap}")
                summary_lines.append(f"  -> Label Distribution: {label_counts}")
                summary_lines.append("-" * 60)  # Separator line
                                
                # Save summary to text file
                with open(log_file, "w") as file:
                    file.write("\n".join(summary_lines))

                print(f"Windowing summary for {subject_folder} saved to: {log_file}")

print("\nWindowing complete.")