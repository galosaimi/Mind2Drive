#!/usr/bin/env python3
"""
normalize.py
------------
Apply z-score normalisation to EEG channel data.

Normalises each EEG channel independently (mean=0, std=1) across all
samples in a session, preserving Timestamp and Label columns unchanged.
Logs mean and standard deviation of normalised data per file.

Inputs  : Processed CSVs 
Outputs : Normalised CSVs and per-subject normalization log files

Author  : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore

# ---------------------------------------------------------------------------
# Configuration: input/output paths
# ---------------------------------------------------------------------------
input_dir = "data/processed"
output_dir = "data/normalized"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def normalize_session(input_csv, output_csv, log_file):
    """
    Apply z-score normalisation to EEG channels in one CSV file.

    Timestamp and Label columns are preserved as-is.
    Logs per-file mean and std of the normalised EEG data.
    """
    # Load EEG Data
    df = pd.read_csv(input_csv)

    # Separate metadata and EEG channels
    timestamps = df.iloc[:, 0]      # Timestamp column
    labels = df.iloc[:, 1]          # Label column 
    eeg_data = df.iloc[:, 2:]       # EEG channel data

    # Apply Z-Score normalization column-wise (Mean = 0, Std Dev = 1)
    eeg_data_normalized = eeg_data.apply(zscore, axis=0)

    # Reconstruct DataFrame with original column order
    df_normalized = pd.DataFrame(eeg_data_normalized, columns=eeg_data.columns)
    df_normalized.insert(0, "Label", labels)
    df_normalized.insert(0, "Timestamp", timestamps)

    # Save normalized data to CSV
    df_normalized.to_csv(output_csv, index=False)
    
    # Log mean and std of normalized EEG data
    mean_data = df_normalized.iloc[:, 2:].mean().mean()     # Mean of the EEG data
    std_data = df_normalized.iloc[:, 2:].std().std()        # Standard deviation of the EEG data

    # Log the results into the corresponding log file
    with open(log_file, 'a') as log:
        log.write(f"File: {os.path.basename(input_csv)}\n")
        log.write(f"Mean of normalized EEG data: {mean_data}\n")
        log.write(f"Standard deviation of normalized EEG data: {std_data}\n\n")

    print("Z-score normalization complete. Normalized data saved to:", output_csv)

# ---------------------------------------------------------------------------
# Main: traverse all label folders and subjects
# ---------------------------------------------------------------------------
for subject_folder in os.listdir(input_dir):
    subject_folder_path = os.path.join(input_dir, subject_folder)
    if os.path.isdir(subject_folder_path):
        # Create the corresponding output folder
        output_subject_folder = os.path.join(output_dir, subject_folder)
        os.makedirs(output_subject_folder, exist_ok=True)

        # Create a log file for the subject folder
        log_file = os.path.join(output_subject_folder, f"{subject_folder}_normalization_log.txt")

        # Loop through each CSV file in the subject folder
        for file_name in os.listdir(subject_folder_path):
            if file_name.endswith("_eeg_Label_action_pyprep.csv"):
                input_csv_path = os.path.join(subject_folder_path, file_name)
                output_csv_path = os.path.join(output_subject_folder, f"{file_name.split('.')[0]}_normalized.csv")

                # Process and normalize the CSV file
                normalize_session(input_csv_path, output_csv_path, log_file)