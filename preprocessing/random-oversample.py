#!/usr/bin/env python3
"""
random-oversample.py
--------------------
Apply random oversampling to balance EEG training windows across classes.

Uses RandomOverSampler to replicate minority class windows until all
classes are balanced. Logs label distributions before and after oversampling.

Inputs  : Windowed train CSVs 
Outputs : Oversampled CSVs, per-file oversampling logs, and an overall summary log

Author  : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""
import os
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

# ---------------------------------------------------------------------------
# Configuration: input/output paths and split types
# ---------------------------------------------------------------------------
INPUT_ROOT = "data/processed-train_split"
OUTPUT_ROOT = "data/oversampled/random/train_split"
splits = ["Custom", "Normal"]

# Ensure output directory exists
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ---------------------------------------------------------------------------
# Main: traverse splits, subjects, and CSV files
# ---------------------------------------------------------------------------
log_file_path = os.path.join(OUTPUT_ROOT, "oversampling_summary.txt")

# Open log file to write processing details
with open(log_file_path, "w") as log_file:
    log_file.write("Oversampling Processing Summary:\n")
    log_file.write("=" * 60 + "\n\n")
    
    # Loop through each split (Custom, Normal)
    for split in splits:
        # Loop through each subject folder in the input directory
        for subject_folder in os.listdir(os.path.join(INPUT_ROOT, split)):
            subject_folder_path = os.path.join(INPUT_ROOT, split, subject_folder)
            if os.path.isdir(subject_folder_path):
                # Create the corresponding output folder for the subject
                output_subject_folder = os.path.join(OUTPUT_ROOT, split, subject_folder)
                os.makedirs(output_subject_folder, exist_ok=True)

                # Create a log file for the subject folder
                log_file_name = os.path.join(output_subject_folder, f"{subject_folder}_oversampling_log.txt")

                # List all CSV files in the subject folder
                csv_files = [f for f in os.listdir(subject_folder_path) if f.endswith(".csv")]

                # Process each CSV file
                for file_name in csv_files:
                    input_csv_path = os.path.join(subject_folder_path, file_name)
                    prefix = file_name.split("_")[0]  # Extract prefix to maintain naming consistency
                    
                    # Load CSV file
                    df = pd.read_csv(input_csv_path)

                    # Separate metadata and EEG channels
                    timestamps = df.iloc[:, 0].values   # First column: Timestamp (not used in training)
                    labels = df.iloc[:, 1].values       # Second column: Labels
                    eeg_data = df.iloc[:, 2:].values    # EEG values (125 timepoints x 16 channels)

                    # Count initial number of windows per label
                    label_counts_before = pd.Series(labels).value_counts()

                    # Reshape EEG data into a feature matrix
                    eeg_data_flat = eeg_data.reshape(eeg_data.shape[0], -1)  # Flatten the EEG data per window

                    # Apply random oversampling
                    ros = RandomOverSampler(random_state=42)
                    eeg_data_resampled, labels_resampled = ros.fit_resample(eeg_data_flat, labels)

                    # Count number of windows per label after oversampling
                    label_counts_after = pd.Series(labels_resampled).value_counts()

                    # Calculate added instances per label
                    added_counts = label_counts_after - label_counts_before

                    # Calculate total number of added instances
                    total_added = added_counts.sum()
                    
                    # Fix timestamp replication to match length
                    repeats = len(labels_resampled) // len(timestamps)
                    remainder = len(labels_resampled) % len(timestamps)
                    new_timestamps = list(timestamps) * repeats + list(timestamps[:remainder])

                    # Generate new filename by adding "over" instead of "_windowed"
                    new_file_name = file_name.replace("_windowed.csv", "_over.csv")

                    # Construct output path
                    output_path = os.path.join(output_subject_folder, new_file_name)

                    # Reconstruct the DataFrame
                    oversampled_df = pd.DataFrame(eeg_data_resampled, columns=df.columns[2:])
                    oversampled_df.insert(0, 'Label', labels_resampled)
                    oversampled_df.insert(0, 'Timestamp', new_timestamps)

                    # Save the oversampled dataset
                    oversampled_df.to_csv(output_path, index=False)
            
                    # Prepare processing summary for this file
                    file_summary = (
                        f"File: {file_name}\n"
                        f"Original label distribution:\n{label_counts_before}\n\n"
                        f"Added instances per label:\n{added_counts}\n\n"
                        f"Final label distribution:\n{label_counts_after}\n\n"
                        f"Total added windows: {total_added}\n"
                        f"Saved as: {new_file_name}\n"
                        + "-" * 60 + "\n"
                    )
                    # Write to log file
                    log_file.write(file_summary)

                    # Print status
                    print(file_summary)

print(f"\nProcessing complete for all CSV files. Summary saved at: {log_file_path}")