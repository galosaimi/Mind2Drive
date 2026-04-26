#!/usr/bin/env python3
"""
splitting.py
------------
Split normalised EEG CSVs into train/test sets using two strategies:

  1. Standard Split    : Simple 70/30 temporal split per file.
  2. Label-Stratified  : Each label group is split into N temporal chunks,
                         each divided 70/30, then recombined chronologically
                         to preserve class balance across train and test sets.

Inputs  : Normalised CSVs 
Outputs : Train/test CSVs and per-subject split log files

Author  : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration: input/output paths
# ---------------------------------------------------------------------------
input_dir = "data/normalized"
output_dir = "data/train_test_splits"

# ---------------------------------------------------------------------------
# Split parameters
# ---------------------------------------------------------------------------
N_CHUNKS   = 5    # Number of temporal chunks per label group 
                  # in stratified splitting only (for better distribution)
TRAIN_FRAC = 0.7  # Train fraction for both strategies

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def standard_split(df):
    """
    Simple 70/30 temporal split.

    Sorts all samples by timestamp and takes the first 70% as train
    and the remaining 30% as test, preserving temporal order.
    """
    df         = df.sort_values("Timestamp").reset_index(drop=True)
    train_size = int(len(df) * TRAIN_FRAC)
    df_train   = df.iloc[:train_size]
    df_test    = df.iloc[train_size:]
    train_dist = df_train["Label"].value_counts().to_dict()
    test_dist  = df_test["Label"].value_counts().to_dict()
    
    return df_train, df_test, train_dist, test_dist

def stratified_split(df):
    """
    Label-stratified temporal 70/30 split.

    Each label group is sorted chronologically and divided into n_chunks
    temporal segments. Each segment is split 70/30 into train and test.
    All segments are recombined and sorted by timestamp.
    """
    # Step 1: Group by label and sort by timestamp
    label_group = {
        label: group.sort_values(by="Timestamp").reset_index(drop=True)
        for label, group in df.groupby("Label")
    }

    # Step 2: Split each label group into num_chunks
    label_chunks = {
        label: np.array_split(group, N_CHUNKS)
        for label, group in label_group.items()
    }
    
    # Step 3: Split each chunk into train/test sets
    label_train_chunks = {label: [] for label in label_chunks}
    label_test_chunks = {label: [] for label in label_chunks}
    train_label_dist = {}
    test_label_dist = {}

    for label, chunks in label_chunks.items():
        total_label_count = 0
        total_test_label_count = 0
        for chunk in chunks:
            n = len(chunk)
            train_size = int(n * 0.7)
            label_train_chunks[label].append(chunk.iloc[:train_size])
            label_test_chunks[label].append(chunk.iloc[train_size:])
            total_label_count += n
            total_test_label_count += n - train_size

        train_label_dist[label] = total_label_count - total_test_label_count
        test_label_dist[label] = total_test_label_count
        
    # Step 4: Combine all chunks across labels
    train_chunks = [chunk for chunks in label_train_chunks.values() for chunk in chunks]
    test_chunks  = [chunk for chunks in label_test_chunks.values() for chunk in chunks]

    combined_train_df = pd.concat(train_chunks).sort_values(by="Timestamp").reset_index(drop=True)
    combined_test_df = pd.concat(test_chunks).sort_values(by="Timestamp").reset_index(drop=True)
    
    return combined_train_df, combined_test_df, train_label_dist, test_label_dist

def save_and_log(df_train, df_test, train_dist, test_dist,
                 output_train_csv, output_test_csv, log_file, input_csv):
    """Save train/test CSVs and log split statistics."""
    df_train.to_csv(output_train_csv, index=False)
    df_test.to_csv(output_test_csv, index=False)
    
    # Logging
    with open(log_file, 'a') as log:
        log.write(f"File: {os.path.basename(input_csv)}\n")
        log.write(f"Total samples: {len(df_train) + len(df_test)}\n")
        log.write(f"Total training samples: {len(df_train)}\n")
        log.write(f"Total testing samples: {len(df_test)}\n")
        log.write(f"Train label distribution: {train_dist}\n")
        log.write(f"Test label distribution: {test_dist}\n")
        log.write("-" * 60 + "\n")

    print(f"Training set saved to: {output_train_csv}")
    print(f"Testing set saved to: {output_test_csv}")                   

# ---------------------------------------------------------------------------
# Main: traverse all label folders and subjects
# ---------------------------------------------------------------------------
for subject_folder in os.listdir(input_dir):
    subject_folder_path = os.path.join(input_dir, subject_folder)
    if os.path.isdir(subject_folder_path):
        # Create the corresponding output folder
        output_subject_folder = os.path.join(output_dir, subject_folder)
        os.makedirs(output_subject_folder, exist_ok=True)
        
        print(f"Processing: {subject_folder}")

        # Create a log file for the subject folder
        log_file = os.path.join(output_subject_folder, f"{subject_folder}_split_log.txt")

        # Loop through each CSV file in the subject folder
        for file_name in os.listdir(subject_folder_path):
            if file_name.endswith("_eeg_labeled_normalized.csv"):
                input_csv = os.path.join(subject_folder_path, file_name)
                session_name = file_name[:7]  # First six characters of the file name
                
                df = pd.read_csv(input_csv)
                
                # --- Standard split ---
                out_norm  = os.path.join(output_dir, "Normal", subject_folder)
                output_train_csv_path = os.path.join(out_norm, f"{session_name}_train.csv")
                output_test_csv_path = os.path.join(out_norm, f"{session_name}_test.csv")
                
                os.makedirs(out_norm, exist_ok=True)
                df_train, df_test, train_dist, test_dist = standard_split(df)
                
                save_and_log(
                    df_train, df_test, train_dist, test_dist,
                    os.path.join(out_norm, f"{session_name}_train.csv"),
                    os.path.join(out_norm, f"{session_name}_test.csv"),
                    os.path.join(out_norm, f"{subject_folder}_split_log.txt"),
                    input_csv
                )

                # --- Label-stratified split ---
                out_strat = os.path.join(output_dir, "Stratified", subject_folder)
                os.makedirs(out_strat, exist_ok=True)
                df_train, df_test, train_dist, test_dist = stratified_split(df)
                save_and_log(
                    df_train, df_test, train_dist, test_dist,
                    os.path.join(out_strat, f"{session_name}_train.csv"),
                    os.path.join(out_strat, f"{session_name}_test.csv"),
                    os.path.join(out_strat, f"{subject_folder}_split_log.txt"),
                    input_csv
                )

print("\nSplitting Done.")