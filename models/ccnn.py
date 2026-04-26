#!/usr/bin/env python3
"""
ccnn.py
-------
CCNN model wrapper and data loader for EEG-based driver intention classification.

Reference: Yang et al. (2018) Continuous convolutional neural network with 3D input for 
           EEG-based emotion recognition[C]//International Conference on Neural 
           Information Processing.
           
Author  : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torcheeg.models import CCNN

# ---------------------------------------------------------------------------
# Anatomical EEG-to-grid mapping (10-20 system, 9x9 grid)
# ---------------------------------------------------------------------------
def map_to_grid(data, grid_size=(9, 9)):
    grid = np.zeros(grid_size)
    
    # Corrected Anatomical Mapping
    mapping = {
        # Front (Row 0-2)
        0: (0, 3), 1: (0, 5),   # FP1, FP2 (Nose area)
        10: (2, 3), 11: (2, 5), # F3, F4
        8: (2, 1), 9: (2, 7),   # F7, F8 (Front sides)

        # Middle (Row 4)
        2: (4, 3), 3: (4, 5),   # C3, C4 (Motor Cortex)
        4: (4, 1), 5: (4, 7),   # T3, T4 (Temporal sides)

        # Back (Row 6-8)
        14: (6, 3), 15: (6, 5), # P3, P4 (Parietal)
        12: (6, 1), 13: (6, 7), # T5, T6 (Rear sides)
        6: (8, 3), 7: (8, 5)    # O1, O2 (Occipital/Back)
    }
    
    for idx, (row, col) in mapping.items():
        grid[row, col] = data[idx]
    return grid

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
def load_data(csv_path, label_mapping):
    """Load windowed EEG CSV, map to 9x9 grid, return tensors."""
    df = pd.read_csv(csv_path)
    
    if "Label" not in df.columns:
        raise ValueError(f"Could not find 'Label' column in {csv_path}.")
    
    labels = df["Label"].values
    
    # 1. Ensure only get the numeric EEG signal columns
    # Drop non-numeric columns to ensure having exactly the 2000 features (125*16)
    eeg_data = df.select_dtypes(include=[np.number]).drop(columns=["Timestamp"], errors="ignore").values

    # 2. Check if the data size matches expectations (N samples * 125 timepoints * 16 channels)
    expected_features = 125 * 16
    if eeg_data.shape[1] != expected_features:
        # If CSV columns are actually just the 16 channels (already averaged), 
        # or if they are structured differently, need to handle that.
        print(f"Warning: Expected {expected_features} features, but found {eeg_data.shape[1]}")
        # If your data is ALREADY (N, 16), skip the reshape and mean:
        if eeg_data.shape[1] == 16:
            eeg_avg = eeg_data
        else:
            raise ValueError(f"Unexpected data shape {eeg_data.shape}. Cannot reshape to (125, 16)")
    else:
        eeg_reshaped = eeg_data.reshape(len(df), 125, 16)
        eeg_avg = np.mean(eeg_reshaped, axis=1)  # Shape: (N, 16)

    # 3. Ensure each sample is treated as a numpy array
    X = []
    for sample in eeg_avg:
        # Force sample to be a numpy array to prevent "scalar" errors
        grid = map_to_grid(np.atleast_1d(sample))
        X.append(grid)
    
    X = np.array(X)

    # Convert to torch tensors
    X = torch.tensor(X[:, np.newaxis], dtype=torch.float32)  # (N, 1, 9, 9)
    y = torch.tensor([label_mapping[label] for label in labels], dtype=torch.long)
    
    return X, y

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CCNNWrapper(nn.Module):
    def __init__(self, in_channels=1, grid_size=(9, 9), num_classes=5, dropout=0.5):
        super(CCNNWrapper, self).__init__()
        self.ccnn = CCNN(in_channels=in_channels, grid_size=grid_size, num_classes=num_classes, dropout=dropout)

    def forward(self, x):
        return self.ccnn(x)