#!/usr/bin/env python3
"""
tsception.py
------------
TSCeption model wrapper and data loader for EEG-based driver intention classification.

TSCeption captures temporal dynamics and spatial asymmetry from EEG signals
using multi-scale temporal and spatial convolutional branches.

Reference: Ding et al. (2022), Tsception: Capturing temporal dynamics and spatial asymmetry 
           from EEG for emotion recognition.
           
Author   : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torcheeg.models import TSCeption

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
def load_data(csv_path, label_mapping, nCH=16):
    df = pd.read_csv(csv_path)
    
    if "Label" not in df.columns:
        raise ValueError(f"Could not find 'Label' column in {csv_path}. Columns found: {df.columns.tolist()}")
    
    labels = df["Label"].values
    
    # Keep only numeric EEG columns
    eeg_data = (
        df
        .drop(columns=["Label", "Source"], errors="ignore")
        .select_dtypes(include=[np.number])
        .values
    )    
    
    n_features = eeg_data.shape[1]
    nTime = n_features // nCH
    
    eeg_data = eeg_data.reshape(len(eeg_data), nTime, nCH)
    eeg_data = np.transpose(eeg_data, (0, 2, 1))
    
    X = torch.tensor(eeg_data, dtype=torch.float32)
    y = torch.tensor([label_mapping[label] for label in labels], dtype=torch.long)
    return X, y

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TSCeptionWrapper(nn.Module):
    def __init__(self, num_classes=3, num_electrodes=16, sampling_rate=125, num_T=15, num_S=15, hid_channels=32, dropout=0.5):
        super(TSCeptionWrapper, self).__init__()
        self.tsception = TSCeption(
            num_classes=num_classes,
            num_electrodes=num_electrodes,
            sampling_rate=sampling_rate,
            num_T=num_T,
            num_S=num_S,
            hid_channels=hid_channels,
            dropout=dropout
        )

    def forward(self, x):
        return self.tsception(x)