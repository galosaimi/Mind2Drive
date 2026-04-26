#!/usr/bin/env python3
"""
gru.py
------
GRU model wrapper and data loader for EEG-based driver intention classification.

Reference: Zhang & Yao (2021), Deep Learning for EEG-Based Brain-Computer Interfaces: 
           Representations, Algorithms and Applications.
           
Author   : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torcheeg.models import GRU

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
def load_data(csv_path, label_mapping, nCH=16):
    df = pd.read_csv(csv_path)
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
class GRUWrapper(nn.Module):
    def __init__(self, num_electrodes=16, hid_channels=64, num_classes=3):
        super(GRUWrapper, self).__init__()
        self.gru = GRU(num_electrodes=num_electrodes, hid_channels=hid_channels, num_classes=num_classes)

    def forward(self, x):
        return self.gru(x)