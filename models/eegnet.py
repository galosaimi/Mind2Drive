#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eegnet.py
---------
EEGNet model wrapper and data loader for EEG-based driver intention classification.

Reference: Lawhern et al. (2018), EEGNet: a compact convolutional neural network for 
           EEG-based brain-computer interfaces.
           
Author   : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torcheeg.models import EEGNet

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
def load_data(csv_path, label_mapping, nCH=16):
    df = pd.read_csv(csv_path)
    labels = df["Label"].values
    # eeg_data = df.drop(columns=["Label"]).values
    
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
class EEGNetWrapper(nn.Module):
    """
    EEGNet wrapper for 16-channel, 125 Hz EEG.
    Input shape: (N, 1, nCH, nTime) — unsqueeze applied in train.py.
    """
    def __init__(self, nCh=16, nTime=125, num_classes=3, dropout=0.5):
        super(EEGNetWrapper, self).__init__()
        self.eegnet = EEGNet(
            chunk_size=nTime,
            num_electrodes=nCh,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x):
        return self.eegnet(x)