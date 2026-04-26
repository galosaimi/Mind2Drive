#!/usr/bin/env python3
"""
stnet.py
--------
STNet model wrapper and data loader for EEG-based driver intention classification.

Uses a 4x4 spatial grid mapping of 16 EEG channels and Focal Loss
to handle class imbalance between Forward and turning manoeuvres.

Reference: Zhang et al. (2022), GANSER: A Self-supervised Data Augmentation Framework for 
           EEG-based Emotion Recognition.
           
Author   : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
from torcheeg.models import STNet
from torcheeg.transforms import ToGrid

# ---------------------------------------------------------------------------
# Spatial grid mapping for 16 EEG channels (4x4 grid)
# ---------------------------------------------------------------------------
custom_channel_location = {
    'Fp1': (0, 0), 'Fp2': (0, 1), 'C3': (0, 2), 'C4': (0, 3),
    'T3': (1, 0), 'T4': (1,1), 'O1': (1, 2), 'O2': (1, 3),
    'F7': (2, 0), 'F8': (2, 1), 'F3': (2, 2), 'F4': (2, 3),
    'T5': (3, 0), 'T6': (3, 1), 'P3': (3, 2), 'P4': (3, 3)
} 

to_grid_transform = ToGrid(custom_channel_location)

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
    
    grid_data = []
    for sample in eeg_data:
        transformed = to_grid_transform(eeg=sample)['eeg']
        grid_data.append(transformed)
        
    X = torch.tensor(np.stack(grid_data), dtype=torch.float32)
    y = torch.tensor([label_mapping[label] for label in labels], dtype=torch.long)
    
    return X, y

# ---------------------------------------------------------------------------
# Focal loss — handles class imbalance between Forward and turning classes
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class STNetWrapper(nn.Module):
    def __init__(self, chunk_size=125, grid_size=(4, 4), num_classes=3, dropout=0.4):
        super(STNetWrapper, self).__init__()
        self.stnet = STNet(chunk_size=chunk_size, grid_size=grid_size, num_classes=num_classes, dropout=dropout)

    def forward(self, x):
        return self.stnet(x)

# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
criterion = FocalLoss(alpha=1, gamma=1)