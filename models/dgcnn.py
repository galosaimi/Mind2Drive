#!/usr/bin/env python3
"""
dgcnn.py
--------
DGCNN model wrapper and data loader for EEG-based driver intention classification.

Uses band differential entropy features (delta, theta, alpha, beta, gamma)
computed from each EEG window before passing to the graph neural network.

Reference: Song et al. (2018), EEG Emotion Recognition Using Dynamical Graph Convolutional 
           Neural Networks.
           
Author   : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torcheeg import transforms
from torcheeg.models import DGCNN

# ---------------------------------------------------------------------------
# Band differential entropy transform (5 frequency bands → 5 features per channel)
# ---------------------------------------------------------------------------
band_transform = transforms.BandDifferentialEntropy(band_dict={
    "delta": [1, 4],
    "theta": [4, 8],
    "alpha": [8, 14],
    "beta": [14, 31],
    "gamma": [31, 49]
})

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

    # Apply band_transform to each sample
    transformed_data = []
    for sample in eeg_data:
        transformed = band_transform(eeg=sample)["eeg"]  # shape: (16, 5)
        transformed_data.append(transformed)

    transformed_data = np.stack(transformed_data)  # shape: (N, 16, 5)
    X = torch.tensor(transformed_data, dtype=torch.float32)
    y = torch.tensor([label_mapping[label] for label in labels], dtype=torch.long)
    return X, y

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DGCNNWrapper(nn.Module):
    def __init__(self, in_channels=5, num_electrodes=16, num_classes=3, num_layers=2, hid_channels=32):
        """
        Wrapper for the DGCNN model.

        Parameters:
        - in_channels (int): The feature dimension for each electrode (default: 5).
        - num_electrodes (int): The number of electrodes (default: 16 for OpenBCI).
        - num_classes (int): The number of output classes (default: 5).
        - num_layers (int): The number of graph convolutional layers (default: 2).
        - hid_channels (int): The number of hidden nodes in the fully connected layer (default: 32).
        """
        super(DGCNNWrapper, self).__init__()
        self.dgcnn = DGCNN(
            in_channels=in_channels,
            num_electrodes=num_electrodes,
            num_classes=num_classes,
            num_layers=num_layers,
            hid_channels=hid_channels
        )

    def forward(self, x):
        return self.dgcnn(x)

    