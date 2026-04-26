#!/usr/bin/env python3
"""
cnn1d.py
--------
CNN1D model wrapper and data loader for EEG-based driver intention classification.

Reference: Taghizadeh et al. (2024), EEG Motor Imagery Classification by Feature Extracted
           Deep 1D-CNN and Semi-Deep Fine-Tuning.
Code     : https://github.com/MohamadTaghizadeh/EEG-1DCNN/blob/main/CNN1D.py

Author   : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn

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
    return X, y, nCH, nTime

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DeepCNN1D(nn.Module):
    def __init__(self, in_channels=2, input_length=640, num_classes=5):
        super(DeepCNN1D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=10, padding='same'),  # same padding
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 64, kernel_size=10),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),

            nn.Conv1d(64, 32, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )

        # Compute flattened size dynamically
        dummy_input = torch.zeros(1, in_channels, input_length)
        with torch.no_grad():
            dummy_output = self.conv_block(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(flattened_size, 306),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(306, 153),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(153, 77),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(77, 77),
            nn.ReLU(),

            nn.Linear(77, num_classes)  # No softmax (use CrossEntropyLoss)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x