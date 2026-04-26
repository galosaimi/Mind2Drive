#!/usr/bin/env python3
"""
deepconvnet.py
--------------
DeepConvNet model wrapper and data loader for EEG-based driver intention classification.

Reference: Schirrmeister et al. (2017), Deep learning with convolutional neural networks 
           for EEG decoding and visualization
Code     : https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py

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
# Utility layers with max-norm constraint
# ---------------------------------------------------------------------------
class Conv2dWithNorm(nn.Conv2d):
    def __init__(self, *args, do_weight_norm=True, max_norm=1.0, p=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, self.p, 0, self.max_norm)
        return super().forward(input)

    def __repr__(self):
        repr = super().__repr__()
        if self.do_weight_norm:
            last_bracket_index = repr.rfind(")")
            self_repr = f", max_norm={self.max_norm}, p={self.p}"
            repr = repr[:last_bracket_index] + self_repr + ")"
        return repr

# Utility function: Linear layer with max-norm constraint
class LinearWithNorm(nn.Linear):
    def __init__(self, *args, do_weight_norm=True, max_norm=1.0, p=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, self.p, 0, self.max_norm)
        return super().forward(input)

    def __repr__(self):
        repr = super().__repr__()
        if self.do_weight_norm:
            last_bracket_index = repr.rfind(")")
            self_repr = f", max_norm={self.max_norm}, p={self.p}"
            repr = repr[:last_bracket_index] + self_repr + ")"
        return repr

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DeepConvNet(nn.Module):
    def __init__(self, nCh: int, nTime: int, nCls: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.nCh = nCh
        self.nTime = nTime
        kernel_size = [1, 5] # Orginal kernal was [1, 10]
        filter_layer = [25, 50, 100, 200]

        first_layer = nn.Sequential(
            Conv2dWithNorm(1, 25, kernel_size, max_norm=2),
            Conv2dWithNorm(25, 25, (nCh, 1), bias=False, max_norm=2),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2)), # Original max pool was (1, 3)
            nn.Dropout(dropout)
        )
        middle_layer = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Dropout(dropout),
                    Conv2dWithNorm(in_f, out_f, kernel_size),
                    nn.BatchNorm2d(out_f),
                    nn.ELU(),
                    nn.MaxPool2d((1, 2)), # Original max pool was (1, 3)
                )
                for in_f, out_f in zip(filter_layer, filter_layer[1:])
            ]
        )
        self.conv_layer = nn.Sequential(first_layer, middle_layer)

        linear_in = self._forward_flatten().shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            LinearWithNorm(linear_in, nCls, max_norm=0.5),
            nn.LogSoftmax(dim=1),
        )

    def _forward_flatten(self):
        x = torch.rand(1, 1, self.nCh, self.nTime)
        for idx, layer in enumerate(self.conv_layer):
            x = layer(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        return x

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.head(x)
        return x

class DeepConvNetWrapper(nn.Module):
    def __init__(self, nTime=125, nCh=16, nCls=5, dropout=0.25):
        super(DeepConvNetWrapper, self).__init__()
        self.deepconvnet = DeepConvNet(
            nCh=nCh,
            nTime=nTime,
            nCls=nCls,
            dropout=dropout
        )

    def forward(self, x):
        return self.deepconvnet(x)
    
# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
criterion = nn.NLLLoss()  # Required because model output uses LogSoftmax