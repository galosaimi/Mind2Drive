#!/usr/bin/env python3
"""
shallowconvnet.py
-----------------
ShallowConvNet model wrapper and data loader for EEG-based driver intention classification.

Adapted for 125 Hz, 16-channel EEG:
  - Temporal filter size reduced from 25 to 14
  - Pooling size (P=35) and stride (S=7) reflect half of original
  - LogSoftmax output requires NLLLoss (set via criterion below)

Reference: Schirrmeister et al. (2017), Deep learning with convolutional neural networks for 
           EEG decoding and visualization
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

class Lambda(nn.Module):
    def __init__(self, func) -> None:
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
    
# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ShallowConvNet(nn.Module):
    def __init__(
        self,
        nCh: int,
        nTime: int,
        nCls: int,
        F: int = 40,
        C: int = 14,
        P: int = 35,
        S: int = 7,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.nCh = nCh
        self.nTime = nTime

        self.conv = nn.Sequential(
            Conv2dWithNorm(1, F, (1, C), max_norm=2, bias=False),
            Conv2dWithNorm(F, F, (nCh, 1), max_norm=2, bias=False, groups=F),
            nn.BatchNorm2d(F),
            Lambda(torch.square),
            nn.AvgPool2d((1, P), stride=(1, S)),
            Lambda(torch.log),
        )

        linear_in = self.forward_flatten().shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            LinearWithNorm(linear_in, nCls, max_norm=0.5),
            nn.LogSoftmax(dim=1),
        )

    def forward_flatten(self):
        x = torch.rand(1, 1, self.nCh, self.nTime)
        x = self.conv(x)
        x = torch.flatten(x, 1, -1)
        return x

    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        return x
    
class ShallowConvNetWrapper(nn.Module):
    def __init__(self, nCh=16, nTime=125, nCls=3, F=40, C=14, P=35, S=7, dropout=0.5):
        super().__init__()
        self.shallowconvnet = ShallowConvNet(
            nCh=nCh,
            nTime=nTime,
            nCls=nCls,
            F=F,
            C=C,
            P=P,
            S=S,
            dropout=dropout,
        )
    
    def forward(self, x):
        return self.shallowconvnet(x)

# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
criterion = nn.NLLLoss()  # Required because model output uses LogSoftmax