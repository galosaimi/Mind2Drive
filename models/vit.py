#!/usr/bin/env python3
"""
vit.py
------
Vision Transformer (ViT) model wrapper and data loader for EEG-based driver intention classification.

Each EEG window is converted to a spatiotemporal grid of shape (nTime, 4, 4)
where 16 channels are mapped to a 4x4 spatial grid at each timepoint.

Reference: Dosovitskiy et al. (2020), An image is worth 16x16 words: Transformers for image recognition 
           at scale.

Author   : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torcheeg.models import ViT

# ---------------------------------------------------------------------------
# Spatial grid mapping: 16 EEG channels → 4x4 grid
# ---------------------------------------------------------------------------
GRID_MAPPING = {
    0: (0, 0),   # Fp1
    1: (0, 1),   # Fp2
    2: (0, 2),   # F7
    3: (0, 3),   # F8
    4: (1, 0),   # T3
    5: (1, 1),   # T4
    6: (1, 2),   # F3
    7: (1, 3),   # F4
    8: (2, 0),   # C3
    9: (2, 1),   # C4
    10: (2, 2),  # T5
    11: (2, 3),  # T6
    12: (3, 0),  # P3
    13: (3, 1),  # P4
    14: (3, 2),  # O1
    15: (3, 3),  # O2
}

def map_to_grid(data, grid_size=(4, 4)):
    """Map 16-channel EEG snapshot to a 4x4 spatial grid."""

    grid = np.zeros(grid_size) 
    for ch, (r, c) in GRID_MAPPING.items():
        grid[r, c] = data[ch]

    return grid

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
    for sample in eeg_data:  # [125, 16]
        temporal_grids = []
        for t in range(125):
            grid = map_to_grid(sample[:, t])   
            temporal_grids.append(grid)
        temporal_grid = np.stack(temporal_grids, axis=0)  # [125, 4, 4] # [125, 9, 9]
        grid_data.append(temporal_grid)

    X = torch.tensor(np.stack(grid_data), dtype=torch.float32)
    y = torch.tensor([label_mapping[label] for label in labels], dtype=torch.long)
    return X, y

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ViTWrapper(nn.Module):
    def __init__(self, chunk_size=125, grid_size=(4, 4), t_patch_size=25, s_patch_size=(2, 2),
                 hid_channels=32, depth=3, heads=4, head_channels=256, mlp_channels=64, 
                 num_classes=3, embed_dropout=0.1, dropout=0.1, pool_func='cls'):
        """
        Wrapper for the Vision Transformer (ViT) model from torcheeg.

        Parameters:
        - chunk_size (int): Number of time points included in each chunk (temporal dimension).
        - grid_size (tuple): Spatial dimensions of grid-like EEG representation.
        - t_patch_size (int): Temporal patch size.
        - s_patch_size (tuple): Spatial patch size (resolution).
        - hid_channels (int): Feature dimension of embedded patches.
        - depth (int): Number of attention layers for each transformer block.
        - heads (int): Number of attention heads for each attention layer.
        - head_channels (int): Dimension of each attention head.
        - mlp_channels (int): Number of hidden nodes in the fully connected layer.
        - num_classes (int): Number of classes for classification.
        - embed_dropout (float): Dropout probability for embedding layers.
        - dropout (float): Dropout probability for transformer layers.
        - pool_func (str): Pooling method ('cls' or 'mean').
        """
        super(ViTWrapper, self).__init__()
        
        # Initialize the ViT model
        self.vit = ViT(
            chunk_size=chunk_size,
            grid_size=grid_size,
            t_patch_size=t_patch_size,
            s_patch_size=s_patch_size,
            hid_channels=hid_channels,
            depth=depth,
            heads=heads,
            head_channels=head_channels,
            mlp_channels=mlp_channels,
            num_classes=num_classes,
            embed_dropout=embed_dropout,
            dropout=dropout,
            pool_func=pool_func
        )
    
    def forward(self, x):
        return self.vit(x)