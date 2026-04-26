#!/usr/bin/env python3
"""
eegconformer.py
---------------
EEGConformer model wrapper and data loader for EEG-based driver intention classification.

Adaptation Notes:
- Original: 22 channels, 250 Hz, input [22, 1000]
- This implementation: 16 channels, 125 Hz, input [16, 125]
- Convolutional kernel adjusted from (22,1) to (16,1)
- EEGConformer uses lr=1e-4 instead of 1e-3 (set via optimizer_config below)

Reference: Song et al. (2023), EEG Conformer: Convolutional Transformer for EEG Decoding 
           and Visualization.
Code     : https://github.com/eeyhsong/EEG-Conformer/blob/main/conformer.py

Author   : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
def load_data(csv_path, label_mapping, nCh=16):
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
    nTime = n_features // nCh
    
    eeg_data = eeg_data.reshape(len(eeg_data), nTime, nCh)
    eeg_data = np.transpose(eeg_data, (0, 2, 1))
    X = torch.tensor(eeg_data, dtype=torch.float32)
    y = torch.tensor([label_mapping[label] for label in labels], dtype=torch.long)
    return X, y

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size, nTime):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (16, 1), (1, 1)), # 16 channels for EEG insteade of 22
            nn.BatchNorm2d(40),
            nn.ELU(),
            # nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.AvgPool2d((1, 25), (1, 12)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)

        scaling = self.emb_size ** 0.5
        att = torch.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=5, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self,
                 depth,
                 emb_size,
                 num_heads=4,
                 drop_p=0.3,
                 forward_expansion=4,
                 forward_drop_p=0.3):
        blocks = [
            TransformerEncoderBlock(
                emb_size,
                num_heads=num_heads,
                drop_p=drop_p,
                forward_expansion=forward_expansion,
                forward_drop_p=forward_drop_p,
            )
            for _ in range(depth)
        ]
        super().__init__(*blocks)

class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        return self.net(x)


class EEGConformer(nn.Module):
    def __init__(self, nCh, nTime, nCls,
                 emb_size=64,
                 depth=4,
                 num_heads=4,
                 drop_p=0.3,
                 forward_expansion=4,
                 forward_drop_p=0.3):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, nTime)
        self.transformer_encoder = TransformerEncoder(
            depth,
            emb_size,
            num_heads=num_heads,
            drop_p=drop_p,
            forward_expansion=forward_expansion,
            forward_drop_p=forward_drop_p,
        )
        self.classification_head = ClassificationHead(emb_size, nCls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) 
        # Not in the original implementation
        # Use global average pooling over tokens, equivalent to the Reduce operation in the original EEG-Conformer implementation, while omitting the auxiliary fully-connected head for fair comparison.
        x = self.classification_head(x)
        return x

# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class EEGConformerWrapper(nn.Module):
    def __init__(
        self,
        nCh=16,
        nTime=125,
        nCls=3,
        emb_size=64,
        depth=4,
        num_heads=4,
        drop_p=0.3,
        forward_expansion=4,
        forward_drop_p=0.3,
    ):
        super().__init__()
        self.eegconformer = EEGConformer(
            nCh=nCh,
            nTime=nTime,
            nCls=nCls,
            emb_size=emb_size,
            depth=depth,
            num_heads=num_heads,
            drop_p=drop_p,
            forward_expansion=forward_expansion,
            forward_drop_p=forward_drop_p,
        )

    def forward(self, x):
        return self.eegconformer(x)

# ---------------------------------------------------------------------------
# Optimizer config — EEGConformer uses lr=1e-4 instead of default 1e-3
# ---------------------------------------------------------------------------
optimizer_config = {"lr": 1e-4, "weight_decay": 1e-4}

