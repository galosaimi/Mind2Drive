#!/usr/bin/env python3
"""
TimeGAN-oversample.py
---------------------
Apply TimeGAN-based synthetic data augmentation to balance EEG training windows.

For each minority class, a TimeGAN model is trained on real windows and used
to generate synthetic samples until all classes reach the majority class count.
Memory is explicitly freed after each subject to support GPU-constrained environments.

Inputs  : Windowed train/test CSVs 
Outputs : Augmented CSVs and per-subject timegan_log.txt files

Author  : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University
"""

import os
import gc
import pandas as pd
import numpy as np
import psutil
import GPUtil
from collections import Counter

from ydata_synthetic.synthesizers.timeseries import TimeGAN
from ydata_synthetic.synthesizers import ModelParameters

import tensorflow as tf
from tensorflow.keras import backend as K

tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration: input/output paths and split types
# ---------------------------------------------------------------------------
INPUT_ROOT = "data/processed-train_split"
OUTPUT_ROOT = "data/oversampled/timegan_500steps/train_split"
splits = ["Custom", "Normal"]

# ---------------------------------------------------------------------------
# TimeGAN parameters
# ---------------------------------------------------------------------------
N_CHANNELS          = 16   # Number of EEG channels
TIMEGAN_TRAIN_STEPS = 500  # Training steps per TimeGAN model

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def print_memory(tag):
    """Print current GPU and RAM usage."""
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        print(
            f"[{tag}] GPU: used={gpu.memoryUsed:.1f}MB / total={gpu.memoryTotal:.1f}MB"
        )

    mem = psutil.virtual_memory()
    print(
        f"[{tag}] RAM: used={mem.used/1e9:.2f}GB / total={mem.total/1e9:.2f}GB"
    )

# Train TimeGAN and generate synthetic data for a given class
def train_timegan_and_generate(X_class_np, seq_len, n_generate):
    """
    Train a TimeGAN model on one class and generate synthetic windows.

    Args:
        X_class_np : Real EEG windows of shape (n_samples, seq_len, n_channels)
        seq_len    : Window length in samples
        n_generate : Number of synthetic windows to generate

    Returns:
        synthetic_flat : Synthetic windows of shape (n_generate, seq_len * n_channels)
    """    
    print(f"  Training TimeGAN on data shape: {X_class_np.shape}")
    
    # ModelParameters without gamma
    gan_args = ModelParameters(
        batch_size=128,
        lr=5e-4,
        noise_dim=32,
        layers_dim=128,
        latent_dim=24
    )
    
    # Initialize TimeGAN: pass all required parameters
    synth = TimeGAN(
        model_parameters=gan_args,
        hidden_dim=128,
        seq_len=seq_len,
        n_seq=N_CHANNELS,
        gamma=1
    )
    
    # Train
    synth.train(data=X_class_np, train_steps=TIMEGAN_TRAIN_STEPS)
    
    # Generate
    synth_data = synth.sample(n_generate)
    print(f"  Generated synthetic data shape: {synth_data.shape}")
    print(
        f"  -> Returned as (n={synth_data.shape[0]}, "
        f"seq_len={synth_data.shape[1]}, "
        f"channels={synth_data.shape[2]})"
    )
    
    # Flatten
    n_samples = synth_data.shape[0]
    try:
        synthetic_flat = synth_data.reshape(n_samples, -1)
    except Exception as e:
        print("  ⚠️ Reshape failed:", e)
        print(f"  n_samples={n_samples}, total size={synth_data.size}")
        raise
    print(f"  Flattened synthetic data shape: {synthetic_flat.shape}")    
    del synth, synth_data
    gc.collect()
    
    return synthetic_flat

# ---------------------------------------------------------------------------
# Main: traverse splits, experiments, and subjects
# ---------------------------------------------------------------------------
for split in splits:
    split_in_dir = os.path.join(INPUT_ROOT, split)
    split_out_dir = os.path.join(OUTPUT_ROOT, split)
    os.makedirs(split_out_dir, exist_ok=True)

    for experiment in sorted(os.listdir(split_in_dir)):
        exp_in = os.path.join(split_in_dir, experiment)
        if not os.path.isdir(exp_in):
            continue

        exp_out = os.path.join(split_out_dir, experiment)
        os.makedirs(exp_out, exist_ok=True)

        print(f"\n=== TimeGAN Oversampling: {split}/{experiment} ===")
        
        for subject in sorted(os.listdir(exp_in)):
            print_memory(f"BEFORE {subject}")

            subj_in = os.path.join(exp_in, subject)
            if not os.path.isdir(subj_in):
                continue

            subj_out = os.path.join(exp_out, subject)
            os.makedirs(subj_out, exist_ok=True)

            log_path = os.path.join(subj_out, "timegan_log.txt")
            with open(log_path, "w") as log:
                for file in sorted(os.listdir(subj_in)):
                    src = os.path.join(subj_in, file)
                    
                    # Copy test files without modification
                    if file.endswith("_test_windows.csv"):
                        base = file.replace("_test_windows.csv", "")
                        dst = os.path.join(subj_out, f"{base}_test.csv")
                        df_test = pd.read_csv(src)                    
                        print(f" Copying {file} -> shape={df_test.shape}")
                        df_test.to_csv(dst, index=False)
                        del df_test
                        continue
                    
                    # Process training files with TimeGAN augmentation
                    if file.endswith("_train_windows.csv"):
                        base = file.replace("_train_windows.csv", "")
                        out_train = os.path.join(subj_out, f"{base}_train.csv")

                        df = pd.read_csv(src)
                        df["Source"] = "real"
            
                        # Print shape
                        print(f" Processing {file} -> shape={df.shape}")
                        
                        before = Counter(df["Label"])

                        X = df.drop(["Label", "Source"], axis=1)
                        y = df["Label"]
                        
                        num_features = X.shape[1]
                        assert num_features % N_CHANNELS == 0, \
                            f"Feature count {num_features} not divisible by {N_CHANNELS}"

                        seq_len = num_features // N_CHANNELS
                        print(f" Processing {file} -> seq_len={seq_len}, channels={N_CHANNELS}")

                        counts = Counter(y)
                        target = max(counts.values())
                        dfs_out = [df.copy()]

                        for label in counts.keys():
                            need = target - counts[label]
                            print(f" Label={label}: have={counts[label]}, target={target}, need={need}")

                            if need <= 0:
                                continue

                            Xc = X[y == label].values
                            if len(Xc) == 0:
                                continue
                            Xc_np = Xc.reshape(len(Xc), seq_len, N_CHANNELS)
                            print(f"  Xc_np shape = {Xc_np.shape}")

                            synth_flat = train_timegan_and_generate(Xc_np, seq_len, need)

                            df_synth = pd.DataFrame(synth_flat, columns=X.columns)
                            df_synth["Label"] = label
                            df_synth["Source"] = "synthetic"
                            dfs_out.append(df_synth)
                            
                            del Xc, Xc_np, synth_flat, df_synth
                            gc.collect()

                        df_final = pd.concat(dfs_out, ignore_index=True)
                        
                        # Trim to exact balance across classes
                        df_final = (
                            df_final
                            .groupby("Label", group_keys=False)
                            .apply(lambda g: g.sample(n=target, random_state=42))
                            .reset_index(drop=True)
                        )

                        after = Counter(df_final["Label"])
                        df_final.to_csv(out_train, index=False)

                        log.write(f"{file}\n")
                        log.write(f"Before: {dict(before)}\n")
                        log.write(f"After:  {dict(after)}\n")
                        log.write("-" * 60 + "\n")
            
            # Free memory after each subject
            try:
                del df
            except NameError:
                pass
            try:
                del X
            except NameError:
                pass
            try:
                del y
            except NameError:
                pass
            try:
                del dfs_out
            except NameError:
                pass
            try:
                del df_final
            except NameError:
                pass
            try:
                del counts, before, after
            except NameError:
                pass
            gc.collect()

            K.clear_session()
            tf.compat.v1.reset_default_graph()

            print("Freed memory after subject:", subject)
            print_memory(f"AFTER {subject}")
            
print("\nTimeGAN Oversampling is DONE.")