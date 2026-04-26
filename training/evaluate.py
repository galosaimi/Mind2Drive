#!/usr/bin/env python3
"""
evaluate.py
-----------
Standalone evaluation script. Loads saved prediction and learning curve CSVs
from a completed training run and regenerates all cumulative plots.

Inputs  : results/{MODEL_NAME}/ directory containing saved CSVs from train.py
Outputs : Regenerated cumulative plots in the same output directory

Author  : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University

Usage:
    python training/evaluate.py \
        --model      {MODEL_NAME} \
        --output_dir results/{MODEL_NAME}
"""

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model",      required=True, help="Model name e.g. ccnn, tsception")
parser.add_argument("--output_dir", required=True, help="Path to results directory")
args = parser.parse_args()

model_name = args.model.upper()
output_dir = args.output_dir

# Label mapping
label_mapping = {"Forward": 0, "Turning Left": 1, "Turning Right": 2}

# ---------------------------------------------------------------------------
# Load saved CSVs
# ---------------------------------------------------------------------------
predictions_path = os.path.join(output_dir, f"{model_name}_AllSession_Predictions.csv")
curves_path      = os.path.join(output_dir, f"{model_name}_AllSession_LearningCurves.csv")
summary_path     = os.path.join(output_dir, f"{model_name}_Summary.csv")

for path in [predictions_path, curves_path, summary_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

combined_preds  = pd.read_csv(predictions_path)
combined_curves = pd.read_csv(curves_path)
summary_df      = pd.read_csv(summary_path)

print(f"Loaded {len(combined_preds)} predictions across {combined_preds['Session'].nunique()} sessions")
print(f"Loaded learning curves for {combined_curves['Session'].nunique()} sessions")

# ---------------------------------------------------------------------------
# Print summary statistics
# ---------------------------------------------------------------------------
print("\n==== Summary Statistics ====")
print(f"Mean Macro-F1        : {summary_df['Best Val Macro-F1'].mean():.4f} ± {summary_df['Best Val Macro-F1'].std():.4f}")
print(f"Mean Balanced Acc    : {summary_df['Best Val Balanced Acc'].mean():.4f} ± {summary_df['Best Val Balanced Acc'].std():.4f}")
print(f"Mean Best Epoch      : {summary_df['Best Epoch'].mean():.1f}")

# ---------------------------------------------------------------------------
# Cumulative learning curve
# ---------------------------------------------------------------------------
avg_curve = combined_curves.groupby("Epoch")[["Train_Loss", "Val_Loss"]].mean()

plt.figure(figsize=(8, 6))
plt.plot(avg_curve.index, avg_curve["Train_Loss"], label="Avg Train Loss")
plt.plot(avg_curve.index, avg_curve["Val_Loss"],   label="Avg Val Loss", linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{model_name} Cumulative Learning Curve Across All Sessions")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{model_name}_Cumulative_Learning_Curve.png"))
plt.close()
print(f"\n✓ Saved cumulative learning curve")

# ---------------------------------------------------------------------------
# Cumulative confusion matrix
# ---------------------------------------------------------------------------
y_true_all = combined_preds["y_true"]
y_pred_all = combined_preds["y_pred"]
conf_matrix = confusion_matrix(y_true_all, y_pred_all, labels=list(label_mapping.values()))
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True) * 100

plt.figure(figsize=(6, 6))
ax = sns.heatmap(conf_matrix_percent, annot=True, fmt=".2f", cmap="Blues",
                 xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
ax.collections[0].colorbar.remove()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"{model_name} Cumulative Confusion Matrix (%)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{model_name}_Cumulative_Confusion_Matrix.png"))
plt.close()
print(f"Saved cumulative confusion matrix")

# ---------------------------------------------------------------------------
# Per-class recall summary
# ---------------------------------------------------------------------------
print("\n==== Per-class Recall ====")
for col in ["Recall_Forward", "Recall_Left", "Recall_Right"]:
    if col in summary_df.columns:
        print(f"{col:20s}: {summary_df[col].mean():.4f} ± {summary_df[col].std():.4f}")

print(f"\nAll plots saved to {output_dir}")