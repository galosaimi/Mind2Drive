#!/usr/bin/env python3
"""
train.py
--------
Training loop and batch processing for EEG-based driver intention classification.

Imports the model and data loader from models/{MODEL_NAME}.py and runs training
across all sessions, saving results and plots to the output directory.

Inputs  : Windowed/oversampled train CSVs and windowed test CSVs
Outputs : Summary CSVs, plots, and TensorBoard logs in results/{MODEL_NAME}/

Author  : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University

Usage:
    python training/train.py \
        --train_base data/oversampled \
        --test_base  data/windowed \
        --output_dir results/{MODEL_NAME} \
"""

import os
import sys
import random
import argparse
import importlib
from xml.parsers.expat import model
import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Import model and data loader from models/
# ---------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--train_base", required=True)
parser.add_argument("--test_base", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Constants and directories
# ---------------------------------------------------------------------------
num_epochs = 2000

train_base = args.train_base
test_base = args.test_base
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "Plots"), exist_ok=True)
model_name = args.model

# Label mapping
label_mapping = {"Forward": 0, "Turning Left": 1, "Turning Right": 2}

# ---------------------------------------------------------------------------
# Dynamic model import based on --model argument
# ---------------------------------------------------------------------------
model_registry = {
    "ccnn":          "models.ccnn",
    "cnn1d":         "models.cnn1d",
    "deepconvnet":   "models.deepconvnet",
    "dgcnn":         "models.dgcnn",
    "eegconformer":  "models.eegconformer",
    "eegnet":        "models.eegnet",
    "gru":           "models.gru",
    "lstm":          "models.lstm",
    "shallowconvnet":"models.shallowconvnet",
    "stnet":         "models.stnet",
    "tsception":     "models.tsception",
    "vit":           "models.vit",
}

model_key = args.model.lower()
if model_key not in model_registry:
    raise ValueError(f"Unknown model '{args.model}'. Choose from: {list(model_registry.keys())}")

model_module  = importlib.import_module(model_registry[model_key])
load_data     = model_module.load_data
ModelWrapper  = model_module.__dict__[[k for k in model_module.__dict__ if "Wrapper" in k][0]]

# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------
def run_model(session_name, train_csv, test_csv, batch_size=128):
    # Load Data
    X_train, y_train = load_data(train_csv, label_mapping)
    X_test,  y_test  = load_data(test_csv,  label_mapping)
    nCH  = X_train.shape[1]
    nTime = X_train.shape[2] if X_train.dim() == 3 else None
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    # Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate model — pass shape if required
    try:
        model = ModelWrapper(nCh=nCH, nTime=nTime, num_classes=len(label_mapping)).to(device)
    except TypeError:
        try:
            model = ModelWrapper(num_classes=len(label_mapping)).to(device)
        except TypeError:
            model = ModelWrapper().to(device)

    # Use model-specific loss if defined
    criterion = getattr(model_module, 'criterion', nn.CrossEntropyLoss())
    opt_cfg   = getattr(model_module, 'optimizer_config', {"lr": 0.001, "weight_decay": 9e-3})
    optimizer = optim.Adam(model.parameters(), **opt_cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5e-6)
    
    # TensorBoard Setup
    log_dir = os.path.join(output_dir, "TensorBoard", session_name)
    writer = SummaryWriter(log_dir=log_dir)

    train_losses, test_losses = [], []
    best_val_f1 = -1
    best_epoch = -1
    best_val_bal_acc = -1    
    best_state_dict = None
    
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct_train, total_train = 0, 0, 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = 100 * correct_train / total_train
        
        # Validation Loop
        model.eval()
        total_val_loss, correct_val, total_val = 0, 0, 0
        y_val_true, y_val_pred = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
                y_val_true.extend(batch_y.cpu().numpy())
                y_val_pred.extend(predicted.cpu().numpy())
                
        val_loss = total_val_loss / len(test_loader)
        test_losses.append(val_loss)
        val_accuracy = 100 * correct_val / total_val
        val_macro_f1 = f1_score(y_val_true, y_val_pred, average='macro')
        val_bal_acc = balanced_accuracy_score(y_val_true, y_val_pred)
        
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_val_bal_acc = val_bal_acc
            best_epoch = epoch + 1
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f}")
        print(f"Best epoch: {best_epoch}, Best Val Macro-F1: {best_val_f1:.4f}")

        # TensorBoard Logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/Val", val_accuracy, epoch)
        writer.add_scalar("F1/Val_Macro", val_macro_f1, epoch)
        writer.add_scalar("BalancedAcc/Val", val_bal_acc, epoch)

    model.load_state_dict(best_state_dict)
    
    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}

    # After training loop
    final_train_loss = train_losses[-1]
    final_val_loss = test_losses[-1]  # test_loader used as validation
    final_train_acc = train_accuracy  # this already stores last epoch's train acc
    final_val_acc = val_accuracy      # this already stores last epoch's val acc
    
    print("\n==== Final Summary ====")
    print(f"Final Training Loss:     {final_train_loss:.4f}")
    print(f"Final Training Accuracy: {final_train_acc:.2f}%")
    print(f"Final Validation Loss:   {final_val_loss:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
 
    # Summary
    summary_row = {
        "Session": session_name,
        "Model": model_name,
        "Train Loss": final_train_loss,
        "Val Loss": final_val_loss,
        "Last Epoch Train Acc (%)": final_train_acc,
        "Last Epoch Val Acc (%)": final_val_acc,
        "Best Epoch": best_epoch,
        "Best Val Macro-F1": best_val_f1,
        "Best Val Balanced Acc": best_val_bal_acc,
    }
    
    # Prediction DataFrame
    df_preds = pd.DataFrame({
        "Session": session_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_label": [inverse_label_mapping[y] for y in y_true],
        "y_pred_label": [inverse_label_mapping[y] for y in y_pred],
    })
    
    # Learning Curve DataFrame
    df_curve = pd.DataFrame({
        "Session": session_name,
        "Epoch": list(range(1, num_epochs + 1)),
        "Train_Loss": train_losses,
        "Val_Loss": test_losses
    })
    
    # Confusion Matrix DataFrame
    labels_list = list(label_mapping.values())
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels_list)

    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=label_mapping.keys(),
        columns=label_mapping.keys()
    )
    conf_matrix_df.insert(0, "Session", session_name)
    
    # Generate classification report dict
    report_dict = classification_report(
        y_true, y_pred,
        labels=list(label_mapping.values()),
        target_names=list(label_mapping.keys()),
        output_dict=True,
        zero_division=0  
    )

    summary_row.update({
        "Recall_Forward": report_dict["Forward"]["recall"],
        "Recall_Left": report_dict["Turning Left"]["recall"],
        "Recall_Right": report_dict["Turning Right"]["recall"],
    })

    # Convert to DataFrame
    report_df = pd.DataFrame(report_dict).transpose().round(2)
    report_df.insert(0, 'Session', session_name)
    
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(conf_matrix_percent, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
    ax.collections[0].colorbar.remove()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix (%) - {model_name} - {session_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Plots", f"{session_name}_conf_matrix.png"))
    plt.close()
    
    # Save learning curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Validation Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Learning Curve - {model_name} - {session_name}")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "Plots", f"{session_name}_learning_curve.png"))
    plt.close()
    
    writer.close()  # Close TensorBoard writer

    return pd.DataFrame([summary_row]), report_df, df_preds, df_curve, conf_matrix_df

# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
all_summaries, all_reports = [], []
all_predictions, all_curves, all_conf_matrices = [], [], []

for root, _, files in os.walk(train_base):
    for file in files:
        if file.endswith("train.csv"):
            session_name = file.replace("_train.csv", "")
            subject_folder = os.path.basename(root)
            train_path = os.path.join(root, file)
            test_path = os.path.join(test_base, subject_folder, f"{session_name}_test.csv")
            if os.path.exists(test_path):
                print(f"Processing {session_name}...")
                summary_df, report_df, df_preds, df_curve, conf_matrix_df = run_model(session_name, train_path, test_path)
                all_summaries.append(summary_df)
                all_reports.append(report_df)
                all_predictions.append(df_preds)
                all_curves.append(df_curve)
                all_conf_matrices.append(conf_matrix_df)
                print(f"Completed {session_name}.")

# ---------------------------------------------------------------------------
# Save all results
# ---------------------------------------------------------------------------
pd.concat(all_summaries).to_csv(os.path.join(output_dir, f"{model_name}_Summary.csv"), index=False)
pd.concat(all_reports).to_csv(os.path.join(output_dir, f"{model_name}_ClassificationReport.csv"), index=False)
pd.concat(all_predictions).to_csv(os.path.join(output_dir, f"{model_name}_AllSession_Predictions.csv"), index=False)
pd.concat(all_curves).to_csv(os.path.join(output_dir, f"{model_name}_AllSession_LearningCurves.csv"), index=False)
pd.concat(all_conf_matrices).to_csv(os.path.join(output_dir, f"{model_name}_AllSession_ConfusionMatrix_Counts.csv"))

# ---------------------------------------------------------------------------
# Cumulative plots
# ---------------------------------------------------------------------------
combined_curves = pd.concat(all_curves)
avg_curve = combined_curves.groupby("Epoch")[["Train_Loss", "Val_Loss"]].mean()

# Plot Cumulative Learning Curve
plt.figure(figsize=(8, 6))
plt.plot(avg_curve.index, avg_curve["Train_Loss"], label="Avg Train Loss")
plt.plot(avg_curve.index, avg_curve["Val_Loss"], label="Avg Val Loss", linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f" {model_name} _ Cumulative Learning Curve Across All Sessions")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, model_name+"_Cumulative_Learning_Curve.png"))
plt.close()

# Cumulative Confusion Matrix
combined_preds = pd.concat(all_predictions)
y_true_all = combined_preds["y_true"]
y_pred_all = combined_preds["y_pred"]
conf_matrix = confusion_matrix(y_true_all, y_pred_all, labels=list(label_mapping.values()))
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True) * 100

# Plot Cumulative Confusion Matrix
plt.figure(figsize=(6, 6))
ax = sns.heatmap(conf_matrix_percent, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
ax.collections[0].colorbar.remove()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"{model_name} _ Cumulative Confusion Matrix (%)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, model_name+"_Cumulative_Confusion_Matrix.png"))
plt.close()