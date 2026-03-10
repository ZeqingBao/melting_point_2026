
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, root_mean_squared_error
import numpy as np
import torch
from pathlib import Path
import copy
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from CNN_model import CNNRegressor
import time
import torch
import optuna
import copy
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator


def find_optimal_clusters(X_scaled, max_k=15, random_state=0, plot=True):
    ks = list(range(2, max_k + 1))
    wcss, sils = [], []

    for k in ks:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=random_state
        )
        labels = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)

        try:
            sils.append(silhouette_score(X_scaled, labels))
        except Exception:
            sils.append(np.nan)

    kn = KneeLocator(ks, wcss, curve="convex", direction="decreasing")
    k_elbow = kn.elbow
    k_sil = ks[int(np.nanargmax(sils))] if np.isfinite(sils).any() else None
    k_opt = k_elbow if k_elbow is not None else k_sil

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ks, wcss, marker="o", label="WCSS")
        if k_elbow is not None:
            ax.axvline(k_elbow, linestyle="--", label=f"Elbow k={k_elbow}")
        ax.set(xlabel="k", ylabel="WCSS", title="Elbow (WCSS)")
        ax.grid(True); ax.legend()
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ks, sils, marker="o", label="Silhouette")
        if k_sil is not None:
            ax.axvline(k_sil, linestyle="--", label=f"Best silhouette k={k_sil}")
        ax.set(xlabel="k", ylabel="Silhouette score", title="Silhouette per k")
        ax.grid(True); ax.legend()
        plt.show()

    print(f"Elbow k: {k_elbow} | Best silhouette k: {k_sil} | Selected k_opt: {k_opt}")
    return k_opt


def plot_training_progress(train_losses, val_losses, early_stop_epoch=None, title="Training and Validation Loss"):

    epochs = range(1, len(train_losses) + 1) 
    
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses,   label="Validation Loss")

    if early_stop_epoch is not None:     # red line for early stopping
        plt.axvline(x=early_stop_epoch, color='r', linestyle='--', label="Early Stop")
    else:                                # otherise, draw gray linne at the end
        plt.axvline(x=len(train_losses), color='gray', linestyle='--', label="End Epoch")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):  

        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps      # eps: a small constant to avoid sqrt(0) / division by zero;  to prevent potential numerical instability or "division by zero" like issues if the MSE happens to be exactly zero 

    def forward(self, y_pred, y_true):
        mse = self.mse(y_pred, y_true)
        rmse = torch.sqrt(mse + self.eps)
        return rmse

class EarlyStopper:
    def __init__(self, patience=30, min_delta=0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.best_loss = float('inf')
        self.stop = False
        self.stop_epoch = None  # remember which epoch was stopped on (for plotting)

    def early_stop(self, val_loss, epoch=None):

        # Improvement means loss decreased by more than min_delta
        if (self.best_loss - val_loss) > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:               # No meaningful improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                self.stop_epoch = epoch
        return self.stop

def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    y_train,
    y_val,
    val_loader,
    fold_idx,
    checkpoint_dir,
    checkpoint_tracking,
    is_final=False,
    ckpt_tag="regular",   # NEW: "regular" or "best" (final uses is_final)
):
    # Calculate val predictions
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            preds = model(xb).detach().cpu().numpy()
            all_preds.append(preds)

    preds_val = np.concatenate(all_preds, axis=0).squeeze()
    y_val_1d = np.asarray(y_val).squeeze()
    y_train_mean = float(np.asarray(y_train).mean())

    # Calculate performance metrics from val predictions
    checkpoint_rmse = root_mean_squared_error(y_val_1d, preds_val)
    checkpoint_r2 = r2_score(y_val_1d, preds_val)

    denom = np.sum((y_val_1d - y_train_mean) ** 2)
    checkpoint_q2 = (
        1 - np.sum((y_val_1d - preds_val) ** 2) / denom
        if denom != 0
        else float("nan")
    )

    # Create checkpoint filename (CONSISTENT NAMING)
    if is_final:
        checkpoint_filename = f"checkpoint_epoch_{epoch:03d}_final.pt"
    else:
        checkpoint_filename = f"checkpoint_epoch_{epoch:03d}_{ckpt_tag}.pt"

    checkpoint_path_full = Path(checkpoint_dir) / checkpoint_filename

    # Save the checkpoint
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "rmse": float(checkpoint_rmse),
        "r2": float(checkpoint_r2),
        "q2": float(checkpoint_q2),
        "fold_idx": int(fold_idx),
        "is_final": bool(is_final),
        "ckpt_tag": str(ckpt_tag),
    }
    torch.save(checkpoint_data, checkpoint_path_full)

    # Record info for spreadsheet
    checkpoint_type = "final" if is_final else ckpt_tag
    checkpoint_info = {
        "Fold": fold_idx,
        "Epoch": epoch,
        "Checkpoint_Type": checkpoint_type,  # NEW (nice for tracking)
        "Checkpoint_Filename": checkpoint_filename,
        "Checkpoint_Path": str(checkpoint_path_full),
        "Train_Loss": round(float(train_loss), 6),
        "Val_Loss": round(float(val_loss), 6),
        "RMSE": round(float(checkpoint_rmse), 6),
        "R2": round(float(checkpoint_r2), 6),
        "Q2": round(float(checkpoint_q2), 6),
        "Is_Final": bool(is_final),
    }
    checkpoint_tracking.append(checkpoint_info)

    pretty_type = "Final" if is_final else checkpoint_type.capitalize()
    print(f"[Fold {fold_idx}] {pretty_type} checkpoint saved at epoch {epoch} - RMSE: {checkpoint_rmse:.4f}")

    return True

def evaluate_fold(
    trial, fold_idx,
    X_train_scaled, y_train,
    X_val_scaled, y_val,
    hidden_layers,              # kept only so your old function calls do not break
    learning_rate, batch_size, dropout_rate, weight_decay,
    max_epochs=10**9, patience=30, min_delta=0,
    X_test_scaled=None, y_test=None,
    save_checkpoints=True, checkpoint_dir="checkpoints", save_every_n_epochs=15
):
    device = torch.device("cpu")
    print(f"Fold {fold_idx}: Training CNN on cpu")

    checkpoint_tracking = []

    if save_checkpoints:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        fold_checkpoint_dir = checkpoint_path / f"fold_{fold_idx}"
        fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints will be saved to: {fold_checkpoint_dir}")

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor   = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor   = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CNNRegressor(
        dropout_rate=dropout_rate
    ).to(device)

    criterion = RMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())

    train_losses, val_losses = [], []
    stop_epoch = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if save_checkpoints and (epoch % save_every_n_epochs == 0 or epoch == 1):
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                y_train, y_val, val_loader, fold_idx,
                fold_checkpoint_dir, checkpoint_tracking, is_final=False
            )

        should_stop = early_stopper.early_stop(val_loss, epoch=epoch)
        if should_stop:
            stop_epoch = early_stopper.stop_epoch
            print(f"[Fold {fold_idx}] Early stopping at epoch {stop_epoch} (best Val Loss: {best_val_loss:.4f})")

            if save_checkpoints and epoch % save_every_n_epochs != 0 and epoch != 1:
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_loss,
                    y_train, y_val, val_loader, fold_idx,
                    fold_checkpoint_dir, checkpoint_tracking, is_final=True
                )
            break

        if epoch % 50 == 0 or epoch == 1:
            print(f"[Fold {fold_idx}] Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ES {early_stopper.counter}/{patience}")

    model.load_state_dict(best_state)
    model.eval()

    if save_checkpoints and checkpoint_tracking:
        df_checkpoints = pd.DataFrame(checkpoint_tracking)
        spreadsheet_file = fold_checkpoint_dir / f"fold_{fold_idx}_checkpoints_high_test.csv"
        df_checkpoints.to_csv(spreadsheet_file, index=False)
        print(f"[Fold {fold_idx}] Checkpoint spreadsheet saved: {spreadsheet_file}")
        print(f"[Fold {fold_idx}] Total checkpoints saved: {len(checkpoint_tracking)}")

    all_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            preds = model(xb).cpu().numpy()
            all_preds.append(preds)

    preds_val = np.concatenate(all_preds)

    rmse = root_mean_squared_error(y_val, preds_val)
    r2 = r2_score(y_val, preds_val)
    q2 = 1 - np.sum((y_val - preds_val) ** 2) / np.sum((y_val - y_train.mean()) ** 2)

    return rmse, r2, q2, model, train_losses, val_losses, stop_epoch

def set_freeze_mode_cnn(model, freeze_level=0):
    for p in model.parameters():
        p.requires_grad = True

    if freeze_level == 0:
        print("Freeze Level 0: all layers trainable")
        return

    # Your conv blocks are:
    # [Conv1d, BatchNorm1d, ReLU, MaxPool/AdaptiveAvgPool]
    block_size = 4
    n_total_layers = len(model.conv_block)
    n_blocks_total = n_total_layers // block_size
    n_blocks = min(freeze_level, n_blocks_total)

    print(f"Freeze Level {freeze_level}: freezing {n_blocks} CNN block(s)")

    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size

        for i in range(start, end):
            layer = model.conv_block[i]
            for p in layer.parameters():
                p.requires_grad = False
                
def evaluate_fold_TL(
    trial, fold_idx,
    X_train_scaled, y_train,
    X_val_scaled, y_val,
    hidden_layers, dropout_rate,   # hidden_layers kept for compatibility
    learning_rate, weight_decay, batch_size,
    freeze_level,
    baseline_ckpt,
    max_epochs=10**9,
    patience=30,
    min_delta=0.0,
    X_test_scaled=None, y_test=None,
    save_checkpoints=False, checkpoint_dir="checkpoints", save_every_n_epochs=15
):
    device = torch.device("cpu")
    print(f"Fold {fold_idx}: CNN TL on cpu | freeze={freeze_level} | lr={learning_rate:g}")

    checkpoint_tracking = []
    fold_checkpoint_dir = None

    if save_checkpoints:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        fold_checkpoint_dir = checkpoint_path / f"fold_{fold_idx}"
        fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints will be saved to: {fold_checkpoint_dir}")

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_tensor   = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
    y_val_tensor   = torch.tensor(y_val, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = CNNRegressor(
    dropout_rate=dropout_rate
    ).to(device)

    state = torch.load(baseline_ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)

    set_freeze_mode_cnn(model, freeze_level)

    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    criterion = RMSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    train_losses, val_losses = [], []
    stop_epoch = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if save_checkpoints and (epoch % save_every_n_epochs == 0 or epoch == 1):
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                y_train, y_val, val_loader, fold_idx,
                fold_checkpoint_dir, checkpoint_tracking, is_final=False
            )

        if early_stopper.early_stop(val_loss, epoch=epoch):
            stop_epoch = early_stopper.stop_epoch
            print(f"[Fold {fold_idx}] Early stopping at epoch {stop_epoch} (best Val Loss: {best_val_loss:.4f})")

            if save_checkpoints and epoch % save_every_n_epochs != 0 and epoch != 1:
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_loss,
                    y_train, y_val, val_loader, fold_idx,
                    fold_checkpoint_dir, checkpoint_tracking, is_final=True
                )
            break

        if epoch % 50 == 0 or epoch == 1:
            print(f"[Fold {fold_idx}] Epoch {epoch:4d} | Train {train_loss:.4f} | Val {val_loss:.4f} | ES {early_stopper.counter}/{patience}")

    model.load_state_dict(best_state)
    model.eval()

    if save_checkpoints and checkpoint_tracking:
        df_checkpoints = pd.DataFrame(checkpoint_tracking)
        spreadsheet_file = fold_checkpoint_dir / f"fold_{fold_idx}_checkpoints.csv"
        df_checkpoints.to_csv(spreadsheet_file, index=False)
        print(f"[Fold {fold_idx}] Checkpoint spreadsheet saved: {spreadsheet_file}")
        print(f"[Fold {fold_idx}] Total checkpoints saved: {len(checkpoint_tracking)}")

    all_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            all_preds.append(model(xb).cpu().numpy())

    preds_val = np.concatenate(all_preds)

    rmse = root_mean_squared_error(y_val, preds_val)
    r2   = r2_score(y_val, preds_val)
    q2   = 1 - np.sum((y_val - preds_val) ** 2) / np.sum((y_val - y_train.mean()) ** 2)

    return rmse, r2, q2, model, train_losses, val_losses, stop_epoch 