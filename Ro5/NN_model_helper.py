
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
from NN_model import ImprovedNN 
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

def evaluate_fold(trial, fold_idx, X_train_scaled, y_train, X_val_scaled, y_val, hidden_layers, learning_rate, batch_size, dropout_rate, weight_decay, max_epochs = 10**9, patience = 30, min_delta = 0, X_test_scaled=None, y_test=None, save_checkpoints=True, checkpoint_dir="checkpoints", save_every_n_epochs=15):

    # Set device to CPU
    device = torch.device("cpu")
    print(f"Fold {fold_idx}: Training on cpu")

    #Setup checkpoint directory and tracking list
    checkpoint_tracking = []  # Empty list to track performance metrics for model checkpointing
    
    # If saving checkpoints is true, we are creating the path checkpoints/fold_{fold_idx}
    if save_checkpoints:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        fold_checkpoint_dir = checkpoint_path / f"fold_{fold_idx}"
        fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints will be saved to: {fold_checkpoint_dir}")

    # Convert data to tensors and move to device
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)      # reshape the targets to column vectors to match the model’s predictions and prevent PyTorch from doing sneaky broadcasting
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor   = torch.tensor(y_val,   dtype=torch.float32).to(device)


    # Load the training df
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    #Load the val df 
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)   
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    #model, optimizer  scheduler, loss (set up training components)
    model     = ImprovedNN(input_size = X_train_scaled.shape[1], hidden_layers=hidden_layers, dropout_rate = dropout_rate).to(device) #A new model is created for each trial run with Optuna, the hyperparameters in each trial is chosen by Optuna, new instance of the model is created, and input size is determined by features in scaled training data, drop out rate is suggested by Optuna
    criterion = RMSELoss() # changed from HuberLoss to RMSELoss 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #Optimizer adjusts the model's internal weights and biases, AdamW is an optimizer, model.parameters() tells optimizer what to optimize, lr = learning_rate uses suggested learning rate by Optuna, same for weight_decay                     
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10) #Automatically adjusts learning rate during training, mode = "min" monitors metric to minimize, factor = 0.5 if monitored metric doesn't improve for a certain amount of epochs reduce lr by 1/2, patience is number of epochs to wait before adjustment by factor 
                                               
    # Set up values for early stopping
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())

    train_losses, val_losses = [], []
    stop_epoch = None

        #-- Model Training ---
    for epoch in range(1, max_epochs + 1): ##for loop represemts the training process for a single model for the current trial, runs for 300 epochs, each epoch indicates that the model has run once, so 12 epoches means the model has been run 12 times 
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader: ##Put model into training mode (dropout turn on), loops over each batch (xb = input, yb = target)
            optimizer.zero_grad() #Clear out any old gradients (a gradient is a piece of information about how much much to change the weights)
            preds = model(xb) #make predictions
            loss  = criterion(preds, yb) #Calculate loss function
            loss.backward() #Back propogate
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) #Prevents exploding gradients which causes the model to become unstable, limits how big the adjustments to weights can be 
            optimizer.step() #Uses calculated gradients to actually update model's weights and biases trying to reduce loss 
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # --- To validate the model ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss  = criterion(preds, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)   
       
        scheduler.step(val_loss)
        
        # Update best model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        
        # Saves checkpoints every n epochs (and at epoch 1)
        if save_checkpoints and (epoch % save_every_n_epochs == 0 or epoch == 1):
            # Calculate metrics for this checkpoint
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, y_train, y_val, val_loader, fold_idx, fold_checkpoint_dir, checkpoint_tracking, is_final=False)

        # Check for early stopping
        should_stop = early_stopper.early_stop(val_loss, epoch=epoch)
        if should_stop:
            stop_epoch = early_stopper.stop_epoch
            print(f"[Fold {fold_idx}] Early stopping  at epoch {stop_epoch} (best Val Loss: {best_val_loss:.4f})")

            # Save final checkpoint on early stop (guarantee last snapshot)
            if save_checkpoints and epoch % save_every_n_epochs != 0 and epoch != 1:
                save_checkpoint(model, optimizer, epoch, train_loss, val_loss, y_train, y_val, 
                              val_loader, fold_idx, fold_checkpoint_dir, checkpoint_tracking, is_final=True)
            
            break

        # Log progress every 50 epochs or first epoch
        if epoch % 50 == 0 or epoch == 1:
            print(f"[Fold {fold_idx}] Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ES {early_stopper.counter}/{patience}")
    
    # Load best model state (from epoch with lowest val loss)
    model.load_state_dict(best_state)
    model.eval()  

    # Save the checkpoint tracking spreadsheet for this fold
    if save_checkpoints and checkpoint_tracking:
        df_checkpoints = pd.DataFrame(checkpoint_tracking)
        spreadsheet_file = fold_checkpoint_dir / f"fold_{fold_idx}_checkpoints_high_test.csv"
        df_checkpoints.to_csv(spreadsheet_file, index=False)
        print(f"[Fold {fold_idx}] Checkpoint spreadsheet saved: {spreadsheet_file}")
        print(f"[Fold {fold_idx}] Total checkpoints saved: {len(checkpoint_tracking)}")
  

    # Final metrics calculation (using the best model)
    all_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            preds = model(xb).cpu().numpy()
            all_preds.append(preds)
    preds_val = np.concatenate(all_preds)
    
    rmse = root_mean_squared_error(y_val, preds_val)
    r2 = r2_score(y_val, preds_val)
    q2 = 1 - np.sum((y_val - preds_val)**2) / np.sum((y_val - y_train.mean())**2)
 
    return rmse, r2, q2, model, train_losses, val_losses, stop_epoch


def set_freeze_mode(model, freeze_level=0):
    """
    freeze_level:
        0 = train all layers
        1 = freeze first hidden block
        2 = freeze first two hidden blocks
        3 = freeze first three hidden blocks (if present)
    """
    block_size = 4  # [Linear, BatchNorm, ReLU, Dropout] per hidden layer

    # Unfreeze everything first
    for p in model.parameters():
        p.requires_grad = True

    if freeze_level == 0:
        print("Freeze Level 0: all layers trainable")
        return

    # How many blocks actually exist?
    n_blocks_total = len(model.network) // block_size  # e.g., 3 blocks for [256,128,64]
    n_blocks = min(freeze_level, n_blocks_total)
    print(f"Freeze Level {freeze_level}: freezing {n_blocks} block(s)")

    for b in range(n_blocks):
        start = b * block_size
        for i in range(start, start + 2):  # [Linear, BatchNorm]
            layer = model.network[i]
            for p in layer.parameters():
                p.requires_grad = False

def evaluate_fold_TL(
    trial, fold_idx,
    X_train_scaled, y_train,
    X_val_scaled,   y_val,
    hidden_layers, dropout_rate,
    learning_rate, weight_decay, batch_size,
    freeze_level,                # 0,1,2,3 → how many feature blocks to freeze
    baseline_ckpt,               # path to medium-range baseline .pth
    max_epochs = 10**9,
    patience   = 30,
    min_delta  = 0.0,
    X_test_scaled=None, y_test=None,
    save_checkpoints=False, checkpoint_dir="checkpoints", save_every_n_epochs=15
):
    """
    Transfer-learning fold trainer using a SINGLE learning rate (no param groups).
    Expects pre-scaled numpy arrays (no scaling here).

    Returns:
        rmse, r2, q2, model, train_losses, val_losses, stop_epoch
    """
    device = torch.device("cpu")
    print(f"Fold {fold_idx}: TL on cpu | freeze={freeze_level} | lr={learning_rate:g}")

    # checkpoint bookkeeping
    checkpoint_tracking = []
    fold_checkpoint_dir = None
    if save_checkpoints:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        fold_checkpoint_dir = checkpoint_path / f"fold_{fold_idx}"
        fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints will be saved to: {fold_checkpoint_dir}")

    # tensors/loaders (inputs are already scaled)
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train,        dtype=torch.float32, device=device)
    X_val_tensor   = torch.tensor(X_val_scaled,   dtype=torch.float32, device=device)
    y_val_tensor   = torch.tensor(y_val,          dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    # --- model: same arch as baseline; load baseline weights ---
    model = ImprovedNN(
        input_size = X_train_scaled.shape[1],
        hidden_layers = hidden_layers,
        dropout_rate  = dropout_rate
    ).to(device)

    state = torch.load(baseline_ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)

    # --- freeze policy ---
    set_freeze_mode(model, freeze_level)

    # --- optimizer: SINGLE LR over all trainable params ---
    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # loss & scheduler & early stopping (same semantics as baseline)
    criterion = RMSELoss()  # your existing class
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    train_losses, val_losses = [], []
    stop_epoch = None

    # --- training loop ---
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)                 # shape [B,] from your ImprovedNN
            loss  = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss  = criterion(preds, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        # save periodic checkpoints
        if save_checkpoints and (epoch % save_every_n_epochs == 0 or epoch == 1):
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                y_train, y_val, val_loader, fold_idx,
                fold_checkpoint_dir, checkpoint_tracking, is_final=False
            )

        # early stopping
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

    # restore best
    model.load_state_dict(best_state)
    model.eval()

    # optional: export the checkpoint-tracking spreadsheet
    if save_checkpoints and checkpoint_tracking:
        df_checkpoints = pd.DataFrame(checkpoint_tracking)
        spreadsheet_file = fold_checkpoint_dir / f"fold_{fold_idx}_checkpoints.csv"
        df_checkpoints.to_csv(spreadsheet_file, index=False)
        print(f"[Fold {fold_idx}] Checkpoint spreadsheet saved: {spreadsheet_file}")
        print(f"[Fold {fold_idx}] Total checkpoints saved: {len(checkpoint_tracking)}")

    # final metrics on validation
    all_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            all_preds.append(model(xb).cpu().numpy())
    preds_val = np.concatenate(all_preds)

    rmse = root_mean_squared_error(y_val, preds_val)
    r2   = r2_score(y_val, preds_val)
    q2   = 1 - np.sum((y_val - preds_val)**2) / np.sum((y_val - y_train.mean())**2)

    return rmse, r2, q2, model, train_losses, val_losses, stop_epoch
