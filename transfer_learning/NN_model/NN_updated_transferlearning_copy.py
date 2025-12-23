import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import root_mean_squared_error

import optuna

from pathlib import Path

import pickle

from NN_model import ImprovedNN



def parse_embeddings(df, emb_col = "embeddings", target_col = "MP"):
    X_Emb_List = []
    y_MP_list = []

    for _, row in df.iterrows():
        emb_str = row[emb_col]
        target = row[target_col]

        # do not include if missing
        if pd.isna(emb_str) or pd.isna(target):
            continue
       
        #grab what is in the [..] in the tensors
        start = emb_str.find('[')
        end = emb_str.find(']')
        if start < 0 or end < 0:
            continue

        list_str = emb_str[start:end+1]
            
         # safely evaluate into a Python list, then cast to np.array
        try:
            emb = np.array(ast.literal_eval(list_str), dtype=np.float32)
        except (ValueError, SyntaxError) as e:
            raise ValueError ("Parsing failed!") from e

        X_Emb_List.append(emb)
        y_MP_list.append(target)
    
    X = np.vstack(X_Emb_List)
    y = np.array(y_MP_list, dtype=np.float32)
    return X, y

# Function to find optimal number of clusters using the elbow method
def find_optimal_clusters(X, max_k=15, plot=True, random_state=0):

    start_time = time.perf_counter()
    wcss = []
    k_list = list(range(2, max_k + 1))
    for k in k_list:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Use KneeLocator to find the optimal number of clusters
    kn = KneeLocator(range(2, max_k + 1), wcss, curve='convex', direction='decreasing')
    k_opt = kn.elbow

    knee_idx = k_list.index(k_opt) 
    knee_y = wcss[knee_idx]

    # Plot the inertia to visualize the elbow
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, wcss, marker='o', label='WCSS')

    plt.axvline(x=k_opt, color='r', linestyle='--', label=f'Optimal k={k_opt}')
    plt.scatter(k_opt, knee_y, color='red', s=100, zorder=5)
    plt.legend()
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel("WCSS")
    plt.grid()
    plt.show()

    
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) / 60.0
    print(f"Elbow method completed in {elapsed_time:.2f} minutes")
    print(f"Optimal number of clusters: {k_opt}")

    return k_opt, wcss, k_list

def plot_cluster_tSNE(X, labels, title="t-SNE Clustering", perplexity=30):
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels, palette='viridis', s=50)
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.show()


def plot_cluster_distribution(df, cluster_col="Structure_Cluster", title="Cluster Distribution"):

    plt.figure(figsize=(10, 6))
    sns.countplot(x=cluster_col, data=df, palette='viridis')
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()  
    

def bin_mp_values(df, mp_col="MP", n_bins=4, strategy='quantile'):

    # Bin the MP values into n_bins using KBinsDiscretizer
    binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    df["MP_bin"] = binner.fit_transform(df[mp_col].values.reshape(-1, 1)).astype(int)

    # Print the bin edges
    bin_edges = binner.bin_edges_[0]
    for i in range(len(bin_edges) - 1):
        print("Bin", i, "goes from", round(bin_edges[i], 2), "to", round(bin_edges[i + 1], 2))
    print(df["MP_bin"].value_counts().sort_index())
    
    return df

def create_stratified_folds(df, save_path, n_splits=10, random_state=42):

    # 1) Build the stratify label by combining the structure cluster and MP bin
    df["Stratify_Label"] = df["Structure_Cluster"].astype(str) + "_" + df["MP_bin"].astype(str)

    # 2) Visualize counts per stratify label 
    counts = df["Stratify_Label"].value_counts()
    plt.figure(figsize=(10, 6))
    sns.countplot(y="Stratify_Label", data=df, order=counts.index, palette='viridis')
    plt.title("Count of Samples per Stratify Label")
    plt.xlabel("Count")
    plt.ylabel("Stratify Label")
    plt.grid(axis='x')
    plt.show()

    # 3) Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # To track which fold each row belongs to in the validation set
    fold_ids = np.empty(len(df), dtype=int)
    #To track train and val folds
    folds = []

    for fold, (train_index, val_index) in enumerate(skf.split(df, df["Stratify_Label"])):
        X_train, X_val = df.iloc[train_index], df.iloc[val_index]
        y_train, y_val = X_train["MP"], X_val["MP"]

        folds.append({"fold": fold, "train_idx": train_index, "val_idx": val_index})
        fold_ids[val_index] = fold  # mark the validation rows' fold id

    # 4) Save the fold id mapping
    df["cv_fold"] = fold_ids
    save_path = Path(save_path)
    cols_to_save = ["SMILES", "MP", "MW", "embeddings", "Structure_Cluster", "MP_bin", "Stratify_Label", "cv_fold"]
    df[cols_to_save].to_csv(save_path, index=False)
    print("Folds saved to final_train_df_with_folds.csv")

    #5) Verify if for each iteration, the whole train df is being used
    for k in range(n_splits):
        val_n = int((df["cv_fold"] == k).sum())
        train_n = int((df["cv_fold"] != k).sum())
        print(f"Fold {k}: train={train_n} | val={val_n}")

    return df, folds

def plot_fold_histograms(df, fold,mp_col="MP", cluster_col="Structure_Cluster", fold_col="cv_fold", mp_bins=None, cluster_levels=None, bins=30):
    train = df[df[fold_col] != fold]
    val   = df[df[fold_col] == fold]

    # Frequency for clusters
    tr_freq = train[cluster_col].value_counts(normalize=True).reindex(cluster_levels, fill_value=0)
    val_freq = val[cluster_col].value_counts(normalize=True).reindex(cluster_levels, fill_value=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # MP histogram 
    ax1.hist(train[mp_col].dropna(), bins=mp_bins, alpha=0.6, label="train", density=True)
    ax1.hist(val[mp_col].dropna(),   bins=mp_bins, alpha=0.6, label="val", density=True)
    ax1.set_title(f"Fold {fold}: {mp_col} distribution (frequency)")
    ax1.set_xlabel(mp_col); ax1.set_ylabel("Frequency"); ax1.legend()

    #  Structure_Cluster Frequencies  
    x = np.arange(len(cluster_levels)); w = 0.4
    ax2.bar(x - w/2, tr_freq.values, width=w, label="train")
    ax2.bar(x + w/2,  val_freq.values, width=w, label="val")
    ax2.set_xticks(x); ax2.set_xticklabels(cluster_levels)
    ax2.set_title(f"Fold {fold}: {cluster_col} frequencies")
    ax2.set_xlabel("Cluster"); ax2.set_ylabel("Proportion"); ax2.legend()

    plt.tight_layout()
    plt.show()


# 2) Preload + scale per fold (fit scaler on TRAIN only)
def preload_and_scale_all_folds(fold_data, emb_col="embeddings", target_col="MP"):
    train_X, train_y, val_X, val_y, scalers = [], [], [], [], []

    print("Pre-processing and scaling all fold data...")
    # Pull train and val dfs per fold
    for fold in sorted(fold_data.keys()):
        train_df = fold_data[fold]["train"]
        val_df   = fold_data[fold]["val"]

        # Parse embeddings -> X (array), y (array/Series)
        X_tr, y_tr = parse_embeddings(train_df, emb_col=emb_col, target_col=target_col)
        X_va, y_va = parse_embeddings(val_df,   emb_col=emb_col, target_col=target_col)

        # Fit scaler on training ONLY (no leakage)
        scaler = RobustScaler().fit(X_tr)

        # Transform both train + val
        train_X.append(scaler.transform(X_tr))
        train_y.append(np.asarray(y_tr))
        val_X.append(scaler.transform(X_va))
        val_y.append(np.asarray(y_va))
        scalers.append(scaler)

        print(f"Fold {fold}: {X_tr.shape[0]} train | {X_va.shape[0]} val | dim={X_tr.shape[1]}")

    return train_X, train_y, val_X, val_y, scalers

# Early Stopping Based on Validation Loss
class EarlyStopper:
    # If the val loss has not been improved (i.e. stayed the same or got worse) for 12 epochs in a row, the training of the model is stopped.

    def __init__(self, patience=12, min_delta=0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.best_loss = float('inf')
        self.stop = False
        self.stop_epoch = None  # remember which epoch we stopped on (for plotting)

    def early_stop(self, val_loss, epoch=None):

        #For each epoch, checks if the validation loss has improved, we reset the counter.
        # We increase the counter if there is no improvement. Once the counter reaches the patience, we stop and remember the epoch.

        # Improvement means the loss decreased by more than min_delta
        if (self.best_loss - val_loss) > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No meaningful improvement this epoch
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                self.stop_epoch = epoch
        return self.stop

def plot_training_progress(train_losses, val_losses, early_stop_epoch=None, title="Training and Validation Loss"):
    #train_losses / val_losses: lists of per-epoch average loss values.
    #early_stop_epoch: integer epoch number (1-based) where early stopping triggered (optional).

    epochs = range(1, len(train_losses) + 1) 
    
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses,   label="Validation Loss")

    if early_stop_epoch is not None:
        plt.axvline(x=early_stop_epoch, color='r', linestyle='--', label="Early Stop")
    else:
    # draw line at last epoch
        plt.axvline(x=len(train_losses), color='gray', linestyle='--', label="End Epoch")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_fold(trial, fold_idx, X_train_scaled, y_train, X_val_scaled, y_val, learning_rate, batch_size, dropout_rate, weight_decay, max_epochs = 100, patience = 12, min_delta = 0.03, X_test_scaled=None, y_test=None):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Fold {fold_idx}: Training on GPU)")
    else:
        device = torch.device("cpu")
        print(f"Fold {fold_idx}: Training on cpu")

    # Convert data to tensors and move to device
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Load the training df
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    #Load the val df 
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)   
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    #model, optimizer  scheduler, loss (set up training components)
    model     = ImprovedNN(input_size = X_train_scaled.shape[1], dropout_rate = dropout_rate).to(device) #A new model is created for each trial run with Optuna, the hyperparameters in each trial is chosen by Optuna, new instance of the model is created, and input size is determined by features in scaled training data, drop out rate is suggested by Optuna
    criterion = nn.HuberLoss(delta=0.5) # More robust to outliers, Defines loss function, how wrong the model's predictions are compared to actual values, loss must be minimized, nn.HuberLoss is a loss function that is less sensitive to extereme errors caused by outlier data points compared to Mean Squared Error 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #Optimizer adjusts the model's internal weights and biases, AdamW is an optimizer, model.parameters() tells optimizer what to optimize, lr = learning_rate uses suggested learning rate by Optuna, same for weight_decay                     
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10) #Automatically adjusts learning rate during training, mode = "min" monitors metric to minimize, factor = 0.5 if monitored metric doesn't improve for a certain amount of epochs reduce lr by 1/2, patience is number of epochs to wait before adjustment by factor 
                                               
    # Set up values for early stopping
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())

    train_losses, val_losses = [], []
    stop_epoch = None

    
    #CHANGED TO 50 PER FOLD TO REDUCE TIME
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

        # Check for early stopping
        should_stop = early_stopper.early_stop(val_loss, epoch=epoch)
        if should_stop:
            stop_epoch = early_stopper.stop_epoch
            print(f"[Fold {fold_idx}] Early stopping at epoch {stop_epoch} (best Val Loss: {best_val_loss:.4f})")
            break

        # Log progress every 50 epochs or first epoch
        if epoch % 50 == 0 or epoch == 1:
            print(f"[Fold {fold_idx}] Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ES {early_stopper.counter}/{patience}")


    # Metrics for validation
    with torch.no_grad():
        preds_val = torch.cat([model(xb).cpu() for xb, _ in val_loader]).numpy()
    rmse = root_mean_squared_error(y_val, preds_val)
    r2   = r2_score(y_val, preds_val)
    q2   = 1 - np.sum((y_val - preds_val)**2) / np.sum((y_val - y_train.mean())**2)
 
    return rmse, r2, q2, model, train_losses, val_losses, stop_epoch



def objective(trial):
    # Suggest hyperparameters
    dropout_rate  = trial.suggest_float("dropout_rate",  0.2, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay  = trial.suggest_float("weight_decay",  1e-6, 1e-2, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [16,32,64])

    # Start trial timer
    start = time.perf_counter()

    # Prefer MPS if available; fall back safely
    device = torch.device("mps")

    rmses = []

    # Run this hyperparameter combo across all folds
    for fold in range(10):
        X_train_scaled = train_X[fold]
        y_train        = train_y[fold]
        X_val_scaled   = val_X[fold]
        y_val          = val_y[fold]

        rmse, _, _, _,  _, _, _ = evaluate_fold(trial=trial, fold_idx=fold,X_train_scaled=X_train_scaled, y_train=y_train,
            X_val_scaled=X_val_scaled,   y_val=y_val, learning_rate=learning_rate, batch_size=batch_size,
            dropout_rate=dropout_rate,   weight_decay=weight_decay)
        rmses.append(rmse)

    elapsed = (time.perf_counter() - start) / 60.0
    trial_times.append(elapsed)
    print(f"Trial {trial.number} finished in {elapsed:.2f} minutes")

    # Return the metric Optuna should minimize
    avg_rmse = float(np.mean(rmses))
    print(f"Trial{trial.number}: Average RMSE = {avg_rmse:.4f}")
    return avg_rmse



def set_optuna_study(n_trials): #SET TO 100 AFTERWARDS
    start_time = time.perf_counter()
    print("Setting up Optuna study...")
    # 1) Set up the Optuna study
    study = optuna.create_study(direction='minimize') #minimize return loss
    study.optimize(objective, n_trials=n_trials)  #CHANGE TO 100 AFTER TESTING
    # 2) Identify the best hyperparameters
    best_params = study.best_params #best_params holds the dropout, learning rate, and weight decay that gave the lowest best_val_loss
    print("Best hyperparameters:", best_params)
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) / 60.0
    print(f"Optuna study completed in {elapsed_time:.2f} minutes")
    
    return best_params, study

def train_and_save_best_general_models(best_params, train_X, train_y, val_X, val_y, save_dir="best_NN_general_fold", device="mps"):
 
    # Convert 'device' to a torch.device object
    device = torch.device(device)

    # Make sure the folder exists
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    results = [] # one row per fold
    n_folds = len(train_X)

    for fold in range(n_folds):
        print(f"\n--- Training final model for Fold {fold} ---")
        start_time = time.perf_counter()

        # Grab data for this fold
        X_tr = train_X[fold]
        y_tr = train_y[fold]
        X_va = val_X[fold]
        y_va = val_y[fold]

        # Train and evaluate one model on this fold
        rmse, r2, q2, model, train_losses, val_losses, stop_epoch = evaluate_fold(trial=None, fold_idx=fold, # not doing Optuna here, just final training, fold_idx=fold,
        X_train_scaled=X_tr, y_train=y_tr, X_val_scaled=X_va, y_val=y_va, learning_rate=best_params["learning_rate"], batch_size=best_params["batch_size"],dropout_rate=best_params["dropout_rate"],weight_decay=best_params["weight_decay"])

        plot_training_progress(train_losses, val_losses, early_stop_epoch=stop_epoch, title=f"Best Params – Fold {fold} Loss Curve")
        
        # Save the trained model weights to a file
        # Example filename: "fold0.pt"
        model_filename = f"fold{fold}.pt"
        model_path = save_path / model_filename

        torch.save({"model_state_dict": model.state_dict(), "input_size": X_tr.shape[1], "hyperparams": dict(best_params),
            "metrics": {"rmse": float(rmse), "r2": float(r2), "q2": float(q2)}, "fold": fold}, model_path)

        print(f"Saved model to: {model_path}")
    
        end_time = time.perf_counter()   
        elapsed_time = (end_time - start_time) / 60.0
        print(f"Model trained and evaluated on validation set{fold} for {elapsed_time:.2f} minutes")

        # Keep results for the summary table
    results.append({"Fold": fold, "Train_Size": int(len(y_tr)),"Val_Size": int(len(y_va)), "Val_RMSE": float(rmse), "Val_R2": float(r2),
    "Val_Q2": float(q2)})


    # Make a small summary CSV 
    results_df = pd.DataFrame(results)
    summary_path = save_path / "summary_metrics.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\nWrote summary files to: {summary_path}")

    return results_df
              
def train_full_test_and_eval(train_df, test_df, best_params, emb_col = "embeddings", train_target_col = "MP", test_target_col = "exp MP",
    mw_col = "MW", mw_category_col = "MW_category_quant", save_pred_csv: str | Path | None = None, save_model_path: str | Path | None = None,
    max_epochs = 100, patience = 12, min_delta = 0):


    # Set the device to MPS and print if using GPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Training on GPU")
    else:
        device = torch.device("cpu")
        print("Training on CPU")

    # Split the train df into train and val by Stratification Label
    tr_df, val_df = train_test_split(train_df, test_size=0.20, random_state=42, stratify=train_df["Stratify_Label"] if "Stratify_Label" in train_df.columns else None)
    print(f"Train rows: {len(tr_df)}  |  Val rows: {len(val_df)}")

    # Plot MP and Cluster distribution for train and Val (modified from previous function)
    mp_bins = np.histogram_bin_edges(train_df["MP"].dropna().to_numpy(), bins=30)
    cluster_levels = np.sort(train_df["Structure_Cluster"].dropna().unique())

    tr_freq = tr_df["Structure_Cluster"].value_counts(normalize=True).reindex(cluster_levels, fill_value=0)
    val_freq = val_df["Structure_Cluster"].value_counts(normalize=True).reindex(cluster_levels, fill_value=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(tr_df["MP"].dropna(),  bins=mp_bins, alpha=0.6, label="train", density=True)
    ax1.hist(val_df["MP"].dropna(), bins=mp_bins, alpha=0.6, label="val",   density=True)
    ax1.set_xlabel("MP"); ax1.set_ylabel("Frequency"); ax1.legend()
    x = np.arange(len(cluster_levels)); w = 0.4
    ax2.bar(x - w/2, tr_freq.values,  width=w, label="train")
    ax2.bar(x + w/2, val_freq.values, width=w, label="val")
    ax2.set_xticks(x); ax2.set_xticklabels(cluster_levels, rotation=45, ha="right")
    ax2.set_xlabel("Cluster"); ax2.set_ylabel("Proportion"); ax2.legend()
    plt.tight_layout(); plt.show()

    # Parse and scale the embeddings
    X_tr_raw,  y_tr  = parse_embeddings(tr_df,  emb_col=emb_col, target_col=train_target_col)
    X_val_raw, y_val = parse_embeddings(val_df, emb_col=emb_col, target_col=train_target_col)
    X_te_raw,  y_te  = parse_embeddings(test_df, emb_col=emb_col, target_col=test_target_col)

    scaler = RobustScaler().fit(X_tr_raw)  # fit on TRAIN only
    X_tr_scaled  = scaler.transform(X_tr_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_te_scaled  = scaler.transform(X_te_raw)

    # Load the best hyperparams from previous 10-fold CV
    batch_size   = int(best_params["batch_size"])
    dropout_rate = float(best_params["dropout_rate"])
    lr           = float(best_params["learning_rate"])
    weight_decay = float(best_params["weight_decay"])

    # Convert to tensors and load the datasets
    X_train_tensor = torch.tensor(X_tr_scaled,  dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_tr,         dtype=torch.float32).to(device)
    X_val_tensor   = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor   = torch.tensor(y_val,        dtype=torch.float32).to(device)
    X_test_tensor  = torch.tensor(X_te_scaled,  dtype=torch.float32).to(device)
    y_test_tensor  = torch.tensor(y_te,         dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_val_tensor,   y_val_tensor), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(X_test_tensor,  y_test_tensor), batch_size=batch_size, shuffle=False, num_workers=0)

    # Set model / loss / optim 
    model = ImprovedNN(input_size=X_tr_scaled.shape[1], dropout_rate=dropout_rate).to(device)
    criterion = nn.HuberLoss(delta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    # Do early stopping based on val loss
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    train_losses, val_losses = [], []
    stop_epoch = None

    # Train loop 
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # validate
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

        # early stopping
        if early_stopper.early_stop(val_loss, epoch=epoch):
            stop_epoch = early_stopper.stop_epoch
            print(f"Early stopping at epoch {stop_epoch} (best Val Loss: {best_val_loss:.4f})")
            break

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ES {early_stopper.counter}/{patience}")

    # Restore best weights before evaluation 
    if best_state is not None:
        model.load_state_dict(best_state)

    #Print validation metrics
    with torch.no_grad():
        preds_val = torch.cat([model(xb).cpu() for xb, _ in val_loader]).numpy()
    rmse_val = root_mean_squared_error(y_val, preds_val)
    r2_val   = r2_score(y_val, preds_val)
    q2_val   = 1 - np.sum((y_val - preds_val)**2) / np.sum((y_val - y_tr.mean())**2)
    print(f"VAL metrics: RMSE={rmse_val:.4f}  R2={r2_val:.4f}  Q2={q2_val:.4f}")

    # Plot train vs val loss to check for overfitting
    plot_training_progress(train_losses, val_losses, early_stop_epoch=stop_epoch, title="Training vs Validation Loss")

    # Evaluate on test set (no early stopping)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            out = model(xb).cpu().numpy().reshape(-1)
            preds.append(out)
    y_pred = np.concatenate(preds)
    y_true = y_te.reshape(-1)

    overall_rmse = float(root_mean_squared_error(y_true, y_pred))
    overall_r2   = float(r2_score(y_true, y_pred))
    overall_q2   = float(1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2))

    metrics = {"TEST overall": {"rmse": overall_rmse, "r2": overall_r2, "q2": overall_q2}, "best_val_loss": float(best_val_loss),
        "epochs_trained": len(train_losses),"val": {"rmse": float(rmse_val), "r2": float(r2_val), "q2": float(q2_val)},}
    print(f"TEST metrics: RMSE={overall_rmse:.4f}  R2={overall_r2:.4f}  Q2={overall_q2:.4f}")

    # Plot residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("True Values")
    plt.ylabel("Residuals (True - Pred)")
    plt.title("Residual Plot (Test Set)")
    plt.tight_layout()
    plt.show()

    # Save predictions
    if save_pred_csv is not None:
        out_path = Path(save_pred_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "index": test_df.index,
            "y_true": y_true,
            "y_pred": y_pred,
            mw_col: test_df[mw_col].values if mw_col in test_df.columns else np.nan,
            mw_category_col: test_df[mw_category_col].values if mw_category_col in test_df.columns else np.nan,
        }).to_csv(out_path, index=False)
        print(f"Saved predictions → {out_path}")

     # Save final TEST model checkpoint (this is the model actually used on test set)
    if save_model_path is not None:
        save_model_path = Path(save_model_path)
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"model_state_dict": model.state_dict(),  # <- weights used on test set
            "model_class": "ImprovedNN", "input_size": X_tr_scaled.shape[1], "dropout_rate": dropout_rate,
            "best_params": best_params, "scaler": scaler, "metrics": metrics,
            "cols": {"emb_col": emb_col, "train_target_col": train_target_col,"test_target_col": test_target_col,},}
        torch.save(checkpoint, save_model_path)
        print(f"Saved TEST model checkpoint → {save_model_path}")

    return model, scaler, metrics, stop_epoch, y_true, y_pred

if __name__ == "__main__":
        # This code will only run when helper_functions.py is executed directly
        print("Running helper_functions.py directly.")
        parse_embeddings, find_optimal_clusters, plot_cluster_tSNE, plot_cluster_distribution, bin_mp_values, create_stratified_folds, plot_fold_histograms, preload_and_scale_all_folds, EarlyStopper, plot_training_progress, evaluate_fold, objective, set_optuna_study, train_and_save_best_general_models, train_full_test_and_eval  
