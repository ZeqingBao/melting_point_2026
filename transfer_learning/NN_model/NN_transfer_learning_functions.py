

#Import all the relevant libraries

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

# 1) Define the base folder *relative* to your notebook/script
current_Path = Path.cwd()
BASE = current_Path.parent.parent


# 2) Import the test file
test_file_dir = BASE / "data_curation" / "original_curated_with_embeddings_and_MW" / "test_predictions" / "consensus_without_data_augmentation.csv"

# 3) Load dataframe 
test_df = pd.read_csv(test_file_dir)
test_df.info()
print(test_df.isna().sum())

# 1) Point to the train & val sub-folders
train_dir = BASE / "data_curation" / "original_curated_with_embeddings_and_MW" / "train_without_data_augmentation"
val_dir   = BASE / "data_curation" / "original_curated_with_embeddings_and_MW" / "val"

# 2) Glob and sort
train_files = sorted(train_dir.glob("train*_curated.csv"))
val_files   = sorted(val_dir.glob("val*_curated.csv"))

# 3) Load all training files into a list 
train_dfs = []
for file in train_files: 
    train_df = pd.read_csv(file)
    train_dfs.append(train_df)

# 4) Load all val  files into a list 
val_dfs = []
for file in val_files: 
    val_df = pd.read_csv(file)
    val_dfs.append(val_df)

#5) Combine all the datasets into a final train df and drop SMILES duplicates 
train_combined_df = pd.concat(train_dfs, ignore_index=True)
val_combined_df = pd.concat(val_dfs, ignore_index=True)
final_train_df = pd.concat([train_combined_df, val_combined_df], ignore_index=True)
print ("Before dropping duplicates: " + str(len(final_train_df)))
final_train_df = final_train_df.drop_duplicates(subset="SMILES")
print ("After dropping duplicates: " + str(len(final_train_df)))

#6) View df
final_train_df.info()
print(final_train_df.isna().sum())
print("Number of unique SMILES values: " + str(final_train_df["SMILES"].nunique()))

# 7) Save the final train df
final_train_df.to_csv(BASE / "data_curation" / "original_curated_with_embeddings_and_MW" / "train_without_data_augmentation" / "final_train_df.csv", index=False)


import ast
import numpy as np 
import pandas as pd

#To convert the X (embeddings) and y (MP) into arrays
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

#Source; https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/
#To make the embeddings into clusters 

import ast 
import numpy as np 
import pandas as pd

from sklearn.cluster import KMeans
from kneed import KneeLocator

import matplotlib.pyplot as plt
import seaborn as sns

import time 

#Parse the embeddings (convert embeddings and MP into arrays)
X, y, = parse_embeddings(final_train_df, emb_col="embeddings", target_col="MP")

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

k_opt, wcss, k_list = find_optimal_clusters(X, max_k=15, plot=True, random_state=0)
final_kmeans = KMeans(n_clusters=k_opt, init="k-means++", n_init=10, random_state=0)
final_train_df["Structure_Cluster"] = final_kmeans.fit_predict(X)


from sklearn.manifold import TSNE

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

plot_cluster_tSNE(X, final_train_df["Structure_Cluster"], title="t-SNE Clustering of Structure Clusters", perplexity=30)

def plot_cluster_distribution(df, cluster_col="Structure_Cluster", title="Cluster Distribution"):

    plt.figure(figsize=(10, 6))
    sns.countplot(x=cluster_col, data=df, palette='viridis')
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()  
    
plot_cluster_distribution(final_train_df, cluster_col="Structure_Cluster", title="Distribution of Structure Clusters")


from sklearn.preprocessing import KBinsDiscretizer

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

# Bin the MP values into quantiles and assign to MP_bin column for training_df
final_train_df = bin_mp_values(final_train_df, mp_col="MP", n_bins=4, strategy='quantile')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

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
    cols_to_save = ["SMILES", "MP", "MW", "embeddings", "Structure_Cluster", "MP_bin", "Stratify_Label", "cv_fold"]
    df[cols_to_save].to_csv(save_path, index=False)
    print("Folds saved to final_train_df_with_folds.csv")

    return df, folds

save_path = BASE / "data_curation" / "original_curated_with_embeddings_and_MW" / "train_without_data_augmentation" / "final_train_df_with_folds.csv"
final_train_df, folds = create_stratified_folds(final_train_df, save_path=save_path, n_splits=10, random_state=42)


import numpy as np
import matplotlib.pyplot as plt

def plot_fold_histograms(df, fold,mp_col="MP", cluster_col="Structure_Cluster", fold_col="cv_fold", mp_bins=None, cluster_levels=None, bins=30):
    train = df[df[fold_col] != fold]
    val   = df[df[fold_col] == fold]

    # counts for clusters
    tr_counts = train[cluster_col].value_counts().reindex(cluster_levels, fill_value=0)
    va_counts = val[cluster_col].value_counts().reindex(cluster_levels, fill_value=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # MP histogram 
    ax1.hist(train[mp_col].dropna(), bins=mp_bins, alpha=0.6, label="train")
    ax1.hist(val[mp_col].dropna(),   bins=mp_bins, alpha=0.6, label="val")
    ax1.set_title(f"Fold {fold}: {mp_col} histogram")
    ax1.set_xlabel(mp_col); ax1.set_ylabel("Count"); ax1.legend()

    #  Structure_Cluster counts  
    x = np.arange(len(cluster_levels)); w = 0.4
    ax2.bar(x - w/2, tr_counts.values, width=w, label="train")
    ax2.bar(x + w/2, va_counts.values, width=w, label="val")
    ax2.set_xticks(x); ax2.set_xticklabels(cluster_levels)
    ax2.set_title(f"Fold {fold}: {cluster_col} counts")
    ax2.set_xlabel("Cluster"); ax2.set_ylabel("Count"); ax2.legend()

    plt.tight_layout()
    plt.show()
    
# Apply funcion to every fold (keeps bins/levels consistent)
for f in sorted(final_train_df["cv_fold"].unique()):
        plot_fold_histograms(final_train_df, f,mp_bins=np.histogram_bin_edges(final_train_df["MP"].dropna().to_numpy(), bins=30), cluster_levels=np.sort(final_train_df["Structure_Cluster"].dropna().unique()))

from sklearn.preprocessing import RobustScaler
import numpy as np

# 1) Build fold_data (from my folds list of indices)
fold_data = {f["fold"]: {"train": final_train_df.iloc[f["train_idx"]].copy(), "val":   final_train_df.iloc[f["val_idx"]].copy()} for f in folds}

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

# 3) Call it with fold_data 
train_X, train_y, val_X, val_y, fold_scalers = preload_and_scale_all_folds(fold_data)


def evaluate_fold(trial, fold_idx, X_train_scaled, y_train, X_val_scaled, y_val, learning_rate, batch_size, dropout_rate, weight_decay, max_epochs = 100, patience = 12, device = torch.device("mps"), X_test_scaled=None, y_test=None):

    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32, device = device), torch.tensor(y_train, dtype=torch.float32, device = device))
    train_loader   = DataLoader(train_dataset, batch_size= batch_size, shuffle = True, num_workers=0) #num_workers=0 for MPS compatibility, shuffle = True to shuffle the data before each epoch, batch_size is set by Optuna hyperparameter suggestion)
    
    #model, optimizer  scheduler, loss (set up training components)
    model     = ImprovedNN(input_size = X_train_scaled.shape[1], dropout_rate = dropout_rate).to(device) #A new model is created for each trial run with Optuna, the hyperparameters in each trial is chosen by Optuna, new instance of the model is created, and input size is determined by features in scaled training data, drop out rate is suggested by Optuna
    criterion = nn.HuberLoss(delta=0.5) # More robust to outliers, Defines loss function, how wrong the model's predictions are compared to actual values, loss must be minimized, nn.HuberLoss is a loss function that is less sensitive to extereme errors caused by outlier data points compared to Mean Squared Error 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)##Optimizer adjusts the model's internal weights and biases, AdamW is an optimizer, model.parameters() tells optimizer what to optimize, lr = learning_rate uses suggested learning rate by Optuna, same for weight_decay                     
    
    scheduler_patience = max(5, patience // 2) # Set a minimum patience of 5 epochs to avoid too early stopping
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.5, patience=scheduler_patience) ##Automatically adjusts learning rate during training, mode = "min" monitors metric to minimize, factor = 0.5 if monitored metric doesn't improve for a certain amount of trained_models[fold] = models reduce lr by 1/2, patience is number of epochs to wait before adjustment by factor 

    # Training Loop
    best_rmse = float('inf')
    epochs_no_improve    = 0
    

    #CHANGED TO 50 PER FOLD TO REDUCE TIME
    #-- Model Training ---
    for epoch in range(1, max_epochs + 1): ##for loop represemts the training process for a single model for the current trial, runs for 300 epochs, each epoch indicates that the model has run once, so 12 epoches means the model has been run 12 times 
        model.train()
        for xb, yb in train_loader: ##Put model into training mode (dropout turn on), loops over each batch (xb = input, yb = target)
            optimizer.zero_grad() #Clear out any old gradients (a gradient is a piece of information about how much much to change the weights)
            preds = model(xb) #make predictions
            loss  = criterion(preds, yb) #Calculate loss function
            loss.backward() #Back propogate
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) #Prevents exploding gradients which causes the model to become unstable, limits how big the adjustments to weights can be 
            optimizer.step() #Uses calculated gradients to actually update model's weights and biases trying to reduce loss 

        # --- To validate the model ----
        model.eval() #dropout is off

        with torch.no_grad(): #Tells pytorch not to calculate gradients, don't need them during validation, just want to see how the model performs 
                preds_val = model(torch.tensor(X_val_scaled, dtype=torch.float32).to(device)).cpu().numpy()
        mse = mean_squared_error(y_val, preds_val)
        rmse = root_mean_squared_error(y_val, preds_val)

        scheduler.step(rmse)
        
        # check for early stopping
        if rmse < best_rmse:
            best_rmse = rmse
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

        if epoch % 50 == 0: #prints the most recent progress every 100 epochs
            print(f"Epoch {epoch:4d}), RMSE: {rmse:.4f}, Best RMSE: {best_rmse:.4f}, No Improve: {epochs_no_improve}/{patience}")

    
    model.load_state_dict(best_model_state)

    # Final validation predictions
    model.eval()
    with torch.no_grad():
        preds_val = model(torch.tensor(X_val_scaled, dtype=torch.float32).to(device)).cpu().numpy()
    
   
    # Final metrics
    rmse = root_mean_squared_error(y_val, preds_val)
    r2 = r2_score(y_val, preds_val)
    q2 = 1 - np.sum((y_val - preds_val)**2) / np.sum((y_val - y_train.mean())**2)
    
    return rmse, r2, q2, model


# Step 3: Hyperparameter optimization
def objective(trial): #defines a function that Optuna (automates hyperparameter optimization) that it will call this function
    # Suggest hyperparameters
    dropout_rate   = trial.suggest_float("dropout_rate",   0.2, 0.5)
    learning_rate  = trial.suggest_float("learning_rate",  1e-5, 1e-3, log=True)
    weight_decay   = trial.suggest_float("weight_decay",   1e-6, 1e-2, log=True)
    batch_size     = trial.suggest_categorical("batch_size", [16,32,64])
    max_epochs = trial.suggest_int("max_epochs", 30, 200, step=10) # CHANGED TO 50 TO REDUCE TIME
    patience = trial.suggest_int("patience", 5, 20, step=1) 
   
    device = torch.device("mps")
    
    rmses = []
    #  Runs hyperparameter combination for all folds
    for fold in range(10):
        # Use pre-processed and scaled data
        X_train_scaled = train_X[fold]
        y_train = train_y[fold]
        X_val_scaled = val_X[fold]
        y_val = val_y[fold]
        
        rmse, _, _, _ = evaluate_fold(trial=trial, fold_idx=fold, X_train_scaled=X_train_scaled, y_train=y_train, X_val_scaled=X_val_scaled, y_val=y_val,
            learning_rate=learning_rate, batch_size=batch_size, dropout_rate=dropout_rate, weight_decay=weight_decay, max_epochs=max_epochs,
            patience=patience, device=device)
        
        rmses.append(rmse)
        
    # Return the average RMSE across folds
    avg_rmse = float(np.mean(rmses))
    print(f"Trial {trial.number}: Average RMSE = {avg_rmse:.4f}")
    return avg_rmse

import time 

import optuna
from optuna.importance import get_param_importances
import optuna.visualization as vis 

device = torch.device("mps")

def set_optuna_study(n_trials=100):
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

best_params, study = set_optuna_study(n_trials=100)

import os
from pathlib import Path
import json
import pandas as pd
import torch

def train_and_save_best_general_models(best_params, train_X, train_y, val_X, val_y, save_dir="best_NN_general_fold", device="mps"):
 
    # Convert 'device' to a torch.device object
    device = torch.device(device)

    # Make sure the folder exists
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    results = []          # we'll fill this with one row per fold

    n_folds = len(train_X)

    for fold in range(n_folds):
        print(f"\n--- Training final model for Fold {fold} ---")

        # Grab data for this fold
        X_tr = train_X[fold]
        y_tr = train_y[fold]
        X_va = val_X[fold]
        y_va = val_y[fold]

        # Train and evaluate one model on this fold
        rmse, r2, q2, model = evaluate_fold(trial=None, fold_idx=fold, # not doing Optuna here, just final training, fold_idx=fold,
        X_train_scaled=X_tr, y_train=y_tr, X_val_scaled=X_va, y_val=y_va, learning_rate=best_params["learning_rate"], batch_size=best_params["batch_size"],dropout_rate=best_params["dropout_rate"],weight_decay=best_params["weight_decay"], max_epochs=best_params["max_epochs"], patience = best_params["patience"], device=device)

        # Save the trained model weights to a file
        # Example filename: "fold0.pt"
        model_filename = f"fold{fold}.pt"
        model_path = save_path / model_filename

        torch.save({"model_state_dict": model.state_dict(), "input_size": X_tr.shape[1], "hyperparams": dict(best_params),
            "metrics": {"rmse": float(rmse), "r2": float(r2), "q2": float(q2)}, "fold": fold}, model_path)

        print(f"Saved model to: {model_path}")

        # Keep results for the summary table
        results.append({"Fold": fold, "Train_Size": int(len(y_tr)),"Val_Size": int(len(y_va)), "Val_RMSE": float(rmse), "Val_R2": float(r2),
            "Val_Q2": float(q2)})


    # Make a small summary CSV and JSON
    results_df = pd.DataFrame(results)
    summary_path = save_path / "summary_metrics.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\nWrote summary files to: {summary_path}")

    return results_df

results_df = train_and_save_best_general_models(best_params, train_X, train_y, val_X, save_dir="best_NN_general_fold",   # this folder will be created
    device="mps")                 

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, root_mean_squared_error


def train_full_test_and_eval(train_df, test_df, best_params, emb_col: str = "embeddings", train_target_col: str = "MP", test_target_col: str = "exp MP",
    mw_col: str = "MW", mw_category_col: str = "MW_category_quant", save_pred_csv: str | Path | None = None, save_model_path: str | Path | None = None,):

    device = torch.device("mps")

    # Parse the embeddings and scale the x and y values
    X_tr_raw, y_tr = parse_embeddings(train_df, emb_col=emb_col, target_col=train_target_col)
    X_te_raw, y_te = parse_embeddings(test_df,  emb_col=emb_col, target_col=test_target_col)

    scaler = RobustScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    # Load best hyperparameters
    batch_size   = int(best_params["batch_size"])
    dropout_rate = float(best_params["dropout_rate"])
    lr           = float(best_params["learning_rate"])
    weight_decay = float(best_params["weight_decay"])
    max_epochs   = int(best_params.get("max_epochs"))

    # --- dataloader (train only) ---
    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32,device = device), torch.tensor(y_tr, dtype=torch.float32,device = device))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # --- model/optim/loss ---
    model     = ImprovedNN(input_size=X_tr.shape[1], dropout_rate=dropout_rate).to(device)
    criterion = nn.HuberLoss(delta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train the model ---
    model.train()
    for epoch in range(1, max_epochs + 1):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        if epoch % 50 == 0 or epoch == max_epochs:
            print(f"Epoch {epoch}/{max_epochs} - TrainLoss: {loss.item():.6f}")

    # Test on test set
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_te, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)

    #Compute metrics
    y_true = y_te.reshape(-1)
    overall_rmse = float(root_mean_squared_error(y_true, y_pred))
    overall_r2   = float(r2_score(y_true, y_pred))
    overall_q2   = float(1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_tr))**2))

    # Break down by MW category (tertiles from TRAIN)
    t1 = float(train_df[mw_col].quantile(1/3)) 
    t2 = float(train_df[mw_col].quantile(2/3)) 
    rmse_by_cat = {}
    if mw_category_col in test_df.columns:
        test_cats = test_df[mw_category_col].astype(str).values
        for cat in ["Low", "Intermediate", "High"]:
            idx = (test_cats == cat)
            rmse_by_cat[cat] = float(root_mean_squared_error(y_true[idx], y_pred[idx])) if idx.sum() > 0 else None

    metrics = {"overall": {"rmse": overall_rmse, "r2": overall_r2, "q2": overall_q2},"by_mw": rmse_by_cat,
        "mw_tertiles_from_train": {"t1": t1, "t2": t2}, "epochs_trained": max_epochs,}

    # Save predictions 
    if save_pred_csv is not None:
        out_path = Path(save_pred_csv); out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame({"index": test_df.index, "y_true": y_true,"y_pred": y_pred, mw_col: test_df[mw_col].values,
            mw_category_col: test_df[mw_category_col].values,})
        pred_df.to_csv(out_path, index=False)

    # Save checkpoint (state_dict + scaler + meta)
    if save_model_path is not None:
        save_model_path = Path(save_model_path); save_model_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"model_state_dict": model.state_dict(), "model_class": "ImprovedNN", "input_size": X_tr.shape[1],"dropout_rate": dropout_rate,
            "best_params": best_params,"scaler": scaler, "metrics": metrics, "train_target_col": train_target_col,"test_target_col": test_target_col,
            "emb_col": emb_col, "mw_col": mw_col, "mw_category_col": mw_category_col,}
        torch.save(checkpoint, save_model_path)

    return model, scaler, metrics

final_model, final_scaler, final_metrics = train_full_test_and_eval(train_df=final_train_df, test_df=test_df, best_params=best_params,
    emb_col="embeddings", train_target_col="MP", test_target_col="exp MP", mw_col="MW", mw_category_col="MW_category_quant",
    save_pred_csv=BASE / "model_training" / "NN_model" / "best_NN_general_full" / "test_set_predictions.csv",
    save_model_path=BASE / "model_training" / "NN_model" / "best_NN_general_full" / "final_model.pt",)


