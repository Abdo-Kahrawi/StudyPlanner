"""
train.py - Trains StudyScoreNet with MSE/RMSE, Adam, and compares to average baseline.
Sanity-check required by spec: NN must beat simple average baseline.

Usage:
  python -m nn.train --seed 42          # reads data/processed/{train,test}_data.csv
  python -m nn.train --train_csv train_data.csv --test_csv test_data.csv --epochs 50 --batch 256

Outputs (default folder nn/weights/, which nn.predictor loads from):
  nn_weights.pt, scaler.npz, train_log.json, baseline_comparison.txt
"""
import argparse, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import config
from .model import get_model, INPUT_DIM, count_params
from .effort import FEATURE_ORDER


def set_seed(seed: int) -> None:
    """Same seed -> same split, initial weights and batch order."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data(train_csv, test_csv=None):
    train_df = pd.read_csv(train_csv)
    # features = FEATURE_ORDER, target = target_score
    missing = [c for c in FEATURE_ORDER if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing columns in train_csv: {missing}. Available: {train_df.columns.tolist()}")
    X_train_full = train_df[FEATURE_ORDER].values.astype(np.float32)
    y_train_full = train_df["target_score"].values.astype(np.float32)

    if test_csv:
        test_df = pd.read_csv(test_csv)
        X_test = test_df[FEATURE_ORDER].values.astype(np.float32)
        y_test = test_df["target_score"].values.astype(np.float32)
        return X_train_full, y_train_full, X_test, y_test, train_df, test_df
    else:
        return X_train_full, y_train_full, None, None, train_df, None

def baseline_predict(train_df, test_df=None, mode="cumulative_avg"):
    """
    Simple average baselines for sanity check:
    - cumulative_avg: use cumulative_avg_score if >0 else prev_score else train mean
    - prev_score: use prev_score
    - train_mean: global mean
    Returns MSE/RMSE on test or val
    """
    train_mean = train_df["target_score"].mean()
    def predict_row(row):
        if mode == "cumulative_avg":
            if row.get("cumulative_avg_score",0) > 0:
                return row["cumulative_avg_score"]
            if row.get("prev_score",0) > 0:
                return row["prev_score"]
            return train_mean
        elif mode == "train_mean":
            return train_mean
        else:
            return row.get("prev_score", train_mean)
    return train_mean, predict_row

def train_loop(args):
    set_seed(args.seed)
    X_full, y_full, X_test, y_test, train_df, test_df = load_data(args.train_csv, args.test_csv)

    # train/val split from full if no external test, else val from train
    if X_test is None:
        X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=args.seed)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.15, random_state=args.seed)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    if X_test is not None:
        X_test_s = scaler.transform(X_test)

    # torch datasets
    train_ds = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    model = get_model(input_dim=INPUT_DIM, hidden1=args.h1, hidden2=args.h2, dropout=args.dropout)
    print(f"Model params: {count_params(model)}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = args.patience
    no_imp = 0
    log = []

    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()*len(xb)
        tr_loss /= len(train_ds)

        # val
        model.eval()
        with torch.no_grad():
            v_loss=0
            for xb,yb in val_loader:
                pred = model(xb)
                v_loss += loss_fn(pred,yb).item()*len(xb)
            v_loss/=len(val_ds)
        rmse_tr = np.sqrt(tr_loss)
        rmse_val = np.sqrt(v_loss)
        log.append({"epoch":epoch, "mse_train":tr_loss, "mse_val":v_loss, "rmse_train":rmse_tr, "rmse_val":rmse_val})
        print(f"Epoch {epoch}: train MSE {tr_loss:.3f} RMSE {rmse_tr:.3f} | val MSE {v_loss:.3f} RMSE {rmse_val:.3f}")

        if v_loss < best_val - 1e-4:
            best_val = v_loss
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            no_imp=0
        else:
            no_imp+=1
            if no_imp>=patience:
                print(f"Early stop at {epoch}")
                break

    # restore best
    if best_state:
        model.load_state_dict(best_state)

    # final eval on test if available
    results = {}
    model.eval()
    with torch.no_grad():
        if X_test is not None:
            X_test_t = torch.from_numpy(X_test_s)
            preds = model(X_test_t).numpy()
            mse = ((preds - y_test)**2).mean()
            rmse = np.sqrt(mse)
            results["test_mse"]=float(mse)
            results["test_rmse"]=float(rmse)
            print(f"\n=== TEST ===")
            print(f"NN MSE: {mse:.3f} RMSE: {rmse:.3f}")

    # Baseline comparisons
    print("\n=== Baseline Comparison (Sanity Check) ===")
    train_mean, _ = baseline_predict(train_df)
    # baseline predictions
    def eval_baseline(df, y_true):
        # cumulative avg baseline
        preds = []
        for _, row in df.iterrows():
            if row.get("cumulative_avg_score",0)>0:
                preds.append(row["cumulative_avg_score"])
            elif row.get("prev_score",0)>0:
                preds.append(row["prev_score"])
            else:
                preds.append(train_mean)
        preds=np.array(preds)
        mse = ((preds - y_true)**2).mean()
        return mse, np.sqrt(mse), preds

    target_df = test_df if test_df is not None else train_df.iloc[-len(y_val):]
    y_target = y_test if y_test is not None else y_val

    # choose df for baseline eval
    if test_df is not None:
        b_mse, b_rmse, _ = eval_baseline(test_df, y_test)
    else:
        # build df from val indices - approximate using full tail
        b_mse, b_rmse = float("nan"), float("nan")

    # global mean baseline
    mean_mse = ((y_target - y_target.mean())**2).mean() if y_target is not None else float("nan")
    mean_mse_train = ((y_target - train_mean)**2).mean() if y_target is not None else float("nan")

    print(f"Baseline (cumulative/prev -> mean) MSE {b_mse:.3f} RMSE {b_rmse:.3f}" if not np.isnan(b_mse) else "Baseline N/A for val split")
    print(f"Baseline (global train mean {train_mean:.2f}) MSE {mean_mse_train:.3f} RMSE {np.sqrt(mean_mse_train):.3f}")
    print(f"NN best val MSE {best_val:.3f} RMSE {np.sqrt(best_val):.3f}")

    is_worth = results.get("test_mse", best_val) < mean_mse_train
    print(f"\nNN worth it? {'YES - beats mean baseline' if is_worth else 'NO - check features/training'}")

    # save
    for path in (args.save_path, args.scaler_path, args.log_path, args.comparison_path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    np.savez(args.scaler_path, mean=scaler.mean_, scale=scaler.scale_, feature_order=np.array(FEATURE_ORDER))
    with open(args.log_path,"w") as f:
        json.dump({"log":log, "test":results, "baseline_mse":float(b_mse) if not np.isnan(b_mse) else None,
                   "baseline_rmse":float(b_rmse) if not np.isnan(b_rmse) else None,
                   "train_mean_mse":float(mean_mse_train), "best_val_mse":float(best_val),
                   "worth_it":bool(is_worth)}, f, indent=2)
    with open(args.comparison_path,"w") as f:
        f.write(f"NN vs Baseline - Study Planner\n")
        f.write(f"NN test MSE: {results.get('test_mse','n/a')} RMSE: {results.get('test_rmse','n/a')}\n")
        f.write(f"NN best val MSE: {best_val:.4f} RMSE: {np.sqrt(best_val):.4f}\n")
        f.write(f"Baseline cumulative_avg MSE: {b_mse} RMSE: {b_rmse}\n")
        f.write(f"Baseline train_mean ({train_mean:.2f}) MSE: {mean_mse_train:.4f}\n")
        f.write(f"Conclusion: {'NN WORTH IT' if is_worth else 'NN NOT WORTH - investigate'}\n")

    print(f"\nSaved weights to {args.save_path}, scaler to {args.scaler_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default=str(config.PROCESSED_DIR / "train_data.csv"))
    ap.add_argument("--test_csv", default=str(config.PROCESSED_DIR / "test_data.csv"))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--h1", type=int, default=64)
    ap.add_argument("--h2", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--save_path", default=str(config.WEIGHTS_DIR / "nn_weights.pt"))
    ap.add_argument("--scaler_path", default=str(config.WEIGHTS_DIR / "scaler.npz"))
    ap.add_argument("--log_path", default=str(config.WEIGHTS_DIR / "train_log.json"))
    ap.add_argument("--comparison_path", default=str(config.WEIGHTS_DIR / "baseline_comparison.txt"))
    args = ap.parse_args()
    train_loop(args)
