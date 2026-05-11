

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

from pathconfig import get_checkpoints_dir, DEFAULT_WF_SCENARIO
from spo_loss import SPOPlusFunction
from training import determine_device
from util import isonow





def SPOPlusFunction_apply(pred, true_wind, prices, times_ns, oracle_val, scenario, target_scaler=None):
    return SPOPlusFunction.apply(pred, true_wind, prices, times_ns, oracle_val, scenario, target_scaler)


def _spo_checkpoint_path(
    model_name: str,
    wf_scenario: str = DEFAULT_WF_SCENARIO,
    scenario_name: str = "default",
) -> Path:
    """Checkpoint path for an SPO+-trained model.
    Includes scenario name so models trained on different LP scenarios don't overwrite each other.
    """
    tag = f"{model_name}_spo_{scenario_name}_best.pt"
    return get_checkpoints_dir(wf_scenario) / tag


def _val_profit(model, val_dataset, scenario, device, target_scaler=None) -> tuple[float, float]:
    """Run LP on every val window and return (mean_profit, mean_value_gap).

    For each window: predict wind -> solve LP -> record achieved profit.
    Value gap = (oracle_val - achieved_profit) / oracle_val.
    Requires val_dataset to be a DFLDataset with oracle_values set.
    """
    from optimization import run_optimization

    if val_dataset.oracle_values is None:
        raise RuntimeError(
            "val_dataset.oracle_values must be set for profit-based validation. "
            "Call val_dataset.set_oracle_values(precompute_oracle_values(...)) first."
        )

    model.eval()
    profits = []
    gaps    = []

    with torch.no_grad():
        for i in tqdm.trange(len(val_dataset), desc="Val profit", leave=False):
            x, _, prices, times_ns, oracle_val = val_dataset[i]
            pred = model(x.unsqueeze(0).to(device))   # (1, N_PREDICT, 1)
            pred_np = pred.squeeze().cpu().numpy()


            if target_scaler is not None:
                scale   = float(target_scaler.scale_[0])
                mean    = float(target_scaler.mean_[0])
                pred_mw = pred_np * scale + mean
            else:
                pred_mw = pred_np

            time_idx  = pd.DatetimeIndex(times_ns.numpy().astype("datetime64[ns]"))
            pred_ser  = pd.Series(np.maximum(0.0, pred_mw), index=time_idx)
            price_ser = pd.Series(prices.numpy(), index=time_idx)

            opt = run_optimization(pred_ser, price_ser, scenario=scenario, silent=True)
            profit = opt["objective_value"]
            del opt["model"]

            oracle = oracle_val.item()
            profits.append(profit)
            gaps.append((oracle - profit) / oracle if oracle != 0 else 0.0)

    return float(np.mean(profits)), float(np.mean(gaps))





def train_model_spo(
    model_name: str,
    model: nn.Module,
    train_dataset,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    val_metric: str = "mse",
    val_loader: DataLoader = None,    # required when val_metric="mse"
    val_dataset = None,               # required when val_metric="profit" (DFLDataset)
    scenario: dict = None,
    wf_scenario: str = DEFAULT_WF_SCENARIO,
    dfl_batch_size: int = 16,
    early_stop_patience: int = 20,
    target_scaler = None,
) -> dict:
    """Train `model` with the SPO+ loss on train_dataset.

    """
    if val_metric not in ("mse", "profit"):
        raise ValueError(f"val_metric must be 'mse' or 'profit', got '{val_metric}'")
    if val_metric == "mse" and val_loader is None:
        raise ValueError("val_loader is required when val_metric='mse'")
    if val_metric == "profit" and val_dataset is None:
        raise ValueError("val_dataset is required when val_metric='profit'")
    if train_dataset.oracle_values is None:
        raise RuntimeError(
            "train_dataset.oracle_values is not set. "
            "Call dataset.set_oracle_values(precompute_oracle_values(...)) first."
        )

    device = determine_device()
    print(f"[DFL] Using device: {device}  |  val_metric: {val_metric}")
    print(f"[DFL] Training start: {isonow()}")
    _t0 = time.time()

    model = model.to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=dfl_batch_size, shuffle=True
    )

    scenario_name    = (scenario or {}).get("name", "default")
    best_val_metric  = float("inf")   # lower is better for both mse and value_gap
    best_ckpt_path   = _spo_checkpoint_path(model_name, wf_scenario, scenario_name)
    epochs_no_improv = 0

    history = {"train_spo_loss": [], "train_grad_norm": []}
    if val_metric == "mse":
        history.update({"val_mse": [], "val_rmse": [], "val_mae": []})
    else:
        history.update({"val_profit": [], "val_value_gap": []})

    for epoch in range(1, num_epochs + 1):



        model.train()
        epoch_spo_acc  = 0.0
        epoch_grad_acc = 0.0
        n_train = 0

        for X, Y, prices, times_ns, oracle_vals in tqdm.tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs} [SPO+ train]",
            leave=False,
        ):
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            preds = model(X)  # (B, N_PREDICT, 1)

            B = X.size(0)
            losses = [
                SPOPlusFunction_apply(
                    preds[i, :, 0],
                    Y[i, :, 0].cpu(),
                    prices[i],
                    times_ns[i],
                    oracle_vals[i],
                    scenario,
                    target_scaler,
                )
                for i in range(B)
            ]
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
         
        
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #clipping to 1.0 to prevent exploding gradients 
            optimizer.step()

            epoch_spo_acc  += batch_loss.item() * B
            epoch_grad_acc += grad_norm.item() * B
            n_train        += B

        train_spo      = epoch_spo_acc  / n_train
        mean_grad_norm = epoch_grad_acc / n_train
        history["train_spo_loss"].append(train_spo)
        history["train_grad_norm"].append(mean_grad_norm)



       
        if val_metric == "mse":
            model.eval()
            val_se_acc = 0.0
            val_ae_acc = 0.0
            n_val = 0

            with torch.no_grad():
                for Xv, Yv in tqdm.tqdm(
                    val_loader,
                    desc=f"Epoch {epoch}/{num_epochs} [val MSE]",
                    leave=False,
                ):
                    Xv = Xv.to(device)
                    Yv = Yv.to(device)
                    preds_v = model(Xv)
                    bs = Xv.size(0)
                    val_se_acc += torch.square(preds_v - Yv).mean().item() * bs
                    val_ae_acc += torch.abs(preds_v - Yv).mean().item() * bs
                    n_val += bs

            val_mse  = val_se_acc / n_val
            val_rmse = math.sqrt(val_mse)
            val_mae  = val_ae_acc / n_val

            history["val_mse"].append(val_mse)
            history["val_rmse"].append(val_rmse)
            history["val_mae"].append(val_mae)

            monitor_val  = val_mse
            val_str = f"Val RMSE={val_rmse:.4f}"

        else:  # profit
            val_profit, val_gap = _val_profit(model, val_dataset, scenario, device, target_scaler=target_scaler)
            history["val_profit"].append(val_profit)
            history["val_value_gap"].append(val_gap)

            monitor_val = -val_profit 
            val_str = f"Val profit=€{val_profit:,.0f}  gap={val_gap*100:.2f}%"



      
        if monitor_val < best_val_metric:
            best_val_metric  = monitor_val
            torch.save(model.state_dict(), best_ckpt_path)
            epochs_no_improv = 0
        else:
            epochs_no_improv += 1

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"SPO+ loss={train_spo:.4f} | "
            f"grad_norm={mean_grad_norm:.3f} | "
            f"{val_str} | "
            f"Epochs since best: {epochs_no_improv}"
        )

        if epochs_no_improv >= early_stop_patience:
            print(f"Early stopping triggered after {early_stop_patience} epochs without improvement.")
            break

    elapsed = time.time() - _t0
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"[DFL] Best val {val_metric}: {best_val_metric:.6f}  |  checkpoint: {best_ckpt_path}")
    print(f"[DFL] Training end: {isonow()}  |  elapsed: {h:02d}:{m:02d}:{s:02d}")
    history["training_time_s"] = round(elapsed, 1)
    return history

