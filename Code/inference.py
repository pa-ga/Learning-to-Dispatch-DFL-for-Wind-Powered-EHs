"""Wind power inference module.

Loads a trained model checkpoint and runs it on the test split,
returning predictions and actuals as DataFrames with a DatetimeIndex.
Shared by both the standard predict-then-optimize (PTO) pipeline
and the decision-focused learning (DFL) pipeline.

Usage:
    from inference import run_inference

    predictions, actuals = run_inference("lstm_medium")
    # predictions: pd.Series [MW], DatetimeIndex, 15-min resolution
    # actuals:     pd.Series [MW], same index
"""

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from configurations import find_cfg
from dataloader import LOOKBACK, N_PREDICT, EnergyDataset
from pathconfig import get_splits_dir, DEFAULT_WF_SCENARIO
from preprocessing import FEATURE_COLS, TARGET_COL
from training import checkpoint_model_path, determine_device


def _load_scaler(wf_scenario: str = DEFAULT_WF_SCENARIO):
    scaler_path = get_splits_dir(wf_scenario) / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            f"Run `python main.py --preprocess --wf-scenario {wf_scenario}` first."
        )
    return joblib.load(scaler_path)


def _load_target_scaler(wf_scenario: str = DEFAULT_WF_SCENARIO):
    scaler_path = get_splits_dir(wf_scenario) / "target_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Target scaler not found at {scaler_path}. "
            "Run `python dataloader.py` first."
        )
    return joblib.load(scaler_path)


def _load_split_df(split: str, wf_scenario: str = DEFAULT_WF_SCENARIO) -> pd.DataFrame:
    path = get_splits_dir(wf_scenario) / f"{split}.parquet"
    return pd.read_parquet(path)[FEATURE_COLS + [TARGET_COL]]


def run_inference(
    model_name: str,
    split: str = "test",
    batch_size: int = 64,
    wf_scenario: str = DEFAULT_WF_SCENARIO,
) -> tuple[pd.Series, pd.Series]:
    """Run inference with a trained model on the given split.

    Parameters
    ----------
    model_name : str
        Name of the model config (e.g. "lstm_medium", "autoreg_medium").
        Must match a trained checkpoint in CHECKPOINTS_DIR.
    split : str
        Data split to run inference on: "train", "val", or "test".
    batch_size : int
        Batch size for the inference DataLoader.

    Returns
    -------
    predictions : pd.Series
        Predicted wind farm power [MW] with DatetimeIndex (15-min resolution).
    actuals : pd.Series
        Ground truth wind farm power [MW] with same DatetimeIndex.
    """
    device = determine_device()

   
    model_cls, cfg = find_cfg(model_name)
    model = model_cls(**cfg["model_kwargs"]).to(device)

    ckpt_path = checkpoint_model_path(model_name, wf_scenario)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Train the model first."
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

  
    scaler = _load_scaler(wf_scenario)
    target_scaler = _load_target_scaler(wf_scenario)
    test_df = _load_split_df(split, wf_scenario)
    test_ds = EnergyDataset(test_df, scaler=scaler, target_scaler=target_scaler)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

  
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(device)
            preds = model(X)                    # (batch, N_PREDICT, 1)
            all_preds.append(preds.cpu().numpy())
            all_actuals.append(Y.numpy())

   
    preds_arr   = np.concatenate(all_preds,   axis=0).squeeze(-1).flatten()
    actuals_arr = np.concatenate(all_actuals, axis=0).squeeze(-1).flatten()

   # inverse-transform to MW
    preds_arr   = target_scaler.inverse_transform(preds_arr.reshape(-1, 1)).squeeze()  
    actuals_arr = target_scaler.inverse_transform(actuals_arr.reshape(-1, 1)).squeeze() 

  
    assert len(preds_arr) % N_PREDICT == 0, \
        f"Flattened predictions length {len(preds_arr)} is not divisible by N_PREDICT={N_PREDICT}"
    n_windows = len(preds_arr) // N_PREDICT
    ts = test_df.index
    start_indices = [LOOKBACK + i * N_PREDICT for i in range(n_windows)]
    index = pd.DatetimeIndex(
        [t for i in start_indices for t in ts[i : i + N_PREDICT]]
    )

    predictions = pd.Series(preds_arr, index=index, name="P_wind_pred_MW")
    actuals     = pd.Series(actuals_arr, index=index, name="P_wind_actual_MW")

    print(f"Inference complete: {len(predictions)} timesteps "
          f"({len(predictions) * 15 / 60:.1f} hours)")

    return predictions, actuals
