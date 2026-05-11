import json
import random

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader

from configurations import find_cfg
from dataloader import load_val, load_test
from dfl_dataloader import DFLDataset, load_train_dfl
from optimization import run_optimization, compute_realized_profit
from pathconfig import (
    RESULTS_DIR,
    get_splits_dir,
    get_checkpoints_dir,
    DEFAULT_WF_SCENARIO,
)
from spo_loss import precompute_oracle_values
from training import determine_device
from training_dfl import train_model_spo, _spo_checkpoint_path
from dataloader import LOOKBACK, N_PREDICT, N_INPUT_FEATURES
from preprocessing import FEATURE_COLS, TARGET_COL
from util import isonow

import numpy as np





def _load_prices(index: pd.DatetimeIndex, split: str = "test", wf_scenario: str = DEFAULT_WF_SCENARIO) -> pd.Series:
    """Load spot prices for `split` aligned to the given DatetimeIndex."""
    prices = pd.read_parquet(get_splits_dir(wf_scenario) / f"{split}.parquet")["spot_price_bornholm"]
    prices = prices.reindex(index)
    if prices.isna().any():
        raise ValueError(
            "Price data has gaps for the requested prediction index. "
            "Check that preprocessing has been run for this wf_scenario."
        )
    return prices


def _run_inference(
    model_name: str,
    split: str = "test",
    wf_scenario: str = DEFAULT_WF_SCENARIO,
    batch_size: int = 64,
    scenario_name: str = "default",
) -> tuple[pd.Series, pd.Series]:
    """Run the SPO+-trained model on `split`.

    Loads the checkpoint saved as {model_name}_spo_{scenario_name}_best.pt.
    Uses the same architecture as the MSE-trained counterpart (same config).
    """
    device = determine_device()

    model_cls, cfg = find_cfg(model_name)
    model = model_cls(**cfg["model_kwargs"]).to(device)

    ckpt_path = _spo_checkpoint_path(model_name, wf_scenario, scenario_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No SPO+ checkpoint at {ckpt_path}. "
            "Run run_dfl() with train=True first."
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"[DFL] Loaded checkpoint: {ckpt_path}")

    # Load scalers fitted on the train split.
    scaler_path = get_splits_dir(wf_scenario) / "scaler.joblib"
    target_scaler_path = get_splits_dir(wf_scenario) / "target_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run preprocessing first.")
    if not target_scaler_path.exists():
        raise FileNotFoundError(f"Target scaler not found at {target_scaler_path}. Run preprocessing first.")
    scaler        = joblib.load(scaler_path)
    target_scaler = joblib.load(target_scaler_path)

    split_path = get_splits_dir(wf_scenario) / f"{split}.parquet"
    split_df   = pd.read_parquet(split_path)[FEATURE_COLS + [TARGET_COL]]

    from dataloader import EnergyDataset
    split_ds    = EnergyDataset(split_df, scaler=scaler, target_scaler=target_scaler)
    split_loader = DataLoader(split_ds, batch_size=batch_size, shuffle=False)

    all_preds   = []
    all_actuals = []
    with torch.no_grad():
        for X, Y in split_loader:
            preds = model(X.to(device))
            all_preds.append(preds.cpu().numpy())
            all_actuals.append(Y.numpy())

    preds_arr   = np.concatenate(all_preds,   axis=0).squeeze(-1).flatten()
    actuals_arr = np.concatenate(all_actuals, axis=0).squeeze(-1).flatten()

   #Reverse scaling to in MW predictions and actuals
    preds_arr   = target_scaler.inverse_transform(preds_arr.reshape(-1, 1)).squeeze()
    actuals_arr = target_scaler.inverse_transform(actuals_arr.reshape(-1, 1)).squeeze()

    
    n_windows    = len(preds_arr) // N_PREDICT
    ts           = split_df.index
    start_idx    = [LOOKBACK + i * N_PREDICT for i in range(n_windows)]
    index        = pd.DatetimeIndex(
        [t for i in start_idx for t in ts[i : i + N_PREDICT]]
    )

    predictions = pd.Series(preds_arr,   index=index, name="P_wind_pred_MW")
    actuals     = pd.Series(actuals_arr, index=index, name="P_wind_actual_MW")

    print(f"[DFL] Inference: {len(predictions)} timesteps "
          f"({len(predictions) * 15 / 60:.1f} h)")
    return predictions, actuals




def _run_eval(model_name, eval_split, wf_scenario, inference_batch_size, scenario, save, scenario_name, history):
    print(f"\n[4/4] Evaluating on {eval_split} split …")
    predictions, actuals = _run_inference(model_name, split=eval_split, wf_scenario=wf_scenario, batch_size=inference_batch_size, scenario_name=scenario_name)
    prices = _load_prices(predictions.index, split=eval_split, wf_scenario=wf_scenario)

    print("[DFL] Running optimization with SPO+-predicted wind …")
    opt = run_optimization(predictions, prices, scenario=scenario)
    print(f"[DFL] Status: {opt['status']} | Objective: €{opt['objective_value']:,.2f}")

    dispatch = pd.DataFrame({
        "P_wind_pred":   predictions,
        "P_wind_actual": actuals,
        "spot_price":    prices,
        "P_grid":  opt["P_grid"],
        "P_curt":  opt["P_curt"],
        "P_ch":    opt["P_ch"],
        "P_dis":   opt["P_dis"],
        "P_ely":   opt["P_ely"],
        "E_bat":   opt["E_bat"],
        "H_prod":  opt["H_prod"],
        "H_sell":  opt["H_sell"],
        "E_H2":    opt["E_H2"],
    })

    realized = compute_realized_profit(dispatch, scenario)
    realized_profit = realized["profit"]
    print(f"[DFL] Realized profit (vs actual wind): €{realized_profit:,.2f}")
    if realized["n_violations"] > 0:
        print(f"[DFL] Warning: {realized['n_violations']} timesteps with unresolvable deficit (worst: {realized['worst_deficit']:.1f} MW)")

   
    dispatch["P_grid_realized"] = realized["P_grid_realized"]
    dispatch["P_curt_realized"] = realized["P_curt_realized"]
    dispatch["P_ely_realized"]  = realized["P_ely_realized"]
    dispatch["P_ch_realized"]   = realized["P_ch_realized"]
    dispatch["H_sell_realized"] = realized["H_sell_realized"]

    results = {
        "predictions":     predictions,
        "actuals":         actuals,
        "prices":          prices,
        "objective_value": opt["objective_value"],
        "realized_profit": realized_profit,
        "status":          opt["status"],
        "dispatch":        dispatch,
        "history":         history,
    }

    if save:
        tag = f"dfl_{wf_scenario}_{model_name}"
        if scenario_name != "default":
            tag += f"_{scenario_name}"
        out_dir = RESULTS_DIR / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        dispatch.to_parquet(out_dir / "dispatch_dfl.parquet")

        summary = {
            "model_name":        model_name,
            "wf_scenario":       wf_scenario,
            "scenario":          scenario_name,
            "objective_value":   opt["objective_value"],
            "realized_profit":   realized_profit,
            "n_violations":      realized["n_violations"],
            "worst_deficit_mw":  realized["worst_deficit"],
            "status":            opt["status"],
            "n_timesteps":       len(predictions),
            "training_epochs":   len(history.get("train_spo_loss", [])),
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        if history:
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=4)

        print(f"\n[DFL] Results saved to {out_dir}")

    return results





def run_dfl(
    model_name: str,
    seed: int = 42,
    save: bool = True,
    scenario: dict = None,
    wf_scenario: str = DEFAULT_WF_SCENARIO,
    pretrained_path: str = None,
    dfl_batch_size: int = 16, #overrides configs batch size 
    num_epochs: int = 30, #overrides config epochs
    early_stop_patience: int = 20, #overrides config patience
    eval_split: str = "test",
    inference_batch_size: int = 64, #overrides configs batch size
    val_metric: str = "mse",
    no_eval: bool = False,
    eval_only: bool = False,
) -> dict:
    """Run the full DFL pipeline: precompute - train (SPO+) - evaluate.

    Parameters
    ----------
    model_name           : Model architecture name, e.g. "medium_autoreg_lstm".
                           The SPO+ checkpoint is saved as {model_name}_spo_best.pt.
    save                 : If True, dispatch decisions and summary are saved to
                           RESULTS_DIR/dfl_{wf_scenario}_{model_name}[_{scenario}]/.
    scenario             : LP scenario config or None (module-level defaults).
    wf_scenario          : Wind farm scenario name.
    dfl_batch_size       : Batch size for SPO+ training (one LP per sample).
    num_epochs           : Maximum training epochs.
    early_stop_patience  : Early stopping patience on val metric.
    eval_split           : Data split to evaluate on ("train", "val", "test").
    inference_batch_size : Batch size for inference forward pass.
    val_metric           : "mse" (fast) or "profit" (decision-consistent, slow).

    Returns
    -------
    dict with keys:
        predictions      : pd.Series  — SPO+-model wind power forecast [MW]
        actuals          : pd.Series  — true wind power [MW]
        prices           : pd.Series  — spot prices [€/MWh]
        objective_value  : float      — achieved profit [€]
        status           : str        — LP solver status
        dispatch         : pd.DataFrame
        history          : dict       — training history
    """
    scenario_name = scenario["name"] if scenario else "default"
    print(f"\n{'='*60}")
    print(f" DFL Pipeline  |  wf: {wf_scenario}  |  model: {model_name}  |  scenario: {scenario_name}")
    print(f"{'='*60}")

    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"      Seed: {seed}")

 

    if eval_only:
        print("\n[eval-only] Skipping training — loading existing SPO+ checkpoint …")
        history = {}
        return _run_eval(model_name, eval_split, wf_scenario, inference_batch_size, scenario, save, scenario_name, history)



    print("\n[1/4] Building DFL training dataset …")
    train_ds, scaler = load_train_dfl(wf_scenario)

 
    scaler_path = get_splits_dir(wf_scenario) / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"      Scaler saved to {scaler_path}")

   


    print("\n[2/4] Precomputing oracle LP values for training windows …")
    oracle_values = precompute_oracle_values(train_ds, scenario=scenario, target_scaler=train_ds.target_scaler)
    train_ds.set_oracle_values(oracle_values)
    print(f"      Mean oracle profit: €{oracle_values.mean():.2f}  "
          f"(min €{oracle_values.min():.2f}, max €{oracle_values.max():.2f})")


  
  
    print(f"\n[3/4] Training {model_name} with SPO+ loss from scratch …")

    model_cls, cfg = find_cfg(model_name)
    model = model_cls(**cfg["model_kwargs"])

    if pretrained_path is not None:
        device = determine_device()
        model.load_state_dict(
            torch.load(pretrained_path, map_location=device)
        )
        print(f"      Warm-started from: {pretrained_path}")

    optimizer = torch.optim.AdamW(
        model.parameters(), **cfg["optimizer_args"]
    )

    if val_metric == "mse":
        val_ds     = load_val(scaler, wf_scenario)
        val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
        val_dfl_ds = None
    else:
        val_path   = get_splits_dir(wf_scenario) / "val.parquet"
        val_raw_df = pd.read_parquet(val_path)
        val_dfl_ds = DFLDataset(val_raw_df[FEATURE_COLS + [TARGET_COL]], scaler=scaler)
        val_oracle = precompute_oracle_values(val_dfl_ds, scenario=scenario,
                                              desc="Precomputing oracle LP values (val)",
                                              target_scaler=None)  # val y is already in MW (not scaled)
        val_dfl_ds.set_oracle_values(val_oracle)
        val_loader = None

    history = train_model_spo(
        model_name          = model_name,
        model               = model,
        train_dataset       = train_ds,
        val_loader          = val_loader,
        val_dataset         = val_dfl_ds,
        val_metric          = val_metric,
        num_epochs          = num_epochs,
        optimizer           = optimizer,
        scenario            = scenario,
        wf_scenario         = wf_scenario,
        dfl_batch_size      = dfl_batch_size,
        early_stop_patience = early_stop_patience,
        target_scaler       = train_ds.target_scaler,
    )

    

    if no_eval:
        print("\n[4/4] Skipping evaluation (--no-eval). Checkpoint saved — run with --eval-only to evaluate later.")
        return {"history": history}

    return _run_eval(model_name, eval_split, wf_scenario, inference_batch_size, scenario, save, scenario_name, history)
