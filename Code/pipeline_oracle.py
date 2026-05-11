import json
import pandas as pd

from optimization import run_optimization
from pathconfig import RESULTS_DIR, get_splits_dir, DEFAULT_WF_SCENARIO
from preprocessing import FEATURE_COLS, TARGET_COL
from dataloader import LOOKBACK, N_PREDICT


def _load_actuals(split: str, wf_scenario: str = DEFAULT_WF_SCENARIO) -> tuple[pd.Series, pd.Series]:
    """Load actual wind power and spot prices from the given split.

    For test split: mirrors the windowing used in inference.py (starts at LOOKBACK).
    For train/val splits: uses the full split.
    """
    df = pd.read_parquet(get_splits_dir(wf_scenario) / f"{split}.parquet")

    df = df.iloc[LOOKBACK:]

    actuals = df[TARGET_COL]
    prices  = df["spot_price_bornholm"]
    return actuals, prices


def run_oracle(split: str, save: bool = True, scenario: dict = None, wf_scenario: str = DEFAULT_WF_SCENARIO) -> dict:
    """Run the perfect-foresight oracle optimization on a chosen split.

    Parameters
    ----------
    split : str
        Which data split to use: "train", "val", or "test". Must be explicit —
        no default to avoid accidentally evaluating on the test set.
    save : bool
        If True, saves dispatch decisions and summary to RESULTS_DIR/oracle_{split}/.
    scenario : dict, optional
        Scenario config from scenarios.find_scenario(). If None, optimisation
        defaults are used.

    Returns
    -------
    dict with keys:
        actuals          : pd.Series — true wind power [MW]
        prices           : pd.Series — spot prices [€/MWh]
        objective_value  : float     — maximum achievable profit [€]
        status           : str       — solver status
        dispatch         : pd.DataFrame — all decision variables over time
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val' or 'test', got '{split}'")

    scenario_name = scenario["name"] if scenario else "default"
    print(f"\n{'='*60}")
    print(f" Oracle Pipeline  |  wf: {wf_scenario}  |  split: {split}  |  scenario: {scenario_name}")
    print(f"{'='*60}")

    actuals, prices = _load_actuals(split, wf_scenario)

    print("\n[Oracle] Running optimization with actual wind power...")
    opt = run_optimization(actuals, prices, scenario=scenario)
    print(f"[Oracle] Status: {opt['status']} | "
          f"Objective: €{opt['objective_value']:,.2f}")

    dispatch = pd.DataFrame({
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

    results = {
        "actuals":          actuals,
        "prices":           prices,
        "objective_value":  opt["objective_value"],
        "status":           opt["status"],
        "dispatch":         dispatch,
    }

    if save:
        folder = f"oracle_{wf_scenario}_{split}" if scenario is None else f"oracle_{wf_scenario}_{split}_{scenario_name}"
        out_dir = RESULTS_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        dispatch.to_parquet(out_dir / "dispatch_oracle.parquet")

        summary = {
            "split":           split,
            "wf_scenario":     wf_scenario,
            "scenario":        scenario_name,
            "objective_value": opt["objective_value"],
            "status":          opt["status"],
            "n_timesteps":     len(actuals),
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        print(f"\nOracle results saved to {out_dir}")

    return results
