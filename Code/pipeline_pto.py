import json
import pandas as pd

from inference import run_inference
from optimization import run_optimization, compute_realized_profit
from pathconfig import RESULTS_DIR, get_splits_dir, DEFAULT_WF_SCENARIO


def _load_prices(index: pd.DatetimeIndex, split: str = "test", wf_scenario: str = DEFAULT_WF_SCENARIO) -> pd.Series:
    """Load spot prices from `split` aligned to the given index."""
    prices = pd.read_parquet(get_splits_dir(wf_scenario) / f"{split}.parquet")["spot_price_bornholm"]
    prices = prices.reindex(index)
    if prices.isna().any():
        raise ValueError(
            "Price data has gaps for the requested prediction index. "
            "Check that preprocessing has been run and the test split is up to date."
        )
    return prices


def run_pto(
    model_name: str,
    save: bool = True,
    scenario: dict = None,
    wf_scenario: str = DEFAULT_WF_SCENARIO,
    split: str = "test",
) -> dict:
    """Run the full predict-then-optimize pipeline on `split`.

    Parameters
    ----------
    model_name : str
        Name of the trained LSTM model (e.g. "lstm_medium", "autoreg_medium").
    save : bool
        If True, saves dispatch decisions and summary metrics to RESULTS_DIR.
    scenario : dict, optional
        Scenario config from scenarios.find_scenario(). If None, optimisation
        defaults are used.

    Returns
    -------
    dict with keys:
        predictions      : pd.Series  — LSTM wind power forecast [MW]
        actuals          : pd.Series  — true wind power [MW]
        prices           : pd.Series  — spot prices [€/MWh]
        objective_value  : float      — achieved profit [€]
        status           : str        — solver status
        dispatch         : pd.DataFrame — all decision variables over time
    """
    scenario_name = scenario["name"] if scenario else "default"
    print(f"\n{'='*60}")
    print(f" PTO Pipeline  |  wf: {wf_scenario}  |  model: {model_name}  |  scenario: {scenario_name}")
    print(f"{'='*60}")

  


    predictions, actuals = run_inference(model_name, split=split, wf_scenario=wf_scenario)
    prices = _load_prices(predictions.index, split=split, wf_scenario=wf_scenario)


    print("\n[PTO] Running optimization with predicted wind power...")
    opt = run_optimization(predictions, prices, scenario=scenario)
    print(f"[PTO] Status: {opt['status']} | "
          f"Objective: €{opt['objective_value']:,.2f}")

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
    print(f"[PTO] Realized profit (vs actual wind): €{realized_profit:,.2f}")
    if realized["n_violations"] > 0:
        print(f"[PTO] Warning: {realized['n_violations']} timesteps with unresolvable deficit (worst: {realized['worst_deficit']:.1f} MW)")


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
    }

    

    if save:
        folder = f"pto_{wf_scenario}_{split}_{model_name}" if scenario is None else f"pto_{wf_scenario}_{split}_{model_name}_{scenario_name}"
        out_dir = RESULTS_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        dispatch.to_parquet(out_dir / "dispatch_pto.parquet")

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
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        print(f"\nResults saved to {out_dir}")

    return results
