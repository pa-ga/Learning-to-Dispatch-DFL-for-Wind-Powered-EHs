"""Main entry point for the pipelines and data simulation setup for the Bornholm Energy Island master thesis project.

Supported functionalities include:
    Windfarm data simulation using ERA5 weather data and a wind farm model.
    Data preprocessing and feature engineering.
    Oracle - Training-agnostic perfect-foresight benchmark (upper bound on achievable profit)
    Training LSTM models for wind power forecasting.
    PTO  — Standard Predict-Then-Optimize (LSTM trained on MSE loss)
    DFL  — Decision-Focused Learning      (LSTM trained on decision quality via SPO+)
    Return a summary of the value gap between the oracle and the PTO/DFL pipelines.

Stages
------
    preprocess   Build data splits (run once before anything else)
    train        Train a single model by name 
    train--all   Train all configs in a family ('lstm' or 'autoreg')
    oracle       Run perfect-foresight benchmark on test split
    pto          Run predict-then-optimize pipeline
    dfl          Run decision-focused learning pipeline
    all          Run preprocess, train, oracle, pto in sequence for a given model.

Usage examples
--------------
    python main.py --preprocess
    python main.py --train medium_lstm
    python main.py --oracle
    python main.py --pto medium_lstm
    python main.py --all medium_lstm
    python main.py --models
"""

import argparse

from configurations import lstm_configs, autoreg_configs
from scenarios import find_scenario, scenarios as all_scenarios
from wf_scenarios import wf_scenarios as all_wf_scenarios
from pathconfig import DEFAULT_WF_SCENARIO


def _print_value_gap(realized_profit: float, oracle_objective: float, model_name: str, planned_profit: float = None):
    gap     = oracle_objective - realized_profit
    gap_pct = gap / oracle_objective * 100 if oracle_objective != 0 else float("nan")
    print(f"\n{'='*60}")
    print(f" Value Gap Summary  |  model: {model_name}")
    print(f"{'='*60}")
    print(f"  Oracle   (perfect foresight)  : €{oracle_objective:>15,.2f}")
    if planned_profit is not None:
        print(f"  Planned  ({model_name:<20s}) : €{planned_profit:>15,.2f}")
    print(f"  Realized ({model_name:<20s}) : €{realized_profit:>15,.2f}")
    print(f"  Gap                            : €{gap:>15,.2f}  ({gap_pct:.2f}%)")
    print(f"{'='*60}\n")




def stage_simulate(wf_scenario: str):
    from wf_scenarios import find_wf_scenario
    from ERA5_DataScaling import ERA5_PROCESSED
    import WindfarmSimulation
    import pandas as pd

    wf_cfg = find_wf_scenario(wf_scenario)
    print(f"\nSimulating wind farm: {wf_cfg['description']}")
    weather_df = pd.read_csv(ERA5_PROCESSED)
    WindfarmSimulation.call_all(weather_df, wf_cfg["WF_cap"], scenario_name=wf_scenario)



def stage_preprocess(wf_scenario: str = DEFAULT_WF_SCENARIO):
    import preprocessing
    import dataloader
    import joblib
    from pathconfig import get_splits_dir

    print(f"\n[1/2] Building dataset and splits (wf: {wf_scenario})...")
    df = preprocessing.build_dataset(wf_scenario)
    preprocessing.split_and_save(df, wf_scenario)

    print(f"\n[2/2] Fitting and saving scalers...")
    train_ds, scaler = dataloader.load_train(wf_scenario)
    splits_dir = get_splits_dir(wf_scenario)
    joblib.dump(scaler, splits_dir / "scaler.joblib")
    joblib.dump(train_ds.target_scaler, splits_dir / "target_scaler.joblib")
    print(f"Scalers saved to {splits_dir}")



def stage_train(model_name: str, wf_scenario: str = DEFAULT_WF_SCENARIO):
    import batch_train
    print(f"\nTraining model: {model_name}  (wf: {wf_scenario})")
    batch_train.train_one(model_name, wf_scenario)


def stage_train_family(family: str, wf_scenario: str = DEFAULT_WF_SCENARIO):
    import batch_train
    print(f"\nTraining all models in family: {family}  (wf: {wf_scenario})")
    batch_train.train_family(family, wf_scenario)


def stage_oracle(split: str = "train", scenario: dict = None, wf_scenario: str = DEFAULT_WF_SCENARIO) -> dict:
    from pipeline_oracle import run_oracle
    return run_oracle(split=split, save=True, scenario=scenario, wf_scenario=wf_scenario)


def stage_pto(model_name: str, scenario: dict = None, wf_scenario: str = DEFAULT_WF_SCENARIO) -> dict:
    from pipeline_pto import run_pto
    return run_pto(model_name, save=True, scenario=scenario, wf_scenario=wf_scenario)


def stage_dfl(model_name: str, scenario: dict = None, wf_scenario: str = DEFAULT_WF_SCENARIO, val_metric: str = "mse", no_eval: bool = False, eval_only: bool = False) -> dict:
    from pipeline_dfl import run_dfl
    return run_dfl(model_name, save=True, scenario=scenario, wf_scenario=wf_scenario, val_metric=val_metric, no_eval=no_eval, eval_only=eval_only)






def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bornholm Energy Island — predict-then-optimize pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--preprocess",
        action="store_true",
        help="Build dataset splits and fit the feature scaler.",
    )
    group.add_argument(
        "--train",
        metavar="MODEL",
        help="Train a single model by name (see --models for valid names).",
    )
    group.add_argument(
        "--train-all",
        metavar="FAMILY",
        help="Train all configs in a family: lstm or autoreg"
    )
    group.add_argument(
        "--oracle",
        action="store_true",
        help="Run the perfect-foresight oracle benchmark.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Data split to use for --oracle (default: test).",
    )
    group.add_argument(
        "--pto",
        metavar="MODEL",
        help="Run the predict-then-optimize pipeline with a trained model.",
    )
    group.add_argument(
        "--all",
        metavar="MODEL",
        help="Run the full pipeline: preprocess, train, oracle, pto.",
    )
    group.add_argument(
        "--models",
        action="store_true",
        help="List all available model names and exit.",
    )
    group.add_argument(
        "--scenarios",
        action="store_true",
        help="List all available optimisation scenario names and exit.",
    )
    group.add_argument(
        "--wf-scenarios",
        action="store_true",
        help="List all available wind farm scenario names and exit.",
    )
    group.add_argument(
        "--dfl",
        metavar="MODEL",
        help="Run the full DFL pipeline (SPO+ training from scratch then evaluate) for MODEL.",
    )
    group.add_argument(
        "--simulate",
        action="store_true",
        help="Run wind farm simulation for --wf-scenario and save WF_Data parquet.",
    )
    parser.add_argument(
        "--val-metric",
        choices=["mse", "profit"],
        default="mse",
        help="Validation metric for --dfl: 'mse' (fast) or 'profit' (decision-consistent, slow).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="For --dfl: skip the final test evaluation. Trains and saves checkpoint only.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="For --dfl: skip training, load existing SPO+ checkpoint and evaluate only.",
    )
    parser.add_argument(
        "--scenario",
        metavar="SCENARIO",
        default=None,
        help="Optimisation scenario name for --oracle/--pto/--all (default: module defaults).",
    )
    parser.add_argument(
        "--wf-scenario",
        metavar="WF_SCENARIO",
        default=DEFAULT_WF_SCENARIO,
        help=f"Wind farm scenario name (default: {DEFAULT_WF_SCENARIO}). "
             "Selects which simulated WF_Data, splits, and checkpoints to use.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.models:
        print("\nAvailable models:")
        print("\n  Standard LSTM:")
        for name in [c["name"] for c in lstm_configs]:
            print(f"    {name}")
        print("\n  Autoregressive Seq2Seq LSTM:")
        for name in [c["name"] for c in autoreg_configs]:
            print(f"    {name}")
        return

    if args.scenarios:
        print("\nAvailable optimisation scenarios:")
        for s in all_scenarios:
            print(f"  {s['name']}")
        return

    if args.wf_scenarios:
        print("\nAvailable wind farm scenarios:")
        for s in all_wf_scenarios:
            print(f"  {s['name']:10s}  —  {s['description']}")
        return

    wf_scenario = args.wf_scenario
    scenario    = find_scenario(args.scenario) if args.scenario else None

    if args.simulate:
        stage_simulate(wf_scenario)
        return

    if args.preprocess:
        stage_preprocess(wf_scenario)
        return

    if args.train:
        stage_train(args.train, wf_scenario)
        return
    
    if args.train_all:
        stage_train_family(args.train_all, wf_scenario)
        return


    if args.oracle:
        stage_oracle(split=args.split, scenario=scenario, wf_scenario=wf_scenario)
        return

    if args.pto:
        pto_results    = stage_pto(args.pto, scenario=scenario, wf_scenario=wf_scenario)
        oracle_results = stage_oracle(split="test", scenario=scenario, wf_scenario=wf_scenario)
        _print_value_gap(
            pto_results["realized_profit"],
            oracle_results["objective_value"],
            args.pto,
            planned_profit=pto_results["objective_value"],
        )
        return

    if args.dfl:
        dfl_results    = stage_dfl(args.dfl, scenario=scenario, wf_scenario=wf_scenario, val_metric=args.val_metric, no_eval=args.no_eval, eval_only=args.eval_only)
        if args.no_eval:
            return
        oracle_results = stage_oracle(split="test", scenario=scenario, wf_scenario=wf_scenario)
        _print_value_gap(
            dfl_results["realized_profit"],
            oracle_results["objective_value"],
            f"{args.dfl} (SPO+)",
            planned_profit=dfl_results["objective_value"],
        )
        return

    if args.all:
        model_name = args.all
        print(f"\nRunning full pipeline for model: {model_name}  (wf: {wf_scenario})\n")
        stage_preprocess(wf_scenario)
        stage_train(model_name, wf_scenario)
        oracle_results = stage_oracle(split="test", scenario=scenario, wf_scenario=wf_scenario)
        pto_results    = stage_pto(model_name, scenario=scenario, wf_scenario=wf_scenario)
        _print_value_gap(
            pto_results["realized_profit"],
            oracle_results["objective_value"],
            model_name,
            planned_profit=pto_results["objective_value"],
        )
        return


if __name__ == "__main__":
    main()
