import argparse
import json
import training
import dataloader
import pathconfig
from util import isonow
from configurations import find_cfg, get_family_cfgs


def load_data(wf_scenario: str = pathconfig.DEFAULT_WF_SCENARIO):
    train, scaler = dataloader.load_train(wf_scenario)
    val = dataloader.load_val(scaler, wf_scenario, target_scaler=train.target_scaler)
    return train, val, scaler


def _save_results(cfg, history, wf_scenario, now):
    """ Save training results conifg and history as JSON to checkpoints dir."""
    results = {
        'config': cfg,
        'history': history,
        'checkpoint_path': str(training.checkpoint_model_path(cfg['name'], wf_scenario)),
    }
    ckpt_dir = pathconfig.get_checkpoints_dir(wf_scenario)
    with ckpt_dir.joinpath(f"{cfg['name']}_results-{now}.json").open('w') as f:
        json.dump(results, fp=f, indent=4)


def train_one(name, wf_scenario: str = pathconfig.DEFAULT_WF_SCENARIO):
    """Train a single model configuration by name."""
    model_cls, cfg = find_cfg(name)
    now = isonow()
    train, val, _ = load_data(wf_scenario)
    _, history = training.train_with_config(model_cls, cfg, train=train, val=val, wf_scenario=wf_scenario)
    _save_results(cfg, history, wf_scenario, now)


def train_family(family: str, wf_scenario: str = pathconfig.DEFAULT_WF_SCENARIO):
    """Train all configurations for a model family ('lstm' or 'autoreg').
       Data is loaded once and reused across all configs.
    """
    model_cls, configs = get_family_cfgs(family)
    print(f"Training all {len(configs)} '{family}' configs for wf_scenario='{wf_scenario}'")
    train, val, _ = load_data(wf_scenario)
    now = isonow()
    results = training.train_all(model_cls, configs, train=train, val=val, wf_scenario=wf_scenario)
    for cfg in configs:
        _save_results(cfg, results[cfg['name']]['history'], wf_scenario, now)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model configurations.")
    parser.add_argument(
        "name", nargs="?",
        help="Name of a single model config to train (e.g. mini_lstm).",
    )
    parser.add_argument(
        "--all", metavar="FAMILY",
        help="Train all configs in a family: 'lstm' or 'autoreg'.",
    )
    parser.add_argument(
        "--wf-scenario", default=pathconfig.DEFAULT_WF_SCENARIO,
        help=f"Wind farm scenario (default: {pathconfig.DEFAULT_WF_SCENARIO}).",
    )
    args = parser.parse_args()

    if args.all:
        train_family(args.all, wf_scenario=args.wf_scenario)
    elif args.name:
        train_one(args.name, wf_scenario=args.wf_scenario)
    else:
        parser.print_help()
