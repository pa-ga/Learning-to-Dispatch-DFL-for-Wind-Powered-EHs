import itertools
from model_lstm import LSTMModel
from model_autoregressive_LSTM import Seq2SeqLSTM



# Standard LSTM configurations


lstm_configs = [
    {
        "name": "mini_lstm",
        "epochs": 10,
        "batch_size": 512,
        "model_kwargs": {"hidden_size": 64,   "num_layers": 2},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-4},
    },
    {
        "name": "small_lstm",
        "epochs": 40,
        "batch_size": 512,
        "model_kwargs": {"hidden_size": 128,  "num_layers": 3},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-4},
    },
    {
        "name": "medium_lstm",
        "epochs": 40,
        "batch_size": 512,
        "model_kwargs": {"hidden_size": 512,  "num_layers": 3},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-4},
    },
    {
        "name": "deeper_lstm",
        "epochs": 90,
        "batch_size": 512,
        "model_kwargs": {"hidden_size": 512,  "num_layers": 5},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-2},
    },
    {
        "name": "even_deeper_lstm",
        "epochs": 200,
        "batch_size": 128,
        "model_kwargs": {"hidden_size": 1024, "num_layers": 8},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-4},
    },
    {
        "name": "medium_lstm_reg",
        "epochs": 150,
        "batch_size": 256,
        "patience": 20,
        "model_kwargs": {"hidden_size": 512, "num_layers": 3, "dropout": 0.5},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-3},
    },
]



# Autoregressive Seq2Seq LSTM configurations


autoreg_configs = [
    {
        "name": "mini_autoreg_lstm",
        "epochs": 10,
        "batch_size": 512,
        "model_kwargs": {"hidden_size": 64,   "num_layers": 2},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-4},
    },
    {
        "name": "small_autoreg_lstm",
        "epochs": 40,
        "batch_size": 512,
        "model_kwargs": {"hidden_size": 128,  "num_layers": 3},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-4},
    },
    {
        "name": "medium_autoreg_lstm",
        "epochs": 40,
        "batch_size": 512,
        "model_kwargs": {"hidden_size": 512,  "num_layers": 3},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-4},
    },
    {
        "name": "deeper_autoreg_lstm",
        "epochs": 90,
        "batch_size": 512,
        "model_kwargs": {"hidden_size": 512,  "num_layers": 5},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-2},
    },
    {
        "name": "deepest_autoreg_lstm",
        "epochs": 1000,
        "batch_size": 128,
        "model_kwargs": {"hidden_size": 1024, "num_layers": 12},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-4},
    },
    {
        "name": "medium_autoreg_lstm_reg",
        "epochs": 150,
        "batch_size": 256,
        "patience": 20,
        "model_kwargs": {"hidden_size": 512, "num_layers": 3, "dropout": 0.5},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-3},
    },
     {
        "name": "medium_autoreg_lstm_reg_overlap",
        "epochs": 150,
        "batch_size": 256,
        "patience": 20,
        "model_kwargs": {"hidden_size": 512, "num_layers": 3, "dropout": 0.2},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-4},
    },
]




# GPU-optimised configurations (cuda_train)

lstm_cuda_configs = [
    {
        "name": "small_lstm_cuda_train_stride24",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 256, "num_layers": 2, "dropout": 0.3},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-3},
    },
    {
        "name": "medium_lstm_cuda_train_stride24",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 512, "num_layers": 3, "dropout": 0.4},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-3},
    },
    {
        "name": "large_lstm_cuda_train_stride24",
        "epochs": 300,
        "batch_size": 512,
        "patience": 30,
        "model_kwargs": {"hidden_size": 1024, "num_layers": 4, "dropout": 0.4},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-3},
    },
]

#Different configs families introduced in a second screening stage to train different strides that were used in the dataloaders, to test if smaller stride leads to better results and have distinct names

lstm_cuda_configs_stride1 = [
    {
        "name": "small_lstm_cuda_train_smallerstride1",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 256, "num_layers": 2, "dropout": 0.3},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-3},
    },
    {
        "name": "medium_lstm_cuda_train_smallerstride1",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 512, "num_layers": 3, "dropout": 0.4},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-3},
    },
    {
        "name": "large_lstm_cuda_train_smallerstride1",
        "epochs": 300,
        "batch_size": 512,
        "patience": 30,
        "model_kwargs": {"hidden_size": 1024, "num_layers": 4, "dropout": 0.4},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-3},
    },
]
lstm_cuda_configs_stride12 = [
    {
        "name": "small_lstm_cuda_train_smallerstride12",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 256, "num_layers": 2, "dropout": 0.3},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-3},
    },
    {
        "name": "medium_lstm_cuda_train_smallerstride12",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 512, "num_layers": 3, "dropout": 0.4},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-3},
    },
    {
        "name": "large_lstm_cuda_train_smallerstride12",
        "epochs": 300,
        "batch_size": 512,
        "patience": 30,
        "model_kwargs": {"hidden_size": 1024, "num_layers": 4, "dropout": 0.4},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-3},
    },
]

lstm_cuda_configs_stride12_12feat = [
    {
        "name": "small_lstm_cuda_train_smallerstride12_12feat",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 256, "num_layers": 2, "dropout": 0.3},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-3},
    },
    {
        "name": "medium_lstm_cuda_train_smallerstride12_12feat",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 512, "num_layers": 3, "dropout": 0.4},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-3},
    },
    {
        "name": "large_lstm_cuda_train_smallerstride12_12feat",
        "epochs": 300,
        "batch_size": 512,
        "patience": 30,
        "model_kwargs": {"hidden_size": 1024, "num_layers": 4, "dropout": 0.4},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-3},
    },
]

autoreg_cuda_configs = [
    {
        "name": "small_autoreg_cuda_train_smallerstride",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 256, "num_layers": 2, "dropout": 0.15},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-3},
    },
    {
        "name": "medium_autoreg_cuda_train_smallerstride",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 512, "num_layers": 3, "dropout": 0.2},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-3},
    },
    {
        "name": "large_autoreg_cuda_train_smallerstride",
        "epochs": 300,
        "batch_size": 512,
        "patience": 30,
        "model_kwargs": {"hidden_size": 1024, "num_layers": 4, "dropout": 0.2},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-3},
    },
]

autoreg_cuda_configs_stride1 = [
    {
        "name": "small_autoreg_cuda_train_smallerstride1",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 256, "num_layers": 2, "dropout": 0.15},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-3},
    },
    {
        "name": "medium_autoreg_cuda_train_smallerstride1",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 512, "num_layers": 3, "dropout": 0.2},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-3},
    },
    {
        "name": "large_autoreg_cuda_train_smallerstride1",
        "epochs": 300,
        "batch_size": 512,
        "patience": 30,
        "model_kwargs": {"hidden_size": 1024, "num_layers": 4, "dropout": 0.2},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-3},
    },
]

autoreg_cuda_configs_stride24 = [
    {
        "name": "small_autoreg_cuda_train_smallerstride24",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 256, "num_layers": 2, "dropout": 0.15},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-3},
    },
    {
        "name": "medium_autoreg_cuda_train_smallerstride24",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 512, "num_layers": 3, "dropout": 0.2},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-3},
    },
    {
        "name": "large_autoreg_cuda_train_smallerstride24",
        "epochs": 300,
        "batch_size": 512,
        "patience": 30,
        "model_kwargs": {"hidden_size": 1024, "num_layers": 4, "dropout": 0.2},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-3},
    },
]
autoreg_cuda_configs_stride12_feat = [
    {
        "name": "small_autoreg_cuda_train_smallerstride12_feat12",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 256, "num_layers": 2, "dropout": 0.15},
        "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-3},
    },
    {
        "name": "medium_autoreg_cuda_train_smallerstride12_feat12",
        "epochs": 200,
        "batch_size": 512,
        "patience": 25,
        "model_kwargs": {"hidden_size": 512, "num_layers": 3, "dropout": 0.2},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-3},
    },
    {
        "name": "large_autoreg_cuda_train_smallerstride12_feat12",
        "epochs": 300,
        "batch_size": 512,
        "patience": 30,
        "model_kwargs": {"hidden_size": 1024, "num_layers": 4, "dropout": 0.2},
        "optimizer_args": {"lr": 5e-4, "weight_decay": 1e-3},
    },
]
_FAMILIES = {
    "lstm":         (LSTMModel,   lstm_configs),
    "autoreg":      (Seq2SeqLSTM, autoreg_configs),
    "lstm_cuda":    (LSTMModel,   lstm_cuda_configs),
    "autoreg_cuda": (Seq2SeqLSTM, autoreg_cuda_configs),
    "small_stride_lstm": (LSTMModel,   lstm_cuda_configs_stride1), 
    "small_stride_autoreg": (Seq2SeqLSTM, autoreg_cuda_configs_stride1),
    "autoreg_cuda_configs_stride12" : (Seq2SeqLSTM, autoreg_cuda_configs_stride24),
    "lstm_cuda_configs_stride12": (LSTMModel,   lstm_cuda_configs_stride12),
    "autoreg_cuda_configs_stride12_12feat": (Seq2SeqLSTM, autoreg_cuda_configs_stride12_feat),
    "lstm_cuda_configs_stride12_12feat": (LSTMModel, lstm_cuda_configs_stride12_12feat)

}

# Lookup Function for Configs


def find_cfg(name: str) -> tuple:
    """Return (ModelClass, config_dict) for the given model name."""
    if "autoreg" in name:
        cfg = next((c for c in autoreg_configs if c["name"] == name), None)
        if cfg is None:
            raise ValueError(f"No autoreg config named '{name}'")
        return Seq2SeqLSTM, cfg
    else:
        cfg = next((c for c in lstm_configs if c["name"] == name), None)
        if cfg is None:
            raise ValueError(f"No lstm config named '{name}'")
        return LSTMModel, cfg

def get_family_cfgs(family: str) -> tuple:
    """Return (ModelClass, list_of_configs) for the given model family.

    Valid families: 'lstm', 'autoreg'.
    """
    entry = _FAMILIES.get(family)
    if entry is None:
        raise ValueError(
            f"Unknown family '{family}'. Choose from: {list(_FAMILIES)}"
        )
    return entry


def validate():
    """Check all configs have the required keys. Returns 0 if valid, 1 otherwise."""
    mandatory_keys      = ["name", "epochs", "batch_size", "model_kwargs", "optimizer_args"]
    mandatory_lstm_args = ["hidden_size", "num_layers"]
    valid = True

    for cfg in itertools.chain(lstm_configs, autoreg_configs, lstm_cuda_configs, autoreg_cuda_configs):
        issues = []
        for key in mandatory_keys:
            if key not in cfg:
                issues.append(f"{key} missing")
        for key in mandatory_lstm_args:
            if key not in cfg["model_kwargs"]:
                issues.append(f"model_kwargs.{key} missing")
        if issues:
            valid = False
            print(f"Config '{cfg.get('name', '?')}': {', '.join(issues)}")

    if valid:
        print("All configs valid.")
    return 0 if valid else 1


if __name__ == "__main__":
    exit(validate())
