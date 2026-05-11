
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from pathconfig import get_splits_dir, DEFAULT_WF_SCENARIO
from preprocessing import FEATURE_COLS, TARGET_COL


# Sliding-window constants

LOOKBACK          = 288   # 72 h at 15-min resolution
N_PREDICT         = 144   # 36 h at 15-min resolution
N_INPUT_FEATURES  = len(FEATURE_COLS)                    # 12 — weather/price/time features
N_OUTPUT_FEATURES = 1                                    # ActivePower_WF_MW
N_MODEL_INPUTS    = N_INPUT_FEATURES + N_OUTPUT_FEATURES # 13 — LSTM input (features + lagged power)


# Sliding-window generator


def sliding_windows(data: np.ndarray, stride: int = N_PREDICT):
    """Generate (x, y) pairs with a configurable stride.

    Parameters
    ----------
    data   : np.ndarray of shape (T, N_INPUT_FEATURES + N_OUTPUT_FEATURES)
        Full time series array.  The last column(s) are the target(s).
    stride : int
        Step size between consecutive windows.  Use N_PREDICT for
        non-overlapping windows (val/test); use a smaller value (e.g. 48 = 12 h)
        for overlapping training windows to increase sample count.

    Yields
    ------
    x : np.ndarray  shape (LOOKBACK,  N_MODEL_INPUTS)   — features + lagged power
    y : np.ndarray  shape (N_PREDICT, N_OUTPUT_FEATURES)
    """
    T = len(data)
    for start in range(0, T - LOOKBACK - N_PREDICT + 1, stride):
        x = data[start : start + LOOKBACK, :]                                         # all cols — features + lagged power
        y = data[start + LOOKBACK : start + LOOKBACK + N_PREDICT, N_INPUT_FEATURES:]  # target only
        yield x, y



# Dataset

class EnergyDataset(Dataset):
    """Sliding-window dataset for wind power forecasting.

    Parameters
    ----------
    df            : DataFrame with columns = FEATURE_COLS + [TARGET_COL]
    scaler        : Pre-fitted StandardScaler for input features.
                    If None and fit_scaler=False, data is not scaled.
    fit_scaler    : If True, fit new scalers on split.
                    Should only be True for the training split.
    target_scaler : Pre-fitted StandardScaler for the target column.
                    Required when scaler is provided (val/test splits).
    """

    scaler: Optional[StandardScaler]
    target_scaler: Optional[StandardScaler]

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        target_scaler: Optional[StandardScaler] = None,
        stride: int = N_PREDICT,
    ):
        arr = df.values.astype(np.float32)  # shape (T, N_INPUT_FEATURES + 1)

        if fit_scaler:
            self.scaler = StandardScaler()
            arr[:, :N_INPUT_FEATURES] = self.scaler.fit_transform(
                arr[:, :N_INPUT_FEATURES]
            )
            self.target_scaler = StandardScaler()
            arr[:, N_INPUT_FEATURES:] = self.target_scaler.fit_transform(
                arr[:, N_INPUT_FEATURES:]
            )
        elif scaler is not None:
            self.scaler = scaler
            arr[:, :N_INPUT_FEATURES] = scaler.transform(arr[:, :N_INPUT_FEATURES])
            self.target_scaler = target_scaler
            if target_scaler is not None:
                arr[:, N_INPUT_FEATURES:] = target_scaler.transform(
                    arr[:, N_INPUT_FEATURES:]
                )
        else:
            self.scaler = None
            self.target_scaler = None

        xs, ys = [], []
        for x, y in sliding_windows(arr, stride=stride):
            xs.append(x)
            ys.append(y)

        self.x = torch.tensor(np.stack(xs), dtype=torch.float32)  # (N, LOOKBACK, 13)
        self.y = torch.tensor(np.stack(ys), dtype=torch.float32)  # (N, N_PREDICT, 1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]





def _load_df(split: str, wf_scenario: str = DEFAULT_WF_SCENARIO) -> pd.DataFrame:
    path = get_splits_dir(wf_scenario) / f"{split}.parquet"
    return pd.read_parquet(path)[FEATURE_COLS + [TARGET_COL]]


def load_train(
    wf_scenario: str = DEFAULT_WF_SCENARIO,
    train_stride: int = 24,
) -> tuple[EnergyDataset, StandardScaler]:
    print(f"Loading TRAIN … (stride={train_stride}, ~{N_PREDICT // train_stride}× more windows than non-overlapping)")
    ds = EnergyDataset(_load_df("train", wf_scenario), fit_scaler=True, stride=train_stride)
    return ds, ds.scaler


def load_val(
    scaler: StandardScaler,
    wf_scenario: str = DEFAULT_WF_SCENARIO,
    target_scaler: Optional[StandardScaler] = None,
) -> EnergyDataset:
    print("Loading VAL …")
    return EnergyDataset(_load_df("val", wf_scenario), scaler=scaler, target_scaler=target_scaler)


def load_test(
    scaler: StandardScaler,
    wf_scenario: str = DEFAULT_WF_SCENARIO,
    target_scaler: Optional[StandardScaler] = None,
) -> EnergyDataset:
    print("Loading TEST …")
    return EnergyDataset(_load_df("test", wf_scenario), scaler=scaler, target_scaler=target_scaler)


if __name__ == "__main__":
    train_ds, scaler = load_train()
    val_ds           = load_val(scaler, target_scaler=train_ds.target_scaler)
    test_ds          = load_test(scaler, target_scaler=train_ds.target_scaler)

    joblib.dump(scaler, get_splits_dir() / "scaler.joblib")
    joblib.dump(train_ds.target_scaler, get_splits_dir() / "target_scaler.joblib")

    print(f"Train windows : {len(train_ds)}")
    print(f"Val windows   : {len(val_ds)}")
    print(f"Test windows  : {len(test_ds)}")
    print(f"x shape : {train_ds.x.shape}")
    print(f"y shape : {train_ds.y.shape}")
