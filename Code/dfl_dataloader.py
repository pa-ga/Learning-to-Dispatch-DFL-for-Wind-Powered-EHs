

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import EnergyDataset, LOOKBACK, N_PREDICT, N_INPUT_FEATURES
from pathconfig import get_splits_dir, DEFAULT_WF_SCENARIO
from preprocessing import FEATURE_COLS, TARGET_COL

# Index of spot_price_bornholm in FEATURE_COLS (column of the raw df values).
_PRICE_COL_IDX = FEATURE_COLS.index("spot_price_bornholm")


class DFLDataset(EnergyDataset):
    """EnergyDataset extended with unscaled prices and timestamps for SPO+ training.

    Parameters
    ----------
    df         : DataFrame with columns FEATURE_COLS + [TARGET_COL] and DatetimeIndex.
    scaler     : Pre-fitted StandardScaler (for features). None -> no scaling.
    fit_scaler : If True, fit a new scaler on this split (train split only).
    """

    def __init__(self, df, scaler=None, fit_scaler: bool = False, stride: int = N_PREDICT):
        # Extract raw (unscaled) prices and timestamps for each forecast window before calling the scaler
        raw = df.values                            # (T, N_INPUT_FEATURES + 1), unscaled
        times_ns = df.index.values.astype(np.int64)  

        T = len(df)
        all_prices: list[np.ndarray]   = []
        all_times_ns: list[np.ndarray] = []

        for start in range(0, T - LOOKBACK - N_PREDICT + 1, stride):
            fc = start + LOOKBACK
            all_prices.append(
                raw[fc : fc + N_PREDICT, _PRICE_COL_IDX].astype(np.float32)
            )
            all_times_ns.append(times_ns[fc : fc + N_PREDICT])

        # Apply standard EnergyDataset initialisation (scaling + window tensors).
        super().__init__(df, scaler=scaler, fit_scaler=fit_scaler, stride=stride)

        # (N_windows, N_PREDICT)
        self.forecast_prices   = torch.tensor(np.stack(all_prices))
        self.forecast_times_ns = torch.from_numpy(np.stack(all_times_ns))

        # Populated later via set_oracle_values().
        self.oracle_values: torch.Tensor | None = None

  
    # Oracle values
  

    def set_oracle_values(self, oracle_values: torch.Tensor) -> None:
        """Attach precomputed oracle LP values (one per window).

        Must be called before this dataset is used in a DataLoader for SPO+
        training.  oracle_values[i] = V(P_wind_true) for window i.
        """
        if len(oracle_values) != len(self):
            raise ValueError(
                f"oracle_values length {len(oracle_values)} "
                f"does not match dataset length {len(self)}."
            )
        self.oracle_values = oracle_values.float()

 
    # Dataset interface


    def __getitem__(self, idx: int):
        """Return (x, y, prices, times_ns, oracle_val) for window `idx`.
         oracle_val is NaN if set_oracle_values() has not been called yet
        (safe for inspection / precompute step, not for training).
        """
        x, y = super().__getitem__(idx)
        prices   = self.forecast_prices[idx]    # (N_PREDICT,)  float32
        times_ns = self.forecast_times_ns[idx]  # (N_PREDICT,)  int64

        if self.oracle_values is not None:
            oracle_val = self.oracle_values[idx]
        else:
            oracle_val = torch.tensor(float("nan"))

        return x, y, prices, times_ns, oracle_val




def _load_df(split: str, wf_scenario: str = DEFAULT_WF_SCENARIO):
    import pandas as pd
    path = get_splits_dir(wf_scenario) / f"{split}.parquet"
    return pd.read_parquet(path)[FEATURE_COLS + [TARGET_COL]]


def load_train_dfl(
    wf_scenario: str = DEFAULT_WF_SCENARIO,
    train_stride: int = 48,
) -> tuple["DFLDataset", object]:
    """Load the training split as a DFLDataset, fitting a new scaler.
    Returns (dataset, scaler). 
    """
    print(f"Loading TRAIN (DFL) … (stride={train_stride})")
    ds = DFLDataset(_load_df("train", wf_scenario), fit_scaler=True, stride=train_stride)
    return ds, ds.scaler
