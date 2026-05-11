import numpy as np
import pandas as pd

from pathconfig import RAW_DATA_DIR, get_wf_data_path, get_splits_dir, DEFAULT_WF_SCENARIO





FREQ = "15min"  # target temporal resolution

FEATURE_COLS = [
    "wind_speed",
    "wind_dir",
    "t2m",                    # 2-m temperature [K]
    "sp",                     # surface pressure [Pa]
    "rho",                    # air density [kg/m³]
    "spot_price_bornholm",    # artificial Bornholm price: avg(DK2, DE/LU) [€/MWh]
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
]

TARGET_COL = "ActivePower_WF_MW"

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test gets the remaining 0.15




def _add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour-of-day, day-of-week and month as sin/cos pairs."""
    idx = df.index
    df["hour_sin"]  = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    return df





def load_wf_data(wf_scenario: str = DEFAULT_WF_SCENARIO) -> pd.DataFrame:
    """Load WF_Data_{wf_scenario}.parquet and interpolate to 15-min resolution."""
    path = get_wf_data_path(wf_scenario)
    df = pd.read_parquet(path)

    df = df.set_index("time").sort_index()
    df.index = pd.to_datetime(df.index)
    df.index.name = "time"

    keep = ["wind_speed", "wind_dir", "t2m", "sp", "rho", TARGET_COL]
    df = df[keep]


    df = df.resample(FREQ).interpolate(method="linear") # upsample to 15-min via linear interpolation.
    return df


def load_spot_prices() -> pd.Series:
    """Parse Spotprices.csv and return the artificial Bornholm price as a 15-min Series.

    """
    path = RAW_DATA_DIR / "Spotprices.csv"

    raw = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",  # handles BOM
        parse_dates=["Datum von"],
        dayfirst=True,
        low_memory=False,
    )

    
    dk2_col = next(c for c in raw.columns if "nemark 2" in c)       
    de_col  = next(c for c in raw.columns if "eutschland" in c)  

    raw = raw[["Datum von", dk2_col, de_col]].copy()
    raw.columns = ["time", "spot_price_DK2", "spot_price_DE"]
    raw["spot_price_DK2"] = pd.to_numeric(raw["spot_price_DK2"], errors="coerce")
    raw["spot_price_DE"]  = pd.to_numeric(raw["spot_price_DE"],  errors="coerce")


    raw["spot_price_bornholm"] = raw[["spot_price_DK2", "spot_price_DE"]].mean(axis=1, skipna=True)

    raw = raw.dropna(subset=["time"])
    raw = raw.set_index("time").sort_index()
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "time"

    raw = raw[~raw.index.duplicated(keep="last")]


    series = raw["spot_price_bornholm"].resample(FREQ).ffill()
    return series





def build_dataset(wf_scenario: str = DEFAULT_WF_SCENARIO) -> pd.DataFrame:
    """Merge all sources into a single 15-min DataFrame."""
    wf     = load_wf_data(wf_scenario)
    prices = load_spot_prices()

    df = wf.join(prices, how="inner")
    df = df.dropna()
    df = _add_cyclic_features(df)

    return df[FEATURE_COLS + [TARGET_COL]]


def split_and_save(df: pd.DataFrame, wf_scenario: str = DEFAULT_WF_SCENARIO) -> None:
    """Split 70 / 15 / 15 chronologically and save to data_splits_{wf_scenario}/."""
    splits_dir = get_splits_dir(wf_scenario)
    n         = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    print(f"Total rows : {n:,}")
    print(f"Train rows : {len(train):,}  ({train.index[0]} → {train.index[-1]})")
    print(f"Val rows   : {len(val):,}   ({val.index[0]} → {val.index[-1]})")
    print(f"Test rows  : {len(test):,}  ({test.index[0]} → {test.index[-1]})")

    train.to_parquet(splits_dir / "train.parquet")
    val.to_parquet(splits_dir / "val.parquet")
    test.to_parquet(splits_dir / "test.parquet")
    print(f"Splits saved to {splits_dir}")


if __name__ == "__main__":
    df = build_dataset()
    split_and_save(df)
