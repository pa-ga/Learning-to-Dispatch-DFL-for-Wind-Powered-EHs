

import zipfile
import numpy as np
import pandas as pd
from pathconfig import RAW_DATA_DIR

WEATHER_DIR    = RAW_DATA_DIR / "Weather Data"
ERA5_ZIP       = WEATHER_DIR / "era5_bornholm.zip"
ERA5_PROCESSED = WEATHER_DIR / "era5_processed.csv"

R_DRY = 287.05  # specific gas constant for dry air [J/(kg*K)]


def _read_csv_from_zip(z: zipfile.ZipFile, pattern: str) -> pd.DataFrame:
    """Find and read the CSV inside the zip from ERA5 API fetch whose name contains `pattern`."""
    match = next((n for n in z.namelist() if pattern in n and n.endswith(".csv")), None)
    if match is None:
        raise FileNotFoundError(
            f"No CSV matching '{pattern}' found in zip. "
            f"Available files: {z.namelist()}"
        )
    with z.open(match) as f:
        return pd.read_csv(f)


def scale_era5(zip_path=ERA5_ZIP) -> pd.DataFrame:
    """Unzip ERA5 archive, merge the 3 CSVs and compute derived features (wind dir, wind speed, rho).
     """
    with zipfile.ZipFile(zip_path) as z:
        wind  = _read_csv_from_zip(z, "sfc-wind")
        temp  = _read_csv_from_zip(z, "2m-temperature")
        press = _read_csv_from_zip(z, "sfc-pressure")

    for df in [wind, temp, press]:
        df["valid_time"] = pd.to_datetime(df["valid_time"])

    merged = (
        wind
        .merge(temp,  on="valid_time", how="inner")
        .merge(press, on="valid_time", how="inner")
    )
    merged = merged.drop(
        columns=[c for c in merged.columns if c.startswith("latitude") or c.startswith("longitude")],
        errors="ignore",
    )


    _H_HUB = 150.0
    _H_REF = 10.0
    _ALPHA = 0.11
    _hub_factor = (_H_HUB / _H_REF) ** _ALPHA  # ≈ 1.34

    merged["wind_speed"] = np.sqrt(merged["u10"]**2 + merged["v10"]**2) * _hub_factor
    merged["wind_dir"]   = (np.degrees(np.arctan2(-merged["u10"], -merged["v10"])) + 360) % 360
 

   
    merged["rho"] = merged["sp"] / (R_DRY * merged["t2m"])

    return merged[["valid_time", "u10", "v10", "wind_speed", "wind_dir", "t2m", "sp", "rho"]]


if __name__ == "__main__":
    if not ERA5_ZIP.exists():
        raise FileNotFoundError(
            f"ERA5 zip not found at {ERA5_ZIP}. Run fetch_era5.py first."
        )

    print(f"Reading ERA5 zip from {ERA5_ZIP}...")
    processed = scale_era5()

    processed.to_csv(ERA5_PROCESSED, index=False)
    print(f"Processed data saved to {ERA5_PROCESSED}")
    print(f"  Rows   : {len(processed):,}")
    print(f"  Period : {processed['valid_time'].iloc[0]} → {processed['valid_time'].iloc[-1]}")
