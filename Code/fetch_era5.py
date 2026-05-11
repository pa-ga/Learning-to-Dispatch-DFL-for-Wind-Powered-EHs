"""ERA5 data fetching via the CDS API.

Requirements:
    pip install cdsapi
    CDS API key configured at ~/.cdsapirc  (see https://cds.climate.copernicus.eu)

    An account needs to be created at the Copernicus Climate Data Store and the API key needs to be configured but the data is free to access.

"""

import argparse
import cdsapi
from pathconfig import RAW_DATA_DIR

WEATHER_DIR  = RAW_DATA_DIR / "Weather Data"
ERA5_OUTPUT  = WEATHER_DIR / "era5_bornholm.zip"

# Bornholm wind farm location (Copernicus snaps to nearest grid point)
LATITUDE  =  55.00000000000135
LONGITUDE =  14.999999999999972

DEFAULT_START = "2018-01-01"
DEFAULT_END   = "2025-12-31"


def fetch_era5(
    start: str = DEFAULT_START,
    end:   str = DEFAULT_END,
    output_path = ERA5_OUTPUT,
) -> None:
    """Download ERA5-Land hourly data for the Bornholm location.

    Parameters
    ----------
    start : str   Start date in YYYY-MM-DD format.
    end   : str   End date   in YYYY-MM-DD format.
    output_path : Path   Where to save the downloaded CSV.
    """
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)

    dataset = "reanalysis-era5-land-timeseries"
    request = {
        "variable": [
            "2m_temperature",
            "surface_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "location": {"longitude": LONGITUDE, "latitude": LATITUDE},
        "date": [f"{start}/{end}"],
        "data_format": "csv",
    }

    print(f"Requesting ERA5 data: {start} → {end}")
    print(f"Location: {LATITUDE}°N, {LONGITUDE}°E (Bornholm)")
    print(f"Output:   {output_path}")

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(str(output_path))

    print(f"Download complete: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 data for Bornholm.")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",   default=DEFAULT_END,   help="End date   (YYYY-MM-DD)")
    args = parser.parse_args()
    fetch_era5(start=args.start, end=args.end)
