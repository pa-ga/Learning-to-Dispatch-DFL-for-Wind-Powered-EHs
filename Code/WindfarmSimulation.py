

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import pyarrow
try:
    from scipy.interpolate import interp1d
except ImportError:
    interp1d = None
from pathconfig import get_wf_data_path






P_RATED_WF_MW = 3000e6 
P_FARM_MW_MAX= 3800e6


WAKE_LOSS = 0.10      # wake loss in %
ELEC_LOSS = 0.03      # electrical losses in %

LOSS_FACTOR = (1 - WAKE_LOSS) * (1 - ELEC_LOSS)


AVAIL_MEAN = 0.98    # average availability
AVAIL_SIGMA = 0.006  # variability around mean 
PHI = 0.97           # persistence (close to 1 => clustered downtime)

NOISE_SIGMA = 0.015  # small multiplicative noise on farm output

SEED = 42
rng = np.random.default_rng(SEED)






V_CI = 3.0        # cut-in 
V_R  = 10.69      # rated wind speed 
V_CO = 25.0       # cut-out 

P_RATED = 15e6    # rated electrical power per WT [W]
ETA_TOT = 0.965    # drivetrain and generator efficiency 
RHO_REF = 1.225   # reference density [kg/m^3]


D_ROTOR = 240.0   
A_ROTOR = np.pi * (D_ROTOR / 2.0) ** 2  # swept area of rotor [m^2]



def wf_capacity(WF_Cap):
    if WF_Cap == P_RATED_WF_MW:
        N_TURBINES = int(round(P_RATED_WF_MW / P_RATED)) # 3 GW / 15 MW
    elif WF_Cap == P_FARM_MW_MAX:
        N_TURBINES = int(round(P_FARM_MW_MAX / P_RATED)) # 3.8 GW / 15 MW
    else:
        raise ValueError(f"Unsupported WF_Cap={WF_Cap}. Expected {P_RATED_WF_MW} or {P_FARM_MW_MAX}.")
    return N_TURBINES



def preprocess_ERA5(df):
    """Reformat time column. Expects wind_speed, wind_dir and rho to already
    be present (computed by ERA5_DataScaling.py)."""
    df["time"] = pd.to_datetime(df["valid_time"])
    df = df.drop(columns=["valid_time"])
    return df



Cp_star = P_RATED / (ETA_TOT * 0.5 * RHO_REF * A_ROTOR * V_R**3)




def wt_power_piecewise(v: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Calculate WT power output using piecewise power curve with cut-in, rated and cut-out wind speeds."""
    v = np.asarray(v, dtype=float)
    rho = np.asarray(rho, dtype=float)

    P = np.zeros_like(v, dtype=float)

    m_part  = (v >= V_CI) & (v < V_R)
    m_rated = (v >= V_R) & (v <= V_CO)

    if np.any(m_part):
        P_rot = 0.5 * rho[m_part] * A_ROTOR * Cp_star * v[m_part]**3
        P[m_part] = np.minimum(ETA_TOT * P_rot, P_RATED)

    P[m_rated] = P_RATED

    return P




def powercurveto_df(df, N_TURBINES):

   df["ActivePower_WT"] = wt_power_piecewise(
    df["wind_speed"].values,
    df["rho"].values
    )/1e6

   df["ActivePower_WF_ideal"] = N_TURBINES * df["ActivePower_WT"]
   df["ActivePower_WF_losses"] = df["ActivePower_WF_ideal"] * LOSS_FACTOR

   T = len(df)
   avail = np.empty(T, dtype=float)
   avail[0] = AVAIL_MEAN
   eps = rng.normal(0.0, AVAIL_SIGMA, size=T)
   for t in range(1, T):
      avail[t] = AVAIL_MEAN + PHI * (avail[t-1] - AVAIL_MEAN) + eps[t]

   df["avail"] = np.clip(avail, 0.90, 1.00)

   noise_factor = 1.0 + rng.normal(0.0, NOISE_SIGMA, size=T)
   df["noise_factor"] = np.clip(noise_factor, 0.90, 1.10)
   

   df["ActivePower_WF_MW"] = (
       df["ActivePower_WF_losses"] * df["avail"] * df["noise_factor"]
   )
   return df






def plot_powercurve(df):
    plt.figure()
    plt.scatter(df["wind_speed"], df["ActivePower_WT"])
    plt.xlabel("Wind speed [m/s]")
    plt.ylabel("Active Power [W]")
    plt.savefig("PowerCurveWT.pdf", format="pdf")


    
    plt.figure()
    plt.scatter(df["wind_speed"], df["ActivePower_WF_MW"])
    plt.xlabel("Wind speed [m/s]")
    plt.ylabel("Active Power [W]")
    plt.savefig("PowerCurveWF.pdf", format="pdf")
    plt.close("all")



def call_all(df, WF_cap, scenario_name: str = "3gw"):
    df = preprocess_ERA5(df)
    df = powercurveto_df(df, N_TURBINES=wf_capacity(WF_cap))
    plot_powercurve(df)
    out_path = get_wf_data_path(scenario_name)
    df.to_parquet(out_path)
    print(f"WF_Data saved to {out_path}  ({len(df):,} rows)")
    return df


if __name__ == '__main__':
    from pathconfig import RAW_DATA_DIR
    from ERA5_DataScaling import ERA5_ZIP, ERA5_PROCESSED, scale_era5
    import fetch_era5

    print("Effective Cp* =", Cp_star)

  
    if not fetch_era5.ERA5_OUTPUT.exists():
        fetch_era5.fetch_era5()

   
    if not ERA5_PROCESSED.exists():
        processed = scale_era5(ERA5_ZIP)
        processed.to_csv(ERA5_PROCESSED, index=False)
        print(f"Processed ERA5 saved to {ERA5_PROCESSED}")


    weather_df = pd.read_csv(ERA5_PROCESSED)
    df = call_all(weather_df, P_RATED_WF_MW, scenario_name="3gw")