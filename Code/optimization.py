

import numpy as np
import linopy as lp
import pandas as pd


# Standard technical parameters can be overridden by scenario dict passed to run_optimization().

P_GRID_MAX     = 3000   # MW
P_ELY_MAX      = 800    # MW
BAT_POWER_MAX  = 300    # MW
BAT_ENERGY_MAX = 600  # MWh
H2_STORAGE_MAX = 2000   # MWh_H2 -> w

pi_H2      = 120    # €/MWh_H2  (~4 €/kg × 30 kg/MWh_H2)
H_sell_max = 600    # MWh_H2/timestep — max production = 800×0.68×0.25 = 136, so 600 is non-binding
E0_H2      = 0 # initial H2 storage level [MWh_H2] 
E0_BAT     = 0 # initial battery SOC [MWh] 
DELTA_T    = 0.25 #15 minute resolutions
ETA_ELY    = 0.68 #electrolysis efficiency
ETA_CH     = 0.95 #charging efficiency
ETA_DIS    = 0.95 #discharging efficiency
R_ELY      = 100 # MW per 15 min — max change in electrolyser power between consecutive timesteps (ramp rate limit)
C_RU       = 5 # €/MW ramping cost for electrolyser 
C_ELY      = 3 # €/MWh cost for using the electrolyser 
C_BAT      = 2 # €/MWh cost for using the battery 
C_CURT     = 100 # €/MWh cost for curtailment





def run_optimization(
    P_wind: pd.Series,
    pi_el: pd.Series,
    solver: str = "highs",
    scenario: dict = None,
    silent: bool = False,
) -> dict:
    """Run the dispatch LP over the horizon defined by P_wind.index.

    Parameters
    ----------
    P_wind : pd.Series (DatetimeIndex)
        Wind farm power output [MW]. Pass actual values or LSTM predictions.
    pi_el  : pd.Series (DatetimeIndex)
        Day-ahead electricity price [€/MWh]. Must share the same index.
    solver : str
        LP solver passed to linopy (default: "highs").
    scenario : dict, optional
        Scenario config from scenarios.find_scenario(). If None, module-level
        defaults are used.

    Returns
    -------
    dict with keys: model, status, objective_value,
                    P_grid, P_curt, P_ch, P_dis, P_ely,
                    E_bat, H_prod, H_sell, E_H2, u
    """
    # resolve parameters — scenario overrides module-level defaults
    s = scenario or {}
    _P_GRID_MAX     = s.get("P_GRID_MAX",     P_GRID_MAX)
    _P_ELY_MAX      = s.get("P_ELY_MAX",      P_ELY_MAX)
    _BAT_POWER_MAX  = s.get("BAT_POWER_MAX",  BAT_POWER_MAX)
    _BAT_ENERGY_MAX = s.get("BAT_ENERGY_MAX", BAT_ENERGY_MAX)
    _H2_STORAGE_MAX = s.get("H2_STORAGE_MAX", H2_STORAGE_MAX)
    _ETA_ELY        = s.get("ETA_ELY",        ETA_ELY)
    _ETA_CH         = s.get("ETA_CH",         ETA_CH)
    _ETA_DIS        = s.get("ETA_DIS",        ETA_DIS)
    _R_ELY          = s.get("R_ELY",          R_ELY)
    _pi_H2          = s.get("pi_H2",          pi_H2)
    _H_sell_max     = s.get("H_sell_max",     H_sell_max)
    _E0_H2          = s.get("E0_H2",          E0_H2)
    _E0_BAT         = s.get("E0_BAT",         E0_BAT)
    _C_RU           = s.get("C_RU",           C_RU)
    _C_ELY          = s.get("C_ELY",          C_ELY)
    _C_BAT          = s.get("C_BAT",          C_BAT)
    _C_CURT         = s.get("C_CURT",         C_CURT)

    T = P_wind.index
    assert isinstance(T, pd.DatetimeIndex), "P_wind must have a DatetimeIndex"
    assert pi_el.index.equals(T), "pi_el must share the same DatetimeIndex as P_wind"

    weekly_end = pd.DatetimeIndex(T.to_series().groupby(T.to_period("W")).idxmax().values)

    m = lp.Model()

  #Variables
    P_grid = m.add_variables(lower=0, upper=_P_GRID_MAX,     coords={"time": T}, dims=["time"], name="P_grid")
    P_curt = m.add_variables(lower=0,                        coords={"time": T}, dims=["time"], name="P_curt")
    P_ch   = m.add_variables(lower=0, upper=_BAT_POWER_MAX,  coords={"time": T}, dims=["time"], name="P_ch")
    P_dis  = m.add_variables(lower=0, upper=_BAT_POWER_MAX,  coords={"time": T}, dims=["time"], name="P_dis")
    P_ely  = m.add_variables(lower=0, upper=_P_ELY_MAX,      coords={"time": T}, dims=["time"], name="P_ely")
    E_bat  = m.add_variables(lower=0, upper=_BAT_ENERGY_MAX, coords={"time": T}, dims=["time"], name="E_bat")
    H_prod = m.add_variables(lower=0,                        coords={"time": T}, dims=["time"], name="H_prod")
    H_sell = m.add_variables(lower=0,                        coords={"time": T}, dims=["time"], name="H_sell")
    E_H2   = m.add_variables(lower=0, upper=_H2_STORAGE_MAX, coords={"time": T}, dims=["time"], name="E_H2")
    r_up   = m.add_variables(lower=0,                        coords={"time": T}, dims=["time"], name="r_up")

 
#Constraints

    m.add_constraints(P_grid + P_ely + P_ch + P_curt == P_wind + P_dis,           name="energy_balance")
    m.add_constraints(P_grid <= _P_GRID_MAX,                                       name="grid_export_limit")
    m.add_constraints(P_ely  <= _P_ELY_MAX,                                        name="ely_capacity")
    m.add_constraints(H_prod == _ETA_ELY * P_ely * DELTA_T,                       name="h2_production")
    
    m.add_constraints(P_ely.isel(time=0) <= _R_ELY,                               name="ely_ramp_initial")
    m.add_constraints((P_ely - P_ely.shift(time=1)) <= _R_ELY,                    name="ely_ramp_limit")
    m.add_constraints(r_up.isel(time=0) >= P_ely.isel(time=0),                       name="ramp_up_aux_initial")
    m.add_constraints(r_up >= (P_ely - P_ely.shift(time=1)),                         name="ramp_up_aux")

 
    m.add_constraints(E_H2 == E_H2.shift(time=1) + H_prod - H_sell,               name="h2_storage_balance")
    m.add_constraints(E_H2   <= _H2_STORAGE_MAX,                                  name="h2_storage_limit")
    m.add_constraints(H_sell <= _H_sell_max,                                       name="h2_offtake_limit")
  
    m.add_constraints(
        E_H2.isel(time=0) == _E0_H2 + H_prod.isel(time=0) - H_sell.isel(time=0), name="h2_initial")
    m.add_constraints(E_H2.loc[weekly_end] == _E0_H2,                             name="weekly_reset")


    m.add_constraints(
        E_bat == E_bat.shift(time=1)
               + _ETA_CH        * P_ch  * DELTA_T
               - (1 / _ETA_DIS) * P_dis * DELTA_T,                                name="battery_balance")
    m.add_constraints(E_bat  <= _BAT_ENERGY_MAX,                                  name="battery_capacity")
    m.add_constraints(P_ch   <= _BAT_POWER_MAX,                                   name="battery_charge_limit")
    m.add_constraints(P_dis  <= _BAT_POWER_MAX,                                   name="battery_discharge_limit")

    m.add_constraints(
        E_bat.isel(time=0) == _E0_BAT
                              + _ETA_CH        * P_ch.isel(time=0)  * DELTA_T
                              - (1 / _ETA_DIS) * P_dis.isel(time=0) * DELTA_T,    name="battery_initial")

    # Objective Function

    profit_t = (
        DELTA_T * (
            pi_el * P_grid
            - _C_ELY * P_ely
            - _C_BAT * (P_ch + P_dis)
            - _C_CURT * P_curt
        )
        + _pi_H2 * H_sell
    )


    m.add_objective(
        profit_t.sum() - _C_RU * r_up.sum(),
        sense="max",
    )

    # Solve Optimisation Model

    solver_opts = {"output_flag": False} if silent else {}
    status, condition = m.solve(solver_name=solver, **solver_opts)

    if str(condition) != "optimal":
        import warnings
        warnings.warn(
            f"LP solve did not reach optimality: status={status}, condition={condition}. "
            f"Returning zero-profit fallback for this window ({T[0]} → {T[-1]})."
        )
        zero = pd.Series(0.0, index=T)
        return {
            "model":           m,
            "status":          str(status),
            "objective_value": 0.0,
            "P_grid": zero, "P_curt": zero, "P_ch":   zero, "P_dis":  zero,
            "P_ely":  zero, "E_bat":  zero, "H_prod": zero, "H_sell": zero,
            "E_H2":   zero, "r_up":      zero,
        }

    def _sol(var) -> pd.Series:
        return pd.Series(var.solution.values, index=T)

    sol_grid  = _sol(P_grid)
    sol_curt  = _sol(P_curt)
    sol_ch    = _sol(P_ch)
    sol_dis   = _sol(P_dis)
    sol_ely   = _sol(P_ely)
    sol_ebat  = _sol(E_bat)
    sol_hprod = _sol(H_prod)
    sol_hsell = _sol(H_sell)
    sol_eh2   = _sol(E_H2)
    sol_r_up     = _sol(r_up)


    
    obj_val = float(
        DELTA_T * (
            (pi_el * sol_grid).sum()
            - _C_ELY  * sol_ely.sum()
            - _C_BAT  * (sol_ch + sol_dis).sum()
            - _C_CURT * sol_curt.sum()
        )
        + _pi_H2 * sol_hsell.sum()
        - _C_RU * sol_r_up.sum()   # OLD (buggy): sol_r_up.iloc[1:] — skipped t=0 ramp cost
    )

    return {
        "model":           m,
        "status":          str(status),
        "objective_value": obj_val,
        "P_grid":  sol_grid,
        "P_curt":  sol_curt,
        "P_ch":    sol_ch,
        "P_dis":   sol_dis,
        "P_ely":   sol_ely,
        "E_bat":   sol_ebat,
        "H_prod":  sol_hprod,
        "H_sell":  sol_hsell,
        "E_H2":    sol_eh2,
        "r_up":    sol_r_up,
    }


def compute_realized_profit(dispatch: pd.DataFrame, scenario: dict = None) -> float:
    """Re-evaluate committed dispatch decisions against actual wind power.

    After PTO / DFL optimises with predicted wind, this re-evaluates those
    decisions with the actual wind that materialised.

    * actual > predicted  -  excess is curtailed (P_curt increases, costs rise)
    * actual < predicted  -  three tier shedding of commitments:
        1. reduce P_grid 
        2. reduce P_ely 
        3. reduce P_ch 
       If deficit remains after zeroing P_grid, it's an unresolvable violation of the dispatch.

    Parameters
    ----------
    dispatch : pd.DataFrame
        Output DataFrame from run_pto / run_dfl.  Required columns:
        P_wind_actual, P_grid, P_ch, P_dis, P_ely, H_sell, spot_price.
    scenario : dict, optional
        Same scenario dict passed to run_optimization.

    Returns
    -------
    dict with keys:
        profit          : float        — realised profit [€]
        P_grid_realized : np.ndarray   — realised grid export [MW] (≤ planned P_grid)
        P_curt_realized : np.ndarray   — realised curtailment [MW] (≥ planned P_curt)
        n_violations    : int          — timesteps where deficit couldn't be covered by zeroing P_grid
        worst_deficit   : float        — largest unresolvable deficit [MW] (0 if none)
    """
    s      = scenario or {}
    _pi_H2  = s.get("pi_H2",  pi_H2)
    _C_ELY  = s.get("C_ELY",  C_ELY)
    _C_BAT  = s.get("C_BAT",  C_BAT)
    _C_CURT = s.get("C_CURT", C_CURT)
    _C_RU   = s.get("C_RU",   C_RU)

    actual = dispatch["P_wind_actual"].values
    pg     = dispatch["P_grid"].values
    p_dis  = dispatch["P_dis"].values
    p_ely  = dispatch["P_ely"].values
    p_ch   = dispatch["P_ch"].values
    h_sell = dispatch["H_sell"].values
    prices = dispatch["spot_price"].values

   
    surplus  = actual + p_dis - pg - p_ely - p_ch


    p_grid_r = np.maximum(pg + np.minimum(surplus, 0.0), 0.0)
    p_curt_r = np.maximum(surplus, 0.0)


    unresolved_deficit = np.maximum(-(surplus + pg), 0.0)

    p_ely_shed = np.minimum(p_ely, unresolved_deficit)
    p_ely_r    = p_ely - p_ely_shed
    remaining  = unresolved_deficit - p_ely_shed
    p_ch_r     = np.maximum(p_ch - remaining, 0.0)


    h_sell_r = np.where(p_ely > 0, h_sell * (p_ely_r / np.maximum(p_ely, 1e-9)), h_sell)
    h_sell_r = np.minimum(h_sell_r, h_sell)


    n_violations  = int((unresolved_deficit > 0.01).sum())
    worst_deficit = float(unresolved_deficit.max()) if n_violations > 0 else 0.0
    if n_violations > 0:
        print(f"[realized profit] Load-shedding applied at {n_violations}/{len(surplus)} timesteps  |  worst: {worst_deficit:.1f} MW  (P_ely/P_ch reduced)")

   
    ramp = np.diff(p_ely_r, prepend=0.0)
    r_r_up  = np.maximum(ramp, 0.0)

  
    profit = (
        DELTA_T * (
            (prices * p_grid_r).sum()
            - _C_ELY  * p_ely_r.sum()
            - _C_BAT  * (p_ch_r + p_dis).sum()
            - _C_CURT * p_curt_r.sum()
        )
        + _pi_H2  * h_sell_r.sum()
        - _C_RU * r_r_up.sum()   
    )

    return {
        "profit":          float(profit),
        "P_grid_realized": p_grid_r,
        "P_curt_realized": p_curt_r,
        "P_ely_realized":  p_ely_r,
        "P_ch_realized":   p_ch_r,
        "H_sell_realized": h_sell_r,
        "n_violations":    n_violations,
        "worst_deficit":   worst_deficit,
    }
