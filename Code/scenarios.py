




scenarios = [
    {
        "name":           "baseline",
   
        "P_GRID_MAX":     3000,    # MW  — export cable capacity
     
        "P_ELY_MAX":      800,     # MW
        "ETA_ELY":        0.68,    # efficiency
        "R_ELY":          100,     # MW/step ramp limit
  
        "BAT_POWER_MAX":  300,     # MW
        "BAT_ENERGY_MAX": 600,    # MWh
        "ETA_CH":         0.95,
        "ETA_DIS":        0.95,
  
        "H2_STORAGE_MAX": 2000,    # MWh_H2
        "H_sell_max":     600,     # MWh_H2/timestep 
        
        "pi_H2":          150,    # €/MWh_H2
        "C_RU":           5,       # €/MW ramp-up penalty
        "C_ELY":          3,       # €/MWh electrolyzer O&M
        "C_BAT":          2,       # €/MWh battery O&M
        "C_CURT":         100,     # €/MWh curtailment penalty
    
        "E0_H2":          0,
        "E0_BAT":         0,
    },
    {
        "name":           "low_h2_price",
        "P_GRID_MAX":     3000,
        "P_ELY_MAX":      800,
        "ETA_ELY":        0.68,
        "R_ELY":          100,
        "BAT_POWER_MAX":  300,     
        "BAT_ENERGY_MAX": 600,    
        "ETA_CH":         0.95,
        "ETA_DIS":        0.95,
        "H2_STORAGE_MAX": 2000,
        "H_sell_max":     600,     
        "pi_H2":          100,
        "C_RU":           5,
        "C_ELY":          3,
        "C_BAT":          2,
        "C_CURT":         100,
        "E0_H2":          0,
        "E0_BAT":         0,
    },
    {
        "name":           "high_h2_price",
        "P_GRID_MAX":     3000,
        "P_ELY_MAX":      800,
        "ETA_ELY":        0.68,
        "R_ELY":          100,
        "BAT_POWER_MAX":  300,
        "BAT_ENERGY_MAX": 600,
        "ETA_CH":         0.95,
        "ETA_DIS":        0.95,
        "H2_STORAGE_MAX": 2000,
        "H_sell_max":     600,    
        "pi_H2":          200.0,    
        "C_RU":           5,
        "C_ELY":          3,
        "C_BAT":          2,
        "C_CURT":         100,
        "E0_H2":          0,
        "E0_BAT":         0,
    },
    {
        "name":           "baseline_3800mw",
        "P_GRID_MAX":     3800,
        "P_ELY_MAX":      800,
        "ETA_ELY":        0.68,
        "R_ELY":          100,
        "BAT_POWER_MAX":  300,
        "BAT_ENERGY_MAX": 600,
        "ETA_CH":         0.95,
        "ETA_DIS":        0.95,
        "H2_STORAGE_MAX": 2000,
        "H_sell_max":     600,    
        "pi_H2":          150,
        "C_RU":           5,
        "C_ELY":          3,
        "C_BAT":          2,
        "C_CURT":         100,
        "E0_H2":          0,
        "E0_BAT":         0,
    },
    {
        "name":           "low_h2_price_3800mw",
        "P_GRID_MAX":     3800,
        "P_ELY_MAX":      800,
        "ETA_ELY":        0.68,
        "R_ELY":          100,
        "BAT_POWER_MAX":  300,
        "BAT_ENERGY_MAX": 600,
        "ETA_CH":         0.95,
        "ETA_DIS":        0.95,
        "H2_STORAGE_MAX": 2000,
        "H_sell_max":     600,     
        "pi_H2":          100,
        "C_RU":           5,
        "C_ELY":          3,
        "C_BAT":          2,
        "C_CURT":         100,
        "E0_H2":          0,
        "E0_BAT":         0,
    },
    {
        "name":           "high_h2_price_3800mw",
        "P_GRID_MAX":     3800,
        "P_ELY_MAX":      800,
        "ETA_ELY":        0.68,
        "R_ELY":          100,
        "BAT_POWER_MAX":  300,
        "BAT_ENERGY_MAX": 600,
        "ETA_CH":         0.95,
        "ETA_DIS":        0.95,
        "H2_STORAGE_MAX": 2000,
        "H_sell_max":     600,     
        "pi_H2":          200.0,
        "C_RU":           5,
        "C_ELY":          3,
        "C_BAT":          2,
        "C_CURT":         100,
        "E0_H2":          0,
        "E0_BAT":         0,
    },
]

DEFAULT_SCENARIO = "baseline"






def find_scenario(name: str) -> dict:
    """Return scenario config dict by name."""
    cfg = next((s for s in scenarios if s["name"] == name), None)
    if cfg is None:
        available = [s["name"] for s in scenarios]
        raise ValueError(f"No scenario named '{name}'. Available: {available}")
    return cfg
