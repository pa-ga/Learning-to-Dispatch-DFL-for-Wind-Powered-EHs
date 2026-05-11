



wf_scenarios = [
    {
        "name":        "3gw",
        "WF_cap":      3000e6,   
        "description": "3 GW baseline  (200 × IEA-15 MW turbines)",
    },
    {
        "name":        "3800mw",
        "WF_cap":      3800e6,  
        "description": "3.8 GW expansion  (253 × IEA-15 MW turbines)",
    },
]

DEFAULT_WF_SCENARIO = "3gw"





def find_wf_scenario(name: str) -> dict:
    """Return WF scenario config dict by name."""
    cfg = next((s for s in wf_scenarios if s["name"] == name), None)
    if cfg is None:
        available = [s["name"] for s in wf_scenarios]
        raise ValueError(f"No WF scenario named '{name}'. Available: {available}")
    return cfg
