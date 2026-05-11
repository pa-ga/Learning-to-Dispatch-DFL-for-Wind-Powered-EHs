
from pathlib import Path

PROJECT_DIR = Path(__file__).parent


ROOT = PROJECT_DIR.parent

RAW_DATA_DIR       = ROOT / "Data"
PROCESSED_DATA_DIR = ROOT / "processed_data"
RESULTS_DIR        = ROOT / "results"

# Default WF scenario, used by legacy constants below and as Command Line Interface default.
DEFAULT_WF_SCENARIO = "3gw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)





def get_wf_data_path(wf_scenario: str = DEFAULT_WF_SCENARIO) -> Path:
    """Path to the simulated wind farm parquet for a given WF scenario."""
    return RAW_DATA_DIR / f"WF_Data_{wf_scenario}.parquet"


def get_splits_dir(wf_scenario: str = DEFAULT_WF_SCENARIO) -> Path:
    """Data-splits directory for a given WF scenario."""
    d = ROOT / f"data_splits_{wf_scenario}"
    d.mkdir(exist_ok=True)
    return d


def get_checkpoints_dir(wf_scenario: str = DEFAULT_WF_SCENARIO) -> Path:
    """Model-checkpoints directory for a given WF scenario."""
    d = ROOT / f"checkpoints_{wf_scenario}"
    d.mkdir(exist_ok=True)
    return d





SPLITS_DIR      = get_splits_dir(DEFAULT_WF_SCENARIO)
CHECKPOINTS_DIR = get_checkpoints_dir(DEFAULT_WF_SCENARIO)
