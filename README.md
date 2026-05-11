
# Learning-to-Dispatch-DFL-for-Wind-Powered-EHs

This repository contains all data and code to run and reproduce the results presented in the master's thesis "Learning to Dispatch: Decision Focused Learning for Wind Powered Energy Hubs - The Energy Island of Bornholm a Hypothetical Case Study".

## Environment Setup

Requires Python 3.11 or higher.

The required packages to install are:
```
pip install numpy pandas scikit-learn joblib tqdm matplotlib pyarrow fastparquet xarray linopy highspy cdsapi scipy torch
```

For GPU support, install PyTorch with the matching CUDA version.

## Problem Statement

The objective of this project is to investigate decision focused learning (DFL) in the form of a SPO+ adaption by [Elmachtoub & Grigas (2022)](https://doi.org/10.1287/mnsc.2020.3922) to the case of energy island optimisation. The SPO approach is then compared to traditional two-stage forecast accuracy based methods (also called predict then optimise pipelines). 

## Reproducing Results

The "evaluation.ipynb" notebook reproduces the obtained results. It can reproduce the different scenario results in detail by changing the name of the scenario in the second cell. 

Note: The full pipeline from downloading the data to training and running the pipelines must be run before the notebook can reproduce results.


## Setup

Before running the models the ERA5 data needs to be fetched and preprocessed via an API setup. 
For the first time API connection an account needs to be created on the Copernicus website and an API key needs to be generated.

Create your:
```
nano ~/.cdsapirc
```

Configure:
```
url: https://cds.climate.copernicus.eu/api
API Key: insert personal API key
```

For detailed instructions and sign up follow the guidelines on [Climate Data Storage CDSAPI setup](https://cds.climate.copernicus.eu/how-to-api)

- Run `fetch_era5.py` and `ERA5_DataScaling.py` once to add a zip and unzipped version of the downloaded ERA5 data (Copernicus CDS, 2025) as well as a preprocessed and scaled "era5_preprocessed.csv".
- Run `main.py --simulate --wf-scenario 3gw` and `main.py --simulate --wf-scenario 3800mw` once to produce and save the synthetic 3GW and 3.8GW aggregated wind farm time-series as parquets.

-  `Data/`: Contains the "Spotprices.csv" which provides the day ahead spot prices needed for the range of 2018-2025 downloaded from SMARD (Bundesnetzagentur, 2026).
-  Run `main.py --preprocess` and `main.py --preprocess --wf-scenario 3800mw` to preprocess the data: Produces and saved preprocessed data splits (train, val and test) and scaler for the two different capacity scenarios.


## Running the Pipeline

- `main.py`: The file is the main file to run the models and preprocessing from.
 
Main Commands Needed To Run Pipelines from `main.py`:


### Single Training Run:
  `python main.py --train INSERT MODEL NAME`
    e.g., for best identified model: `python main.py --train medium_autoreg_lstm_reg_overlap`

### Run Family of Models:
  `python main.py --train-all INSERT MODEL FAMILY NAME`
    checkout family names in `configurations.py`

For running training on the 3,800 MW scenario add `--wf-scenario 3800mw`
  e.g., `python main.py --train medium_autoreg_lstm_reg_overlap --wf-scenario 3800mw`


### Run Optimisation on Oracle
  `python main.py --oracle --split INSERT SPLIT` 
    e.g., `python main.py --oracle --split test`

For running on the 3,800 MW scenario add `--wf-scenario 3800mw` to the command.
For running on different hydrogen scenarios add `--scenario SCENARIO NAME` to the command.
  e.g., `python main.py --oracle --split test --wf-scenario 3800mw --scenario high_h2_price`

The following hydrogen scenarios are available:  {low_h2_price, baseline, high_h2_price}

### Run Predict-Then-Optimise Pipeline
  `python main.py --pto INSERT MODEL NAME`
    e.g., for best identified model: `python main.py --pto medium_autoreg_lstm_reg_overlap`

  For running on different hydrogen scenarios or capacity scenarios add them as indicated before.
    e.g., `python main.py --pto medium_autoreg_lstm_reg_overlap --wf-scenario 3800mw --scenario baseline_3800mw`

  Requires the LSTM to already be trained. The pipeline runs on test data split. 

  Alternatively `python main.py --all INSERT MODEL NAME` trains on the indicated model and then runs the optimisation on test in sequence.

### Run Decision-Focused Learning Pipeline

`python main.py --dfl INSERT MODEL NAME --scenario INSERT SCENARIO NAME`

Important: This trains the model and then directly runs evaluation, to solely run training add after the scenario indication `--no-eval`
  - e.g., `python main.py --dfl medium_autoreg_lstm_reg_overlap --scenario low_h2_price --no-eval`
    
For running on different scenario add `--wf-scenario 3800mw` before the scenario indication.
  - e.g., `python main.py --dfl medium_autoreg_lstm_reg_overlap --wf-scenario 3800mw --scenario low_h2_price --no-eval`


To run evaluation on test split replace `--no-eval` with `--eval-only`. 
If indication is missing the model automatically runs training and then test evaluation in sequence.
  


## File Overview    

- `Code/` contains the `main.py` and all supporting functions as well as the data fetching and preprocessing py files. All included files are listed alphabetically below:
  -   `batch_train.py` -  LSTM model training script
  -   `configurations.py` - Model configurations registry
  -   `dataloader.py` - Sliding window PyTorch dataset for PTO pipeline
  -   `dfl_dataloader.py` - DFLDataset extending the PTO pipeline dataset with unscaled prices and timestamps for SPO+
  -   `ERA5_DataScaling.py` - ERA5 zip processing
  -   `evaluation.ipynb` - Reproduction of results notebook
  -   `fetch_era5.py` - CDS API download script
  -   `inference.py` - model inference PTO
  -   `main.py` - CLI entry point
  -   `model_autoregressive_LSTM.py` - Seq2Seq encoder-decoder LSTM
  -   `model_lstm.py` - standard LSTM
  -   `model_selection.py` - ranks models based on MSE and produces loss plots
  -   `optimization.py` - LP dispatch model
  -   `pathconfig.py` - centralised path configuration
  -   `pipeline_dfl.py` - decision focused learning (SPO+) pipeline
  -   `pipeline_oracle.py` - perfect-foresight benchmark
  -   `pipeline_pto.py` - predict-then-optimise pipeline
  -   `preprocessing.py` - data merging, 70/15/15 splitting and data scaling
  -   `scenarios.py` - optimisation scenario configurations (LP parameters)
  -   `spo_loss.py` - SPOPlusLossFunction, SPOPlusLoss and Oracle Precompute
  -   `training_dfl.py` - SPO+ training loop
  -   `training.py` - training loop for LSTMs with early stopping
  -   `util.py` - general purpose helper functions
  -   `wf_scenarios.py` - wind farm scenario configurations
  -   `WindfarmSimulation.py` - ERA5 based wind farm power simulation









