

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm

from optimization import run_optimization






class SPOPlusFunction(torch.autograd.Function):
    """Single-sample SPO+ forward + backward.

    Inputs (via apply):
        pred         (N_PREDICT,) float tensor  — LSTM prediction [MW], requires_grad
        true_wind    (N_PREDICT,) float tensor  — ground truth [MW]
        prices       (N_PREDICT,) float tensor  — spot prices [€/MWh]
        times_ns     (N_PREDICT,) int64 tensor  — UTC timestamps as int64 nanoseconds
        oracle_value  float                     — V(P_true), precomputed
        scenario      dict | None               — optimization scenario config

    Returns:
        loss  scalar float tensor  = V(P_true) − V(2·P̂ − P_true)
    """

    @staticmethod
    def forward(ctx, pred, true_wind, prices, times_ns, oracle_value, scenario, target_scaler):
        pred_np  = pred.detach().cpu().numpy() # Move to CPU numpy for the LP solver.
        true_np  = true_wind.detach().cpu().numpy()
        price_np = prices.detach().cpu().numpy()
        time_idx = pd.DatetimeIndex(times_ns.numpy().astype("datetime64[ns]"))

        
        if target_scaler is not None: # Inverse-transform from scaled space to MW so the LP receives physical values.
            scale   = float(target_scaler.scale_[0])
            mean    = float(target_scaler.mean_[0])
            pred_mw = pred_np * scale + mean
            true_mw = true_np * scale + mean
        else:
            scale   = 1.0
            pred_mw = pred_np
            true_mw = true_np


       
        perturbed_np = np.maximum(0.0, 2.0 * pred_mw - true_mw)  # The LP requires P_wind ≥ 0; clamping restricts SPO+ to the physical domain.
        perturbed    = pd.Series(perturbed_np, index=time_idx)
        price_ser    = pd.Series(price_np,     index=time_idx)

        opt = run_optimization(perturbed, price_ser, scenario=scenario, silent=True)

        
        duals = (
            opt["model"]
            .constraints["energy_balance"]
            .dual
            .values
            .astype(np.float32)           # (N_PREDICT,) - Extract dual variables of the energy balance constraint 
        )

      
        del opt["model"]

        ctx.save_for_backward(torch.from_numpy(duals))
   
        ctx.target_scaler_scale = scale

        loss = float(oracle_value) - opt["objective_value"]
        return torch.tensor(loss, dtype=pred.dtype, device=pred.device)

    @staticmethod
    def backward(ctx, grad_output):
     
        duals, = ctx.saved_tensors
        scale = ctx.target_scaler_scale
        grad_pred = -2.0 * duals.to(grad_output.device) * grad_output * scale
 
        return grad_pred, None, None, None, None, None, None






class SPOPlusLoss(nn.Module):
    """SPO+ surrogate loss for end-to-end decision-focused learning.

    Loops over samples in the batch (LP solves cannot be batched), then
    returns the mean loss across the batch.

    """

    def __init__(self, scenario: dict = None, target_scaler=None):
        super().__init__()
        self.scenario = scenario
        self.target_scaler = target_scaler

    def forward(
        self,
        pred: torch.Tensor,       # (B, N_PREDICT) or (N_PREDICT,)  [scaled]
        true_wind: torch.Tensor,  # same shape                       [scaled]
        prices: torch.Tensor,     # same shape                       [€/MWh]
        times_ns: torch.Tensor,   # same shape                       [int64 ns]
        oracle_values: torch.Tensor,  # (B,) or scalar               [€]
    ) -> torch.Tensor:
        if pred.dim() == 1:
            # Single sample.
            return SPOPlusFunction.apply(
                pred, true_wind, prices, times_ns,
                oracle_values.item(), self.scenario, self.target_scaler,
            )


        B = pred.size(0)
        losses = [
            SPOPlusFunction.apply(
                pred[i], true_wind[i], prices[i], times_ns[i],
                oracle_values[i].item(), self.scenario, self.target_scaler,
            )
            for i in range(B)
        ]
        return torch.stack(losses).mean()





def precompute_oracle_values(
    dataset,
    scenario: dict = None,
    desc: str = "Precomputing oracle LP values",
    target_scaler=None,
) -> torch.Tensor:
    """Solve the oracle LP (true wind) for every window in `dataset`.

    This is done once before SPO+ training.  The results are passed to
    dataset.set_oracle_values() and used as the fixed V(P_true) term during
    each training step.

    """
    values = []

    for i in tqdm.trange(len(dataset), desc=desc):
        _, y, prices, times_ns, _ = dataset[i]

        time_idx = pd.DatetimeIndex(times_ns.numpy().astype("datetime64[ns]"))
        y_np     = y.squeeze(-1).numpy()

        # Inverse-transform from scaled space to MW.
        if target_scaler is not None:
            scale = float(target_scaler.scale_[0])
            mean  = float(target_scaler.mean_[0])
            y_mw  = y_np * scale + mean
        else:
            y_mw = y_np

        true_wind = pd.Series(y_mw,            index=time_idx)
        price_ser = pd.Series(prices.numpy(),  index=time_idx)

        opt = run_optimization(true_wind, price_ser, scenario=scenario, silent=True)
        values.append(opt["objective_value"])
        del opt["model"]

    return torch.tensor(values, dtype=torch.float32)
