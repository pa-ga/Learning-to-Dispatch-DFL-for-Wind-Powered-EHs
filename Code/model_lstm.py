import torch.nn as nn
from dataloader import N_MODEL_INPUTS, N_PREDICT


N_OUTPUT_FEATURES = 1    # ActivePower_WF_MW


class LSTMModel(nn.Module):
    """Standard (non-autoregressive) many-to-many LSTM.

    Reads the full look-back window and projects the last hidden state to the
    entire forecast horizon in one shot.

    """

    def __init__(self, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=N_MODEL_INPUTS,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, #(batch, 288,  13)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.l_out = nn.Linear(
            in_features=hidden_size,
            out_features=N_PREDICT * N_OUTPUT_FEATURES,
            bias=False,
        )

    def forward(self, x):
        out, _ = self.lstm(x)                        # (B, LOOKBACK, hidden)
        last   = out[:, -1, :]                       # (B, hidden)
        y      = self.l_out(last)                    # (B, N_PREDICT)
        return y.view(-1, N_PREDICT, N_OUTPUT_FEATURES)
