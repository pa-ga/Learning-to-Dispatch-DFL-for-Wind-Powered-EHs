import torch
import torch.nn as nn
from dataloader import N_MODEL_INPUTS, N_PREDICT

N_OUTPUT_FEATURES = 1    # ActivePower_WF_MW


class Seq2SeqLSTM(nn.Module):
    """Autoregressive encoder-decoder LSTM (Seq2Seq).

    Encoder
    -------
    Reads the full look-back window of weather + time features and compresses
    it into a hidden state (h, c).

    Decoder
    -------
    Generates the forecast autoregressively: at each step it receives its own
    previous prediction (wind power, 1 feature) and the carried hidden state.
    The first decoder step starts from a zero token.


    """

    def __init__(self, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        _dropout = dropout if num_layers > 1 else 0.0

        self.encoder_lstm = nn.LSTM(
            input_size=N_MODEL_INPUTS, #number of features per timestep
            hidden_size=hidden_size, #how many hidden units the LSTM should use
            num_layers=num_layers, #number of layers that are stacked in the LSTM
            batch_first=True, #input shape: (B, 288, 13) 
            dropout=_dropout,
        )

        #  (not full features).
        self.decoder_lstm = nn.LSTM(
            input_size= N_OUTPUT_FEATURES, #Decoder receives only the previous prediction input size = 1 (ActivePower_WF_MW)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=_dropout,
        )

        self.l_out = nn.Linear(hidden_size, N_OUTPUT_FEATURES) 

    def forward(self, x):
        B = x.size(0)

        
        _, (h, c) = self.encoder_lstm(x)   # h, c: (num_layers, B, hidden)

       
        decoder_input = torch.zeros(B, 1, N_OUTPUT_FEATURES, device=x.device) #shape: B × 1 × 1

        outputs = []
        for _ in range(N_PREDICT):
            out, (h, c) = self.decoder_lstm(decoder_input, (h, c))
            y = self.l_out(out)             # (B, 1, 1)
            outputs.append(y)
            decoder_input = y               # autoregressive feedback

        return torch.cat(outputs, dim=1)    # (B, N_PREDICT, 1)
