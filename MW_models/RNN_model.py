import torch
import torch.nn as nn


class RNNRegressor(nn.Module):
    def __init__(
        self,
        input_size,
        dropout_rate=0.2,
        hidden_layers=None,
        rnn_hidden_size=128,
        rnn_num_layers=2,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 32]

        # Treat each original feature as one timestep with 1 feature per step
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            nonlinearity="relu",
            batch_first=True,
            dropout=dropout_rate if rnn_num_layers > 1 else 0.0,
        )

        layerlist = []
        n_in = rnn_hidden_size

        for h in hidden_layers:
            layerlist += [
                nn.Linear(n_in, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ]
            n_in = h

        layerlist.append(nn.Linear(n_in, 1))
        self.network = nn.Sequential(*layerlist)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: (batch, input_size)
        x = x.unsqueeze(-1)          # -> (batch, seq_len=input_size, 1)
        rnn_out, _ = self.rnn(x)
        x = rnn_out[:, -1, :]        # take last timestep
        return self.network(x).squeeze(-1)