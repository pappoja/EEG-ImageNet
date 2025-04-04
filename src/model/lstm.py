import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, args, num_classes, chans=62, num_freq_bands=5, dropout_rate=0.5):
        super(LSTM, self).__init__()

        self.chans = chans
        self.num_freq_bands = num_freq_bands
        self.input_size = chans * num_freq_bands  # Input size per timestep
        self.hidden_size = 128
        self.num_layers = 2

        # Two-layer LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout_rate
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, chans * bands, time]
        x = x.permute(0, 2, 1)  # --> [batch_size, time, features]

        lstm_out, _ = self.lstm(x)  # [batch_size, time, hidden]
        last_timestep = lstm_out[:, -1, :]  # Use final timestep
        out = self.dropout(last_timestep)
        out = self.fc(out)
        return out