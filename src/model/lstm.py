import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, args, num_classes, chans=62, dropout_rate=0.5):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.chans = chans
        self.hidden_size = 256
        self.num_freq_bands = 5  # Delta, Theta, Alpha, Beta, Gamma
        self.input_size = chans * self.num_freq_bands  # 62 channels * 5 frequency bands
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final linear layer
        self.fc = nn.Linear(self.hidden_size, num_classes)
    
    def forward(self, x):
        # Input shape: [batch_size, channels*bands, freq_points]
        # Reshape to [batch_size, freq_points, channels*bands] for LSTM
        x = x.permute(0, 2, 1)
        
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        # Take the final time step's output
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
