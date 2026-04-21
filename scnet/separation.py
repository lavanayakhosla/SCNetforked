import torch
import torch.nn as nn
from torch.nn.modules.rnn import LSTM


class CumulativeLayerNorm(nn.Module):
    """
    Cumulative Layer Normalization (cLN) for causal/real-time processing.
    At each time step t, normalization statistics are computed using only
    frames 0..t (no future information).
    
    Supports 3D input (B, C, T) and 4D input (B, C, F, T).
    """
    def __init__(self, num_channels, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        if x.dim() == 3:
            return self._forward_3d(x)
        elif x.dim() == 4:
            return self._forward_4d(x)
        else:
            raise ValueError(f"CumulativeLayerNorm expects 3D or 4D input, got {x.dim()}D")

    def _forward_3d(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        step = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, 1, T)
        
        # Cumulative sum along time
        cum_sum = torch.cumsum(x, dim=-1)        # (B, C, T)
        cum_sum_sq = torch.cumsum(x.pow(2), dim=-1)  # (B, C, T)
        
        # Mean and variance over C, cumulative over T
        count = step * C  # (1, 1, T)
        cum_mean = cum_sum.sum(dim=1, keepdim=True) / count    # (B, 1, T)
        cum_var = cum_sum_sq.sum(dim=1, keepdim=True) / count - cum_mean.pow(2)
        cum_var = torch.clamp(cum_var, min=0)
        
        x = (x - cum_mean) / (cum_var + self.eps).sqrt()
        x = x * self.weight.view(1, C, 1) + self.bias.view(1, C, 1)
        return x

    def _forward_4d(self, x):
        # x: (B, C, F, T)
        B, C, F, T = x.shape
        step = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, 1, 1, T)
        
        # Cumulative sum along time
        cum_sum = torch.cumsum(x, dim=-1)        # (B, C, F, T)
        cum_sum_sq = torch.cumsum(x.pow(2), dim=-1)  # (B, C, F, T)
        
        # Mean and variance over C and F, cumulative over T
        count = step * C * F  # (1, 1, 1, T)
        cum_mean = cum_sum.sum(dim=(1, 2), keepdim=True) / count    # (B, 1, 1, T)
        cum_var = cum_sum_sq.sum(dim=(1, 2), keepdim=True) / count - cum_mean.pow(2)
        cum_var = torch.clamp(cum_var, min=0)
        
        x = (x - cum_mean) / (cum_var + self.eps).sqrt()
        x = x * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        return x


class DualPathRNN(nn.Module):
    """  
    Causal Dual-Path RNN for Online SCNet.
    
    - Frequency-path: Bidirectional LSTM (full frequency band is available).
    - Time-path: Unidirectional LSTM (causal — no future frames).
    - Uses Cumulative Layer Normalization (cLN) instead of GroupNorm.

    Args:
        d_model (int): The number of expected features in the input.
        expand (int): Expansion factor for LSTM hidden size.
    """
    def __init__(self, d_model, expand):
        super(DualPathRNN, self).__init__()

        self.d_model = d_model
        self.hidden_size = d_model * expand

        # Frequency-path: bidirectional (full frequency info is available)
        self.lstm_freq = LSTM(d_model, self.hidden_size, num_layers=1,
                              bidirectional=True, batch_first=True)
        self.linear_freq = nn.Linear(self.hidden_size * 2, d_model)
        self.norm_freq = CumulativeLayerNorm(d_model)

        # Time-path: unidirectional (causal — no future information)
        self.lstm_time = LSTM(d_model, self.hidden_size, num_layers=1,
                              bidirectional=False, batch_first=True)
        self.linear_time = nn.Linear(self.hidden_size, d_model)
        self.norm_time = CumulativeLayerNorm(d_model)

    def forward(self, x):
        B, C, F, T = x.shape

        # Frequency-path
        original_x = x
        x = self.norm_freq(x)
        x = x.transpose(1, 3).contiguous().view(B * T, F, C)
        x, _ = self.lstm_freq(x)
        x = self.linear_freq(x)
        x = x.view(B, T, F, C).transpose(1, 3)
        x = x + original_x

        # Time-path (causal)
        original_x = x
        x = self.norm_time(x)
        x = x.transpose(1, 2).contiguous().view(B * F, C, T).transpose(1, 2)
        x, _ = self.lstm_time(x)
        x = self.linear_time(x)
        x = x.transpose(1, 2).contiguous().view(B, F, C, T).transpose(1, 2)
        x = x + original_x

        return x


class SeparationNet(nn.Module):
    """
    Causal Separation Network for Online SCNet.
    
    - All DPRNN layers use the same channel dimension (no FFT doubling).
    - FeatureConversion (FFT/iFFT) between layers is REMOVED for real-time operation.
    
    Args:
    - channels (int): Number of input channels.
    - expand (int): Expansion factor for LSTM hidden size.
    - num_layers (int): Number of dual-path layers.
    """
    def __init__(self, channels, expand=1, num_layers=6):
        super(SeparationNet, self).__init__()

        self.num_layers = num_layers

        # All layers use same channels (no FFT-based channel doubling)
        self.dp_modules = nn.ModuleList([
            DualPathRNN(channels, expand) for _ in range(num_layers)
        ])

    def forward(self, x):
        for dp in self.dp_modules:
            x = dp(x)
        return x
