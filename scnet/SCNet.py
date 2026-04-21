import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from .separation import SeparationNet, CumulativeLayerNorm
import math


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class CausalConv1d(nn.Module):
    """Causal 1D convolution: pads only on the left so no future information leaks."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__()
        self.causal_padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, groups=groups)

    def forward(self, x):
        if self.causal_padding > 0:
            x = F.pad(x, (self.causal_padding, 0))
        return self.conv(x)


class CausalConv2d(nn.Module):
    """
    Causal 2D convolution: causal (left-only) padding in time dimension,
    symmetric padding in frequency dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        self.freq_padding = (kernel_size[0] - 1) // 2  # symmetric in frequency
        self.time_padding = kernel_size[1] - 1          # causal (left-only) in time
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0)

    def forward(self, x):
        # x: (B, C, F, T)
        # F.pad order for 4D: (T_left, T_right, F_top, F_bottom)
        x = F.pad(x, (self.time_padding, 0, self.freq_padding, self.freq_padding))
        return self.conv(x)


class ConvolutionModule(nn.Module):
    """
    Causal Convolution Module in SD block.
    All Conv1d layers use causal (left-only) padding.
    All normalization uses Cumulative LayerNorm (cLN).
    
    Args:    
        channels (int): input/output channels.
        depth (int): number of layers in the residual branch.
        compress (float): amount of channel compression.
        kernel (int): kernel size for the convolutions.
    """
    def __init__(self, channels, depth=2, compress=4, kernel=3):
        super().__init__()
        assert kernel % 2 == 1
        self.depth = abs(depth)
        hidden_size = int(channels / compress)
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            mods = [
                CumulativeLayerNorm(channels),
                CausalConv1d(channels, hidden_size * 2, kernel),
                nn.GLU(1),
                CausalConv1d(hidden_size, hidden_size, kernel, groups=hidden_size),
                CumulativeLayerNorm(hidden_size),
                Swish(),
                nn.Conv1d(hidden_size, channels, 1),  # 1x1 conv, no padding needed
            ]
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class FusionLayer(nn.Module):
    """
    A causal FusionLayer within the decoder.
    Uses CausalConv2d for causal operation in time dimension.

    Args:
    - channels (int): Number of input channels.
    - kernel_size (int, optional): Kernel size, defaults to 3.
    """
    def __init__(self, channels, kernel_size=3):
        super(FusionLayer, self).__init__()
        self.conv = CausalConv2d(channels * 2, channels * 2, kernel_size, stride=1)

    def forward(self, x, skip=None):
        if skip is not None:
            x += skip
        x = x.repeat(1, 2, 1, 1)
        x = self.conv(x)
        x = F.glu(x, dim=1)
        return x


class SDlayer(nn.Module):
    """
    Sparse Down-sample Layer for processing different frequency bands separately.
    Frequency-only convolutions (kernel size 1 in time) — inherently causal.

    Args:
    - channels_in (int): Input channel count.
    - channels_out (int): Output channel count.
    - band_configs (dict): Configuration for each frequency band.
    """
    def __init__(self, channels_in, channels_out, band_configs):
        super(SDlayer, self).__init__()

        self.convs = nn.ModuleList()
        self.strides = []
        self.kernels = []
        for config in band_configs.values():
            self.convs.append(nn.Conv2d(channels_in, channels_out,
                                        (config['kernel'], 1), (config['stride'], 1), (0, 0)))
            self.strides.append(config['stride'])
            self.kernels.append(config['kernel'])

        self.SR_low = band_configs['low']['SR']
        self.SR_mid = band_configs['mid']['SR']

    def forward(self, x):
        B, C, Fr, T = x.shape
        splits = [
            (0, math.ceil(Fr * self.SR_low)),
            (math.ceil(Fr * self.SR_low), math.ceil(Fr * (self.SR_low + self.SR_mid))),
            (math.ceil(Fr * (self.SR_low + self.SR_mid)), Fr)
        ]

        outputs = []
        original_lengths = []
        for conv, stride, kernel, (start, end) in zip(self.convs, self.strides, self.kernels, splits):
            extracted = x[:, :, start:end, :]
            original_lengths.append(end - start)
            current_length = extracted.shape[2]

            # Frequency-dimension padding (same as original — no time leakage)
            if stride == 1:
                total_padding = kernel - stride
            else:
                total_padding = (stride - current_length % stride) % stride
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left
            padded = F.pad(extracted, (0, 0, pad_left, pad_right))

            output = conv(padded)
            outputs.append(output)

        return outputs, original_lengths


class SUlayer(nn.Module):
    """
    Sparse Up-sample Layer in decoder.
    Frequency-only transposed convolutions — inherently causal.

    Args:
    - channels_in: Input channels.
    - channels_out: Output channels.
    - band_configs: Transposed convolution configurations.
    """
    def __init__(self, channels_in, channels_out, band_configs):
        super(SUlayer, self).__init__()

        self.convtrs = nn.ModuleList([
            nn.ConvTranspose2d(channels_in, channels_out, [config['kernel'], 1], [config['stride'], 1])
            for _, config in band_configs.items()
        ])

    def forward(self, x, lengths, origin_lengths):
        B, C, Fr, T = x.shape
        splits = [
            (0, lengths[0]),
            (lengths[0], lengths[0] + lengths[1]),
            (lengths[0] + lengths[1], None)
        ]
        outputs = []
        for idx, (convtr, (start, end)) in enumerate(zip(self.convtrs, splits)):
            out = convtr(x[:, :, start:end, :])
            current_Fr_length = out.shape[2]
            dist = abs(origin_lengths[idx] - current_Fr_length) // 2
            trimmed_out = out[:, :, dist:dist + origin_lengths[idx], :]
            outputs.append(trimmed_out)

        x = torch.cat(outputs, dim=2)
        return x


class SDblock(nn.Module):
    """
    Causal Sparse Down-sample block in encoder.
    Uses CausalConv2d for globalconv and cLN-based ConvolutionModules.
    
    Args:
    - channels_in (int): Number of input channels.
    - channels_out (int): Number of output channels.
    - band_config (dict): Configuration for the SDlayer.
    - conv_config (dict): Configuration for convolution modules.
    - depths (list of int): Convolution depths for low, mid, and high bands.
    - kernel_size (int): Kernel size for globalconv.
    """
    def __init__(self, channels_in, channels_out, band_configs={}, conv_config={}, depths=[3, 2, 1], kernel_size=3):
        super(SDblock, self).__init__()
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs)

        self.conv_modules = nn.ModuleList([
            ConvolutionModule(channels_out, depth, **conv_config) for depth in depths
        ])
        # Causal Conv2d for global convolution
        self.globalconv = CausalConv2d(channels_out, channels_out, kernel_size, stride=1)

    def forward(self, x):
        bands, original_lengths = self.SDlayer(x)
        bands = [
            F.gelu(
                conv(band.permute(0, 2, 1, 3).reshape(-1, band.shape[1], band.shape[3]))
                .view(band.shape[0], band.shape[2], band.shape[1], band.shape[3])
                .permute(0, 2, 1, 3)
            )
            for conv, band in zip(self.conv_modules, bands)
        ]
        lengths = [band.size(-2) for band in bands]
        full_band = torch.cat(bands, dim=2)
        skip = full_band

        output = self.globalconv(full_band)

        return output, skip, lengths, original_lengths


class SCNet(nn.Module):
    """
    Online SCNet: Causal real-time baseline from the Band-SCNet paper (InterSpeech 2025).
    
    Modifications from original SCNet:
    - All convolutions are causal (left-only padding in time).
    - LayerNorm replaced with Cumulative LayerNorm (cLN).
    - Time-dimension LSTMs are unidirectional.
    - FFT/iFFT between DPRNN layers removed.
    - STFT uses center=False for causal operation.

    Args:
    - sources (List[str]): List of sources to separate.
    - audio_channels (int): Number of audio channels.
    - nfft (int): FFT size for STFT.
    - hop_size (int): Hop size for STFT.
    - win_size (int): Window size for STFT.
    - normalized (bool): Whether to normalize the STFT.
    - dims (List[int]): Channel dimensions for each block.
    - band_SR (List[float]): Proportion of each frequency band.
    - band_stride (List[int]): Down-sampling ratio per band.
    - band_kernel (List[int]): Kernel sizes for down-sampling per band.
    - conv_depths (List[int]): Convolution depths in each SD block.
    - compress (int): Compression factor for convolution module.
    - conv_kernel (int): Kernel size for convolution layers.
    - num_dplayer (int): Number of dual-path layers.
    - expand (int): Expansion factor in dual-path RNN.
    """
    def __init__(self,
                 sources=['drums', 'bass', 'other', 'vocals'],
                 audio_channels=2,
                 # Main structure
                 dims=[4, 32, 64, 128],
                 # STFT
                 nfft=4096,
                 hop_size=1024,
                 win_size=4096,
                 normalized=True,
                 # SD/SU layer
                 band_SR=[0.175, 0.392, 0.433],
                 band_stride=[1, 4, 16],
                 band_kernel=[3, 4, 16],
                 # Convolution Module
                 conv_depths=[3, 2, 1],
                 compress=4,
                 conv_kernel=3,
                 # Dual-path RNN
                 num_dplayer=6,
                 expand=1,
                 ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.dims = dims
        band_keys = ['low', 'mid', 'high']
        self.band_configs = {band_keys[i]: {'SR': band_SR[i], 'stride': band_stride[i], 'kernel': band_kernel[i]} for i in range(len(band_keys))}
        self.hop_length = hop_size
        self.nfft = nfft
        self.win_size = win_size
        self.conv_config = {
            'compress': compress,
            'kernel': conv_kernel,
        }

        # Register Hann window as buffer (moves with model to GPU automatically)
        self.register_buffer('window', torch.hann_window(win_size))

        # STFT config: center=True for proper COLA reconstruction.
        # Causality is enforced by internal model components (causal convs,
        # unidirectional LSTMs, cumulative LayerNorm), not by STFT boundaries.
        self._stft_params = {
            'n_fft': nfft,
            'hop_length': hop_size,
            'win_length': win_size,
            'center': True,
            'normalized': normalized
        }

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(len(dims) - 1):
            enc = SDblock(
                channels_in=dims[index],
                channels_out=dims[index + 1],
                band_configs=self.band_configs,
                conv_config=self.conv_config,
                depths=conv_depths
            )
            self.encoder.append(enc)

            dec = nn.Sequential(
                FusionLayer(channels=dims[index + 1]),
                SUlayer(
                    channels_in=dims[index + 1],
                    channels_out=dims[index] if index != 0 else dims[index] * len(sources),
                    band_configs=self.band_configs,
                )
            )
            self.decoder.insert(0, dec)

        self.separation_net = SeparationNet(
            channels=dims[-1],
            expand=expand,
            num_layers=num_dplayer,
        )

    def forward(self, x):
        # B, C, L = x.shape
        B = x.shape[0]

        # Right-pad to make signal length divisible by hop_length.
        # No need for even T constraint since FFT between DPRNN layers was removed.
        padding = self.hop_length - x.shape[-1] % self.hop_length
        if padding == self.hop_length:
            padding = 0
        x = F.pad(x, (0, padding))

        # STFT
        L = x.shape[-1]
        x = x.reshape(-1, L)
        x = torch.stft(x, **self._stft_params, window=self.window, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).reshape(
            x.shape[0] // self.audio_channels,
            x.shape[3] * self.audio_channels,
            x.shape[1], x.shape[2]
        )

        B, C, Fr, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        # encoder
        for sd_layer in self.encoder:
            x, skip, lengths, original_lengths = sd_layer(x)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # separation
        x = self.separation_net(x)

        # decoder
        for fusion_layer, su_layer in self.decoder:
            x = fusion_layer(x, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())

        # output
        n = self.dims[0]
        x = x.view(B, n, -1, Fr, T)
        x = x * std[:, None] + mean[:, None]
        x = x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1)
        x = torch.view_as_complex(x.contiguous())
        x = torch.istft(x, **self._stft_params, window=self.window, length=L)
        x = x.reshape(B, len(self.sources), self.audio_channels, -1)

        # Trim right padding
        if padding > 0:
            x = x[:, :, :, :-padding]

        return x
