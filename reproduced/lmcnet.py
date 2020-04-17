import scipy.signal as sps

import torch
import torch.nn.functional as F

import ignite_trainer as it

from utils import features

from typing import Tuple
from typing import Union
from typing import Optional


class LMCNet(it.AbstractNet):

    def __init__(self,
                 num_channels: int = 1,
                 num_classes: int = 10,
                 sample_rate: int = 44100,
                 norm: Union[str, float] = 'inf',
                 n_fft: int = 2048,
                 hop_length: int = 1024,
                 win_length: int = 2048,
                 window: str = 'hann',
                 n_mels: int = 128,
                 tuning: float = 0.0,
                 n_chroma: int = 12,
                 ctroct: float = 5.0,
                 octwidth: float = 2.0,
                 base_c: bool = True,
                 freq: Optional[torch.Tensor] = None,
                 fmin: float = 200.0,
                 fmax: Optional[float] = None,
                 n_bands: int = 6,
                 quantile: float = 0.02,
                 linear: bool = False):

        super(LMCNet, self).__init__()

        norm = float(norm)

        self.lmc = features.LMC(
            sample_rate=sample_rate,
            norm=norm,
            n_fft=n_fft,
            n_mels=n_mels,
            tuning=tuning,
            n_chroma=n_chroma,
            ctroct=ctroct,
            octwidth=octwidth,
            base_c=base_c,
            freq=freq,
            fmin=fmin,
            fmax=fmax,
            n_bands=n_bands,
            quantile=quantile,
            linear=linear
        )

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        window_buf = sps.get_window(window, win_length, False)
        self.register_buffer('window', torch.from_numpy(window_buf).to(torch.get_default_dtype()))

        self.conv1 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.bn1 = torch.nn.BatchNorm2d(num_features=self.conv1.out_channels)
        self.activation1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=self.conv1.out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.bn2 = torch.nn.BatchNorm2d(num_features=self.conv2.out_channels)
        self.activation2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1))

        self.conv3 = torch.nn.Conv2d(
            in_channels=self.conv2.out_channels,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.bn3 = torch.nn.BatchNorm2d(num_features=self.conv3.out_channels)
        self.activation3 = torch.nn.ReLU()

        self.conv4 = torch.nn.Conv2d(
            in_channels=self.conv3.out_channels,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.bn4 = torch.nn.BatchNorm2d(num_features=self.conv4.out_channels)
        self.activation4 = torch.nn.ReLU()
        self.pool4 = torch.nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1))

        self.fc1 = torch.nn.Linear(in_features=11 * 22 * self.conv4.out_channels, out_features=1024)
        self.activation5 = torch.nn.Sigmoid()

        self.fc2 = torch.nn.Linear(in_features=self.fc1.out_features, out_features=num_classes)

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        spectrogram = torch.stft(
            x.view(x.shape[0], -1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True
        )
        spectrogram = spectrogram[..., 0] ** 2 + spectrogram[..., 1] ** 2
        spectrogram = spectrogram.view(x.shape[0], -1, *spectrogram.shape[1:])
        spectrogram = torch.where(spectrogram == 0.0, spectrogram + 1e-10, spectrogram)

        return spectrogram

    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        x = self.spectrogram(x)
        x = self.lmc(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)

        x = F.dropout2d(x, p=0.5, training=self.training)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation3(x)

        x = F.dropout2d(x, p=0.5, training=self.training)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation4(x)
        x = self.pool4(x)

        x = x.view(x.shape[0], -1)

        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc1(x)
        x = self.activation5(x)
        y_pred = self.fc2(x)

        loss = None
        if y is not None:
            loss = self.loss_fn(y_pred, y).mean()

        return y_pred if loss is None else (y_pred, loss)

    def loss_fn(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss_pred = F.cross_entropy(y_pred, y)

        return loss_pred

    @property
    def loss_fn_name(self) -> str:
        return 'Cross Entropy'
