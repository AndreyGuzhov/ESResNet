import math

import torch
import torch.nn.functional as F

import ignite_trainer as it

from utils import transforms, features

from typing import Tuple
from typing import Union
from typing import Optional


class Block(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 pooling_size: Tuple[int, int]):

        super(Block, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv1x1 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1))
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.activation = torch.nn.LeakyReLU()
        self.pooling = torch.nn.MaxPool2d(kernel_size=pooling_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pooling(x)

        return x


class Attention(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 pooling_size: Tuple[int, int]):

        super(Attention, self).__init__()

        self.pool = torch.nn.MaxPool2d(kernel_size=pooling_size)
        self.conv_depth = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels
        )
        self.conv_point = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(1, 1)
        )
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv_depth(x)
        x = self.conv_point(x)
        x = self.bn(x)
        x = self.activation(x)

        return x


class DCNN5(it.AbstractNet):

    def __init__(self,
                 num_channels: int = 1,
                 sample_rate: int = 32000,
                 n_fft: int = 256,
                 hop_length: Optional[int] = None,
                 window: Optional[str] = None,
                 num_classes: int = 10):

        super(DCNN5, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        if hop_length is None:
            hop_length = int(math.floor(n_fft / 4))

        if window is None:
            window = 'boxcar'

        self.log10_eps = 1e-18

        self.mfcc = features.MFCC(
            sample_rate=sample_rate,
            n_mfcc=128,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window
        )

        self.block1 = Block(self.num_channels, 32, (3, 1), (2, 1))
        self.block2 = Block(32, 32, (1, 5), (1, 4))
        self.block3 = Block(32, 64, (3, 1), (2, 1))
        self.block4 = Block(64, 64, (1, 5), (1, 4))
        self.block5 = Block(64, 128, (3, 5), (1, 1))
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 4))

        self.drop1 = torch.nn.Dropout(p=0.25)
        self.fc1 = torch.nn.Linear(in_features=128 * 12 * 2, out_features=256)
        self.fc2 = torch.nn.Linear(in_features=self.fc1.out_features, out_features=self.num_classes)

        self.activation = torch.nn.LeakyReLU()

        self.l2_lambda = 0.1

    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        x = self.mfcc(x)
        x = transforms.scale(
            x,
            x.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values.min(dim=-3, keepdim=True).values,
            x.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values.max(dim=-3, keepdim=True).values,
            0.0,
            1.0
        )

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.max_pool(self.block5(x))

        x = x.view(x.shape[0], -1)
        x = self.drop1(x)

        x = self.fc1(x)
        x = self.activation(x)

        y_pred = self.fc2(x)

        loss = None
        if y is not None:
            loss = self.loss_fn(y_pred, y).sum()

        return y_pred if loss is None else (y_pred, loss)

    def loss_fn(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss_pred = F.cross_entropy(y_pred, y)

        loss_l2 = 0.0
        loss_l2_params = list(self.fc1.parameters())
        for p in loss_l2_params:
            loss_l2 = p.norm(2) + loss_l2

        loss_pred = loss_pred + self.l2_lambda * loss_l2

        return loss_pred

    @property
    def loss_fn_name(self) -> str:
        return 'Cross Entropy'


class ADCNN5(DCNN5):

    def __init__(self,
                 num_channels: int = 1,
                 n_fft: int = 1024,
                 hop_length: Optional[int] = None,
                 window: Optional[str] = None,
                 num_classes: int = 10):

        super(ADCNN5, self).__init__(
            num_channels=num_channels,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            num_classes=num_classes
        )

        self.attn1 = Attention(self.num_channels, 32, (3, 1), (2, 1))
        self.attn2 = Attention(32, 32, (1, 3), (1, 4))
        self.attn3 = Attention(32, 64, (3, 1), (2, 1))
        self.attn4 = Attention(64, 64, (1, 3), (1, 4))
        self.attn5 = Attention(64, 128, (3, 3), (2, 4))
        self.attn5.pool = torch.nn.Identity()
        self.attn5 = torch.nn.Sequential(
            self.attn5,
            torch.nn.AdaptiveMaxPool2d(output_size=(12, 2))
        )

    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        x = self.mfcc(x)
        x = transforms.scale(
            x,
            x.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values.min(dim=-3, keepdim=True).values,
            x.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values.max(dim=-3, keepdim=True).values,
            0.0,
            1.0
        )

        x = self.attn1(x) * self.block1(x)
        x = self.attn2(x) * self.block2(x)
        x = self.attn3(x) * self.block3(x)
        x = self.attn4(x) * self.block4(x)
        x = self.attn5(x) * self.max_pool(self.block5(x))

        x = x.view(x.shape[0], -1)
        x = self.drop1(x)

        x = self.fc1(x)
        x = self.activation(x)

        y_pred = self.fc2(x)

        loss = None
        if y is not None:
            loss = self.loss_fn(y_pred, y).sum()

        return y_pred if loss is None else (y_pred, loss)
