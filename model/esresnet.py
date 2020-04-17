import termcolor

import numpy as np
import scipy.signal as sps

import torch
import torch.nn.functional as F

import torchvision as tv

import ignite_trainer as it

from model import attention

from typing import Tuple
from typing import Union
from typing import Optional
from typing import Sequence


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(torch.nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(torch.nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = torch.nn.BatchNorm2d(planes * self.expansion)
        self.relu = torch.nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(it.AbstractNet):

    def __init__(self,
                 block: Union[BasicBlock, Bottleneck],
                 layers: Sequence[int],
                 num_channels: int = 3,
                 num_classes: int = 1000):

        super(ResNet, self).__init__()

        self.inplanes = 64

        self.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                torch.nn.BatchNorm2d(planes * block.expansion)
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        y_pred = self.fc(x)

        loss = None
        if y is not None:
            loss = self.loss_fn(y_pred, y).sum()

        return y_pred if loss is None else (y_pred, loss)

    def loss_fn(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss_pred = F.cross_entropy(y_pred, y)

        return loss_pred

    @property
    def loss_fn_name(self) -> str:
        return 'Cross Entropy'


class ResNet50(ResNet):

    def __init__(self, num_channels: int = 3, num_classes: int = 1000):
        super(ResNet50, self).__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            num_channels=num_channels,
            num_classes=num_classes
        )


class ResNetWithAttention(ResNet):

    def __init__(self,
                 block: Union[BasicBlock, Bottleneck],
                 layers: Sequence[int],
                 num_channels: int = 3,
                 num_classes: int = 1000):

        super(ResNetWithAttention, self).__init__(
            block=block,
            layers=layers,
            num_channels=num_channels,
            num_classes=num_classes
        )

        self.att1 = attention.Attention2d(
            in_channels=64,
            out_channels=64 * block.expansion,
            num_kernels=1,
            kernel_size=(3, 1),
            padding_size=(1, 0)
        )
        self.att2 = attention.Attention2d(
            in_channels=64 * block.expansion,
            out_channels=128 * block.expansion,
            num_kernels=1,
            kernel_size=(1, 5),
            padding_size=(0, 2)
        )
        self.att3 = attention.Attention2d(
            in_channels=128 * block.expansion,
            out_channels=256 * block.expansion,
            num_kernels=1,
            kernel_size=(3, 1),
            padding_size=(1, 0)
        )
        self.att4 = attention.Attention2d(
            in_channels=256 * block.expansion,
            out_channels=512 * block.expansion,
            num_kernels=1,
            kernel_size=(1, 5),
            padding_size=(0, 2)
        )
        self.att5 = attention.Attention2d(
            in_channels=512 * block.expansion,
            out_channels=512 * block.expansion,
            num_kernels=1,
            kernel_size=(3, 5),
            padding_size=(1, 2)
        )

    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_att = x.clone()
        x = self.layer1(x)
        x_att = self.att1(x_att, x.shape[-2:])
        x = x * x_att

        x_att = x.clone()
        x = self.layer2(x)
        x_att = self.att2(x_att, x.shape[-2:])
        x = x * x_att

        x_att = x.clone()
        x = self.layer3(x)
        x_att = self.att3(x_att, x.shape[-2:])
        x = x * x_att

        x_att = x.clone()
        x = self.layer4(x)
        x_att = self.att4(x_att, x.shape[-2:])
        x = x * x_att

        x_att = x.clone()
        x = self.avgpool(x)
        x_att = self.att5(x_att, x.shape[-2:])
        x = x * x_att
        x = x.view(x.size(0), -1)

        y_pred = self.fc(x)

        loss = None
        if y is not None:
            loss = self.loss_fn(y_pred, y).sum()

        return y_pred if loss is None else (y_pred, loss)


class ResNet50WithAttention(ResNetWithAttention):

    def __init__(self, num_channels: int = 3, num_classes: int = 1000):
        super(ResNet50WithAttention, self).__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            num_channels=num_channels,
            num_classes=num_classes
        )


class _ESResNet(ResNet):

    def __init__(self,
                 block: Union[BasicBlock, Bottleneck],
                 layers: Sequence[int],
                 n_fft: int = 256,
                 hop_length: Optional[int] = None,
                 win_length: Optional[int] = None,
                 window: Optional[str] = None,
                 normalized: bool = False,
                 onesided: bool = True,
                 spec_height: int = 224,
                 spec_width: int = 224,
                 num_classes: int = 1000,
                 pretrained: Union[bool, str] = False,
                 lock_pretrained: Optional[bool] = None):

        super(_ESResNet, self).__init__(
            block=block,
            layers=layers,
            num_channels=3,
            num_classes=num_classes
        )

        self.num_classes = num_classes

        self.fc = torch.nn.Identity()
        self.classifier = torch.nn.Linear(
            in_features=512 * block.expansion,
            out_features=self.num_classes
        )

        if hop_length is None:
            hop_length = int(np.floor(n_fft / 4))

        if win_length is None:
            win_length = n_fft

        if window is None:
            window = 'boxcar'

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.normalized = normalized
        self.onesided = onesided

        self.spec_height = spec_height
        self.spec_width = spec_width

        self.pretrained = pretrained
        if pretrained:
            err_msg = self.load_pretrained()

            unlocked_weights = list()

            for name, p in self.named_parameters():
                if lock_pretrained and name not in err_msg:
                    p.requires_grad_(False)
                else:
                    unlocked_weights.append(name)

            print(f'Following weights are unlocked: {unlocked_weights}')

        window_buffer: torch.Tensor = torch.from_numpy(
            sps.get_window(window=window, Nx=win_length, fftbins=True)
        ).to(torch.get_default_dtype())
        self.register_buffer('window', window_buffer)

        self.log10_eps = 1e-18

    def load_pretrained(self) -> str:
        if isinstance(self.pretrained, bool):
            state_dict = self.loading_func(pretrained=True).state_dict()
        else:
            state_dict = torch.load(self.pretrained, map_location='cpu')

        err_msg = ''
        try:
            self.load_state_dict(state_dict=state_dict, strict=True)
        except RuntimeError as ex:
            err_msg += f'While loading some errors occurred.\n{ex}'
            print(termcolor.colored(err_msg, 'red'))

        return err_msg

    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        pow_spec = self.spectrogram(x)
        x_db = torch.log10(pow_spec).mul(10.0)

        outputs = list()
        for ch_idx in range(x_db.shape[1]):
            ch = x_db[:, ch_idx]
            out = super(_ESResNet, self).forward(ch)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=-1).sum(dim=-1)
        y_pred = self.classifier(outputs)

        loss = None
        if y is not None:
            loss = self.loss_fn(y_pred, y).mean()

        return y_pred if loss is None else (y_pred, loss)

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x.view(-1, x.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            pad_mode='reflect',
            normalized=self.normalized,
            onesided=True
        )

        if not self.onesided:
            spec = torch.cat((torch.flip(spec, dims=(-3,)), spec), dim=-3)

        spec_height_3_bands = spec.shape[-3] // 3
        spec_height_single_band = 3 * spec_height_3_bands
        spec = spec[:, :spec_height_single_band]

        spec = spec.reshape(x.shape[0], -1, spec.shape[-3] // 3, *spec.shape[-2:])

        spec_height = spec.shape[-3] if self.spec_height < 1 else self.spec_height
        spec_width = spec.shape[-2] if self.spec_width < 1 else self.spec_width

        pow_spec = spec[..., 0] ** 2 + spec[..., 1] ** 2

        if spec_height != pow_spec.shape[-2] or spec_width != pow_spec.shape[-1]:
            pow_spec = F.interpolate(
                pow_spec,
                size=(spec_height, spec_width),
                mode='bilinear',
                align_corners=True
            )

        pow_spec = torch.where(pow_spec > 0.0, pow_spec, torch.full_like(pow_spec, self.log10_eps))

        pow_spec = pow_spec.view(x.shape[0], -1, 3, *pow_spec.shape[-2:])

        return pow_spec


class ESResNet(_ESResNet):

    loading_func = staticmethod(tv.models.resnet50)

    def __init__(self,
                 n_fft: int = 256,
                 hop_length: Optional[int] = None,
                 win_length: Optional[int] = None,
                 window: Optional[str] = None,
                 normalized: bool = False,
                 onesided: bool = True,
                 spec_height: int = 224,
                 spec_width: int = 224,
                 num_classes: int = 1000,
                 pretrained: bool = False,
                 lock_pretrained: Optional[bool] = None):

        super(ESResNet, self).__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=onesided,
            spec_height=spec_height,
            spec_width=spec_width,
            num_classes=num_classes,
            pretrained=pretrained,
            lock_pretrained=lock_pretrained
        )


class ESResNetAttention(_ESResNet, ResNetWithAttention):

    loading_func = staticmethod(tv.models.resnet50)

    def __init__(self,
                 n_fft: int = 256,
                 hop_length: Optional[int] = None,
                 win_length: Optional[int] = None,
                 window: Optional[str] = None,
                 normalized: bool = False,
                 onesided: bool = True,
                 spec_height: int = 224,
                 spec_width: int = 224,
                 num_classes: int = 1000,
                 pretrained: bool = False,
                 lock_pretrained: Optional[bool] = None):

        super(ESResNetAttention, self).__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=onesided,
            spec_height=spec_height,
            spec_width=spec_width,
            num_classes=num_classes,
            pretrained=pretrained,
            lock_pretrained=lock_pretrained
        )

    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        pow_spec = self.spectrogram(x)
        x_db = torch.log10(pow_spec).mul(10.0)

        outputs = list()
        for ch_idx in range(x_db.shape[1]):
            ch = x_db[:, ch_idx]
            out = super(_ESResNet, self).forward(ch)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=-1).sum(dim=-1)
        y_pred = self.classifier(outputs)

        loss = None
        if y is not None:
            loss = self.loss_fn(y_pred, y).mean()

        return y_pred if loss is None else (y_pred, loss)
