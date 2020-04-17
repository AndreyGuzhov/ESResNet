import numpy as np
import scipy.fft as spf
import scipy.signal as sps

import librosa

import torch
import torch.nn.functional as F

from utils import transforms

from typing import Optional


def fft_frequencies(sample_rate: int = 22050, n_fft: int = 2048) -> torch.Tensor:
    return torch.linspace(0, sample_rate * 0.5, int(1 + n_fft // 2))


def power_to_db(spectrogram: torch.Tensor, ref: float = 1.0, amin: float = 1e-10, top_db: float = 80.0) -> torch.Tensor:
    log_spec = 10.0 * torch.log10(torch.max(torch.full_like(spectrogram, amin), spectrogram))
    log_spec -= 10.0 * torch.log10(torch.full_like(spectrogram, max(amin, ref)))

    log_spec = torch.max(
        log_spec,
        log_spec.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values - top_db
    )

    return log_spec


class MFCC(torch.nn.Module):

    def __init__(self,
                 sample_rate: int = 22050,
                 n_mfcc: int = 128,
                 n_fft: int = 1024,
                 hop_length: int = 512,
                 window: str = 'hann'):

        super(MFCC, self).__init__()

        mel_filterbank = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mfcc
        )
        mel_filterbank = torch.from_numpy(mel_filterbank).to(torch.get_default_dtype())
        self.register_buffer('mel', mel_filterbank)

        dct_buf = spf.dct(np.eye(n_mfcc), type=2, norm='ortho').T
        dct_buf = torch.from_numpy(dct_buf).to(torch.get_default_dtype())
        self.register_buffer('dct_mat', dct_buf)

        window_buffer: torch.Tensor = torch.from_numpy(
            sps.get_window(window=window, Nx=n_fft, fftbins=True)
        ).to(torch.get_default_dtype())
        self.register_buffer('window', window_buffer)

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length

    def dct2(self, x):
        x_dct = self.dct_mat.view(1, *self.dct_mat.shape) @ x

        return x_dct

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x.view(-1, x.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            normalized=True
        )

        power_spec = spec[..., 0] ** 2 + spec[..., 1] ** 2
        log_power_spec = 10 * torch.log10(power_spec.add(1e-18))

        mel_spec = self.mel.view(1, *self.mel.shape) @ log_power_spec
        mfcc = self.dct2(mel_spec)
        mfcc = mfcc.view(x.shape[0], 1, *mfcc.shape[-2:])

        return mfcc


class Chroma(torch.nn.Module):

    def __init__(self,
                 sample_rate: int = 22050,
                 norm: float = float('inf'),
                 n_fft: int = 2048,
                 tuning: float = 0.0,
                 n_chroma: int = 12,
                 ctroct: float = 5.0,
                 octwidth: float = 2.0,
                 base_c: bool = True):

        super(Chroma, self).__init__()

        chroma_fb_buf = librosa.filters.chroma(
            sr=sample_rate,
            n_fft=n_fft,
            n_chroma=n_chroma,
            tuning=tuning,
            ctroct=ctroct,
            octwidth=octwidth,
            norm=norm,
            base_c=base_c
        )
        self.register_buffer('chroma_fb', torch.from_numpy(chroma_fb_buf).to(torch.get_default_dtype()))

        self.norm = norm

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        chroma = self.chroma_fb @ spectrogram
        chroma = chroma / torch.norm(chroma, p=self.norm, dim=-2, keepdim=True)

        return chroma


class Tonnetz(Chroma):

    def __init__(self,
                 sample_rate: int = 22050,
                 norm: float = float('inf'),
                 n_fft: int = 2048,
                 tuning: float = 0.0,
                 n_chroma: int = 12,
                 ctroct: float = 5.0,
                 octwidth: float = 2.0,
                 base_c: bool = True):

        super(Tonnetz, self).__init__(
            sample_rate=sample_rate,
            norm=norm,
            n_fft=n_fft,
            tuning=tuning,
            n_chroma=n_chroma,
            ctroct=ctroct,
            octwidth=octwidth,
            base_c=base_c
        )

        # Generate Transformation matrix
        dim_map = np.linspace(0, 12, n_chroma, endpoint=False)

        scale = np.asarray([7. / 6, 7. / 6,
                            3. / 2, 3. / 2,
                            2. / 3, 2. / 3])

        V = np.multiply.outer(scale, dim_map)

        # Even rows compute sin()
        V[::2] -= 0.5

        R = np.array([1, 1,  # Fifths
                      1, 1,  # Minor
                      0.5, 0.5])  # Major

        phi_buf = R[:, np.newaxis] * np.cos(np.pi * V)

        self.register_buffer('phi', torch.from_numpy(phi_buf).to(torch.get_default_dtype()))

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        chroma = super(Tonnetz, self).forward(spectrogram)
        chroma = chroma / torch.norm(chroma, p=1, dim=-2, keepdim=True)
        tonnetz = self.phi @ chroma

        return tonnetz


class SpectralContrast(torch.nn.Module):

    def __init__(self,
                 sample_rate: int = 22050,
                 n_fft: int = 2048,
                 freq: Optional[torch.Tensor] = None,
                 fmin: float = 200.0,
                 n_bands: int = 6,
                 quantile: float = 0.02,
                 linear: bool = False):

        super(SpectralContrast, self).__init__()

        # Compute the center frequencies of each bin
        if freq is None:
            freq = fft_frequencies(sample_rate=sample_rate, n_fft=n_fft)

        self.register_buffer('freq', freq)

        if n_bands < 1 or not isinstance(n_bands, int):
            raise ValueError('n_bands must be a positive integer')

        self.n_bands = n_bands

        if not 0.0 < quantile < 1.0:
            raise ValueError('quantile must lie in the range (0, 1)')

        self.quantile = quantile

        if fmin <= 0:
            raise ValueError('fmin must be a positive number')

        octa_buf = torch.zeros(n_bands + 2)
        octa_buf[1:] = fmin * (2.0 ** torch.arange(0, n_bands + 1, dtype=torch.float32))

        if torch.any(octa_buf[:-1] >= 0.5 * sample_rate):
            raise ValueError('Frequency band exceeds Nyquist. Reduce either fmin or n_bands.')

        self.register_buffer('octa', octa_buf)

        self.linear = linear

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        valley = torch.zeros(
            *spectrogram.shape[:-2], self.n_bands + 1, spectrogram.shape[-1],
            dtype=spectrogram.dtype,
            device=spectrogram.device
        )
        peak = torch.zeros_like(valley)

        for k, (f_low, f_high) in enumerate(zip(self.octa[:-1], self.octa[1:])):
            current_band: torch.Tensor = (self.freq >= f_low) & (self.freq <= f_high)

            idx = torch.nonzero(torch.flatten(current_band))

            if k > 0:
                current_band[idx[0] - 1] = True

            if k == self.n_bands:
                current_band[idx[-1] + 1:] = True

            sub_band = spectrogram[..., current_band, :]

            if k < self.n_bands:
                sub_band = sub_band[..., :-1, :]

            # Always take at least one bin from each side
            idx = np.rint(self.quantile * torch.sum(current_band).item())
            idx = int(np.maximum(idx, 1))

            sortedr, _ = torch.sort(sub_band, dim=-2)

            valley[..., k, :] = torch.mean(sortedr[..., :idx, :], dim=-2)
            peak[..., k, :] = torch.mean(sortedr[..., -idx:, :], dim=-2)

        if self.linear:
            return peak - valley
        else:
            return power_to_db(peak) - power_to_db(valley)


class Melspectrogram(torch.nn.Module):

    def __init__(self,
                 sample_rate: int = 22050,
                 n_fft: int = 2048,
                 n_mels: int = 128,
                 fmin: float = 0.0,
                 fmax: Optional[float] = None):

        super(Melspectrogram, self).__init__()

        mel_fb_buf = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )
        self.register_buffer('mel_fb', torch.from_numpy(mel_fb_buf).to(torch.get_default_dtype()))

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        lm = self.mel_fb @ spectrogram
        lm = power_to_db(lm)

        lm = transforms.scale(
            lm,
            lm.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values,
            lm.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values,
            -1.0,
            1.0
        )

        return lm


class CST(torch.nn.Module):

    def __init__(self,
                 sample_rate: int = 22050,
                 norm: float = float('inf'),
                 n_fft: int = 2048,
                 tuning: float = 0.0,
                 n_chroma: int = 12,
                 ctroct: float = 5.0,
                 octwidth: float = 2.0,
                 base_c: bool = True,
                 freq: Optional[torch.Tensor] = None,
                 fmin: float = 200.0,
                 n_bands: int = 6,
                 quantile: float = 0.02,
                 linear: bool = False):

        super(CST, self).__init__()

        self.chroma = Chroma(
            sample_rate=sample_rate,
            norm=norm,
            n_fft=n_fft,
            tuning=tuning,
            n_chroma=n_chroma,
            ctroct=ctroct,
            octwidth=octwidth,
            base_c=base_c
        )
        self.spectral_contrast = SpectralContrast(
            sample_rate=sample_rate,
            n_fft=n_fft,
            freq=freq,
            fmin=fmin,
            n_bands=n_bands,
            quantile=quantile,
            linear=linear
        )
        self.tonnetz = Tonnetz(
            sample_rate=sample_rate,
            norm=norm,
            n_fft=n_fft,
            tuning=tuning,
            n_chroma=n_chroma,
            ctroct=ctroct,
            octwidth=octwidth,
            base_c=base_c
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        chroma = self.chroma(spectrogram)
        spectral_contrast = self.spectral_contrast(spectrogram)
        tonnetz = self.tonnetz(spectrogram)

        chroma = transforms.scale(
            chroma,
            chroma.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values,
            chroma.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values,
            -1.0,
            1.0
        )
        spectral_contrast = transforms.scale(
            spectral_contrast,
            spectral_contrast.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values,
            spectral_contrast.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values,
            -1.0,
            1.0
        )
        tonnetz = transforms.scale(
            tonnetz,
            tonnetz.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values,
            tonnetz.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values,
            -1.0,
            1.0
        )

        cst = torch.cat((
            tonnetz,
            spectral_contrast,
            chroma
        ), dim=-2)

        return cst


class LMC(torch.nn.Module):

    def __init__(self,
                 sample_rate: int = 22050,
                 norm: float = float('inf'),
                 n_fft: int = 2048,
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

        super(LMC, self).__init__()

        self.lm = Melspectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )

        self.cst = CST(
            sample_rate=sample_rate,
            norm=norm,
            n_fft=n_fft,
            tuning=tuning,
            n_chroma=n_chroma,
            ctroct=ctroct,
            octwidth=octwidth,
            base_c=base_c,
            freq=freq,
            fmin=fmin,
            n_bands=n_bands,
            quantile=quantile,
            linear=linear
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        lm = self.lm(spectrogram)
        cst = self.cst(spectrogram)

        lmc = torch.cat((
            cst,
            lm
        ), dim=-2)

        return lmc
