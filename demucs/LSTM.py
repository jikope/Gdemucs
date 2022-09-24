import logging

import torch
import torch.nn.functional as F

from torch import nn
from openunmix.filtering import wiener

from .states import capture_init
from .DPRNN import Encoder, Decoder
from .spec import spectro, ispectro

logger = logging.getLogger(__name__)


class BasicLSTM(nn.Module):
    @capture_init
    def __init__(self,
                 sources,
                 audio_channels=2,
                 channels=48,
                 samplerate=44100,
                 segment=4 * 10,
                 # Conv
                 kernel_size=8,
                 stride=4,
                 # STFT
                 nfft=4096,
                 # RNN
                 in_channels=256,
                 out_channels=64,
                 hidden_size=128,
                 bidirectional=True,
                 # Masking
                 wiener_iters=0
                 ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment
        self.wiener_iters = wiener_iters

        # Conv
        self.kernel_size = kernel_size
        self.stride = stride

        # STFT
        self.nfft = nfft
        self.hop_length = nfft // 4

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        chin = audio_channels
        chout = channels
        self.encoder.append(Encoder(chin, chout, norm=False))
        self.decoder.insert(0, Decoder(chout, audio_channels * len(self.sources), norm=False, last=True))
        chin = chout
        chout = chin * 2
        self.encoder.append(Encoder(chin, chout, norm=False))
        self.decoder.insert(0, Decoder(chout, chin, norm=False))
        chin = chout
        chout = chin * 2
        self.encoder.append(Encoder(chin, chout, norm=False))
        self.decoder.insert(0, Decoder(chout, chin, norm=False))
        chin = chout
        chout = chin * 2
        self.encoder.append(Encoder(chin, chout, norm=False))
        self.decoder.insert(0, Decoder(chout, chin, norm=False))
        chin = chout
        chout = chin * 2
        self.encoder.append(Encoder(chin, chout, norm=True, pad=False))
        self.decoder.insert(0, Decoder(chout, chin, norm=True, pad=False))
        chin = chout
        chout = chin * 2
        self.encoder.append(Encoder(chin, chout, norm=True, freq=False, kernel_size=4, stride=2))
        self.decoder.insert(0, Decoder(chout, chin, norm=True, freq=False, kernel_size=4, stride=2))

        self.lstm = nn.LSTM(input_size=chout, hidden_size=chout // 2, bidirectional=bidirectional)
        self.fc = nn.Linear(chout, out_features=chout)

    @staticmethod
    def _mask(m, z, niters):
        spec_type = z.dtype
        wiener_win_len = 300

        n_samples, n_sources, n_channels, n_bins, n_frames = m.shape
        mag_out = m.permute(0, 4, 3, 2, 1)
        mix_stft = torch.view_as_real(z.permute(0, 3, 2, 1))

        outs = []
        for sample in range(n_samples):
            pos = 0
            out = []
            for pos in range(0, n_frames, wiener_win_len):
                frame = slice(pos, pos + wiener_win_len)
                z_out = wiener(mag_out[sample, frame], mix_stft[sample, frame], niters, residual=False)
                out.append(z_out.transpose(-1, -2))
            outs.append(torch.cat(out, dim=0))
        out = torch.view_as_complex(torch.stack(outs, 0))
        out = out.permute(0, 4, 3, 2, 1).contiguous()

        assert list(out.shape) == [n_samples, n_sources, n_channels, n_bins, n_frames]
        return out.to(spec_type)

    def forward(self, mix):
        x = mix
        length = x.shape[-1]
        # logger.info(f"input shape {mix.shape}")

        z = spectro(x, self.nfft, self.hop_length)[..., :-1, :]
        # logger.info(f"spectro shape {z.shape}")
        x = z.abs()  # Magnitude spectrogram

        n_samples, n_channels, n_bins, n_frames = x.shape

        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        # Layer Norm
        # x = self.iLN(z)

        saved = []  # skip connections, freq.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        for idx, encoder in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            x = encoder(x)
            saved.append(x)
            # logger.info(f"after encoder {idx} {x.shape}")

        # TODO: Add LSTM Layer
        layer_norm = nn.LayerNorm(x.shape[-1])
        lstm_out = self.lstm(x.permute(2, 0, 1))[0]
        lstm_out = self.fc(lstm_out.contiguous())
        lstm_out = layer_norm(lstm_out.permute(1, 2, 0))

        x = x + lstm_out

        for idx, decoder in enumerate(self.decoder):
            skip = saved.pop(-1)
            x = decoder(x, skip, lengths.pop(-1))
            # logger.info(f"after decoder {idx} {x.shape}")

        n_sources = len(self.sources)
        x = x.view(n_samples, n_sources, -1, n_bins, n_frames)
        x = x * std[:, None] + mean[:, None]

        zout = self._mask(x, z, self.wiener_iters)

        # logger.info(f"mask {zout.shape}")

        hl = self.hop_length // (4 ** 0)
        zout = F.pad(zout, (0, 0, 0, 1))
        x = ispectro(zout, self.hop_length, length)
        # logger.info(f"wave back {x.shape}")

        return x
