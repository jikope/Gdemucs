import logging

import torch
from torch import nn
from torch.nn.modules.rnn import LSTM
from torchinfo import summary
import torch.nn.functional as F
from torch.nn.functional import batch_norm, fold, unfold

from openunmix.filtering import wiener
from .norms import GlobLN
from .states import capture_init
from .spec import spectro, ispectro

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self,
                 chin,
                 chout,
                 kernel_size=8,
                 stride=4,
                 freq=True,
                 pad=True,
                 norm=False,
                 norm_group=4,
                 context=0):
        super().__init__()
        self.freq = freq
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0

        kernel_size = kernel_size
        self.stride = stride
        klass = nn.Conv1d

        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad = [pad, 0]
            klass = nn.Conv2d

        self.conv = klass(chin, chout, kernel_size, stride, pad)

        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_group, d)

        self.norm1 = norm_fn(chout)
        
        self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
        self.norm2 = norm_fn(2 * chout)

    def forward(self, x):
        if not self.freq:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)
            le = x.shape[-1]
            x = F.pad(x, (0, self.stride - (le % self.stride)))

        y = self.conv(x)
        y = F.gelu(self.norm1(y))
        z = self.norm2(self.rewrite(y))
        z = F.glu(z, dim=1)
        
        return z


class Decoder(nn.Module):
    def __init__(self,
                 chin,
                 chout,
                 kernel_size=8,
                 stride=4,
                 nfft=4096,
                 freq=True,
                 norm=False,
                 pad=True,
                 norm_group=4,
                 emtpy=False,
                 last=True,
                 context=0
                 ):
        super().__init__()
        self.chin = chin
        self.empty = emtpy
        self.freq = freq
        self.last = last
        self.pad = pad

        kernel_size = kernel_size
        stride = stride
        klass = nn.ConvTranspose1d

        if pad:
            pad = kernel_size // 4
        else:
            pad = 0

        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad = [pad, 0]
            klass = nn.ConvTranspose2d

        self.conv_tr = klass(chin, chout, kernel_size, stride, pad)
        norm_fn = lambda d: nn.Identity()
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_group, d)

        self.norm1 = norm_fn(chout)
        self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
        self.norm2 = norm_fn(2 * chout)

    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3:
            B, C, T =  x.shape
            x = x.view(B, self.chin, -1, T)

        if not self.empty:
            x = x + skip

        x = self.conv_tr(x)
        z = self.norm1(x)

        if self.freq:
            j = 0
            # if self.pad:
            # z = z[..., self.pad:-self.pad, :]
            # logger.info(f"freq pad {z.shape}")
        else:
            z = z[..., self.pad:self.pad + length]
            assert z.shape[-1] == length, (z.shape[-1], length)

        if not self.last:
            z = F.gelu(z)

        return z


class DPRNNBlock(nn.Module):
    def __init__(self,
                 input_size,
                 n_frames=33,
                 width=2048,
                 channels=2,
                 ):
        super(DPRNNBlock, self).__init__()
        self.input_size = input_size
        self.n_frames = n_frames
        self.width = width
        self.channels = channels
        norm_fn = lambda d: nn.GroupNorm(4, d)

        self.intra_rnn = nn.LSTM(input_size=input_size, hidden_size=input_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(input_size, out_features=input_size)
        # self.intra_ln = nn.LayerNorm(17)

        self.inter_rnn = nn.LSTM(input_size=input_size, hidden_size=input_size // 2, bidirectional=True)
        self.inter_fc = nn.Linear(input_size, out_features=input_size)
        # self.inter_ln = nn.LayerNorm(17)

    def forward(self, x):
        """
         Input shape x -> (n_samples, n_channels, n_frames) (4, 1024, 17)
        """
        n_samples, n_channels, n_frames = x.shape
        # x = x.permute(2, 0, 1)
        layer_norm = nn.LayerNorm(n_frames)

        intra_out = self.intra_rnn(x.permute(2, 0, 1))[0]
        intra_out = self.intra_fc(intra_out.contiguous())
        # logger.info(intra_out.shape)
        intra_out = intra_out.permute(1, 2, 0)
        intra_out = layer_norm(intra_out)

        intra_out = intra_out + x

        inter_out = self.inter_rnn(intra_out.permute(2, 0, 1))[0]
        inter_out = self.inter_fc(inter_out.contiguous())
        inter_out = inter_out.permute(1, 2, 0)
        inter_out = layer_norm(inter_out)

        out = intra_out + inter_out

        return out


class DPRNN(nn.Module):
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
                 wiener_iters=0,
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

        # DPRNN Block
        self.dprnn_block = DPRNNBlock(chout)

        self.lstm = LSTM(in_channels, hidden_size)
        self.iLN = nn.LayerNorm(1, 3)  # dim = 2

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
        logger.info(f"input shape {mix.shape}")

        z = spectro(x, self.nfft, self.hop_length)[..., :-1, :]
        logger.info(f"spectro shape {z.shape}")
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

        x = self.dprnn_block(x)
        # logger.info(f"after dprnn {x.shape}")

        x = torch.zeros_like(x)
        for idx, decoder in enumerate(self.decoder):
            skip = saved.pop(-1)
            x = decoder(x, skip, lengths.pop(-1))
            # logger.info(f"after decoder {idx} {x.shape}")

        ## logger.info(f"after decoder {x.shape}")
        n_sources = len(self.sources)
        x = x.view(n_samples, n_sources, -1, n_bins, n_frames)

        zout = self._mask(x, z, self.wiener_iters)

        # logger.info(f"mask {zout.shape}")

        hl = self.hop_length // (4 ** 0)
        zout = F.pad(zout, (0, 0, 0, 1))
        x = ispectro(zout, self.hop_length, length)
        # logger.info(f"wave back {x.shape}")

        return x


if __name__ == "__main__":
    model = DPRNN(['BALUNGAN'])
    summary(model)
