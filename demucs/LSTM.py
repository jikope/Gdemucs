import logging

import torch
import torch.nn.functional as F

from torch import nn
from openunmix.filtering import wiener

from .demucs import BLSTM
from .hdemucs import HEncLayer, HDecLayer, ScaledEmbedding
from .states import capture_init
from .spec import spectro, ispectro

logger = logging.getLogger(__name__)


def rescale_conv(conv, reference):
    """Rescale initial weight scale. It is unclear why it helps but it certainly does.
    """
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            rescale_conv(sub, reference)


class Encoder(nn.Module):
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1, context=0, norm=True, pad=True, rewrite=True, freq=True):
        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0

        self.stride = stride
        self.norm = norm
        self.pad = pad
        self.freq = freq

        kernel_size = [kernel_size, 1]
        stride = [stride, 1]
        pad = [pad, 0]
        klass = nn.Conv2d

        self.conv = klass(chin, chout, kernel_size, stride, pad)
        self.norm1 = norm_fn(chout)

        self.rewrite = None
        if rewrite:
            self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
            self.norm2 = norm_fn(2 * chout)


    def forward(self, x):
        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)

        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))
        # y = self.conv(x)
        y = self.conv(x)
        y = F.gelu(self.norm1(y))

        if self.rewrite:
            z = self.norm2(self.rewrite(y))
            z = F.glu(z, dim=1)
        else:
            z = y

        return z


class Decoder(nn.Module):
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1, last=False, empty=False, norm=True, pad=True, rewrite=True, freq=True, context=1, context_freq=True):
        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0

        self.stride = stride
        self.norm = norm
        self.last = last
        self.pad = pad
        self.freq = freq
        self.empty = empty
        self.rewrite = rewrite

        kernel_size = [kernel_size, 1]
        stride = [stride, 1]
        pad = [pad, 0]
        klass = nn.Conv2d
        klass_tr = nn.ConvTranspose2d

        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)

        self.rewrite = None
        if rewrite:
            if context_freq:
                self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            else:
                self.rewrite = klass(chin, 2 * chin, [1, 1 + 2 * context], 1,
                                     [0, context])
            self.norm1 = norm_fn(2 * chin)

    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)

        if not self.empty:
            x = x + skip

            if self.rewrite:
                y = F.glu(self.norm1(self.rewrite(x)), dim=1)
            else:
                y = x
        else:
            y = x
            assert skip is None

        z = self.norm2(self.conv_tr(y))

        if self.pad:
            z = z[..., self.pad:-self.pad, :]

        if not self.last:
            z = F.gelu(z)

        return z, y


class BasicLSTM(nn.Module):
    @capture_init
    def __init__(self,
                 sources,
                 # Channels
                 audio_channels=2,
                 channels=48,
                 growth=2,
                 # STFT
                 nfft=4096,
                 wiener_iters=0,
                 end_iters=0,
                 wiener_residual=False,
                 cac=True,
                 # Architecture
                 depth=6,
                 rewrite=True,
                 # Convolutions
                 kernel_size=8,
                 time_stride=2,
                 stride=4,
                 context=1,
                 context_enc=0,
                 # Frequency Smoothing
                 multi_freqs=None,
                 multi_freqs_depth=2,
                 freq_emb=0.2,
                 emb_scale=10,
                 emb_smooth=True,
                 # Activation
                 glu=True,
                 gelu=True,
                 # Normalization
                 norm_starts=4,
                 norm_groups=4,
                 # Weight init
                 rescale=0.1,
                 # Metadata
                 samplerate=44100,
                 segment=4 * 10):

        super().__init__()
        self.cac = cac
        self.sources = sources
        self.audio_channels = audio_channels
        self.channels = channels
        self.segment = segment
        self.samplerate = samplerate

        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_residual = wiener_residual
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # self.lstm = nn.LSTM(768, 768)
        self.linear = nn.Linear(1536, 768)
        self.blstm = BLSTM(channels * channels, layers=2, max_steps=200, skip=True)

        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1
        if gelu:
            act2 = nn.GELU
        else:
            act2 = nn.ReLU

        chin_z = audio_channels  # number of channels for the freq branch
        if self.cac:
            chin_z *= 2
        chout = channels
        chout_z = channels
        freqs = nfft // 2

        # Loop for assigning encoders 
        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size

            if not freq:
                assert freqs == 1
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                'kernel_size': ker,
                'stride': stri,
                'freq': freq,
                'pad': pad,
                'norm': norm,
                'rewrite': rewrite,
                'norm_groups': norm_groups,
            }
            kw_dec = dict(kw)

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            # enc = Encoder(chin_z, chout_z, **kw)
            enc = HEncLayer(chin_z, chout_z, dconv=False, context=context_enc, **kw)
            self.encoder.append(enc)
            
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2

            # dec = Decoder(chout_z, chin_z, last=index == 0, **kw_dec)
            dec = HDecLayer(chout_z, chin_z, dconv=False, context=context, last=index == 0, **kw_dec)
            self.decoder.insert(0, dec)

            # Increase channels
            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride

                    
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z, smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        if rescale:
            rescale_module(self, reference=rescale)

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft

        z = spectro(x, nfft, hl)[..., :-1, :]
        return z

    def _ispec(self, z, length=None, scale=0):
        hl = self.hop_length

        z = F.pad(z, (0, 0, 0, 1))
        x = ispectro(z, hl, length)

        return x

    def _magnitude(self, z):

        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m

    def _mask(self, z, m):
        # Apply masking given the mixture spectrogram `z` and the estimated mask `m`.
        # If `cac` is True, `m` is actually a full spectrogram and `z` is ignored.
        niters = self.wiener_iters
        if self.cac:
            B, S, C, Fr, T = m.shape
            out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
            out = torch.view_as_complex(out.contiguous())
            return out
        if self.training:
            niters = self.end_iters
        if niters < 0:
            z = z[:, None]
            return z / (1e-8 + z.abs()) * m
        else:
            return self._wiener(m, z, niters)

    def _wiener(self, mag_out, mix_stft, niters):
        # apply wiener filtering from OpenUnmix.
        init = mix_stft.dtype
        wiener_win_len = 300
        residual = self.wiener_residual

        B, S, C, Fq, T = mag_out.shape
        mag_out = mag_out.permute(0, 4, 3, 2, 1)
        mix_stft = torch.view_as_real(mix_stft.permute(0, 3, 2, 1))

        outs = []
        for sample in range(B):
            pos = 0
            out = []
            for pos in range(0, T, wiener_win_len):
                frame = slice(pos, pos + wiener_win_len)
                z_out = wiener(
                    mag_out[sample, frame], mix_stft[sample, frame], niters,
                    residual=residual)
                out.append(z_out.transpose(-1, -2))
            outs.append(torch.cat(out, dim=0))
        out = torch.view_as_complex(torch.stack(outs, 0))
        out = out.permute(0, 4, 3, 2, 1).contiguous()
        if residual:
            out = out[:, :-1]
        assert list(out.shape) == [B, S, C, Fq, T]
        return out.to(init)

    def forward(self, mix):
        x = mix
        length = x.shape[-1]
        # logger.info(f"shape of input {mix.shape}")

        # _spec
        z = self._spec(mix)
        # _magnitude
        mag = self._magnitude(z)
        x = mag

        B, C, Fq, T = x.shape

        # Normalize
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        saved = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        lengths_t = []  # saved lengths for time branch.

        # encoder loop
        for idx, encode in enumerate(self.encoder):
            # logger.info(f"encoder {idx}")
            lengths.append(x.shape[-1])
            x = encode(x)

            if idx == 0 and self.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

            saved.append(x)

        # Latent space
        # logger.info(f"x shape {x.shape}")
        # x = x.permute(2, 0, 1)
        x = self.blstm(x)
        # x = self.linear(x)
        # x = x.permute(1, 2, 0)
        # logger.info(f"x shape {x.shape}")

        x = torch.zeros_like(x)
        # decoder loop
        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, _ = decode(x, skip, lengths.pop(-1))

        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        # mask
        zout = self._mask(z, x)

        # inverse spec
        x = self._ispec(zout, length)

        return x
