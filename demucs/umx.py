import logging

import torch
from torch import nn, stft, istft
from torch.nn import functional as F
from openunmix.filtering import wiener

logger = logging.getLogger(__name__)


class UMX(nn.Module):

    def __init__(self,
                 sources,
                 audio_channels=2,
                 channels=48,
                 samplerate=44100,
                 segment=4 * 10,
                 # STFT
                 nfft=4096,
                 # LSTM
                 hidden_size=512,
                 layers=3
                 ):
        super(UMX, self).__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.channels = channels
        self.segment = segment
        self.samplerate = samplerate

        # STFT
        self.nfft = nfft
        self.hop_length = nfft // 4

        # LSTM
        self.hidden_size = hidden_size

        # Layer 1 blocks
        self.fc1 = nn.Linear(nfft * audio_channels, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        # LSTM Block
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=layers,
            bidirectional=True,
            batch_first=False,
            dropout=0.4 if layers > 1 else 0
        )

        # Layer 2 blocks
        self.fc2 = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        # Layer 3 blocks
        self.fc3 = nn.Linear(in_features=hidden_size, out_features=nfft * audio_channels, bias=False)
        self.bn3 = nn.BatchNorm1d(nfft * audio_channels)

    @staticmethod
    def _stft(x, nfft, hl):
        """
        Args:
            x (Tensor): (n_samples, n_channels, n_timesteps)
        """
        n_samples, n_channels, n_timesteps = x.shape
        x = x.reshape(-1, n_timesteps)
        z = stft(x,
                 n_fft=nfft,
                 hop_length=hl,
                 window=torch.hann_window(nfft).to(x),
                 win_length=nfft,
                 normalized=True,
                 center=True,
                 return_complex=True,
                 pad_mode='reflect')

        # z = torch.view_as_real(z)
        _, freqs, frame = z.shape

        return z.view(n_samples, n_channels, freqs, frame)

    @staticmethod
    def _mask(m, z, niters):
        """
        Args:
            m (Tensor): Estimate mask.
            z (Tensor): Original Spectrogram. (n_samples, n_channels, n_bins, n_frames)
        """
        wiener_win_length = 300

        outs = []
        for sample in range(B):
            pos = 0
            out = []
            for pos in range(0, T, wiener_win_length):
                frame = slice(pos, pos + wiener_win_length)
                z_out = wiener(m[sample, frame])

        return x

    def forward(self, mix):
        # Mix shape (4, 2, 33516)
        x = mix
        length = mix.shape[-1]
        n_sources = len(self.sources)

        z = self._stft(x, self.nfft, self.hop_length)[:, :, :-1, :]  # Trim last freq
        z = torch.abs(z)  # Spec shape (4, 2, 2048, 33)

        x = z.permute(3, 0, 1, 2)
        logger.info(f"After permute {x.shape}")
        nb_frames, nb_samples, nb_channels, nb_bins = x.shape
        x_spec = x.detach().clone()

        x = x[..., :self.nfft]
        logger.info(f"After crop {x.shape}")

        # LAYER 1
        x = x.reshape(-1, nb_channels * self.nfft)
        x = self.fc1(x)
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_channels, self.hidden_size)
        x = torch.tanh(x)
        logger.info(f"After tanh {x.shape}")

        # LSTM
        lstm_out = self.lstm(x)
        x = torch.cat([x, lstm_out[0]], -1)
        logger.info(f"LSTM output concat {x.shape}")

        # Layer 2
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        logger.info(f"after fc2 {x.shape}")
        x = self.bn2(x)
        logger.info(f"after bn2 {x.shape}")
        x = F.relu(x)
        logger.info(f"after relu {x.shape}")

        # Layer 3
        x = self.fc3(x)
        logger.info(f"after fc3 {x.shape}")
        x = self.bn3(x)
        logger.info(f"after bn3 {x.shape}")

        x = x.reshape(nb_frames, nb_samples, nb_channels, nb_bins)
        logger.info(f"back to original {x.shape}")

        x = F.relu(x) * x_spec
        logger.info(f"after mix {x.shape}")

        x = x.permute(1, 2, 3, 0)
        logger.info(f"back to original {x.shape}")

        x = x.view(nb_samples, n_sources, nb_channels, nb_bins, nb_frames)
        logger.info(f"reshape to sources {x.shape}")

        zout = self._mask(x, z)

        # Back to time domain
        x = self.istft(zout, length)

        return x
