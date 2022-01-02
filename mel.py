import torch
import torchaudio
from torch import nn


class MEL(nn.Module):
    """
    Helper class for performing the flavor of MEL-coding that this model expects. This is a fairly standard set of
    parameters for speech processing, but specifically does log-based range compression and further normalizes each
    filter bank by a set of factors derived from a very large set of speech files.
    """
    def __init__(self, mel_norms_file="data/mel_norms.pth", filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, mel_fmin=0, mel_fmax=8000, sampling_rate=22050, normalize=False):
        super().__init__()
        self.mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=filter_length, hop_length=hop_length,
                                                             win_length=win_length, power=2, normalized=normalize,
                                                             sample_rate=sampling_rate, f_min=mel_fmin,
                                                             f_max=mel_fmax, n_mels=n_mel_channels,
                                                             norm="slaney")
        self.mel_norms = torch.load(mel_norms_file)

    def forward(self, inp):
        if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        self.mel_stft = self.mel_stft.to(inp.device)
        mel = self.mel_stft(inp)
        # Perform dynamic range compression
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if self.mel_norms is not None:
            self.mel_norms = self.mel_norms.to(mel.device)
            mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
        return mel