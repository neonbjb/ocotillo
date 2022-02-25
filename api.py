import torch
import torchaudio

from model_loader import load_model
from utils import load_audio


class Transcriber:
    def __init__(self, on_cuda=True, cuda_device=0):
        if on_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model, self.processor = load_model(self.device)

    def transcribe(self, audio_data, sample_rate):
        """
        Transcribes audio_data at the given sample_rate. audio_data is expected to be a list of floats, a torch tensor
        or a numpy array. audio_data must be either 1d mono audio data or 2d stereo data (one channel is thrown out).
        The channel dimension must be first. audio_data must be normalized to [-1,1]. One-shot transcription is
        length-limited by the model.
        """
        if not isinstance(audio_data, torch.Tensor):
            audio_data = torch.tensor(audio_data, dtype=torch.float)
        audio_data = audio_data.to(self.device).unsqueeze(0)
        return self.transcribe_batch(audio_data, sample_rate)[0]

    def transcribe_batch(self, audio_data, sample_rate):
        """
        Transcribes audio_data at the given sample_rate. audio_data is expected to be a torch tensor
        or a numpy array. audio_data must be either 2d mono audio data or 3d stereo data (one channel is thrown out).
        The batch dimension is first. The channel dimension is second. audio_data must be normalized to [-1,1].
        One-shot transcription is  length-limited by the model.
        """
        if not isinstance(audio_data, torch.Tensor):
            audio_data = torch.tensor(audio_data, dtype=torch.float)  # This makes valid inputs either a torch tensor a numpy array.
        if sample_rate != 16000:
            audio_data = torchaudio.functional.resample(audio_data, sample_rate, 16000)
        audio_data = audio_data.to(self.device)
        with torch.no_grad():
            logits = self.model(audio_data)[0]
            tokens = torch.argmax(logits, dim=-1)
        return self.processor.decode(tokens[0])


if __name__ == '__main__':
    transcriber = Transcriber(on_cuda=False)
    audio = load_audio('data/obama.mp3', 44100)
    print(transcriber.transcribe(audio, 44100))