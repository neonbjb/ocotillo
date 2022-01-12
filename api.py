import os

import torch
import torchaudio

from tqdm import tqdm

from gpt_asr_hf import GptAsrHf, MODEL_CONFIGS
from mel import MEL
from tokenizer import VoiceBpeTokenizer
from utils import load_audio

PRETRAINED_MODELS_IDS = {
    # Base URL to use (as of 1/2022.. damn you Google..) https://drive.google.com/file/d/{id}
    'medium': '1BW-inL6PJra_rjK1q9gzKOakmgv13nyd',
    'large': '1tXm6vZt-jwkvfYC4hkW7OGkylvH2Xdo2',
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class Transcriber:
    def __init__(self, model_path=None, num_beams=1, model_config='large',
                 tokenizer_params_path='data/bpe_lowercase_asr_256.json',
                 mel_norms_path='data/mel_norms.pth', on_cuda=True, cuda_device=0,
                 pretrained_models_download_path='.weights/'):
        self.tokenizer = VoiceBpeTokenizer(tokenizer_params_path)
        self.mel = MEL(mel_norms_path)
        self.model = GptAsrHf(**MODEL_CONFIGS[model_config]).eval()
        if model_path is None:
            pretrained_model_file = f'{pretrained_models_download_path}/{model_config}.pth'
            if not os.path.exists(pretrained_model_file):
                print("Downloading pretrained model for use with transcription..")
                os.makedirs(pretrained_models_download_path, exist_ok=True)
                id = PRETRAINED_MODELS_IDS[model_config]
                import gdown  # If you do not wish to use this dep, download the files yourself.
                gdown.download(output=pretrained_model_file, quiet=False, id=id)
                print("Done.")
            model_path = pretrained_model_file

        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.num_beams = num_beams
        if on_cuda:
            self.model = self.model.cuda(cuda_device)
            self.mel = self.mel.cuda(cuda_device)
            self.device = 'cuda'
        else:
            self.device = 'cpu'

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
        if sample_rate != self.mel.mel_stft.sample_rate:
            audio_data = torchaudio.functional.resample(audio_data, sample_rate, self.mel.mel_stft.sample_rate)
        audio_data = audio_data.to(self.device)
        mels = self.mel(audio_data)
        with torch.no_grad():
            tokens = self.model.inference(mels, num_beams=self.num_beams)
        return [self.tokenizer.decode(toks) for toks in tokens]


if __name__ == '__main__':
    transcriber = Transcriber(num_beams=4, on_cuda=False)
    audio = load_audio('data/obama.mp3', 44100)
    print(transcriber.transcribe(audio, 44100))