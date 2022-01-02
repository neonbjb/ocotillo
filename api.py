import os
import urllib
import torch
import torchaudio

from tqdm import tqdm

from gpt_asr_hf import GptAsrHf, MODEL_CONFIGS
from mel import MEL
from tokenizer import VoiceBpeTokenizer

PRETRAINED_MODELS_URLS = {
    'medium': 'https://www.nonint.com/downloads/asr/gpt_asr_medium.pth',
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class Transcriber:
    def __init__(self, model_path=None, num_beams=1, model_config='medium',
                 tokenizer_params_path='data/bpe_lowercase_asr_256.json',
                 mel_norms_path='data/mel_norms.pth', on_cuda=True, cuda_device=0,
                 pretrained_models_download_path='.weights/'):
        self.tokenizer = VoiceBpeTokenizer(tokenizer_params_path)
        self.mel = MEL(mel_norms_path)
        self.model = GptAsrHf(**MODEL_CONFIGS[model_config])
        if model_path is None:
            pretrained_model_file = f'{pretrained_models_download_path}/{model_config}.pth'
            if not os.path.exists(pretrained_model_file):
                print("Downloading pretrained model for use with transcription..")
                os.makedirs(pretrained_models_download_path, exist_ok=True)
                url = PRETRAINED_MODELS_URLS[model_config]
                with DownloadProgressBar(unit='B', unit_scale=True,
                                         miniters=1, desc=url.split('/')[-1]) as t:
                    urllib.request.urlretrieve(url, filename=pretrained_model_file, reporthook=t.update_to)
                print("Done.")
            model_path = pretrained_model_file

        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.num_beams = num_beams
        if on_cuda:
            self.model = self.model.cuda(cuda_device)
            self.mel = self.mel.cuda(cuda_device)

    def transcribe(self, audio_data, sample_rate):
        audio_data = audio_data.unsqueeze(0)
        return self.transcribe_batch(audio_data, sample_rate)[0]

    def transcribe_batch(self, audio_data, sample_rate):
        if sample_rate != self.mel.mel_stft.sample_rate:
            audio_data = torchaudio.functional.resample(audio_data, sample_rate, self.mel.mel_stft.sample_rate)
        mels = self.mel(audio_data)
        tokens = self.model.inference(mels, num_beams=self.num_beams)
        return [self.tokenizer.decode(toks) for toks in tokens]


if __name__ == '__main__':
    # Hack because on many systems Python SSL extensions do not work properly.
    import ssl
    from utils import load_audio
    ssl._create_default_https_context = ssl._create_unverified_context

    transcriber = Transcriber(num_beams=4, on_cuda=False)
    audio = load_audio('data/obama.mp3', 44100)
    print(transcriber.transcribe(audio, 44100))