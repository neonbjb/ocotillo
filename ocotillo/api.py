from time import time

import torch
import torchaudio

from ocotillo.model_loader import load_model
from ocotillo.utils import load_audio

def _append_with_at_least_one_space(text, new_text):
    if not text.endswith(' ') and not new_text.startswith(' '):
        text = text + " "
    return text + new_text

class Transcriber:
    def __init__(self, phonetic=False, on_cuda=True, cuda_device=0):
        if on_cuda:
            self.device = f'cuda:{cuda_device}'
        else:
            self.device = 'cpu'
        self.model, self.processor = load_model(self.device, phonetic=phonetic)

    def transcribe(self, audio_data, sample_rate):
        """
        Transcribes audio_data at the given sample_rate. audio_data is expected to be a list of floats, a torch tensor
        or a numpy array. audio_data must be either 1d mono audio data or 2d stereo data (one channel is thrown out).
        The channel dimension must be first.
        """
        if not isinstance(audio_data, torch.Tensor):
            audio_data = torch.tensor(audio_data, dtype=torch.float)
        audio_data = audio_data.to(self.device).unsqueeze(0)
        if audio_data.shape[-1] > 30 * sample_rate:
            return self._process_large_clip(audio_data, sample_rate)
        else:
            return self.transcribe_batch(audio_data, sample_rate)[0]

    def transcribe_batch(self, audio_data, sample_rate):
        """
        Transcribes audio_data at the given sample_rate. audio_data is expected to be a torch tensor
        or a numpy array. audio_data must be either 2d mono audio data or 3d stereo data (one channel is thrown out).
        The batch dimension is first. The channel dimension is second. transcribe_batch() will not handle audio clips
        of length > 30 seconds. Transcribe long clips individually using transcribe.
        """
        if not isinstance(audio_data, torch.Tensor):
            audio_data = torch.tensor(audio_data, dtype=torch.float)  # This makes valid inputs either a torch tensor a numpy array.
        if sample_rate != 16000:
            audio_data = torchaudio.functional.resample(audio_data, sample_rate, 16000)
        assert audio_data.shape[-1] < 30 * 16000
        audio_data = audio_data.to(self.device)
        clip_norm = (audio_data - audio_data.mean()) / torch.sqrt(audio_data.var() + 1e-7)
        with torch.no_grad():
            logits = self.model(clip_norm)[0]
            tokens = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(tokens)

    def _process_large_clip(self, audio_data, sample_rate):
        # Handle extra-long clips by breaking them into chunks with 2 seconds of shared audio on each end.
        if sample_rate != 16000:
            audio_data = torchaudio.functional.resample(audio_data, sample_rate, 16000)
        wav2vec_logit_unit = 320
        chunk_sz = wav2vec_logit_unit * 1000  # Roughly 20 seconds
        logits_overlap_sz = 100
        overlap_sz = wav2vec_logit_unit * logits_overlap_sz  # Roughly 2 seconds
        chunked_clips = []
        i = 0
        while i < audio_data.shape[-1]:
            chunked_clips.append(audio_data[:, max(0, i - overlap_sz):i + chunk_sz])
            i += chunk_sz
        # Pad the first and last elements so they can be batched together.
        chunked_clips[0] = torch.nn.functional.pad(chunked_clips[0], (overlap_sz,0))
        chunked_clips[-1] = torch.nn.functional.pad(chunked_clips[-1],(0,overlap_sz+chunk_sz-chunked_clips[-1].shape[-1]))
        chunked_clips = torch.cat(chunked_clips, dim=0)
        # Further split up the tensor into sub-batches that can fit in device memory. <8> is arbitrarily chosen
        # because it can fit in 12GiB of GPU memory.
        chunked_clips = torch.split(chunked_clips, 8, dim=0)
        text_out = ""
        last_logits = None
        for batch in chunked_clips:
            with torch.no_grad():
                batch_norm = (batch - batch.mean()) / torch.sqrt(batch.var() + 1e-7)
                batched_logits = self.model(batch_norm)[0]
                for logits in batched_logits:
                    if last_logits is None:
                        last_logits = logits
                    else:
                        # Use the overlap to finalize last_logits into text.
                        last_logits[-logits_overlap_sz:] = (last_logits[-logits_overlap_sz:] + logits[:logits_overlap_sz]) / 2
                        tokens = torch.argmax(last_logits, dim=-1)
                        text_out = _append_with_at_least_one_space(text_out, self.processor.decode(tokens))
                        last_logits = logits[logits_overlap_sz:]
        # Don't forget about the remaining logits!
        tokens = torch.argmax(last_logits, dim=-1)
        text_out = _append_with_at_least_one_space(text_out, self.processor.decode(tokens))
        return text_out


if __name__ == '__main__':
    transcriber = Transcriber(on_cuda=True, phonetic=True)
    audio = load_audio('../data/obama.mp3', 44100)
    print(transcriber.transcribe(audio, 44100))
    start = time()
    audio = load_audio('../data/obama_long.mp3', 16000)
    print(transcriber.transcribe(audio, 16000))
    print(f"Elapsed: {time() - start}")