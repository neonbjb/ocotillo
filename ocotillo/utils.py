import pathlib

import numpy as np
import torch
import torchaudio
from audio2numpy import open_audio
from scipy.io.wavfile import read


#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#'''                       AUDIO UTILS                          '''
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def find_audio_files(base_path, globs=['*.wav', '*.mp3', '*.ogg', '*.flac']):
    path = pathlib.Path(base_path)
    paths = []
    for glob in globs:
        paths.extend([str(f) for f in path.rglob(glob)])
    return paths


def load_audio(audiopath, sampling_rate, raw_data=None):
    if raw_data is not None:
        # Assume the data is wav format. SciPy's reader can read raw WAV data from a BytesIO wrapper.
        audio, lsr = load_wav_to_torch(raw_data)
    else:
        if audiopath[-4:] == '.wav':
            audio, lsr = load_wav_to_torch(audiopath)
        else:
            audio, lsr = open_audio(audiopath)
            audio = torch.FloatTensor(audio)

    # Remove any channel data.
    if len(audio.shape) > 1:
        if audio.shape[0] < 5:
            audio = audio[0]
        else:
            assert audio.shape[1] < 5
            audio = audio[:, 0]

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '2' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1, 1)

    return audio


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    if data.dtype == np.int32:
        norm_fix = 2 ** 31
    elif data.dtype == np.int16:
        norm_fix = 2 ** 15
    elif data.dtype == np.float16 or data.dtype == np.float32:
        norm_fix = 1.
    else:
        raise NotImplemented(f"Provided data dtype not supported: {data.dtype}")
    return (torch.FloatTensor(data.astype(np.float32)) / norm_fix, sampling_rate)
