import pathlib

import torch
import torchaudio
from scipy.io.wavfile import read
import numpy as np
from audio2numpy import open_audio


#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#'''                       AUDIO UTILS                          '''
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def find_audio_files(base_path, globs=['*.wav', '*.mp3', '*.ogg', '*.flac']):
    path = pathlib.Path(base_path)
    paths = []
    for glob in globs:
        paths.extend([str(f) for f in path.rglob(glob)])
    return paths


def load_audio(audiopath, sampling_rate):
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


#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#'''                       TEXT UTILS                           '''
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

valid_symbols = [
  'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
  'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
  'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
  'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
  'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
  'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
  'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''

  _id_to_symbol = {i: s for i, s in enumerate(symbols)}

  result = ''
  for symbol_id in sequence:
    if isinstance(symbol_id, torch.Tensor):
      symbol_id = symbol_id.item()
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')