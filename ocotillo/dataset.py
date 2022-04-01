import sys
import torch.utils.data
from ocotillo.utils import find_audio_files, load_audio


class AudioFolderDataset(torch.utils.data.Dataset):
    def __init__(self, path, sampling_rate, pad_to, skip=0):
        self.audiopaths = find_audio_files(path)[skip:]
        self.sampling_rate = sampling_rate
        self.pad_to = pad_to

    def __getitem__(self, index):
        try:
            path = self.audiopaths[index]
            audio_norm = load_audio(path, self.sampling_rate)
        except:
            print(f"Error loading audio for file {path} {sys.exc_info()}")
            # Recover gracefully. It really sucks when we outright fail.
            return self[index+1]

        orig_length = audio_norm.shape[-1]
        if audio_norm.shape[-1] > self.pad_to:
            print(f"Warning - {path} has a longer audio clip than is allowed: {audio_norm.shape[-1]}; allowed: {self.pad_to}. "
                  f"Truncating the clip, though this will likely invalidate the prediction.")
            audio_norm = audio_norm[:self.pad_to]
        else:
            padding = self.pad_to - audio_norm.shape[-1]
            if padding > 0:
                audio_norm = torch.nn.functional.pad(audio_norm, (0, padding))

        return {
            'clip': audio_norm,
            'samples': orig_length,
            'path': path,
        }

    def __len__(self):
        return len(self.audiopaths)
