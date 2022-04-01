import argparse

from ocotillo.api import Transcriber
from ocotillo.utils import load_audio

if __name__ == '__main__':
    """
    Simple tool that transcribes the given audio file and prints the transcription to standard out.
    
    Operations are performed on CPU. If you want/need CUDA, modify this script or consider using transcribe.py, which
    is better for batched operations anyways.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input audio file.')
    args = parser.parse_args()

    transcriber = Transcriber(on_cuda=False)
    audio = load_audio(args.input, 16000)
    print(transcriber.transcribe(audio, 16000))
