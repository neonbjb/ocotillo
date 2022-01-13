"""
A utility script for transcribing an entire folder of audio files. This script was used to transcribe the files from
the librispeech and CV test datasets; the results were then separately fed through a word error rate script.
"""

import argparse
import os
from time import time

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AudioFolderDataset
from gpt_asr_hf import GptAsrHf, MODEL_CONFIGS
from mel import MEL
import torch

from tokenizer import VoiceBpeTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Input folder containing audio files you want transcribed.')
    parser.add_argument('--model_path', help='Pretrained model path.')
    parser.add_argument('--model_type', default='large', help='Name of the MODEL_CONFIGS entry describing this model.')
    parser.add_argument('--output_file', default='results.tsv', help='Where transcriptions will be placed.')
    parser.add_argument('--resume', default=0, type=int, help='Skip the first <n> audio tracks.')
    parser.add_argument('--batch_size', default=8, type=int, help='Number of audio files to process at a time. Larger batches are more efficient on a GPU.')
    parser.add_argument('--num_beams', default=1, type=int, help='The number of beams to use when decoding with beam search. Higher numbers of beams '
                                                       'improve accuracy but take longer to compute. This is subject to diminishing returns.')
    parser.add_argument('--cuda', default=-1, type=int, help='The cuda device to perform inference on. -1 (or default) means use the CPU.')
    args = parser.parse_args()

    model = GptAsrHf(**MODEL_CONFIGS[args.model_type]).eval()
    model.load_state_dict(torch.load(args.model_path))
    stft = MEL()
    tokenizer = VoiceBpeTokenizer()

    dataset = AudioFolderDataset(args.path, sampling_rate=22050, pad_to=780283, skip=args.resume)
    dataloader = DataLoader(dataset, args.batch_size, num_workers=4)

    if args.cuda >= 0:
        model = model.cuda(args.cuda)
        stft = stft.cuda(args.cuda)

    start = None
    total_duration = 0
    output = open(args.output_file, 'w')
    with torch.no_grad():
        for e, batch in enumerate(tqdm(dataloader)):
            if start is None:
                start = time()  # Do this here because the first batch often takes a **long** time to load and we are not measuring dataloader performance.
            clip = batch['clip']
            total_duration += clip.shape[0] * clip.shape[-1] / 22050
            if args.cuda >= 0:
                clip = clip.cuda(args.cuda)
            mels = stft(clip)
            tokens = model.inference(mels, num_beams=args.num_beams)
            for b in range(tokens.shape[0]):
                text = tokenizer.decode(tokens[b])
                relpath = os.path.relpath(batch['path'][b], args.path).replace('\\', '/')
                output.write(f'{text}\t{relpath}\n')
            output.flush()
    stop = time()
    elapsed = stop - start
    print(f'Total elapsed: {elapsed}, processing time per second of audio input: {elapsed / total_duration} RTPR: {total_duration / elapsed}')

