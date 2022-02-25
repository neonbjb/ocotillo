"""
A utility script for transcribing an entire folder of audio files. This script was used to transcribe the files from
the librispeech and CV test datasets; the results were then separately fed through a word error rate script.
"""

import argparse
import os
from time import time

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer

from dataset import AudioFolderDataset
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Input folder containing audio files you want transcribed.')
    parser.add_argument('--output_file', default='results.tsv', help='Where transcriptions will be placed.')
    parser.add_argument('--resume', default=0, type=int, help='Skip the first <n> audio tracks.')
    parser.add_argument('--batch_size', default=8, type=int, help='Number of audio files to process at a time. Larger batches are more efficient on a GPU.')
    parser.add_argument('--cuda', default=-1, type=int, help='The cuda device to perform inference on. -1 (or default) means use the CPU.')
    parser.add_argument('--output_tokens', default=False, type=bool, help='Whether or not to output the CTC codes. Useful for text alignment.')
    args = parser.parse_args()

    model = Wav2Vec2ForCTC.from_pretrained("jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli").to("cuda")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(f"facebook/wav2vec2-large-960h")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('jbetker/tacotron_symbols')
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)

    dataset = AudioFolderDataset(args.path, sampling_rate=16000, pad_to=566400, skip=args.resume)
    dataloader = DataLoader(dataset, args.batch_size, num_workers=2)

    if args.cuda >= 0:
        model = model.cuda(args.cuda)

    start = None
    total_duration = 0
    mode = 'w' if args.resume == 0 else 'a'
    output = open(args.output_file, mode, encoding='utf-8')
    with torch.no_grad():
        for e, batch in enumerate(tqdm(dataloader)):
            if start is None:
                start = time()  # Do this here because the first batch often takes a **long** time to load and we are not measuring dataloader performance.
            max_sample_size = batch['samples'].max()
            clip = batch['clip'][:, :max_sample_size]
            total_duration += clip.shape[0] * clip.shape[-1] / 16000
            clip = [b.numpy() for b in clip]  # Because the processor takes in numpy values.
            clip = processor(clip, return_tensors='pt', padding='longest', sampling_rate=16000).input_values
            if args.cuda >= 0:
                clip = clip.cuda(args.cuda)

            logits = model(clip).logits
            tokens = torch.argmax(logits, dim=-1)
            for b in range(tokens.shape[0]):
                # Chop off all the padding for each batch element.
                usage_percent = batch['samples'][b] / max_sample_size
                sub_tokens = tokens[b, :int(usage_percent*tokens.shape[-1])]
                # Decode and write to the output file.
                text = processor.decode(sub_tokens)
                relpath = os.path.relpath(batch['path'][b], args.path).replace('\\', '/')
                if args.output_tokens:
                    output.write(f'{text.lower()}\t{relpath}\t{sub_tokens.tolist()}\n')
                else:
                    output.write(f'{text.lower()}\t{relpath}\n')
            output.flush()
    stop = time()
    elapsed = stop - start
    print(f'Total elapsed: {elapsed}, processing time per second of audio input: {elapsed / total_duration} RTPR: {total_duration / elapsed}')

