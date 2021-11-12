import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AudioFolderDataset
from gpt_asr_hf import GptAsrHf, MODEL_CONFIGS
from stft import MELSTFT
from utils import sequence_to_text
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Input folder containing audio files you want transcribed.')
    parser.add_argument('--model_path', help='Pretrained model path.')
    parser.add_argument('--output_file', default='results.tsv', help='Where transcriptions will be placed.')
    parser.add_argument('--resume', default=0, type=int, help='Skip the first <n> audio tracks.')
    parser.add_argument('--batch_size', default=8, type=int, help='Number of audio files to process at a time. Larger batches are more efficient on a GPU.')
    parser.add_argument('--model_type', default='medium', help='Name of the MODEL_CONFIGS entry describing this model.')
    parser.add_argument('--do_sampling', default=False, type=bool, help='Whether or not the model should randomly sample from the models predictions. If '
                                                             'False, the model simply picks the most likely prediction.')
    parser.add_argument('--temperature', default='1.0', type=float, help='The softmax temperature to use. Lower values make the model more ' \
                                                            'likely to "explore" alternative transcription possibilities. Only effective '
                                                             'if do_sampling is enabled.')
    parser.add_argument('--num_beams', default=1, type=int, help='The number of beams to use when decoding with beam search. Higher numbers of beams '
                                                       'improve accuracy but take longer to compute. This is subject to diminishing returns.')
    parser.add_argument('--cuda', default=-1, type=int, help='The cuda device to perform inference on. -1 (or default) means use the CPU.')
    args = parser.parse_args()

    model = GptAsrHf(**MODEL_CONFIGS[args.model_type])
    model.load_state_dict(torch.load(args.model_path))
    stft = MELSTFT()

    dataset = AudioFolderDataset(args.path, sampling_rate=22050, pad_to=358395, skip=args.resume)
    dataloader = DataLoader(dataset, args.batch_size, num_workers=2)

    if args.cuda >= 0:
        model = model.cuda(args.cuda)
        stft = stft.cuda(args.cuda)

    output = open(args.output_file, 'w')
    with torch.no_grad():
        for e, batch in enumerate(tqdm(dataloader)):
            clip = batch['clip']
            if args.cuda >= 0:
                clip = clip.cuda(args.cuda)
            mels = stft(clip)
            tokens = model.inference(mels, do_sample=args.do_sampling, temperature=args.temperature, num_beams=args.num_beams)
            for b in range(tokens.shape[0]):
                text = sequence_to_text(tokens[b]).replace('_', '')
                output.write(f'{batch["path"][b]}\t{text}\n')
            output.flush()

