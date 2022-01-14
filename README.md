# ocotillo - A fast, accurate and super simple speech recognition model

This repo is for ocotillo, a pytorch-based ML model that does state of the art English speech recognition. The goal of
this repo is to distribute a pretrained model and scripts which can be used to decode your audio files. 

## Model Description

ocotillo uses a causal transformer to decode audio spectrograms into text. To simplify the code, the [GPT-2 model](https://huggingface.co/docs/transformers/model_doc/gpt2) from
HuggingFace's transformer repo is used to do most of the heavy lifting. What code is left over is a lightweight MEL
encoder that sits at the bottom of the model and some format shifting logic.

One nifty thing about this approach is that it is likely to have near-infinite scalability, as has been shown by OpenAI
with its relentless drive towards larger and larger language models. Unfortunately, freely available transcribed speech
data is hard to come by and I found that larger models simply overfit my available data quickly. 

I also use [spectral augmentation](https://arxiv.org/abs/1904.08779) to normalize the audio inputs.

## Data

This model was trained on a dataset I call the "BigASR" dataset. This consists of a concatenation of the following
publicly available datasets:

- [Mozilla CommonVoice](https://arxiv.org/abs/1912.06670)
- [VoxPopuli](https://arxiv.org/abs/2101.00390)
- [LibriSpeech](https://arxiv.org/abs/1904.02882)
- [HiFi TTS](https://arxiv.org/abs/2104.01497)
- [LJSpeech](https://arxiv.org/pdf/1903.11269.pdf)
- [Tedlium](https://arxiv.org/abs/1805.04699)
- [Voxforge](https://arxiv.org/pdf/1805.08615.pdf)

In addition, the model is trained on [bookscorpus](https://huggingface.co/datasets/bookcorpus) in a special text-only mode.
This happens concurrent with speech->text training.

I am extremely grateful to all the authors that produced these datasets. They are incredible resources for the
community and their quality is the primary reason that we can create such incredible transcription models.

It is not easy to gather all of this data in one place. In the interest of allowing others to train their own versions
of this model, I am seeding a torrent to a tar file containing all of this data that is usable by the training instructions
below.

## Performance

There are currently two ocotillo models: large, a 68M parameter model, and small, a 21M parameter model that was distilled
from large.

| Model                      | Params (Million) | MAC (Billion) | WER Librispeech Clean Test | WER Mozilla CV Test | RTPR
|----------------------------|------------------|---------------|----------------------------|---------------------|-----------------------------
| ocotillo small 1-beam      | 21               | 5.0           |                            |                     | 720                        
| ocotillo small 8-beam      | 21               | 5.0           |                            |                     | 140.8                     
| ocotillo large 1-beam      | 68               | 16.4          |                            |                     | 441.6
| ocotillo large 16-beam     | 68               | 16.4          | 6.57                       |                     | 42.2
| wav2vec2 base 960h*        | 94               | 74.1          | 2.1                        |                     |
| deepspeech**               | 87               | 15.7          | 7.1                        | N/A                 |

*: Best score; Base-LV60k from https://arxiv.org/pdf/2006.11477v3.pdf. Model size metrics computed from HuggingFace implementation (does not include Language Model).

**: https://github.com/SeanNaren/deepspeech.pytorch/releases

### Real time processing rate (RTPR)

This is the amount of seconds of audio the model can transcribe with one second of compute. A RTPR of 1 means that the
model can process in realtime. A value of 100 means that the model can process audio at 100x real time. The values in
the above table were averaged across the librispeech-test-clean dataset from an RTX 3090. In addition to the above values,
I measured the RTPR of ocotillo-small and ocotillo-large on a Ryzen 5800x CPU below:

| Model                     | RTPR on CPU
|---------------------------|-----------------------------
| ocotillo small 1-beam     | 32.0
| ocotillo large 1-beam     |

*Note that audio inputs to the model are padded to the same size as the largest clip in a batch. This results in
slightly faster results than you might see in real tests, since this padding does not directly result in text output.*

## Instructions for use

I provide two entry points for using these models:

### API

This repo contains a class called transcribe.Transcriber, which can be used to transcribe audio
data into text. Usage looks like the following:

```python
from transcribe import Transcriber
transcriber = Transcriber(num_beams=4, on_cuda=False)
audio = load_audio('data/obama.mp3', 44100)
print(transcriber.transcribe(audio, 44100))
```

This will automatically download the 'large' model and use it to perform transcription on the CPU.
options to specify a smaller model, perform transcription on a GPU, and perform batch transcription
are available. See api.py.

44100 is the sampling rate of the audio. Transcriber works with numpy arrays and torch arrays.
Audio data must be fp32 on the range [-1,1]. A demo colab notebook that uses the API is included:
asr_demo.ipynb.

### CLI

A script is included, transcribe.py. This script searches for all audio files in a directory and
transcribes all the files found. Sample usage:

```shell
python transcribe.py --path /my/audio/folder --model_path pretrained_model_path.pth --model_type large
                     --cuda=0
```

This will use a GPU to transcribe audio files found in /my/audio/folder. Transcription results
will be written to results.tsv.

### Limitations

The primary limitation of ocotillo is an input size restriction. Speech audio is limited to 35 seconds.
Text output is limited to 300 tokens, which is about 400 characters. It is entirely possible to 
make this model work with infinite-length sequences by using a sliding frame. This is something I'd like
to build in the future.

## Training

ocotillo was trained within my own bespoke training environment, called DLAS. This environment is similar to Pytorch
Lightning in that it abstracts out most of the nitty-gritty details involved in training torch models.

DLAS is available on GitHub but is very much not ready for public consumption. I regularly make changes as needed for 
my personal projects. I do not use robust testing practices. As a result, I make no promises that changes I make won't
break ocotillo training. Therefore, the recommendations that follow suggest you use an exact snapshot in time to perform
training.

1. Download and extract the BigASR dataset.
2. Clone DLAS: `git clone <> && cd DLArtSchool && git reset --hard 009a1e840427a94857c81c2da3920e2f45e93511` 
3. `cd codes && pip install -r requirements.txt`
4. Copy `train/train_large.yml` from this repo to `DLArtSchool/codes`
5. Copy `data/mel_norms.pth` from this repo to `DLArtSchool/codes`
6. Start training: `python train.py -o train_large.yml`

The provided config assumes you have an RTX 3090. Training will converge around 35k iterations. On a single RTX3090,
expect to train for a little less than a week. Checkpoints will be saved every 500 steps. Increase `mega_batch_factor`
in the config to `8` or `16` to support GPUs with less VRAM. Your final model will be found in 
`DLArtSchool/experiments/train_gpt_asr_mass_hf2/models/<last_step>_gpt_ema.pth`.

Training curves for step 5k an onwards can be found here: [https://wandb.ai/neonbjb/gpt_asr/runs/v5802rxo](https://wandb.ai/neonbjb/gpt_asr/runs/v5802rxo).
Training curves for earlier steps are available in [https://wandb.ai/neonbjb/gpt_asr](https://wandb.ai/neonbjb/gpt_asr)
in runs with the name `run16_5gpu_20lyr_huge_receptive_area`.

## Future work

While I am not actively working on ASR, it would be intriguing to try the following things to further improve performance:

* Integrate unsupervised audio pre-training a la wav2vec to the training flow.
* Try out improved decoder architectures, for example some of the advances found in [X-transformers](https://github.com/lucidrains/x-transformers).
* To reduce the context size, experiment with further MEL reductions in the convolutional encoder (currently 8x reduction is used), 
  and experiment with denser tokenizers.
* Run some hyper-parameter sweeps. I did hardly any of these.

I would also like to build a streaming transcriber that works on infinite-length audio.