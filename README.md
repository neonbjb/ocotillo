# ocotillo - A fast, accurate and super simple speech recognition model

This repo is for ocotillo, a pytorch-based ML model that does state-of-the-art English speech transcription. While this
is not necessarily difficult to accomplish with the libraries available today, every one that I have run to is 
excessively complicated and therefore difficult to use. Ocotillo is dirt simple. It does its job in under 500 lines
of code, and it does a **good** job.

## Model Description

ocotillo uses a model pre-trained with [wav2vec2](https://arxiv.org/abs/2006.11477) and fine-tuned for speech recognition.
This model is hosted by HuggingFace's transformers API, and pretrained weights have been provided by Facebook/Meta.
The specific model being used is [jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli](https://huggingface.co/jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli),
which I personally fine-tuned from existing wav2vec2 checkpoints to also predict punctuation. This makes ocotillo useful
for generating transcriptions which will be used for TTS.

A special thanks goes out to [Patrick von Platen](https://huggingface.co/patrickvonplaten), who contributed (wrote?) the model to huggingface and maintains
the API that does all the heavy lifting. You could easily re-build this repo by simply reading all of his fantastic
blog posts. In particular, [this one on finetuning wav2vec](https://huggingface.co/blog/fine-tune-wav2vec2-english)
and [this one on leveraging a language model with wav2vec](https://huggingface.co/blog/wav2vec2-with-ngram).

## Instructions for use

I provide two entry points for using these models:

### API

This repo contains a class called transcribe.Transcriber, which can be used to transcribe audio
data into text. Usage looks like the following:

```python
from transcribe import Transcriber
transcriber = Transcriber(model_config='large', on_cuda=False)
audio = load_audio('data/obama.mp3', 44100)
print(transcriber.transcribe(audio, sample_rate=44100))
```

This will automatically download the 'large' model and use it to perform transcription on the CPU.
Options to specify a smaller model, perform transcription on a GPU, and perform batch transcription
are available. See api.py.

Transcriber works with numpy arrays and torch arrays. Audio data must be fp32 on the range [-1,1]. A demo colab 
notebook that uses the API is included:
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

ocotillo currently has input size limitations. Because it is built on a transformer architecture, the memory cost with
respect to sequence size is quadratic in nature. In the future I would like to build automatic chunking or streaming 
ASR which could be used to transcribe large contiguous audio samples.
