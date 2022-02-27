# 🌵 ocotillo - A fast, accurate and super simple speech recognition model

This repo is for ocotillo, a pytorch-based ML model that does state-of-the-art English speech transcription. While this
is not necessarily difficult to accomplish with the libraries available today, every one that I have run to is 
excessively complicated and therefore difficult to use. Ocotillo is dirt simple. The APIs I offer have almost no
configuration options: just feed your speech in and go.

It's also fast. It traces the underlying model to torchscript. This means most of the heavy lifting is done in C++.
The transcribe.py script achieves a processing rate 329x faster than realtime on an NVIDIA A5000 GPU when transcribing
batches of 16 audio files at once.

## Model Description

ocotillo uses a model pre-trained with [wav2vec2](https://arxiv.org/abs/2006.11477) and fine-tuned for speech recognition.
This model is hosted by HuggingFace's transformers API, and pretrained weights have been provided by Facebook/Meta.
The specific model being used is [jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli](https://huggingface.co/jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli),
which I personally fine-tuned from existing wav2vec2 checkpoints to also predict punctuation. This makes ocotillo useful
for generating transcriptions which will be used for TTS.

A special thanks goes out to [Patrick von Platen](https://huggingface.co/patrickvonplaten), who contributed (wrote?) the model to huggingface and maintains
the API that does all the heavy lifting. His fantastic blog posts were instrumental in building this repo. 
In particular, [this one on finetuning wav2vec](https://huggingface.co/blog/fine-tune-wav2vec2-english)
and [this one on leveraging a language model with wav2vec](https://huggingface.co/blog/wav2vec2-with-ngram).

## Instructions for use

There are several ways to use ocotillo, described below. First you need to clone it and install its dependencies:

```shell
git clone https://github.com/neonbjb/ocotillo.git
cd ocotillo
pip install -r requirements.txt
```

### Simple CLI

This is the most dead-simple way to get started with ocotillo. Find an audio clip on your computer, and run:

```shell
ocotillo path/to/audio/clip.mp3
```

### Batch CLI

A script is included, transcribe.py. This script searches for all audio files in a directory and
transcribes all the files found. Sample usage:

```shell
python transcribe.py --path /my/audio/folder --model_path pretrained_model_path.pth --cuda=0
```

This will use a GPU to transcribe audio files found in /my/audio/folder. Transcription results
will be written to results.tsv.


### API

This repo contains a class called transcribe.Transcriber, which can be used to transcribe audio
data into text. Usage looks like the following:

```python
from transcribe import Transcriber
transcriber = Transcriber(on_cuda=False)
audio = load_audio('data/obama.mp3', 44100)
print(transcriber.transcribe(audio, sample_rate=44100))
```

This will automatically download the 'large' model and use it to perform transcription on the CPU.
Options to specify a smaller model, perform transcription on a GPU, and perform batch transcription
are available. See api.py.

Transcriber works with numpy arrays and torch arrays. Audio data must be fp32 on the range [-1,1]. A demo colab 
notebook that uses the API is included:
asr_demo.ipynb.

### HTTP server with [Mycroft](https://github.com/MycroftAI) support

This will allow you to run a speech-to-text server that operates the ocotillo model. The protocol was specifically
designed to work with the open source assistant Mycroft.

This server does not need to run on the same device as you run mycroft (but your mycroft device needs to be on the
same network, or you need to expose your server to the web - not recommended).

Responses are fast and high quality. On a modern x86 CPU, expect responses to most queries in under a second. On CUDA,
responses take less than a tenth of a second (most of which is data processing - model inference is on the order of 
10s of milliseconds). I have not tested ocotillo on embedded hardware like the Pi.

1. Install Flask: `pip install flask`.
2. Start server: `python stt_server.py`. CUDA device 0 is used by default, specify `--cuda=-1` to run on CPU.
3. (optional) Install Mycroft: https://mycroft.ai/get-started/
4. From mycroft build directory: `bin/mycroft-config edit user`
5. Add the following code:
    ```json
    {
      "stt": {
        "deepspeech_server": {
          "uri": "http://<your_ip_address>/stt"
        },
        "module": "deepspeech_server"
      },
    }
    ```
6. Restart mycroft: `./stop-mycroft.sh && ./start-mycroft.sh`
