# GPT ASR - A fast, accurate and super simple speech recognition model

This repo is for GPT ASR, a pytorch-based ML model that does state of the art English speech recognition. The goal of
this repo is to distribute a pretrained model and scripts which can be used to decode your audio files. Training is
more complicated and can be done within my training repo, [DL Art School](https://github.com/neonbjb/DL-Art-School).
Instructions for how to train it are forthcoming.

## Model Description

GPT-ASR uses a causal transformer to decode audio spectrograms into text. To simplify the code, the GPT-2 model from
HuggingFace's transformer repo is used to do most of the heavy lifting. What is left over to do is a lightweight MEL
encoder that sits at the bottom of the model and some format shifting logic.

One nifty thing about this approach is that it is likely to have near-infinite scalability, as has been shown by OpenAI
with its relentless drive towards larger and larger language models. Unfortunately, freely available transcribed speech
data is hard to come by and I found that larger models simply overfit my available data quickly. 

This model was trained on a dataset I termed the "BigASR" dataset. This consists of a concatenation of the following
publicly available datasets:

- LibriTTS (not LibriSpeech - although this design choice was merely because I had LibriTTS on hand and the two are very similar)
- Mozilla CommonVoice
- Facebook's (err Meta's) LibriVox
- Tedlium
- LJSpeech

While these datasets are publicly available, they come in a smattering of formats which makes them very difficult to
work with. BigASR remediates this by following a single unified format for both audio and text. I plan to open this 
dataset up to the community once I can figure out how to host 700GiB of data.

I also use spectral augmentation to normalize the dataset.

## Performance

There are three performance aspects to consider when it comes to these types of ML models:

### Accuracy

This refers to how accurate the model's transcriptions are. Accuracy is generally measured in "word error rate", or the
percentage of words that the model gets wrong.

This model achieves state-of-the-art results for accuracy. On the LibriTTS test-clean test set, the "medium" model
achieves 2.008% word error rate with 8 beams. The smaller "medium-distilled" model achieves a 2.311% word error rate
on the same test set with 1 beam. "Beams" will be defined later.

### Speed

This refers to the speed at which the model can produce predictions. The model is currently only capable of processing
audio ~16 seconds at a time. On a modern computer, this model can transcribe faster than real time. On a GPU, this model
can transcribe **significantly** faster than real time. Here are a few performance metrics, for beam_size=1, batch_size=8,
model=medium:

- Ryzen 7 5800X CPU: .55sec/transcription
- RTX 3090: .23sec/transcription
- RTX 3090, batch_size=128: .04sec/transcription
- Tesla P100: <>

### Size

These are fairly compact models. The "medium" model is 173MiB. The "medium-distilled" model is 97MiB.

## Instructions for use
