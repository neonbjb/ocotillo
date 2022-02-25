from io import BytesIO
from time import time

import torch
import torchaudio
from flask import Flask, request
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor

from utils import load_audio

def create_app(cfg={}):
    app = Flask("ocotillo wav2vec2 STT")
    model = Wav2Vec2ForCTC.from_pretrained("jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(f"facebook/wav2vec2-large-960h")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('jbetker/tacotron_symbols')
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)

    use_cuda = 'cuda' in cfg.keys()
    if use_cuda:
        cuda_dev = f"cuda:{cfg['cuda']}"
        model = model.to(cuda_dev)

    @app.route('/stt', methods=['POST'])
    def stt():
        start_request = time()
        wav_wrapper = BytesIO(request.data)
        clip = load_audio("", 16000, wav_wrapper).unsqueeze(0)
        # Normalize, which obviates the need for to shuffle bytes around in processor.
        clip_norm = (clip - clip.mean()) / torch.sqrt(clip.var() + 1e-7)
        if use_cuda:
            clip_norm = clip_norm.to(cuda_dev)
        logits = model(clip_norm).logits
        tokens = torch.argmax(logits, dim=-1)
        text = processor.decode(tokens[0])
        request_elapsed = time() - start_request
        print(f"Recognized text: {text}; request latency: {request_elapsed}")

        return {"text": text}

    return app
