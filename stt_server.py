import argparse
from io import BytesIO
from time import time

import torch
from flask import Flask, request

from ocotillo.model_loader import load_model
from ocotillo.utils import load_audio


def create_app(cfg={}):
    app = Flask("ocotillo wav2vec2 STT")

    use_cuda = 'cuda' in cfg.keys()
    dev = f"cuda:{cfg['cuda']}" if use_cuda else 'cpu'
    model, processor = load_model(dev)

    if use_cuda:
        # Forward prop the model to pre-load it into the GPU. This greatly speeds up the first inference.
        model(torch.randn((1,16000), device=dev))

    @app.route('/stt', methods=['POST'])
    def stt():
        with torch.no_grad():
            start_request = time()
            wav_wrapper = BytesIO(request.data)
            clip = load_audio("", 16000, wav_wrapper).unsqueeze(0)
            # Normalize, which obviates the need for to shuffle bytes around in processor.
            clip_norm = (clip - clip.mean()) / torch.sqrt(clip.var() + 1e-7)
            if use_cuda:
                clip_norm = clip_norm.to(dev)
            model_inference_start = time()
            logits = model(clip_norm)[0]
            inference_latency = time() - model_inference_start
            tokens = torch.argmax(logits, dim=-1)
            text = processor.decode(tokens[0])
            request_elapsed = time() - start_request
            print(f"Recognized text: {text}; request latency: {request_elapsed}; model_inference_latency: {inference_latency}")

        return {"text": text}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=0, type=int, help='Cuda device to use for inference, or -1 if cpu is to be used')
    parser.add_argument('--port', default=80, type=int, help='Port the server will bind to')
    parser.add_argument('--bind_addr', default="0.0.0.0", type=str, help='IP address to bind to. Use 0.0.0.0 to accept remote connections, otherwise localhost.')
    args = parser.parse_args()

    flask_cfg = {}
    if args.cuda >= 0:
        flask_cfg['cuda'] = str(args.cuda)
    app = create_app(flask_cfg)
    app.run(host=args.bind_addr, port=args.port, debug=True)