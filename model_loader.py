import functools
import os

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor


def load_model(device, use_torchscript=False):
    """
    Utility function to load the model and corresponding processor to the specified device. Supports loading
    torchscript models when they have been pre-built (which is accomplished by running this file.)
    """
    if use_torchscript:
        model = trace_torchscript_model('cuda' if 'cuda' in device else 'cpu')
        model = model.to(device)
    else:
        model = Wav2Vec2ForCTC.from_pretrained("jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli").to(device)
        model.config.return_dict = False
        model.eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(f"facebook/wav2vec2-large-960h")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('jbetker/tacotron_symbols')
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)
    return model, processor


def trace_torchscript_model(dev_type='cpu', load_from_cache=True):
    output_trace_cache_file = f'torchscript/traced_model_{dev_type}.pth'
    if load_from_cache and os.path.exists(output_trace_cache_file):
        return torch.jit.load(output_trace_cache_file)

    print("Model hasn't been traced. Doing so now.")
    model, extractor = load_model(dev_type, use_torchscript=False)
    with torch.autocast(dev_type) and torch.no_grad():
        traced_model = torch.jit.trace(model, (torch.randn((1,16000), device=dev_type)))
    os.makedirs('torchscript', exist_ok=True)
    torch.jit.save(traced_model, output_trace_cache_file)
    print("Done tracing.")
    return model


if __name__ == '__main__':
    trace_torchscript_model('cuda', load_from_cache=False)