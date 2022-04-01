import os
from time import time

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
    os.makedirs('../torchscript', exist_ok=True)
    torch.jit.save(traced_model, output_trace_cache_file)
    print("Done tracing.")
    return model


def trace_onnx_model(dev_type='cpu'):
    model, extractor = load_model(dev_type, use_torchscript=False)
    torch.onnx.export(model, torch.randn((2,16000), device=dev_type), 'ocotillo.onnx', export_params=True, opset_version=13,
                      do_constant_folding=True, input_names=['input'], output_names=['logits'],
                      dynamic_axes={'input': {0: 'batch_size', 1: 'input_length'}, 'output': {0: 'batch_size', 1: 'sequence_length'}})


def test_onnx_model():
    # Test whether the model can be loaded and use the ONNX checker.
    import onnx
    model = onnx.load("ocotillo.onnx")
    onnx.checker.check_model(model)

    import onnxruntime
    from ocotillo.utils import load_audio
    from tqdm import tqdm
    onnx_model = onnxruntime.InferenceSession('../ocotillo.onnx')
    torch_model, _ = load_model('cpu', use_torchscript=True)

    audio = load_audio('data/obama.mp3', 16000).unsqueeze(0)
    audio_norm = (audio - audio.mean()) / torch.sqrt(audio.var() + 1e-7)
    with torch.no_grad():
        start = time()
        for k in tqdm(range(100)):
            logits = torch_model(audio_norm)[0]
        print(f'Elapsed torchscript: {time()-start}')
        tokens = torch.argmax(logits, dim=-1)

        onnx_inputs = {'input': audio_norm.numpy()}
        start = time()
        for k in tqdm(range(100)):
            onnx_outputs = onnx_model.run(None, onnx_inputs)
        print(f'Elapsed ONNX: {time() - start}')

        onnx_tokens = torch.argmax(torch.tensor(onnx_outputs[0]), dim=-1)
        assert torch.all(onnx_tokens == tokens)


if __name__ == '__main__':
    trace_onnx_model()
    test_onnx_model()
    #trace_torchscript_model('cuda', load_from_cache=False)