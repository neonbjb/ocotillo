from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor


def load_model(device, use_torchscript=False):
    """
    Utility function to load the model and corresponding processor to the specified device. Supports loading
    torchscript models when they have been pre-built (which is accomplished by running this file.)
    """
    model = Wav2Vec2ForCTC.from_pretrained("jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli").to(device)
    model.eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(f"facebook/wav2vec2-large-960h")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('jbetker/tacotron_symbols')
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)
    return model, processor


def trace_torchscript_model():
    model, extractor = load_model('cpu')


if __name__ == '__main__':
    trace_torchscript_model()