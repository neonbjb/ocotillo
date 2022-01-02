import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config

from inference_model import GPT2InferenceModel

# Pretrained model kwargs.
MODEL_CONFIGS = {
    'medium': {
        'layers': 12,
        'model_dim': 512,
        'heads': 8
    },
    'large': {
        'max_symbols_per_phrase': 500,
        'max_mel_frames': 3200,
        'layers': 20,
        'model_dim': 512,
        'heads': 8
    }
}

class ResBlock(nn.Module):
    """
    Basic residual convolutional block that uses GroupNorm.
    """
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan//8, chan),
            nn.ReLU(),
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan//8, chan)
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class LeanMelEncoder(nn.Module):
    """
    Encodes a BxCxS MEL tensor into a latent space suitable for use with a transformer.
    """
    def __init__(self, channels, mel_channels=80, resblocks_per_reduction=1):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(nn.Conv1d(mel_channels, channels//2, kernel_size=5, stride=2, padding=1),
                                     nn.GroupNorm(channels//16, channels//2),
                                     nn.ReLU(),
                                     nn.Sequential(*[ResBlock(channels//2) for _ in range(resblocks_per_reduction)]),
                                     nn.Conv1d(channels//2, channels, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels//8, channels),
                                     nn.ReLU(),
                                     nn.Sequential(*[ResBlock(channels) for _ in range(resblocks_per_reduction)]),
                                     nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels//8, channels),
                                     nn.ReLU(),
                                     nn.Sequential(*[ResBlock(channels) for _ in range(resblocks_per_reduction)]),
                                     )
        self.reduction = 8

    def forward(self, x):
        for e in self.encoder:
            x = e(x)
        return x


def null_position_embeddings(range, dim):
    """
    Helper method which simply returns a range-shaped tensor filled with zeros. Useful for emulating a no-effect
    embedding.
    """
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


class GptAsrHf(nn.Module):
    """
    Core module that encapsulates a set of embeddings, a MEL encoder, a GPT-style transformer and the head needed to
    make its output useful.
    """
    def __init__(self, layers=8, model_dim=512, heads=8, max_symbols_per_phrase=350, max_mel_frames=1600,
                 number_text_tokens=256, start_token=255, stop_token=0, checkpointing=False):
        super().__init__()
        self.number_text_tokens = number_text_tokens
        self.start_token = start_token
        self.stop_token = stop_token
        self.max_symbols_per_phrase = max_symbols_per_phrase
        self.model_dim = model_dim
        self.mel_encoder = LeanMelEncoder(model_dim)
        self.max_mel_frames = max_mel_frames // self.mel_encoder.reduction
        seq_length = 2+self.max_symbols_per_phrase+self.max_mel_frames
        self.gpt_config = GPT2Config(vocab_size=self.number_text_tokens,
                                     n_positions=seq_length,
                                     n_ctx=seq_length,
                                     n_embd=model_dim,
                                     n_layer=layers,
                                     n_head=heads,
                                     gradient_checkpointing=checkpointing,
                                     use_cache=not checkpointing)
        self.gpt = GPT2Model(self.gpt_config)
        # Override the built in positional embeddings
        del self.gpt.wpe
        self.gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)

        # This model uses its own positional embeddings, which helps discriminate between text and audio MELs.
        self.text_pos_embedding = nn.Embedding(self.max_symbols_per_phrase + 1, model_dim)
        self.mel_pos_embedding = nn.Embedding(self.max_mel_frames, model_dim)
        self.text_solo_embedding = nn.Parameter(torch.randn(1,1,512) * self.gpt.config.initializer_range, requires_grad=True)

        # Head layers
        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens)

        # Initialize the embeddings per the GPT-2 scheme
        for module in [self.text_pos_embedding, self.mel_pos_embedding]:
            module.weight.data.normal_(mean=0.0, std=self.gpt.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        """
        Helper function for producing inputs and outputs for the GPT model.
        """
        inp = F.pad(input, (1,0), value=start_token)
        tar = F.pad(input, (0,1), value=stop_token)
        return inp, tar

    def get_logits(self, mel_inputs, text_emb, get_attns=False):
        """
        Helper function for producing text logits.
        """
        if mel_inputs is None:
            emb = text_emb
            mel_len = 0
        else:
            mel_emb = self.mel_encoder(mel_inputs)
            assert mel_emb.shape[-1] <= self.max_mel_frames, f'{mel_emb.shape[-1]} > {self.max_mel_frames}'
            mel_emb = mel_emb.permute(0,2,1).contiguous()
            mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_emb.device))
            emb = torch.cat([mel_emb, text_emb], dim=1)
            mel_len = mel_emb.shape[1]
        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)
        if get_attns:
            return gpt_out.attentions
        enc = gpt_out.last_hidden_state
        text_logits = self.final_norm(enc[:, mel_len:])
        text_logits = self.text_head(text_logits)
        text_logits = text_logits.permute(0,2,1)
        return text_logits

    def forward(self, mel_inputs, text_inputs, return_attentions=False):
        """
        "Normal" forward pass which produces a text loss when given a MEL-encoded audio clip and transcribed text
        targets.
        """
        assert text_inputs.shape[1] <= self.max_symbols_per_phrase, str(text_inputs.shape[1])
        assert text_inputs.max() <= self.number_text_tokens, str(text_inputs.max())

        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_token, self.stop_token)
        text_emb = self.gpt.get_input_embeddings()(text_inputs) + \
                   self.text_pos_embedding(torch.arange(text_inputs.shape[1], device=text_inputs.device))
        text_logits = self.get_logits(mel_inputs, text_emb, get_attns=return_attentions)

        if return_attentions:
            return text_logits  # These weren't really the logits, they were actually attentions despite the variable name.
        loss_text = F.cross_entropy(text_logits, text_targets.long())
        return loss_text.mean(), text_logits

    def text_only(self, text_inputs):
        """
        Used to train on only text inputs.
        """
        assert text_inputs.shape[1] <= self.max_symbols_per_phrase, str(text_inputs.shape[1])
        assert text_inputs.max() <= self.number_text_tokens, str(text_inputs.max())

        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_token, self.stop_token)
        text_emb = self.gpt.get_input_embeddings()(text_inputs) + \
                   self.text_pos_embedding(torch.arange(text_inputs.shape[1], device=text_inputs.device)) + \
                   self.text_solo_embedding
        text_logits = self.get_logits(None, text_emb)
        loss_text = F.cross_entropy(text_logits, text_targets.long())
        return loss_text.mean(), text_logits

    def inference(self, mel_inputs, num_beams=8):
        """
        Performs inference by transcribing mel_inputs into text. Returns the text tokens.
        """
        if not hasattr(self, 'inference_model'):
            self.inference_model = GPT2InferenceModel(self.gpt_config, self.gpt, self.text_pos_embedding, self.final_norm, self.text_head)

        mel_emb = self.mel_encoder(mel_inputs)
        assert mel_emb.shape[-1] <= self.max_mel_frames
        mel_emb = mel_emb.permute(0,2,1).contiguous()
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_emb.device))
        self.inference_model.store_mel_emb(mel_emb)

        # "fake_inputs" are stand-ins for the MEL frames, which will be injected with the prep_inputs function above.
        fake_inputs = torch.full((mel_emb.shape[0],mel_emb.shape[1]+1,), fill_value=1, dtype=torch.long, device=mel_inputs.device)
        fake_inputs[:,-1] = self.start_token
        gen = self.inference_model.generate(fake_inputs, bos_token_id=self.start_token, pad_token_id=0, eos_token_id=0,
                                            max_length=self.max_symbols_per_phrase+mel_emb.shape[1], num_beams=num_beams,
                                            use_cache=True)
        return gen[:, mel_emb.shape[1]+1:]


if __name__ == '__main__':
    gpt = GptAsrHf(max_symbols_per_phrase=250, max_mel_frames=1400, layers=16, model_dim=512, heads=8)
    l = gpt(torch.randn(2,80,640), torch.randint(high=100, size=(2,80)))
    gpt.text_only(torch.randint(high=100, size=(2,120)))