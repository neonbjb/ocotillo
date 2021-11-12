import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config

from inference_model import GPT2InferenceModel
from utils import symbols


# Pretrained model kwargs.
MODEL_CONFIGS = {
    'medium': {
        'layers': 12,
        'model_dim': 512,
        'heads': 8
    },
    'medium-distilled': {
        'layers': 6,
        'model_dim': 512,
        'heads': 8
    }
}


class ResBlock(nn.Module):
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


class MelEncoder(nn.Module):
    def __init__(self, channels, mel_channels=80):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(nn.Conv1d(mel_channels, channels//4, kernel_size=5, padding=2),
                                     ResBlock(channels//4),
                                     ResBlock(channels//4),
                                     nn.Conv1d(channels//4, channels//2, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels//16, channels//2),
                                     nn.ReLU(),
                                     ResBlock(channels//2),
                                     ResBlock(channels//2),
                                     nn.Conv1d(channels//2, channels, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels//8, channels),
                                     nn.ReLU(),
                                     ResBlock(channels),
                                     ResBlock(channels)
                                     )

    def forward(self, x):
        return self.encoder(x)


class GptAsrHf(nn.Module):
    NUMBER_SYMBOLS = len(symbols)
    NUMBER_TEXT_TOKENS = NUMBER_SYMBOLS+1

    def __init__(self, layers=8, model_dim=512, heads=8, max_symbols_per_phrase=250, max_mel_frames=1400, checkpointing=False):
        super().__init__()
        self.max_mel_frames = max_mel_frames // 4  # Mel frames are reduced by a factor of 4 during encoding.
        self.max_symbols_per_phrase = max_symbols_per_phrase

        self.model_dim = model_dim
        self.max_mel_frames = self.max_mel_frames
        self.mel_encoder = MelEncoder(model_dim)
        self.text_pos_embedding = nn.Embedding(self.max_symbols_per_phrase + 1, model_dim)
        self.mel_pos_embedding = nn.Embedding(self.max_mel_frames, model_dim)
        seq_length = 2+self.max_symbols_per_phrase+self.max_mel_frames
        self.gpt_config = GPT2Config(vocab_size=self.NUMBER_TEXT_TOKENS,
                                        n_positions=seq_length,
                                        n_ctx=seq_length,
                                        n_embd=model_dim,
                                        n_layer=layers,
                                        n_head=heads,
                                        gradient_checkpointing=checkpointing,
                                        use_cache=not checkpointing)
        self.gpt = GPT2Model(self.gpt_config)
        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.NUMBER_TEXT_TOKENS)


    def get_logits(self, mel_inputs, text_targets, get_attns=False):
        # Pad front and back. Pad at front is the "START" token.
        text_targets = F.pad(text_targets, (1,0), value=self.NUMBER_SYMBOLS)
        text_targets = F.pad(text_targets, (0, self.max_symbols_per_phrase - text_targets.shape[1]))
        text_emb = self.gpt.get_input_embeddings()(text_targets)
        text_emb = text_emb + self.text_pos_embedding(torch.arange(text_emb.shape[1], device=text_targets.device))
        mel_emb = self.mel_encoder(mel_inputs)
        mel_emb = F.pad(mel_emb, (0, self.max_mel_frames - mel_emb.shape[-1]))
        mel_emb = mel_emb.permute(0,2,1).contiguous()
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_emb.device))
        emb = torch.cat([mel_emb, text_emb], dim=1)
        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)
        if get_attns:
            return gpt_out.attentions
        enc = gpt_out.last_hidden_state
        text_logits = self.final_norm(enc[:, self.max_mel_frames:])
        text_logits = self.text_head(text_logits)
        text_logits = text_logits.permute(0,2,1)
        return text_logits

    def forward(self, mel_inputs, text_targets, return_attentions=False):
        text_logits = self.get_logits(mel_inputs, text_targets, get_attns=return_attentions)
        if return_attentions:
            return text_logits  # These weren't really the logits.
        loss_text = F.cross_entropy(text_logits, text_targets.long())
        return loss_text.mean(), text_logits

    def inference(self, mel_inputs, cond_text=None, do_sample=False, temperature=1.0, num_beams=8):
        if not hasattr(self, 'inference_model'):
            self.inference_model = GPT2InferenceModel(self.gpt_config, self.gpt, self.text_pos_embedding, self.final_norm, self.text_head)

        mel_emb = self.mel_encoder(mel_inputs)
        assert mel_emb.shape[-1] <= self.max_mel_frames
        mel_emb = F.pad(mel_emb, (0, self.max_mel_frames - mel_emb.shape[-1]))
        mel_emb = mel_emb.permute(0,2,1).contiguous()
        mel_emb = mel_emb + self.mel_pos_embedding(torch.arange(mel_emb.shape[1], device=mel_emb.device))
        self.inference_model.store_mel_emb(mel_emb)

        # "fake_inputs" are stand-ins for the MEL frames, which will be injected with the prep_inputs function above.
        if cond_text is None:
            fake_inputs = torch.full((mel_inputs.shape[0],self.max_mel_frames+1,), fill_value=1, dtype=torch.long, device=mel_inputs.device)
            fake_inputs[:,-1] = self.NUMBER_SYMBOLS
        else:
            cond_used = 10
            fake_inputs = torch.full((mel_inputs.shape[0],self.max_mel_frames+1+cond_used,), fill_value=1, dtype=torch.long, device=mel_inputs.device)
            fake_inputs[:,-1-cond_used] = self.NUMBER_SYMBOLS
            fake_inputs[:, -cond_used:] = cond_text[:, :cond_used]
        gen = self.inference_model.generate(fake_inputs, do_sample=do_sample, bos_token_id=self.NUMBER_SYMBOLS, pad_token_id=0, eos_token_id=0,
                          max_length=self.max_symbols_per_phrase+self.max_mel_frames, temperature=temperature, num_beams=num_beams, use_cache=True)
        return gen[:, self.max_mel_frames:]
