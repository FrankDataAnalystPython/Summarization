import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        self.device = device

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(self.device)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)


    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        # Note the output shape is Batch_size FIRST
        return x


class Transformer(nn.Module):
    def __init__(self,
                 embed_size,
                 vocab_size,
                 num_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 forward_expansion,
                 dropout,
                 device,
                 pad_idx,
                 unk_idx
                 ):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embed_size,
                                      padding_idx = pad_idx
                                      )
        self.embedding.weight.data[unk_idx] = 0
        self.pos = PositionalEncoding(d_model = embed_size,
                                      device = device
                                      )
        self.device = device
        self.transformer = nn.Transformer(
            d_model = embed_size,
            nhead = num_heads,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward = forward_expansion * embed_size,
            dropout = dropout
        )

        self.fc = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def forward(self,
                context,
                summary,    # Note the summary is not None during predict, because the first token is sos
                context_padding_mask = None,
                summary_padding_mask = None
                ):
        batch_size, summary_length = summary.shape
        context_embed = self.embedding(context)
        context_embed = self.pos(context_embed)
        context_embed = self.dropout(context_embed).to(self.device)

        summary_embed = self.embedding(summary)
        summary_embed = self.pos(summary_embed)
        summary_embed = self.dropout(summary_embed).to(self.device)

        summary_attention_mask = self.transformer.generate_square_subsequent_mask(summary_length).to(self.device)

        context_embed = torch.einsum('ijk->jik', context_embed)
        summary_embed = torch.einsum('ijk->jik', summary_embed)

        out = self.transformer(src = context_embed,
                               tgt = summary_embed,
                               tgt_mask = summary_attention_mask,
                               src_key_padding_mask = context_padding_mask,
                               tgt_key_padding_mask = summary_padding_mask
                               )

        out = torch.einsum('jik->ijk', out)
        out = self.fc(out)
        return out
