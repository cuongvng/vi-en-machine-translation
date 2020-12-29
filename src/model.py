import torch
import torch.nn as nn
import math
import sys
sys.path.append("../")
from CONFIG import MAX_LENGTH, EMBEDDING_SIZE, N_TRANSFORMER_LAYERS, N_HEADS, DIM_FEEDFORWARD, VI2EN, EN2VI

class NMT(nn.Module):
    def __init__(self, tgt_vocab_size):
        super(NMT, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size=tgt_vocab_size)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        return self.decoder(tgt, memory)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.positional_encoder = PositionalEncoder(d_model=EMBEDDING_SIZE)

        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBEDDING_SIZE, nhead=N_HEADS, dim_feedforward=DIM_FEEDFORWARD)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, N_TRANSFORMER_LAYERS)

    def forward(self, src_tokens):
        X, _ = self.embedder(src_tokens) # (batch_size, max_len, embedding_size)
        X = X + self.positional_encoder(X)
        return self.transformer_encoder(X)

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.positional_encoder = PositionalEncoder(d_model=EMBEDDING_SIZE)

        decoder_layer = nn.TransformerDecoderLayer(d_model=EMBEDDING_SIZE, nhead=N_HEADS, dim_feedforward=DIM_FEEDFORWARD)
        self.transfomer_decoder = nn.TransformerDecoder(decoder_layer, N_TRANSFORMER_LAYERS)

        self.fc = nn.Linear(in_features=EMBEDDING_SIZE, out_features=vocab_size)

    def forward(self, tgt_tokens, encoder_last_state):
        X, valid_len = self.embedder(tgt_tokens)
        X = X + self.positional_encoder(X)
        X = self.transfomer_decoder(tgt=X, memory=encoder_last_state)
        return self.fc(X), valid_len

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=MAX_LENGTH):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)