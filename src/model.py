import torch
import torch.nn as nn
import math
import sys
sys.path.append("../")
from embedder import BERT, PhoBERT
from CONFIG import MAX_LENGTH, EMBEDDING_SIZE, N_TRANSFORMER_LAYERS, N_HEADS, DIM_FEEDFORWARD, VI2EN, EN2VI

class NMT(nn.Module):
    def __init__(self, mode, tgt_vocab_size):
        super(NMT, self).__init__()
        assert mode == VI2EN or mode == EN2VI
        self.mode = mode
        self.bert = BERT()
        self.phobert = PhoBERT()
        self.positional_encoder = PositionalEncoder(d_model=EMBEDDING_SIZE)
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size=tgt_vocab_size)

    def forward(self, src, tgt):
        # Embedding
        if self.mode == EN2VI:
            src = self.bert(src)
            tgt = self.phobert(tgt)
        else:
            src = self.phobert(src)
            tgt = self.bert(tgt)

        # Positional encoding
        src = src + self.positional_encoder(src)
        tgt = src + self.positional_encoder(tgt)

        memory = self.encoder(src)
        return self.decoder(tgt, memory)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBEDDING_SIZE, nhead=N_HEADS, dim_feedforward=DIM_FEEDFORWARD)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, N_TRANSFORMER_LAYERS)

    def forward(self, X):
        return self.transformer_encoder(X)

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()

        decoder_layer = nn.TransformerDecoderLayer(d_model=EMBEDDING_SIZE, nhead=N_HEADS, dim_feedforward=DIM_FEEDFORWARD)
        self.transfomer_decoder = nn.TransformerDecoder(decoder_layer, N_TRANSFORMER_LAYERS)

        self.fc = nn.Linear(in_features=EMBEDDING_SIZE, out_features=vocab_size)

    def forward(self, X, encoder_last_state):
        decoder_state = self.transfomer_decoder(tgt=X, memory=encoder_last_state)
        return decoder_state, self.fc(decoder_state)

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