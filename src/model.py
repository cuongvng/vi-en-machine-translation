import torch
import torch.nn as nn
import math
import sys
sys.path.append("../")
from CONFIG import MAX_LENGTH, EMBEDDING_SIZE, N_TRANSFORMER_LAYERS, N_HEADS, DIM_FEEDFORWARD, VI2EN, EN2VI

class NMT(nn.Module):
    def __init__(self, mode, src_vocab_size, tgt_vocab_size):
        super(NMT, self).__init__()
        assert mode == VI2EN or mode == EN2VI
        self.mode = mode

        self.encoder = Encoder(src_vocab_size=src_vocab_size)
        self.decoder = Decoder(tgt_vocab_size=tgt_vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = self.encoder(src, src_key_padding_mask)
        return self.decoder(tgt, memory, tgt_key_padding_mask)

class Encoder(nn.Module):
    def __init__(self, src_vocab_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=EMBEDDING_SIZE)
        self.positional_encoder = PositionalEncoder(d_model=EMBEDDING_SIZE)
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBEDDING_SIZE, nhead=N_HEADS,
                                                   dim_feedforward=DIM_FEEDFORWARD)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, N_TRANSFORMER_LAYERS)

    def forward(self, X, src_key_padding_mask=None):
        X = self.embedding(X)
        X = X + self.positional_encoder(X)
        X = X.permute(1, 0, 2) # (seq_len, batch_size, embedding_size)
        return self.transformer_encoder(X, src_key_padding_mask=src_key_padding_mask)

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=tgt_vocab_size, embedding_dim=EMBEDDING_SIZE)
        self.positional_encoder = PositionalEncoder(d_model=EMBEDDING_SIZE)
        decoder_layer = nn.TransformerDecoderLayer(d_model=EMBEDDING_SIZE, nhead=N_HEADS,
                                                   dim_feedforward=DIM_FEEDFORWARD)
        self.transfomer_decoder = nn.TransformerDecoder(decoder_layer, N_TRANSFORMER_LAYERS)

        self.fc = nn.Linear(in_features=EMBEDDING_SIZE, out_features=tgt_vocab_size)

    def forward(self, X, encoder_last_state, tgt_key_padding_mask=None):
        X = self.embedding(X)
        X = X + self.positional_encoder(X)
        X = X.permute(1, 0, 2) # (seq_len, batch_size, embedding_size)
        decoder_state = self.transfomer_decoder(tgt=X, memory=encoder_last_state,
                                                tgt_key_padding_mask=tgt_key_padding_mask)
        decoder_state = decoder_state.permute(1, 0, 2) # (batch_size, seq_len, embedding_size)
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