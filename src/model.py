import torch
import torch.nn as nn
import math
import sys
sys.path.append("../")
from CONFIG import MAX_LENGTH


class En2ViEncoder(nn.Module):
    pass

class En2ViDecoder(nn.Module):
    pass

class Vi2EnEncoder(nn.Module):
    pass

class Vi2EnDecoder(nn.Module):
    pass

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

def xavier_init_weights(model):
    if isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)
    return