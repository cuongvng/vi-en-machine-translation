import torch
import os
import pandas as pd
from pyvi import ViTokenizer
from torchnlp.encoders.text import StaticTokenizerEncoder, DEFAULT_PADDING_INDEX, DEFAULT_EOS_INDEX

def load_data(data_path):
    with open(data_path, 'r') as fr:
        list_sentences = fr.readlines()
    sentence_series = pd.Series(list_sentences)
    return sentence_series

def pad_or_truncate(tokenized_sentence, target_len):
    if isinstance(tokenized_sentence, torch.Tensor):
        tokenized_sentence = tokenized_sentence.tolist()

    if len(tokenized_sentence) >= target_len: # Truncate
        res = tokenized_sentence[:target_len-1]
    else: # Pad
        res = tokenized_sentence[:-1] + [DEFAULT_PADDING_INDEX] * (target_len - len(tokenized_sentence))
    return torch.tensor(res + [DEFAULT_EOS_INDEX], dtype=torch.int64)

def get_encoder(list_training_sentences):
    encoder = StaticTokenizerEncoder(sample=list_training_sentences, min_occurrences=2,
                                     append_sos=True, append_eos=True)
    return encoder

def tokenize_en(data_en, encoder_en):
    pass