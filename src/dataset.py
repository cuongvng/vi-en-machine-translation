import torch
from torch.utils.data import Dataset
import pandas as pd
from pyvi import ViTokenizer
from torchnlp.encoders.text import StaticTokenizerEncoder, DEFAULT_PADDING_INDEX, DEFAULT_EOS_INDEX
import sys
sys.path.append("../")
from CONFIG import TARGET_VALID_LEN

class EnViTrainingDataSet(Dataset):
    def __init__(self, en_path, vi_path, target_len=TARGET_VALID_LEN+2):
        data_en = load_data(en_path)
        data_vi = load_data(vi_path)
        assert len(data_en) == len(data_vi), "Numbers of vietnamese and english sentences do not match!"

        self.encoder_en = get_encoder(data_en)
        self.encoder_vi = get_encoder(data_vi)
        self.vocab_en = self.encoder_en.vocab
        self.vocab_vi = self.encoder_vi.vocab

        self.token_en = data_en.apply(lambda s: encode_en(s, self.encoder_en, target_len))
        self.token_vi = data_vi.apply(lambda s: encode_vi(s, self.encoder_vi, target_len))

    def __len__(self):
        return len(self.token_en)

    def __getitem__(self, index):
        return self.token_en[index], self.token_vi[index]

class EnViDevTestDataset(Dataset):
    def __init__(self, en_path, vi_path, encoder_en, encoder_vi, target_len=TARGET_VALID_LEN+2):
        data_en = load_data(en_path)
        data_vi = load_data(vi_path)
        assert len(data_en) == len(data_vi), "Numbers of vietnamese and english sentences do not match!"

        self.token_en = data_en.apply(lambda s: encode_en(s, encoder_en, target_len))
        self.token_vi = data_vi.apply(lambda s: encode_vi(s, encoder_vi, target_len))

    def __len__(self):
        return len(self.token_en)

    def __getitem__(self, index):
        return self.token_en[index], self.token_vi[index]


def encode_en(sentence, encoder_en, target_len):
    """
    :input: "If you're a doctor you can do some good things"
    :return: 1D Tensor of indices in the encoder's dictionary of all tokens (words) of the sentence.
    tensor([3, 1, 57, 328, 27, 481, 57, 345, 11, 6, 189, 76, 482, 465, 2])
    """
    encoded_sen = encoder_en.encode(sentence)
    return pad_or_truncate(encoded_sen, target_len)

def encode_vi(sentence, encoder_vi, target_len):
    """
    :input: ""Bà nói với tất cả những đứa cháu rằng chúng đặc biệt"
    :tokenized_value: "Bà nói với tất_cả những đứa cháu rằng chúng đặc_biệt"
    :return: tensor([ 3, 133, 332,  42, 600,  18,  43, 662, 106, 377, 557, 2])
    """
    tokenized_sen = ViTokenizer.tokenize(sentence)
    encoded_sen = encoder_vi.encode(tokenized_sen)
    return pad_or_truncate(encoded_sen, target_len)

def decode_en(sentence, encoder_en):
    """
    :input: tensor([3, 1, 57, 328, 27, 481, 57, 345, 11, 6, 189, 76, 482, 465, 2])
    :return: "If you're a doctor you can do some good things"
    """
    return encoder_en.decode(sentence)

def decode_vi(sentence, encoder_vi):
    return encoder_vi.decode(sentence).replace('_', ' ')

def load_data(data_path):
    with open(data_path, 'r') as fr:
        list_sentences = fr.readlines()
    sentence_series = pd.Series(list_sentences)
    return sentence_series

def get_encoder(list_training_sentences):
    encoder = StaticTokenizerEncoder(sample=list_training_sentences, min_occurrences=2,
                                     append_sos=True, append_eos=True)
    return encoder

def pad_or_truncate(tokenized_sentence, target_len):
    if isinstance(tokenized_sentence, torch.Tensor):
        tokenized_sentence = tokenized_sentence.tolist()

    if len(tokenized_sentence) >= target_len: # Truncate
        res = tokenized_sentence[:target_len-1]
    else: # Pad
        res = tokenized_sentence[:-1] + [DEFAULT_PADDING_INDEX] * (target_len - len(tokenized_sentence))

    return torch.tensor(res + [DEFAULT_EOS_INDEX], dtype=torch.int64)
