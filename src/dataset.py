import torch
from torch.utils.data import Dataset
from torchnlp.encoders.text import StaticTokenizerEncoder, DEFAULT_PADDING_INDEX, DEFAULT_EOS_INDEX
import pandas as pd
from pyvi import ViTokenizer
import sys
sys.path.append("../")
from CONFIG import MAX_LENGTH

class IWSLT15EnViDataSet(Dataset):
    def __init__(self, en_path, vi_path):
        self.data_en = load_data(en_path)
        data_vi = load_data(vi_path)
        # Segment Vietnamese words
        self.data_vi = data_vi.apply(lambda s: ViTokenizer.tokenize(s))

        # Tokenizing
        self.tokenizer_en = get_tokenizer(self.data_en)
        self.tokenizer_vi = get_tokenizer(self.data_vi)

        assert self.data_en.shape[0] == self.data_vi.shape[0], \
            "Numbers of vietnamese and english sentences do not match!"

        self.en_vocab_size = self.tokenizer_en.vocab_size
        self.vi_vocab_size = self.tokenizer_vi.vocab_size

        print("English vocab size:", self.en_vocab_size)
        print("Vietnamese vocab size:", self.vi_vocab_size)

    def __len__(self):
        return self.data_vi.shape[0]

    def __getitem__(self, index):
        tokens_en, valid_len_en = convert_tokens_to_indices(self.data_en[index], self.tokenizer_en)
        tokens_vi, valid_len_vi = convert_tokens_to_indices(self.data_vi[index], self.tokenizer_vi)
        return tokens_en, valid_len_en, tokens_vi, valid_len_vi

    def get_vocab_en(self):
        return self.tokenizer_en.vocab

    def get_vocab_vi(self):
        return self.tokenizer_vi.vocab

def get_tokenizer(list_training_sentences):
    tokenizer = StaticTokenizerEncoder(sample=list_training_sentences, min_occurrences=2,
                                     append_sos=False, append_eos=True)
    return tokenizer

def convert_tokens_to_indices(sentence, tokenizer, max_len=MAX_LENGTH):
    """
    :input: "If you're a doctor you can do some good things"
    :return: tensor([3, 1, 57, 328, 27, 481, 57, 345, 11, 6, 189, 76, 482, 465, 2]), valid

    :input: "Bà nói với tất_cả những đứa cháu rằng chúng đặc_biệt"
    :return: tensor([ 3, 133, 332,  42, 600,  18,  43, 662, 106, 377, 557, 2])
    """
    tokens = tokenizer.encode(sentence)
    return pad_or_truncate(tokens, max_len)

def convert_indices_to_tokens(indices, tokenizer):
    """
    :input: tensor([3, 1, 57, 328, 27, 481, 57, 345, 11, 6, 189, 76, 482, 465, 2])
    :return: "If you're a doctor you can do some good things"
    """
    return tokenizer.decode(indices)

def pad_or_truncate(tokenized_sentence, max_len):
    if isinstance(tokenized_sentence, torch.Tensor):
        tokenized_sentence = tokenized_sentence.tolist()

    if len(tokenized_sentence) >= max_len: # Truncate
        res = tokenized_sentence[:max_len-1] + [DEFAULT_EOS_INDEX]
        valid_len = max_len
    else: # Pad
        res = tokenized_sentence + [DEFAULT_PADDING_INDEX]*(max_len - len(tokenized_sentence))
        valid_len = len(tokenized_sentence)

    return torch.tensor(res, dtype=torch.int64), valid_len

def load_data(data_path):
    with open(data_path, 'r') as fr:
        list_sentences = fr.readlines()
    sentence_series = pd.Series(list_sentences)
    return sentence_series