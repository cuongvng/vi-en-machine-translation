import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from pyvi import ViTokenizer
from torchnlp.encoders.text import StaticTokenizerEncoder
import sys
sys.path.append("../")
from CONFIG import MAX_LENGTH, PhoBERT_REPO, BERT_REPO, BERT_PADDING_INDEX, PhoBERT_PADDING_INDEX

class IWSLT15EnViDataSet(Dataset):
    def __init__(self, en_path, vi_path):
        data_en = load_data(en_path)
        data_vi = load_data(vi_path)
        # Segment Vietnamese words
        data_vi = data_vi.apply(lambda s: ViTokenizer.tokenize(s))

        self.en_vocab_size = _get_vocab_size(data_en)
        self.vi_vocab_size = _get_vocab_size(data_vi)

        # Tokenizing
        self.bert_tokenizer = BertTokenizer()
        self.phobert_tokenizer = PhoBertTokenizer()

        self.tokens_en, self.valid_len_en = self.bert_tokenizer(data_en.tolist())
        self.tokens_vi, self.valid_len_vi = self.phobert_tokenizer(data_vi.tolist())

        assert self.tokens_en.shape[0] == self.tokens_vi.shape[0], "Numbers of vietnamese and english sentences do not match!"

    def __len__(self):
        return self.tokens_vi.shape[0]

    def __getitem__(self, index):
        return self.tokens_en[index], self.valid_len_en[index], self.tokens_vi[index], self.valid_len_vi[index]

class TokenizerBase(torch.nn.Module):
    tokenizer = None
    padding_index = None

    def forward(self, list_of_sentences):
        tokens = self.get_token_indices(list_of_sentences)
        valid_len = torch.where(tokens != self.padding_index, torch.tensor(1), torch.tensor(0)).sum(dim=1)
        return tokens, valid_len

    def get_token_indices(self, list_of_sentences):
        with torch.no_grad():
            tokens = self.tokenizer(list_of_sentences, truncation=True,
                                    padding="max_length", max_length=MAX_LENGTH)
        return torch.tensor(tokens['input_ids'])

class BertTokenizer(TokenizerBase):
    """
    def forward(self, list_of_sentences):
        :param list_of_sentences: e.g. [
            "Hello world \n",
            "This is something he has to do with air quality or smog .\n'"
        ]
        :return: torch Tensor of shape (batch_size, max_length)
    """
    tokenizer = AutoTokenizer.from_pretrained(BERT_REPO)
    padding_index = BERT_PADDING_INDEX

class PhoBertTokenizer(TokenizerBase):
    """
    def forward(self, list_of_segmented_sentences):
        :param list_of_segmented_sentences: e.g. [
            "Tôi là sinh_viên trường đại_học Công_nghệ , còn đây là em_trai tôi , đang học cấp 3 .",
            "Bạn là ai vậy ?"
        ]
        :return: torch Tensor of shape (batch_size, max_length)
    """
    tokenizer = AutoTokenizer.from_pretrained(PhoBERT_REPO)
    padding_index = PhoBERT_PADDING_INDEX

def load_data(data_path):
    with open(data_path, 'r') as fr:
        list_sentences = fr.readlines()
    sentence_series = pd.Series(list_sentences)
    return sentence_series

def _get_vocab_size(data):
    encoder = StaticTokenizerEncoder(sample=data, min_occurrences=2,
                                         append_sos=True, append_eos=True)
    return encoder.vocab_size
