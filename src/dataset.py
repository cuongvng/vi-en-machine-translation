import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from pyvi import ViTokenizer
import sys
sys.path.append("../")
from CONFIG import MAX_LENGTH, BERT_REPO, PhoBERT_REPO

class IWSLT15EnViDataSet(Dataset):
    def __init__(self, en_path, vi_path):
        data_en = load_data(en_path)
        data_vi = load_data(vi_path)
        # Segment Vietnamese words
        data_vi = data_vi.apply(lambda s: ViTokenizer.tokenize(s))

        # Tokenizing
        self.bert_tokenizer = BertTokenizer()
        self.phobert_tokenizer = PhoBertTokenizer()

        self.tokens_en, self.valid_len_en = self.bert_tokenizer(data_en.tolist())
        self.tokens_vi, self.valid_len_vi = self.phobert_tokenizer(data_vi.tolist())

        assert self.tokens_en.shape[0] == self.tokens_vi.shape[0], \
            "Numbers of vietnamese and english sentences do not match!"

        self.en_vocab = self.bert_tokenizer.get_vocab()
        self.vi_vocab = self.phobert_tokenizer.get_vocab()
        self.en_vocab_size = self.bert_tokenizer.get_vocab_size()
        self.vi_vocab_size = self.phobert_tokenizer.get_vocab_size()
        print("English vocab size:", self.en_vocab_size)
        print("Vietnamese vocab size:", self.vi_vocab_size)

    def __len__(self):
        return self.tokens_vi.shape[0]

    def __getitem__(self, index):
        return self.tokens_en[index], self.valid_len_en[index], self.tokens_vi[index], self.valid_len_vi[index]

class TokenizerBase(torch.nn.Module):
    tokenizer = None
    BOS_INDEX = None
    EOS_INDEX = None
    PADDING_INDEX = None
    BOS_TOKEN = None
    EOS_TOKEN = None
    PADDING_TOKEN = None

    def forward(self, list_of_sentences):
        tokens = self.get_token_indices(list_of_sentences)
        valid_len = torch.where(tokens != self.PADDING_INDEX, torch.tensor(1), torch.tensor(0)).sum(dim=1)
        return tokens, valid_len

    def get_token_indices(self, list_of_sentences):
        with torch.no_grad():
            tokens = self.tokenizer(list_of_sentences, truncation=True,
                                    padding="max_length", max_length=MAX_LENGTH)
        return torch.tensor(tokens['input_ids'])

    def get_vocab(self)->dict:
        return self.tokenizer.get_vocab()

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

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
    BOS_INDEX = 101
    EOS_INDEX = 102
    PADDING_INDEX = 0
    BOS_TOKEN = '[CLS]'
    EOS_TOKEN = '[SEP]'
    PADDING_TOKEN = '[PAD]'

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
    BOS_INDEX = 0
    EOS_INDEX = 2
    PADDING_INDEX = 1
    BOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'
    PADDING_TOKEN = '<pad>'

def load_data(data_path):
    with open(data_path, 'r') as fr:
        list_sentences = fr.readlines()
    sentence_series = pd.Series(list_sentences)
    return sentence_series