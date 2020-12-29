import torch
from transformers import AutoTokenizer, AutoModel
import sys
sys.path.append("../")
from CONFIG import MAX_LENGTH, PhoBERT_REPO, BERT_REPO, BERT_PADDING_INDEX, PhoBERT_PADDING_INDEX

class EmbedderBase(object):
    tokenizer = None
    embedder = None
    padding_index = None

    def __call__(self, list_of_sentences):
        tokens = self.get_token_indices(list_of_sentences)
        valid_len = torch.where(tokens != self.padding_index, torch.tensor(1), torch.tensor(0)).sum(dim=1)
        embedding = self.embed(tokens)
        return embedding, valid_len

    def get_token_indices(self, list_of_sentences):
        with torch.no_grad():
            tokens = self.tokenizer(list_of_sentences, truncation=True,
                                    padding="max_length", max_length=MAX_LENGTH)
        return torch.tensor(tokens['input_ids'])

    def embed(self, tokens):
        with torch.no_grad():
            features = self.embedder(tokens)
        return features['last_hidden_state']

class PhoBERT(EmbedderBase):
    """
    def __call__(self, list_of_segmented_sentences):
        :param list_of_segmented_sentences: e.g. [
            "Tôi là sinh_viên trường đại_học Công_nghệ , còn đây là em_trai tôi , đang học cấp 3 .",
            "Bạn là ai vậy ?"
        ]
        :return: torch Tensor of shape (batch_size, max_length, embedding_size)
    """
    tokenizer = AutoTokenizer.from_pretrained(PhoBERT_REPO)
    embedder = AutoModel.from_pretrained(PhoBERT_REPO)
    padding_index = PhoBERT_PADDING_INDEX

class BERT(EmbedderBase):
    """
    def __call__(self, list_of_sentences):
        :param list_of_sentences: e.g. [
            "Hello world \n",
            "This is something he has to do with air quality or smog .\n'"
        ]
        :return: torch Tensor of shape (batch_size, max_length, embedding_size)
    """
    tokenizer = AutoTokenizer.from_pretrained(BERT_REPO)
    embedder = AutoModel.from_pretrained(BERT_REPO)
    padding_index = BERT_PADDING_INDEX