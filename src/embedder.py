import torch
from transformers import AutoTokenizer, AutoModel
import sys
sys.path.append("../")
from CONFIG import MAX_LENGTH, PhoBERT_REPO, BERT_REPO, BERT_PADDING_INDEX, PhoBERT_PADDING_INDEX

class PhoBERT(object):
    phobert_tokenizer = AutoTokenizer.from_pretrained(PhoBERT_REPO)
    phobert_embedder = AutoModel.from_pretrained(PhoBERT_REPO)

    def __call__(self, list_of_segmented_sentences):
        """
        :param list_of_segmented_sentences: e.g. [
            "Tôi là sinh_viên trường đại_học Công_nghệ .",
            "Bạn là ai ?"
        ]
        :return: torch Tensor of shape (batch_size, max_length, embedding_size)
        """
        tokens = self.phobert_tokenizer(list_of_segmented_sentences, truncation=True,
                                        padding="max_length", max_length=MAX_LENGTH)
        tokens = torch.tensor(tokens['input_ids'])
        valid_len = torch.where(tokens != PhoBERT_PADDING_INDEX, torch.tensor(1), torch.tensor(0)).sum(dim=1)

        with torch.no_grad():
            features = self.phobert_embedder(tokens)
        return features['last_hidden_state'], valid_len

class BERT(object):
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_REPO)
    bert_embedder = AutoModel.from_pretrained(BERT_REPO)

    def __call__(self, list_of_sentences):
        """
        :param list_of_sentences: e.g. [
            "Hello world \n",
            "This is something he has to do with air quality or smog .\n'"
        ]
        :return: torch Tensor of shape (batch_size, max_length, embedding_size)
        """
        tokens = self.bert_tokenizer(list_of_sentences, truncation=True,
                                        padding="max_length", max_length=MAX_LENGTH)
        tokens = torch.tensor(tokens['input_ids'])
        valid_len = torch.where(tokens!=BERT_PADDING_INDEX, torch.tensor(1), torch.tensor(0)).sum(dim=1)

        with torch.no_grad():
            features = self.bert_embedder(tokens)
        return features['last_hidden_state'], valid_len