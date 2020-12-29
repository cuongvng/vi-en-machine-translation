import torch
from transformers import AutoModel
import sys
sys.path.append("../")
from CONFIG import PhoBERT_REPO, BERT_REPO

class EmbedderBase(torch.nn.Module):
    def __init__(self):
        super(EmbedderBase, self).__init__()
        self.embedder = None

    def forward(self, tokens):
        with torch.no_grad():
            features = self.embedder(tokens)
        return features['last_hidden_state']

class PhoBERT(EmbedderBase):
    """
    def forward(self, tokens):
        :input shape: (batch_size, max_length)
        :return: torch Tensor of shape (batch_size, max_length, embedding_size)
    """
    def __init__(self):
        super(PhoBERT, self).__init__()
        self.embedder = AutoModel.from_pretrained(PhoBERT_REPO)

class BERT(EmbedderBase):
    """
    def forward(self, tokens):
        :input shape: (batch_size, max_length)
        :return: torch Tensor of shape (batch_size, max_length, embedding_size)
    """
    def __init__(self):
        super(BERT, self).__init__()
        self.embedder = AutoModel.from_pretrained(BERT_REPO)