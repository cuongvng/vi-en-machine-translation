from torch.utils.data import Dataset
import pandas as pd
from pyvi import ViTokenizer
from torchnlp.encoders.text import StaticTokenizerEncoder
from .embedder import PhoBERT, BERT
import sys
sys.path.append("../")

class IWSLT15EnViDataSet(Dataset):
    def __init__(self, en_path, vi_path):
        self.data_en = load_data(en_path).tolist()
        data_vi = load_data(vi_path)
        # Segment Vietnamese words
        self.data_vi = data_vi.apply(lambda s: ViTokenizer.tokenize(s)).tolist()
        assert len(self.data_en) == len(self.data_vi), "Numbers of vietnamese and english sentences do not match!"

    def __len__(self):
        return len(self.data_vi)

    def __getitem__(self, index):
        """
        Return tuple of English and segmented Vietnamese sentences, e.g.
        ('They are both two branches of the same field of atmospheric science .\n',
        'Cả hai đều là một nhánh của cùng một lĩnh_vực trong ngành khoa_học khí_quyển .')
        """
        return self.data_en[index], self.data_vi[index]

    def get_vocab_size(self, lang):
        assert lang == 'en' or lang == 'vi', "Only 'en' or 'vi' are valid!"
        if lang == 'vi':
            encoder = StaticTokenizerEncoder(sample=self.data_vi, min_occurrences=2,
                                             append_sos=True, append_eos=True)
        else:
            encoder = StaticTokenizerEncoder(sample=self.data_en, min_occurrences=2,
                                             append_sos=True, append_eos=True)
        return encoder.vocab_size

def get_labels(batch_of_sentences, lang, device):
    assert lang == 'en' or lang == 'vi', "Only 'en' or 'vi' are valid!"
    if lang == 'vi':
        embedder = PhoBERT().to(device)
    else:
        embedder = BERT().to(device)
    labels = embedder.get_token_indices(batch_of_sentences)
    return labels

def load_data(data_path):
    with open(data_path, 'r') as fr:
        list_sentences = fr.readlines()
    sentence_series = pd.Series(list_sentences)
    return sentence_series