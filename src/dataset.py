from torch.utils.data import Dataset
import pandas as pd
from pyvi import ViTokenizer
import sys
sys.path.append("../")

class IWSLT15EnViDataSet(Dataset):
    def __init__(self, en_path, vi_path):
        self.data_en = load_data(en_path)
        data_vi = load_data(vi_path)
        # Segment Vietnamese words
        self.data_vi = data_vi.apply(lambda s: ViTokenizer.tokenize(s))
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


def load_data(data_path):
    with open(data_path, 'r') as fr:
        list_sentences = fr.readlines()
    sentence_series = pd.Series(list_sentences)
    return sentence_series

