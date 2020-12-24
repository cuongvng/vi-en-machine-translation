import unittest
import sys
sys.path.append("../")
from src.dataset import *
from torchnlp.encoders.text import DEFAULT_SOS_INDEX

class TestDataPreprocessor(unittest.TestCase):
    def test_pad(self):
        tokenized_sentence = torch.tensor([3, 1, 57, 328, 27, 481, 57, 2]) # 8
        padded_sen = pad_or_truncate(tokenized_sentence, target_len=10)
        self.assertListEqual(padded_sen.tolist(), [3, 1, 57, 328, 27, 481, 57, 0, 0, 2])

    def test_truncate(self):
        tokenized_sentence = torch.tensor([3, 1, 57, 328, 27, 481, 57, 345, 11, 6, 189, 76, 482, 465, 2])  # 15
        trucated_sen = pad_or_truncate(tokenized_sentence, target_len=10)
        self.assertListEqual(trucated_sen.tolist(), [3, 1, 57, 328, 27, 481, 57, 345, 11, 2])

    def test_data(self):
        training_set = EnViTrainingDataSet(en_path="../data/train-en-vi/train.en",
                                           vi_path="../data/train-en-vi/train.vi")

        for i in range(10):
            en_token = training_set[i][0].tolist()
            vi_token = training_set[i][1].tolist()

            self.assertEqual(len(en_token), TARGET_VALID_LEN+2)
            self.assertEqual(len(vi_token), TARGET_VALID_LEN + 2)

            self.assertEqual(en_token[0], DEFAULT_SOS_INDEX)
            self.assertEqual(en_token[-1], DEFAULT_EOS_INDEX)

            self.assertEqual(vi_token[0], DEFAULT_SOS_INDEX)
            self.assertEqual(vi_token[-1], DEFAULT_EOS_INDEX)

if __name__ == '__main__':
    unittest.main()


