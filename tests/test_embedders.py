import unittest
import torch
from CONFIG import MAX_LENGTH, EMBEDDING_SIZE
import sys
sys.path.append("../")
from src.embedder import BERT, PhoBERT

class TestEmbedders(unittest.TestCase):
    def test_BERT(self):
        bert = BERT()
        inputs = {
            "Hello world \n": 4,
            "This is something the government have to do with air quality or smog .\n": 12
        }
        sentences, valid_lengths = list(inputs.keys()), list(inputs.values())

        embedding, valid_len = bert(inputs)
        self.assertEqual(embedding.shape, (len(sentences), MAX_LENGTH, EMBEDDING_SIZE))
        self.assertListEqual(valid_len.tolist(), valid_lengths)

    def test_PhoBERT(self):
        phobert = PhoBERT()
        inputs = {
            "Tôi là sinh_viên trường đại_học Công_nghệ , còn đây là em_trai tôi , đang học cấp 3 .": 12,
            "Bạn là ai vậy ?": 7
        }
        sentences, valid_lengths = list(inputs.keys()), list(inputs.values())

        embedding, valid_len = phobert(inputs)
        self.assertEqual(embedding.shape, (len(sentences), MAX_LENGTH, EMBEDDING_SIZE))
        self.assertListEqual(valid_len.tolist(), valid_lengths)


if __name__ == '__main__':
    unittest.main()