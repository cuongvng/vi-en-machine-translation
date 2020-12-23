import unittest
import sys
sys.path.append("../")
from src.dataset import *

en = [
    'If you &apos;re a doctor you can do some good things, but if you &apos;re a caring doctor you can do some other things.',
    'sHe &apos;d squEEze me',
    'If I said , &quot; No , &quot; she &apos;d assault me again',
    'I said , &quot; Okay , Mama . &quot;'
]
expected_en = [
]

vi = [
    'Mẹ tôi hỏi nhà sư , &quot; Tại sao tay mẹ tôi ấm thế mà cả người lại lạnh ngắt ? &quot;',
    '&quot; Các chị kHông hề buông tay. &quot;',
    'Giờ đây tôi cũng muốn có con , và tôi băn khoăn về con thuyền .'
]
expected_vi = [

]
TARGET_LEN = 10

class TestDataPreprocessor(unittest.TestCase):
    def test_pad(self):
        tokenized_sentence = torch.tensor([3, 1, 57, 328, 27, 481, 57, 2]) # 8
        padded_sen = pad_or_truncate(tokenized_sentence, target_len=TARGET_LEN)
        self.assertListEqual(padded_sen.tolist(), [3, 1, 57, 328, 27, 481, 57, 0, 0, 2])

    def test_truncate(self):
        tokenized_sentence = torch.tensor([3, 1, 57, 328, 27, 481, 57, 345, 11, 6, 189, 76, 482, 465, 2])  # 15
        trucated_sen = pad_or_truncate(tokenized_sentence, target_len=TARGET_LEN)
        self.assertListEqual(trucated_sen.tolist(), [3, 1, 57, 328, 27, 481, 57, 345, 11, 2])

    def test_apos_quot(self):
        pass
    def test_lower(self):
        pass
    def test_tokenize_en(self):
        pass
    def test_tokenize_vi(self):
        pass

if __name__ == '__main__':
    unittest.main()


