from torchnlp.encoders.text import DEFAULT_PADDING_INDEX
import torch.nn as nn
import torch

class MaskedPaddingCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, device):
        """
        :param pred: shape (batch_size, vocab_size)
        :param label: shape (batch_size,)
        """
        self.reduction = 'none'
        masks = (label != DEFAULT_PADDING_INDEX).type(torch.int8)
        unmasked_loss = super(MaskedPaddingCrossEntropyLoss, self).forward(input=pred, target=label)
        masked_padding_loss = unmasked_loss * masks
        return masked_padding_loss