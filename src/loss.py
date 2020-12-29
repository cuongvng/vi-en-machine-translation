import torch
import torch.nn as nn

class MaskedPaddingCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len, device):
        """
        :param pred: shape (batch_size, seq_len, vocab_size)
        :param label: shape (batch_size, seq_len)
        :param valid_len: shape (batch_size,)
        """
        self.reduction = 'none'
        masks = self._mask_padding_tokens(torch.ones_like(label), valid_len, device)
        unmasked_loss = super(MaskedPaddingCrossEntropyLoss, self).forward(input=pred.permute(0, 2, 1), target=label)
        masked_padding_loss = unmasked_loss * masks
        mean_loss_over_seq = masked_padding_loss.mean(dim=1)
        return mean_loss_over_seq

    def _mask_padding_tokens(self, X, valid_len, device, value=0):
        """
        Set values on positions of padding tokens to 0
        :param X: shape (batch_size, seq_len); e.g.
        ```
        tensor([[1, 2, 3],
                [4, 5, 6]])
        ```
        :param valid_len: shape (batch_size,), e.g.
        ```
        tensor([1, 2] # `valid_len = 1` for the first sequence, `valid_len = 2` for the second sequence
        ```
        :return: shape (batch_size, seq_len), e.g.
        ```
        tensor([[1, 0, 0],
                [4, 5, 0]])
        ```
        """
        X = X.to(device)
        valid_len = valid_len.to(device)

        masks = torch.arange(X.shape[1]).unsqueeze(dim=0).to(device) # (1, seq_len)
        valid_len = torch.unsqueeze(valid_len, dim=1) # (batch_size, 1)

        masks = masks < valid_len
        X[~masks] = value
        return X