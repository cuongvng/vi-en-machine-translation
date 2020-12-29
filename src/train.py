import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import IWSLT15EnViDataSet
from model import NMT
from loss import MaskedPaddingCrossEntropyLoss
import sys
sys.path.append("../")
from CONFIG import N_EPOCHS, BATCH_SIZE, EN2VI, MODE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def train():
    data_train = IWSLT15EnViDataSet(en_path="../data/train-en-vi/train.en",
                                    vi_path="../data/train-en-vi/train.vi")
    data_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    tgt_vocab_size = data_train.vi_vocab_size if MODE == EN2VI else data_train.en_vocab_size

    model = NMT(tgt_vocab_size=tgt_vocab_size)
    model.apply(xavier_init_weights)
    model.to(device)

    criterion = MaskedPaddingCrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters())

    for epoch in range(N_EPOCHS):
        print(f"\nEpoch: {epoch+1}")

        for b, (en_tokens, en_valid_len, vi_tokens, vi_valid_len) in enumerate(data_loader):
            print(f"\tBatch: {b+1}")

            if MODE == EN2VI:
                src, tgt = en_tokens.to(device), vi_tokens.to(device)
                valid_lengths = vi_valid_len.to(device)
            else:
                src, tgt = vi_tokens.to(device), en_tokens.to(device)
                valid_lengths = en_valid_len.to(device)

            optimizer.zero_grad()
            logit_outputs = model(src, tgt)
            loss = criterion(pred=logit_outputs, label=tgt, valid_len=valid_lengths).sum()
            loss.backward()
            optimizer.step()

            if b%100 == 0:
                print(f"\tLoss: {loss:.4f}")
            ## Free up GPU memory
            del src, tgt, logit_outputs, loss
            torch.cuda.empty_cache()

def xavier_init_weights(model):
    if isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)
    return

if __name__ == "__main__":
    train()