import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchsummary import summary
import os
from pathlib import Path
from dataset import IWSLT15EnViDataSet
from model import NMT
from loss import MaskedPaddingCrossEntropyLoss
import sys
sys.path.append("../")
from CONFIG import N_EPOCHS, BATCH_SIZE, EN2VI, MODE, MAX_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

CHECKPOINT_DIR = "../model/"
if not Path(CHECKPOINT_DIR).exists():
    os.mkdir(CHECKPOINT_DIR)
CHECKPOINT_PATH = Path(CHECKPOINT_DIR + "model.pt")

def train():
    # Data
    data_train = IWSLT15EnViDataSet(en_path="../data/train-en-vi/train.en",
                                    vi_path="../data/train-en-vi/train.vi")
    data_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    tgt_vocab_size = data_train.vi_vocab_size if MODE == EN2VI else data_train.en_vocab_size

    # Model & Optimizer
    model = NMT(mode=MODE, tgt_vocab_size=tgt_vocab_size)
    model.to(device)

    criterion = MaskedPaddingCrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters())

    prev_epochs = 0
    if CHECKPOINT_PATH.exists():  # Resume training
        model, optimizer, prev_epoch = load_checkpoint(model, optimizer)
    else:
        model.apply(xavier_init_weights)

    summary(model, input_size=MAX_LENGTH, batch_size=BATCH_SIZE, device=str(device))

    model.train()
    for epoch in range(N_EPOCHS-prev_epochs):
        print(f"\nEpoch: {epoch+prev_epochs+1}")

        for b, (en_tokens, en_valid_len, vi_tokens, vi_valid_len) in enumerate(data_loader):
            if MODE == EN2VI:
                src, tgt = en_tokens.to(device), vi_tokens.to(device)
                valid_lengths = vi_valid_len.to(device)
            else:
                src, tgt = vi_tokens.to(device), en_tokens.to(device)
                valid_lengths = en_valid_len.to(device)

            optimizer.zero_grad()
            logit_outputs = model(src, tgt)
            loss = criterion(pred=logit_outputs, label=tgt, valid_len=valid_lengths, device=device).sum()
            loss.backward()
            optimizer.step()

            if b%300 == 0:
                print(f"\tBatch {b}; Loss: {loss:.2f}")

            ## Free up GPU memory
            del src, tgt, logit_outputs, loss
            torch.cuda.empty_cache()

        save_checkpoint(model, optimizer, prev_epoch+epoch+1)

def save_checkpoint(model, optimizer, epoch):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

def load_checkpoint(model, optimizer):
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    prev_epoch = checkpoint["epoch"]
    print("Loaded checkpoints successfully!")
    return model, optimizer, prev_epoch

def xavier_init_weights(model):
    if isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)
    return

if __name__ == "__main__":
    train()