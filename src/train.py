import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
import os
import argparse
from pathlib import Path
from dataset import IWSLT15EnViDataSet
from model import NMT
from loss import MaskedPaddingCrossEntropyLoss
import sys
sys.path.append("../")
from CONFIG import N_EPOCHS, BATCH_SIZE, EN2VI, VI2EN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def train(mode, checkpoint_path):
    # Data
    data_train = IWSLT15EnViDataSet(en_path="../data/dev-2012-en-vi/tst2012.en",
                                    vi_path="../data/dev-2012-en-vi/tst2012.vi")
    data_loader = DataLoader(data_train, batch_size=BATCH_SIZE,
                             shuffle=False, drop_last=False)
    tgt_vocab_size = data_train.vi_vocab_size if mode == EN2VI else data_train.en_vocab_size
    print("Loading data done!")

    # Model & Optimizer
    model = NMT(mode=mode, tgt_vocab_size=tgt_vocab_size)
    model.to(device)

    criterion = MaskedPaddingCrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters())

    prev_epoch = 0
    if checkpoint_path.exists():  # Resume training
        model, optimizer, prev_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resume training from {prev_epoch} epochs!")
    else:
        model.apply(xavier_init_weights)
        print("Training from start!")

    model.train()
    for epoch in range(N_EPOCHS-prev_epoch):
        print(f"\nEpoch: {epoch+prev_epoch+1}")

        for b, (en_tokens, en_valid_len, vi_tokens, vi_valid_len) in enumerate(data_loader):
            if mode == EN2VI:
                src, tgt = en_tokens.to(device), vi_tokens.to(device)
                valid_lengths = vi_valid_len.to(device)
            else:
                src, tgt = vi_tokens.to(device), en_tokens.to(device)
                valid_lengths = en_valid_len.to(device)

            optimizer.zero_grad()
            decoder_state, logit_outputs = model(src, tgt)
            loss = criterion(pred=logit_outputs, label=tgt, valid_len=valid_lengths, device=device).sum()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if b%1000 == 0:
                print(f"\tBatch {b}; Loss: {loss:.2f}")

            ## Free up GPU memory
            del src, tgt, decoder_state, logit_outputs, loss
            torch.cuda.empty_cache()

        save_checkpoint(model, optimizer, prev_epoch+epoch+1, checkpoint_path)

def grad_clipping(model, theta):
    assert isinstance(model, torch.nn.Module)
    params = [p for p in model.parameters() if p.requires_grad]

    norms = torch.sqrt(sum(torch.sum(p**2) for p in params))
    if norms > theta:
        for p in params:
            p.grad.data.clamp_()

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, checkpoint_path)
    print("Checkpoint saved!")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[EN2VI, VI2EN], type=str, default=EN2VI,
                        help="Choose source and target language")
    parser.add_argument("--checkpoint_dir", type=str, default="../model/",
                        help="Directory to save checkpoints")
    args = parser.parse_args()
    mode = args.mode
    checkpoint_dir = args.checkpoint_dir
    if not Path(checkpoint_dir).exists():
        os.mkdir(checkpoint_dir)
    CHECKPOINT_PATH = Path(os.path.join(checkpoint_dir, f"model_{mode}.pt"))

    print("MODE:", mode)
    train(mode=mode, checkpoint_path=CHECKPOINT_PATH)