import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torchnlp.encoders.text import DEFAULT_SOS_INDEX
import os
import argparse
from pathlib import Path
from dataset import IWSLT15EnViDataSet
from model import NMT
from loss import MaskedPaddingCrossEntropyLoss
from eval import translate_en2vi
import sys
sys.path.append("../")
from CONFIG import N_EPOCHS, BATCH_SIZE, EN2VI, VI2EN, MAX_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ens = [
    "I go.",
    "My grandmother never let me forget his life . ",
    "I was so shocked .",
    "As you can see , the river can be very narrow at certain points , allowing North Koreans to secretly cross .",
    "I could have never imagined that it would take 14 years to live together .",
    "But many die ."
]


def train(mode, checkpoint_path):
    # Data
    data_train = IWSLT15EnViDataSet(en_path="../data/train-en-vi/train.en",
                                    vi_path="../data/train-en-vi/train.vi")
    data_loader = DataLoader(data_train, batch_size=BATCH_SIZE,
                             shuffle=False, drop_last=False)
    if mode == EN2VI:
        src_vocab_size, tgt_vocab_size = data_train.en_vocab_size, data_train.vi_vocab_size
    else:
        src_vocab_size, tgt_vocab_size = data_train.vi_vocab_size, data_train.en_vocab_size
    print("Loading data done!")

    # Model & Optimizer
    model = NMT(mode=mode, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
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
            en_tokens, vi_tokens = en_tokens.to(device), vi_tokens.to(device)
            en_valid_len, vi_valid_len = en_valid_len.to(device), vi_valid_len.to(device)

            en_padding_masks = mask_padding(en_tokens, en_valid_len, device)
            vi_padding_masks = mask_padding(vi_tokens, vi_valid_len, device)

            if mode == EN2VI:
                src, tgt = en_tokens, vi_tokens
                tgt_valid_len = vi_valid_len
                src_masks, tgt_masks = en_padding_masks, vi_padding_masks
            else:
                src, tgt = vi_tokens, en_tokens
                tgt_valid_len = en_valid_len
                src_masks, tgt_masks = vi_padding_masks, en_padding_masks

            optimizer.zero_grad()

            # Encoder's forward pass:
            encoder_state = model.encoder(src, src_masks)
            # Decoder's forward pass
            decoder_X = torch.tensor([[DEFAULT_SOS_INDEX]*tgt.shape[0]], device=device).reshape(-1, 1)
            decoder_state = encoder_state

            loss = torch.tensor(0, device=device, dtype=torch.float)
            for i in range(1, tgt.shape[1]):
                decoder_state, logit_pred = model.decoder(decoder_X, decoder_state)
                loss += criterion(pred=logit_pred[:, 0, :], label=tgt[:, i], device=device).sum()
                # Teacher forcing
                decoder_X = tgt[:, i].reshape(-1, 1)

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if b%50 == 0:
                seq_loss = loss/(MAX_LENGTH-1)
                print(f"\tBatch {b}; Loss: {seq_loss:.2f}; "
                      f"Mean Token Loss: {seq_loss/tgt_valid_len.sum():.4f}")

            ## Free up GPU memory
            del src, tgt, en_valid_len, vi_valid_len, decoder_state, logit_pred, loss
            torch.cuda.empty_cache()

        save_checkpoint(mode, src_vocab_size, tgt_vocab_size, model, optimizer, data_train.tokenizer_en,
                        data_train.tokenizer_vi, prev_epoch+epoch+1, checkpoint_path)

        for en in ens:
            vi = translate_en2vi(en_sentence=en, length=MAX_LENGTH, model=model,
                                 tokenizer_en=data_train.tokenizer_en,
                                 tokenizer_vi=data_train.tokenizer_vi, device=device)
            print("en:", en, "=> vi:", vi)


def mask_padding(X, valid_len, device):
    positions = torch.arange(X.shape[1]).unsqueeze(dim=0).to(device)  # (1, seq_len)
    valid_len = torch.unsqueeze(valid_len, dim=1)  # (batch_size, 1)

    masks = positions >= valid_len
    return masks

def save_checkpoint(mode, src_vocab_size, tgt_vocab_size, model,
                    optimizer, tokenizer_en, tokenizer_vi, epoch, checkpoint_path):
    checkpoint = {
        "mode": mode,
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "tokenizer_en": tokenizer_en,
        "tokenizer_vi":tokenizer_vi
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