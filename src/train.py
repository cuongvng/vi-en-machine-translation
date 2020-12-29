import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import IWSLT15EnViDataSet
from embedder import PhoBERT, BERT
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

    tgt_lang = "vi" if MODE == EN2VI else "en"
    tgt_vocab_size = data_train.get_vocab_size(lang=tgt_lang)

    model = NMT(mode=MODE, tgt_vocab_size=tgt_vocab_size)
    model.apply(xavier_init_weights)
    model.to(device)

    criterion = MaskedPaddingCrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters())

    for epoch in range(N_EPOCHS):
        print(f"\nEpoch: {epoch+1}")

        for b, (en_batch, vi_batch) in enumerate(data_loader):
            print(f"\tBatch: {b+1}")

            if tgt_lang == "vi":
                src, tgt = en_batch.to(device), vi_batch.to(device)
            else:
                src, tgt = vi_batch.to(device), en_batch.to(device)
            labels = get_labels(tgt, lang=tgt_lang, device=device)

            optimizer.zero_grad()
            logit_outputs, valid_lengths = model(src, tgt)
            loss = criterion(pred=logit_outputs, label=labels, valid_len=valid_lengths).sum()
            loss.backward()
            optimizer.step()

            if b%100 == 0:
                print(f"\tLoss: {loss:.4f}")
            ## Free up GPU memory
            del src, tgt, logit_outputs, labels, loss
            torch.cuda.empty_cache()

def get_labels(batch_of_sentences, lang, device):
    assert lang == 'en' or lang == 'vi', "Only 'en' or 'vi' are valid!"
    if lang == 'vi':
        embedder = PhoBERT().to(device)
    else:
        embedder = BERT().to(device)
    labels = embedder.get_token_indices(batch_of_sentences)
    return labels

def xavier_init_weights(model):
    if isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)
    return

if __name__ == "__main__":
    train()