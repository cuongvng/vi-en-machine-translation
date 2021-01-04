import torch
from model import NMT, PositionalEncoder
from dataset import PhoBertTokenizer, BertTokenizer
from embedder import BERT, PhoBERT
from train import mask_padding
import sys
sys.path.append("../")
from CONFIG import EMBEDDING_SIZE, EN2VI, VI2EN, MAX_LENGTH

def translate_en2vi(model, en_sentence, length, device):
    assert isinstance(model, NMT), "Incompatible model!"

    en_tokenizer = BertTokenizer()
    vi_tokenizer = PhoBertTokenizer()
    en_embedder = BERT()
    vi_embedder = PhoBERT()
    vi_embedder.to(device)

    en_tokens, valid_len = en_tokenizer([en_sentence])
    en_padding_mask = mask_padding(en_tokens, valid_len, device)
    en_embedding = en_embedder(en_tokens)
    en_embedding = en_embedding.to(device)

    ps = PositionalEncoder(d_model=EMBEDDING_SIZE)
    ps.to(device)

    pred_indices = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        en_embedding = en_embedding + ps(en_embedding)
        encoder_state = model.encoder(en_embedding, en_padding_mask)
        # Initialize the input and memory of the decoder by the BOS token and `encoder_state`
        decoder_X = torch.zeros((1, length), device=device)
        cur_token_idx = vi_tokenizer.BOS_INDEX
        for i in range(length):
            decoder_X[0][i] = cur_token_idx

            # Embedding + Positional Encoding
            decoder_X = vi_embedder(decoder_X)
            decoder_X = decoder_X + ps(decoder_X)

            # Decoder forward pass
            decoder_memory, logit_outputs = model.decoder(decoder_X, encoder_state)
            # Use the token with highest probability as the input of the next time step
            cur_token_idx = logit_outputs[:, i, :].argmax(dim=1)
            pred_idx = decoder_X.squeeze(dim=0).to(torch.int32).item()

            if pred_idx == vi_tokenizer.EOS_INDEX:
                break
            pred_indices.append(pred_idx)

    translated_sentence = " ".join(vi_tokenizer.convert_ids_to_meaningful_tokens(pred_indices))
    translated_sentence.replace('_', ' ') # Remove '_' of segmented words
    return translated_sentence

def translate_vi2en(vi_sentences, model):
    model.eval()


def calculate_bleu(pred, label, k):
    """
    :param pred: tokens
    :param label: tokens
    :param k: maximum n-grams to match
    :return:
    """
    pass

def _get_n_grams_precision(pred, label, n):
    """
    Ratio of the number of matched n_grams
    to the number of n_grams (not necessarily unique) in the predicted sequence.
    """

    pass

def main():
    model = NMT(mode=EN2VI, tgt_vocab_size=64000)
    checkpoint = torch.load("../model/model_en2vi.pt")
    model.load_state_dict(checkpoint["model"])

    en = "I go."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vi = translate_en2vi(model, en_sentence=en, length=MAX_LENGTH, device=device)
    print("vi:", vi)

if __name__ == '__main__':
    main()