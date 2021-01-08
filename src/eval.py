import torch
from torchnlp.encoders.text import DEFAULT_SOS_INDEX, DEFAULT_EOS_INDEX
from model import NMT
from dataset import IWSLT15EnViDataSet, convert_indices_to_tokens, convert_tokens_to_indices
import argparse
import sys
sys.path.append("../")
from CONFIG import MAX_LENGTH

def translate_en2vi(en_sentence, length, model, tokenizer_en, tokenizer_vi, device):
    assert isinstance(model, NMT), "Incompatible model!"

    en_tokens, valid_len = convert_tokens_to_indices(en_sentence, tokenizer_en)
    en_tokens = en_tokens.reshape(-1, 1).to(device)
    # en_padding_mask = mask_padding(en_tokens, valid_len, device)

    pred_indices = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        encoder_state = model.encoder(en_tokens)

        # Initialize the input and memory of the decoder by the BOS token and `encoder_state`
        dec_X = torch.unsqueeze(torch.tensor(
            [DEFAULT_SOS_INDEX], dtype=torch.long, device=device), dim=0)
        decoder_memory = encoder_state

        for i in range(length):
            # Decoder forward pass
            decoder_memory, logit_outputs = model.decoder(dec_X, decoder_memory)
            # Use the token with highest probability as the input of the next time step
            dec_X = logit_outputs[:, i, :].argmax(dim=1)
            pred_idx = dec_X.squeeze(dim=0).to(torch.int32).item()

            if pred_idx == DEFAULT_EOS_INDEX:
                break
            pred_indices.append(pred_idx)

    translated_sentence = "".join(convert_indices_to_tokens(pred_indices, tokenizer_vi))
    translated_sentence = translated_sentence.replace('_', ' ') # Remove '_' of segmented words
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

def main(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = NMT(mode=checkpoint["mode"],
                src_vocab_size=checkpoint["src_vocab_size"],
                tgt_vocab_size=checkpoint["tgt_vocab_size"])
    model.load_state_dict(checkpoint["model"])
    tokenizer_en = checkpoint["tokenizer_en"]
    tokenizer_vi = checkpoint["tokenizer_vi"]

    ens = [
      "I go.",
      "My grandmother never let me forget his life . ",
      "I was so shocked .",
      "As you can see , the river can be very narrow at certain points , allowing North Koreans to secretly cross .",
      "I could have never imagined that it would take 14 years to live together .",
      "But many die ."
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for en in ens:
      vi = translate_en2vi(en_sentence=en, length=MAX_LENGTH, model=model,
                          tokenizer_en=tokenizer_en, tokenizer_vi=tokenizer_vi, device=device)
      print("vi:", vi)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    args = parser.parse_args()
    main(args.checkpoint_path)