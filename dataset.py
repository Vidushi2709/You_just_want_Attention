import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len=128) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Extract raw data
        src_target_pair = self.ds[idx]
        src_txt = src_target_pair['translation'][self.src_lang]
        tgt_txt = src_target_pair['translation'][self.tgt_lang]

        # Text -> Token IDs
        enc_ip_tokens = self.tokenizer_src.encode(src_txt).ids
        dec_ip_tokens = self.tokenizer_tgt.encode(tgt_txt).ids

        # Padding length
        enc_num_pads_tokens = self.seq_len - len(enc_ip_tokens) - 2
        dec_nums_pads_tokens = self.seq_len - len(dec_ip_tokens) - 1

        # Check for overly long sentences
        if enc_num_pads_tokens < 0 or dec_nums_pads_tokens < 0:
            raise ValueError("Sentence too long")

        # Encoder input
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_ip_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token.item()] * enc_num_pads_tokens, dtype=torch.int64),
        ], 
        dim=0,
        )

        # Decoder input
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_ip_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token.item()] * dec_nums_pads_tokens, dtype=torch.int64),
        ],
        dim=0,
        )

        # Decoder label
        label = torch.cat([
            torch.tensor(dec_ip_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token.item()] * dec_nums_pads_tokens, dtype=torch.int64),
        ],
        dim=0,
        )

        # Sanity check
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_txt,
            "tgt_text": tgt_txt,
        }

def causal_mask(size):
    # mask with False above the diagonal
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0  # True where we allow attending


        
