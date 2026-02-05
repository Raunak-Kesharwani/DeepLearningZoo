import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer

from models.question_answering.seq2seq_attention.model import Seq2Seq


def export_seq2seq_attention():
    # ---- Load tokenizer ----
    tokenizer_path = (
        "models/question_answering/seq2seq_attention/tokenizer.json"
    )
    tokenizer = Tokenizer.from_file(tokenizer_path)

    vocab_size = tokenizer.get_vocab_size()
    pad_idx = tokenizer.token_to_id("<pad>")

    # ---- Model hyperparams (MUST match training) ----
    embed_dim = 256
    hidden_size = 128
    num_layers = 2

    # ---- Build model ----
    model = Seq2Seq(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        pad_idx=pad_idx,
    ).cpu()

    # ---- Load checkpoint ----
    model.load_state_dict(
        torch.load(
            "models/question_answering/seq2seq_attention/seq2seq_attention_best.pth",
            map_location="cpu",
        )
    )
    model.eval()

    # ---- Dummy inputs (batch=1) ----
    enc_x = torch.zeros((1, 128), dtype=torch.long)
    dec_x = torch.zeros((1, 32), dtype=torch.long)

    # ---- TorchScript ----
    scripted = torch.jit.trace(model, (enc_x, dec_x))
    scripted.save(
        "models/question_answering/seq2seq_attention/model.pt"
    )

    print("✅ Seq2Seq QA (with attention) TorchScript exported")


if __name__ == "__main__":
    export_seq2seq_attention()
