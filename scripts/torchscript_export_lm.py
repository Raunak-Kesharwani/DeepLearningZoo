import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer
from models.language_modeling.model import RNNLanguageModel


def export_language_model(
    checkpoint_path,
    save_path,
    tokenizer_path,
    rnn_type
):
    # Load tokenizer correctly
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    model = RNNLanguageModel(
        vocab_size=vocab_size,
        rnn_type=rnn_type
    ).cpu()

    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")
    )
    model.eval()

    # Dummy input must match training: (batch=1, seq_len=60)
    dummy_input = torch.zeros((1, 60), dtype=torch.long)

    # IMPORTANT: do not pass hidden
    scripted = torch.jit.trace(model, dummy_input)

    scripted.save(save_path)
    print(f"✅ {rnn_type.upper()} TorchScript saved")


if __name__ == "__main__":

    TOKENIZER_PATH = "models/language_modeling/tokenizer.json"

    export_language_model(
        "models/language_modeling/gru/GRU_WikiText2.pth",
        "models/language_modeling/gru/model.pt",
        TOKENIZER_PATH,
        rnn_type="gru"
    )

    export_language_model(
        "models/language_modeling/lstm/LSTM_WikiText2.pth",
        "models/language_modeling/lstm/model.pt",
        TOKENIZER_PATH,
        rnn_type="lstm"
    )
