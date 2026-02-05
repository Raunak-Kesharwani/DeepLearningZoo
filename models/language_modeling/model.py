import torch
from torch import nn 


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, rnn_type="lstm", embed_dim=256, hidden_size=128, num_layers=2 ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                embed_dim, hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.3
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                embed_dim, hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.3
            )
        else:
            raise ValueError("Invalid rnn_type")

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.rnn_type = rnn_type

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.fc(out)
        return logits, hidden
