import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 256, padding_idx=pad_idx)
        self.rnn = nn.LSTM(256, 128, num_layers=2, batch_first=True)

    def forward(self, x):
        _, h = self.rnn(self.emb(x))
        return h


class Decoder(nn.Module):
    def __init__(self, vocab_size, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 256, padding_idx=pad_idx)
        self.rnn = nn.LSTM(256, 128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, vocab_size)

    def forward(self, x, h):
        out, _ = self.rnn(self.emb(x), h)
        return self.fc(out)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, pad_idx):
        super().__init__()
        self.enc = Encoder(vocab_size, pad_idx)
        self.dec = Decoder(vocab_size, pad_idx)

    def forward(self, enc_x, dec_x):
        h = self.enc(enc_x)
        return self.dec(dec_x, h)
