import torch
from torch import nn


# ---------------------------
# Encoder
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.rnn = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        x = self.embedding(x)
        outputs, hidden = self.rnn(x)
        return outputs, hidden


# ---------------------------
# Bahdanau Attention
# ---------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_enc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_dec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, enc_mask):
        # dec_hidden: (B, H)
        # enc_outputs: (B, T, H)
        dec_hidden = dec_hidden.unsqueeze(1)

        energy = torch.tanh(
            self.W_enc(enc_outputs) + self.W_dec(dec_hidden)
        )

        scores = self.v(energy).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~enc_mask, -1e9)

        attn_weights = torch.softmax(scores, dim=1)

        context = torch.sum(
            attn_weights.unsqueeze(-1) * enc_outputs,
            dim=1
        )

        return context, attn_weights


# ---------------------------
# Decoder (FIXED)
# ---------------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )

        self.attn = BahdanauAttention(hidden_size)

        self.rnn = nn.LSTM(
            embed_dim + hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden, enc_outputs, enc_mask):
        """
        x: (B, 1)
        hidden: (h, c)
        enc_outputs: (B, T, H)
        enc_mask: (B, T)
        """

        x = self.embedding(x)              # (B, 1, E)

        # last-layer hidden state
        dec_hidden = hidden[0][-1]         # (B, H)

        context, attn = self.attn(
            dec_hidden, enc_outputs, enc_mask
        )                                  # (B, H)

        # ❗ NO squeeze() — explicit indexing
        x_t = x[:, 0, :]                   # (B, E)

        rnn_input = torch.cat(
            [x_t, context], dim=1
        ).unsqueeze(1)                     # (B, 1, E+H)

        out, hidden = self.rnn(
            rnn_input, hidden
        )

        out_t = out[:, 0, :]               # (B, H)

        logits = self.fc(
            torch.cat([out_t, context], dim=1)
        )                                  # (B, vocab)

        return logits, hidden, attn


# ---------------------------
# Seq2Seq
# ---------------------------
class Seq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_size,
        num_layers,
        pad_idx,
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size, embed_dim, hidden_size, num_layers, pad_idx
        )

        self.decoder = Decoder(
            vocab_size, embed_dim, hidden_size, num_layers, pad_idx
        )

        self.pad_idx = pad_idx
        self.vocab_size = vocab_size

    def forward(self, enc_x, dec_x):
        B, T = dec_x.size()

        enc_outputs, hidden = self.encoder(enc_x)
        enc_mask = (enc_x != self.pad_idx)

        outputs = torch.zeros(
            B, T, self.vocab_size, device=enc_x.device
        )

        for t in range(T):
            logits, hidden, _ = self.decoder(
                dec_x[:, t:t+1],
                hidden,
                enc_outputs,
                enc_mask
            )
            outputs[:, t] = logits

        return outputs
