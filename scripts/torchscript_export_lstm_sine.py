import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from models.timeseries.model import LSTM_Model


def export_lstm_sine():
    model = LSTM_Model(input_size=1, hidden_size=64, num_layers=1).cpu()

    model.load_state_dict(
        torch.load(
            "models/timeseries/LSTM_sequence.pth",
            map_location="cpu"
        )
    )

    model.eval()

    # Dummy input must EXACTLY match training shape
    dummy = torch.randn(1, 20, 1)

    scripted = torch.jit.trace(model, dummy)
    scripted.save("models/timeseries/model.pt")

    print("✅ LSTM sine TorchScript saved")


if __name__ == "__main__":
    export_lstm_sine()
