import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from backend.loader import load_torchscript_model
from backend.inference import infer_lstm_timeseries
from backend.preprocessing.timeseries import preprocess_sine_window
import numpy as np

# Load model
model = load_torchscript_model(
    "models/timeseries/model.pt"
)

# Generate sample sine window
x = np.sin(0.02 * np.arange(100, 120))
x = preprocess_sine_window(x.tolist())

# Inference
pred = infer_lstm_timeseries(model, x)
print("Predicted next value:", pred)
