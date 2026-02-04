import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from backend.loader import load_torchscript_model
from backend.preprocessing.image import preprocess_lenet_mnist

# Load model
model = load_torchscript_model(
    "models/image/lenet_mnist/model.pt"
)

# Load sample image (MNIST-like)
image = Image.open("image.png")  # any digit image

# Preprocess
x = preprocess_lenet_mnist(image)  # (1, 32, 32)

# Inference
with torch.no_grad():
    logits = model(x.unsqueeze(0))
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(1).item()

print("Prediction:", pred)
print("Probabilities:", probs)
