import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from models.image.lenet_mnist.lenet import LeNet5

def export_lenet_mnist():
    model = LeNet5().cpu()
    model.load_state_dict(torch.load("models/image/lenet_mnist/lenet_best.pth", map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 1, 32, 32)

    scripted_model = torch.jit.trace(model, dummy_input)
    scripted_model.save("models/image/lenet_mnist/model.pt")

    print("✅ TorchScript model saved")

if __name__ == "__main__":
    export_lenet_mnist()
