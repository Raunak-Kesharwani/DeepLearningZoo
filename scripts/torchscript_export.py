import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from models.image.vgg_cnn_cifar10.vgg import VGG_CNN  

def export_vgg_cnn():
    model = VGG_CNN().cpu()
    model.load_state_dict(
        torch.load("models/image/vgg_cnn_cifar10/VGG_CNN.pth", map_location="cpu")
    )
    model.eval()

    dummy = torch.randn(1, 3, 32, 32)
    scripted = torch.jit.trace(model, dummy)
    scripted.save("models/image/vgg_cnn_cifar10/model.pt")

    print("✅ VGG_CNN TorchScript saved")

if __name__ == "__main__":
    export_vgg_cnn()

