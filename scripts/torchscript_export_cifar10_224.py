import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from models.image.resnet_cifar10_224.resnet import ResNet18FineTune


def export_cifar10_224():
    model = ResNet18FineTune().cpu()
    model.load_state_dict(
    torch.load(
        "models/image/resnet_cifar10_224/tune_Resnet.pth",
        map_location="cpu"
    )
    )


    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    scripted = torch.jit.trace(model, dummy)

    scripted.save("models/image/resnet_cifar10_224/model.pt")
    print("✅ CIFAR-10 224 TorchScript saved")


if __name__ == "__main__":
    export_cifar10_224()
