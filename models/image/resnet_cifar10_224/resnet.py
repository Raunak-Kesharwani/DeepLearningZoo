import torch
from torch import nn
from torchvision.models import resnet18


class ResNet18FineTune(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # IMPORTANT:
        # self *is* the resnet, not a wrapper around it
        base = resnet18(weights=None)

        # Copy all layers to self (this preserves key names)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.avgpool = base.avgpool

        # Classifier head (must match fc.1.*)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(base.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
