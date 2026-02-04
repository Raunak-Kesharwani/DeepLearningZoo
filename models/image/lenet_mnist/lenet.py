import torch
import torch
from torch import nn

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1= nn.Sequential(
            nn.Conv2d(1 , 6 ,kernel_size=5 , stride=1 , padding = 0 ),
            nn.ReLU(),
            nn.AvgPool2d(2 ,2),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(16 , 120 , kernel_size=5 , stride=1 , padding=0)
        )
        self.fc_1 = nn.Linear(120, 84)
        self.fc_2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc_1(x))
        x = self.fc_2(x)            

        return x