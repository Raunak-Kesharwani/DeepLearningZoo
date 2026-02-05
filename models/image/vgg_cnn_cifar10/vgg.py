import torch 
from torch import nn


class VGG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1= nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size=3 , stride=1 , padding = 1 ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64 , 64 , kernel_size=3 , stride=1 , padding = 1 ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2 ,2),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2 ,2),
        )
        self.layer_3 = nn.Sequential(
             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2 ,2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.fc_1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.35)
        )
                                  
        self.fc_2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.gap(x)
        
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = self.fc_2(x)            

        return x