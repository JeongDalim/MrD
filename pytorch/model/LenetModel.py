import torch
import torch.nn as nn
import torch.nn.functional as F


class LenetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # 28 x 28
        # in_channels = 1 : 흑백이 니까 1로 시작 합세
        # out_channels = filter 6개로 할 것
        # kernel_size 5*5
        # stride =1  1씩 이동
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=5,
                               stride=1)

        self.avg_pool1 = nn.AvgPool2d(kernel_size=2,
                                      stride=2)

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5,
                               stride=1)

        self.avg_pool2 = nn.AvgPool2d(kernel_size=2,
                                      stride=2)

        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=120,
                               kernel_size=4,
                               stride=1)

        #1열로 만들어 준 다음 fully connect
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=120,
                             out_features=84)

        self.fc2 = nn.Linear(in_features=84,
                             out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.avg_pool1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.avg_pool2(x)
        x = torch.tanh(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x