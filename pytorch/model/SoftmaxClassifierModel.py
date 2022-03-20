import torch.nn as nn
import torch.nn.functional as F


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hiddenLayer1 = nn.Linear(784, 128)
        self.drop_out1 = nn.Dropout(p=0.3)
        self.hiddenLayer2 = nn.Linear(128, 64)
        self.drop_out2 = nn.Dropout(p=0.3)
        self.hiddenLayer3 = nn.Linear(64, 10)
        # dim = 1 : 첫번 째 배치에(10개) softmax를 하는 것 dim = 0 : 배치 전체를 softmax 하는 것
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hiddenLayer1(x)
        x = self.drop_out1(x)
        x = F.relu(x)
        x = self.hiddenLayer2(x)
        x = self.drop_out2(x)
        x = F.relu(x)
        x = self.hiddenLayer3(x)
        ## 데이터셋의 이미지상에 객체가 1개가 나오는걸 보장
        x = self.softmax(x)

        return x
