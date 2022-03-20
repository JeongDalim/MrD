import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
import cv2 as cv2
import random

from model.SoftmaxClassifierModel import SoftmaxClassifierModel


# GPU를 사용가능 하면 True, 아니 라면 False를 리턴
USE_CUDA = torch.cuda.is_available()

# GPU 사용 가능 하면 사용 하고 아니면 CPU 사용
device = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습 합니다:", device)
print("Torch version:{}".format(torch.__version__))

# parameters
training_epochs = 100
batch_size = 100
learning_rate = 0.001
target_accuracy = 0.98

# minist 데이터 셋을 불러 올거야
# root: 데이터셋 위치
# torch의 경우 이미지를 사용할 때 0 ~ 1 숫자의, (Chanel x Hight x Width)를 사용하지만 일반적의 이미지의 경우 0 ~ 255 숫자와,
# (Hight x Hight x Width )를 이용해 이미지를 표현한다. 이를 torch에서 사용 하기 위해서 변환이 필요
mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

model = SoftmaxClassifierModel().to(device)

# define cost/Loss & optimizer
cost_fn = nn.CrossEntropyLoss().to(device)
# Adam = SGD에 비해 관성이 있다.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(training_epochs):
    avg_cost = 0
    avg_acc = 0
    total_batch = len(data_loader)
    model.train()

    for X, Y in data_loader:
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        # cost 계산
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = cost_fn(hypothesis, Y)
        cost.backward()
        optimizer.step()

        # accuracy 계산
        model.eval()
        prediction = model(X)
        correct_prediction = torch.argmax(prediction, 1) == Y
        accuracy = correct_prediction.float().mean()  # Batch 사이즈 정확도

        avg_cost += cost / total_batch
        avg_acc += (accuracy / total_batch)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'acc =', '{:.9f}'.format(avg_acc))
    if avg_acc > target_accuracy:
        break;

print('Learning finished')

# 테스트 데이터를 사용 하여 모델을 테스트 한다.
with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행 하지 않는다.
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    for i in range(500):
        r = random.randint(0, len(mnist_test) - 1)
        X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
        Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

        single_prediction = model(X_single_data)
        print('Label: ', Y_single_data.item(), '    Prediction: ', torch.argmax(single_prediction, 1).item())

        test_image = mnist_test.test_data[r:r + 1].permute(1, 2, 0).numpy()
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result', 512, 512)
        cv2.imshow('result', test_image)
        cv2.waitKey(33)