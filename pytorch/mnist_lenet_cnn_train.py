import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
import cv2 as cv2
import random

from model.LenetModel import LenetModel

USE_CUDA = torch.cuda.is_available()

device = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습 합니다:", device)
print("Torch version:{}".format(torch.__version__))

# parameters
training_epochs = 30
batch_size = 100
learning_rate = 0.001
target_accuracy = 0.99

mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

model = LenetModel().to(device)

# define cost/Loss & optimizer
cost_fn = nn.CrossEntropyLoss().to(device)
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(training_epochs):
    avg_cost = 0
    avg_acc = 0
    total_batch = len(data_loader)
    model.train()

    for X, Y in data_loader:
        X = X.to(device)
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

# 테스트 데이터를 사용하여 모델을 테스트한다.
with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
    model.eval()

    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    for i in range(500):
        r = random.randint(0, len(mnist_test) - 1)
        X_single_data = mnist_test.test_data[r:r + 1].to(device)
        Y_single_data = mnist_test.test_labels[r:r + 1].to(device)
        X_single_data = X_single_data.unsqueeze(0).float()
        Y_single_data = Y_single_data.unsqueeze(0).float()

        single_prediction = model(X_single_data)
        print('Label: ', Y_single_data.item(), '    Prediction: ', torch.argmax(single_prediction, 1).item())

        test_image = mnist_test.test_data[r:r + 1].permute(1, 2, 0).numpy()
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result', 512, 512)
        cv2.imshow('result', test_image)
        cv2.waitKey(33)