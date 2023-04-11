import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

hidenSize = 500
epochs = 50
batchSize = 200
learningRate = 0.001


class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.l1 = nn.Linear(28 * 28, hidenSize)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidenSize, 10)

    def forward(self, y):
        outp = self.l1(y.reshape(-1, 28 * 28))
        outp = self.relu(outp)
        outp = self.l2(outp)
        return outp


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.l1 = nn.Linear(28 * 28, 1000)
        self.l2 = nn.Linear(1000, 1500)
        self.l3 = nn.Linear(1500, 1000)
        self.l4 = nn.Linear(1000, 500)
        self.l5 = nn.Linear(500, 10)

    def forward(self, y):
        outp = F.relu(self.l1(y.reshape(-1, 28 * 28)))
        outp = F.relu(self.l2(outp))
        outp = F.relu(self.l3(outp))
        outp = F.relu(self.l4(outp))
        outp = (self.l5(outp))
        return outp

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 40, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(40, 200, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(200, 800, kernel_size=3)
        self.fc1 = nn.Linear(800*4, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(batchSize, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = (self.fc4(x))
        return x

def train():
    for epoch in range(epochs):
        totalLoss = 0
        for x, (images, labels) in enumerate(trainLoader):
            image = images.to(device)
            label = labels.to(device)
            outp = model(image)
            loss = criter(outp, label)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            totalLoss += loss.item()

        print(f'Epochs [{epoch + 1}/{epochs}], Step[{x + 1}/{trainNum}], Losses:{totalLoss / (trainNum * batchSize)}')
        test(epoch)


def test(epoch):
    correct = 0
    for image, label in testLoader:
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        _, predict = torch.max(output.data, 1)
        correct += (predict == label).sum()
        accuracy = correct / (testNum * batchSize)
    print(f"accuracy:{accuracy}")
    accuracyArray[epoch] = accuracy


def plotAccuracy():
    x = np.arange(1, epochs + 1, 1, dtype=int)
    yTicks = np.arange(0.800, 0.950, 0.005, dtype=float)
    figure = plt.plot(x, accuracyArray)
    ax = plt.gca()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_yticks(yTicks)
    plt.savefig("./accu2_2.png", dpi=400)
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 判断能否调用GPU
    print(device)

    trainDataset = torchvision.datasets.FashionMNIST(root="./dataset/train", train=True, transform=trans.ToTensor(),
                                                     download=True)
    testDataset = torchvision.datasets.FashionMNIST(root="./dataset/test", train=False, transform=trans.ToTensor(),
                                                    download=True)
    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False)
    trainNum = len(trainLoader)
    testNum = len(testLoader)

    # model = Net0(inputSize, hidenSize, classNum)
    model = Net2()
    criter = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learningRate)
    model = model.to(device)
    accuracyArray = np.zeros((epochs,), dtype=float)

    train()
    plotAccuracy()
