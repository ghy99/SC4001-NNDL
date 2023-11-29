from typing import Any
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pylab as plt

from sklearn import datasets, model_selection as ms


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class Softmax(nn.Module):
    def __init__(self, no_features, no_labels):
        super().__init__()
        self.softmax_layer = nn.Sequential(
            nn.Linear(no_features, no_labels), nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.softmax_layer(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    train_loss /= size
    correct /= size
    return train_loss, correct


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    no_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    return test_loss, correct


def plot_err_entropy(train_acc, train_loss, test_acc, test_loss, no_epochs):
    plt.figure(2)
    plt.plot(range(no_epochs), train_loss, label='train', color = 'blue')
    plt.plot(range(no_epochs), test_loss, label='test', color = 'red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.figure(3)
    plt.plot(range(no_epochs), train_acc, label='train', color = 'blue')
    plt.plot(range(no_epochs), test_acc, label='test', color = 'red')
    plt.xlabel("Epochs")
    plt.ylabel("Classification Accuracy")
    plt.show()


def plot_data(X, Y):
    plt.figure(1)
    plt.plot(X[Y == 0, 0], X[Y == 0, 1], "b^", label="Iris-Setosa")
    plt.plot(X[Y == 1, 0], X[Y == 1, 1], "ro", label="Iris-Versicolour")
    plt.plot(X[Y == 2, 0], X[Y == 2, 1], "gx", label="Iris-Virginica")

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": "{:12.8f}".format})

    no_epochs = 1000
    batch_size = 16
    no_inputs = 4
    no_classes = 3
    learning_rate = 0.1

    SEED = 10
    np.random.seed(SEED)

    iris = datasets.load_iris()
    X = iris["data"]
    Y = iris["target"]

    # for key, val in iris.items():
    #     print(f"{key}\n{val}\n")

    # plot_data(X, Y)

    x_train, x_test, y_train, y_test = ms.train_test_split(
        X, Y, test_size=0.4, random_state=2
    )

    print(f"x_train shape: {np.shape(x_train)}")
    print(f"y_test shape: {np.shape(y_test)}")

    train_data = MyDataset(x_train, y_train)
    test_data = MyDataset(x_test, y_test)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = Softmax(no_inputs, no_classes)
    print(f"Model Structure: {model}\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name}\tSize: {param.size()}\tValues: {param}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loss_, train_acc_, test_loss_, test_acc_ = [], [], [], []

    for epoch in range(no_epochs):
        train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)
        
        train_loss_.append(train_loss), train_acc_.append(train_acc)
        test_loss_.append(test_loss), test_acc_.append(test_acc)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch + 1}")
            print(f"train loss: {train_loss:.4f}\ttrain accuracy: {train_acc:.4f}")
            print(f"test loss: {test_loss:.4f}\ttest accuracy: {test_acc:.4f}")

    plot_err_entropy(train_acc_, train_loss_, test_acc_, test_loss_, no_epochs)
    w = model.state_dict()['softmax_layer.0.weight'].numpy()
    b = model.state_dict()['softmax_layer.0.bias'].numpy()

    print(f"\n\nConverged weight:\n{w}\nConverged bias:\n{b}")