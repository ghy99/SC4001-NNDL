import torch
from torch import nn

import numpy as np
import matplotlib.pylab as plt

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )

    def forward(self, x):
        logits = self.relu_stack(x)
        return logits


def train_loop(X, Y, model, loss_fn, optimizer):
    pred = model(X)
    loss = loss_fn(pred, Y)

    # back-propagation, getting the gradients.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def plotpoints(X):
    plt.figure()
    plt.plot(X[:, 0], X[:, 1], "rx")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("training inputs for z = 0.8$x^2$ - $y^3$ + 2.5xy")
    plt.show()


def plot3D(X, Y):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(X[:, 0], X[:, 1], Y[:, 0], color="blue", marker=".")
    ax.set_title("Target Output for z = 0.8$x^2$ - $y^3$ + 2.5xy")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_xticks(np.arange(-1, 1.001, 0.5))
    ax.set_yticks(np.arange(-1, 1.001, 0.5))
    plt.show()

def plot_loss(no_epochs, train_loss):
    plt.figure()
    plt.plot(range(no_epochs), train_loss)
    plt.xlabel("No. of epochs")
    plt.ylabel("Mean Square Error")
    plt.title("GD Training Loss")
    plt.show()

def plot_multiple_losses(no_epochs, train_losses, lr):
    plt.figure()
    i = 0
    for train_loss in train_losses:
        plt.plot(range(no_epochs), train_loss, label=lr[i])
        i += 1
    plt.xlabel("No. of epochs")
    plt.ylabel("Mean Square Error")
    plt.title("GD Training Loss")
    plt.legend()
    plt.show()

def plot_pred(X, Y, pred):
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.scatter(X[:,0], X[:,1], Y[:,0], color='blue', marker='.', label='targets')
    ax.scatter(X[:,0], X[:,1], pred[:,0].detach().numpy(), color='red', marker='x', label='predictions')
    ax.set_title('Targets and Predictions')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$\phi$')
    ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.legend()
    plt.show()

if __name__ == "__main__":
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    no_epochs = 5000
    lr = 0.01
    learning_rates = [0.005, 0.001, 0.01, 0.05]
    learning_rates.sort()

    X = np.zeros((9 * 9, 2)).astype(np.float32)
    p = 0
    for i in np.arange(-1, 1.001, 0.25):
        for j in np.arange(-1, 1.001, 0.25):
            X[p] = [i, j]
            p += 1

    # print(f"X:\n{X}")
    np.random.shuffle(X)
    Y = np.zeros((9 * 9, 1)).astype(np.float32)
    Y[:, 0] = 0.8 * X[:, 0] ** 2 - X[:, 1] ** 3 + 2.5 * X[:, 0] * X[:, 1]
    # plotpoints(X)
    # plot3D(X, Y)

    train_loss_per_lr = []

    for learning_rate in learning_rates:
        model = FFN()
        # print(f"Model Structure: {model}\n\n")
        # for name, param in model.named_parameters():
        #     print(f"Layer: {name}\tSize: {param.size()}\tValues: {param}")

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        train_loss = []

        for epoch in range(no_epochs):
            train_loss_ = train_loop(torch.tensor(X), torch.tensor(Y), model, loss_fn, optimizer)
            train_loss.append(train_loss_)

            if epoch > no_epochs - 2:
                print(f"Epoch: {epoch}\tLearning Rate: {learning_rate}\tTrain loss: {train_loss_:.4f}")
        
        # plot_loss(no_epochs, train_loss)
        train_loss_per_lr.append(train_loss)
        print(f"Training loss: {train_loss[-1]:.4f}")

        pred = model(torch.tensor(X))

        # plot_pred(X, Y, pred)
    plot_multiple_losses(no_epochs, train_loss_per_lr, learning_rates)