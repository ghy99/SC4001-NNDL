from typing import Any
import torch
import numpy as np
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import copy
import math

np.set_printoptions(formatter={"float": "{:6.2f}".format})


def generate_datapoints():
    """
    Generate my X matrix from range 0-1 for both x and y as X>=0, y<=1.
    Calculates targets based on given formula.

    Returns:
        X: Matrix of input feature
        d: Output label
    """
    x = np.random.uniform(0, 1, 100).reshape(100, 1)
    y = np.random.uniform(0, 1, 100).reshape(100, 1)
    d = 1.5 + (3.3 * x) - (2.5 * y) + (1.2 * x * y)
    # print(f"x: {x.shape}, y: {y.shape}")
    X = np.column_stack((x, y))
    # print(f"X: {X.shape}, {type(X)}")
    return X, d


class Linear:
    def __init__(self) -> None:
        self.w = torch.tensor(
            np.random.rand(2, 1), dtype=torch.double, requires_grad=True
        )
        self.b = torch.tensor(0.0, dtype=torch.double, requires_grad=True)

    def __call__(self, x):
        pass


class Perceptron:
    def __init__(self) -> None:
        self.w = torch.tensor(
            np.random.rand(2, 1), dtype=torch.double, requires_grad=True
        )
        self.b = torch.tensor(0.0, dtype=torch.double, requires_grad=True)

    def __call__(self, x, d):
        """
        u = Xw + b1p
        f(u) = 1 / (1 + e^-u), with adjustments to amplitude and graph shifting.
        I took the ceiling of the max value for the amplitude,
        and the floor of the min value for how much the graph shifts downwards.

        Args:
            x (matrix): input features
            d (vector): target labels
        """
        u = torch.matmul(torch.tensor(x), self.w) + self.b
        # print(f"d max: {d.max(), math.ceil(d.max())}, \tmin: {d.min(), math.floor(d.min())}")
        y = math.ceil(d.max()) * torch.sigmoid(u) - math.floor(d.min())
        return u, y


def loss(predicted, target):
    """
    J = (1 / P) * Summation from 1 to P((dp - yp)^2)
    """
    return torch.mean(torch.square(predicted - torch.tensor(target)))


def train(model, x, d, learning_rate):
    """
    Training goes through here.
    Calculate loss function J in this training function.
    loss_.backward() gives the gradient of J.
    model.w and model.b are updated with the formula of
        w = w - a*(gradient of J w.r.t. w),
        b = b - a*(gradient of J w.r.t. b)

    Returns:
        loss_ (J): The value of the cost
    """
    u, y = model(x, d)
    loss_ = loss(y_, d)
    loss_.backward()
    # grad_w = -torch.matmul(
    #     torch.transpose(torch.tensor(x), 0, 1), (torch.tensor(d) - y) * model.w.grad
    # )
    # grad_b = -torch.sum((torch.tensor(d) - y) * model.b.grad)
    with torch.no_grad():
        model.w -= learning_rate * model.w.grad
        model.b -= learning_rate * model.b.grad

    model.w.grad = None
    model.b.grad = None

    return loss_


def mean_square_error(mse):
    """
    Calculate final mean square error
    """
    return torch.mean(torch.tensor(mse))


def plot_learning_curve(no_epochs, mse):
    """
    Plot learning curve of epoch vs mean square error.
    """
    plt.figure(1)
    plt.plot(range(no_epochs), torch.tensor(mse).detach().numpy())
    plt.xlabel("epochs")
    plt.ylabel("mse")
    plt.savefig("./graphs/Tutorial 2 - Question 4 (GD Perceptron) (epoch vs mse).png")
    # plt.show()


def plot_target_vs_prediction(X, Y):
    """
    Plot target vs prediction in a 3D graph.
    """
    u, y = model(X, Y)
    y = y.detach().numpy()[:, 0]

    fig = plt.figure(2)
    ax = plt.axes(projection="3d")
    ax.scatter(X[:, 0], X[:, 1], Y[:, 0], color="blue", marker="x", label="targets")
    ax.scatter(X[:, 0], X[:, 1], y, color="red", marker=".", label="predicted")
    ax.set_title("Targets and Predictions")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.legend()
    plt.savefig(
        "./graphs/Tutorial 2 - Question 4 (GD Perceptron)) (targets vs prediction).png"
    )
    # plt.show()


def plot_hyperplane(w, b, targets):
    """
    Plot the hyperplane.
    first Z is actually u = w^Tx + b
    second Z calculates the sigmoid function with adjustments to the amplitude and graph shift. 
    """
    fig = plt.figure(3)
    ax = plt.axes(projection="3d")
    X1 = np.arange(0, 1, 0.05)
    X2 = np.arange(0, 1, 0.05)
    X1, X2 = np.meshgrid(X1, X2)
    Z = w.detach().numpy()[0] * X1 + w.detach().numpy()[1] * X2 + b.detach().numpy()
    # Z = 6/(1+np.exp(-Z))-1.0
    Z = math.ceil(targets.max()) * torch.sigmoid(torch.tensor(Z)) - math.floor(
        targets.min()
    )
    regression_plane = ax.plot_surface(X1, X2, Z)
    ax.xaxis.set_major_locator(ticker.IndexLocator(base=0.2, offset=0.0))
    ax.yaxis.set_major_locator(ticker.IndexLocator(base=0.2, offset=0.0))
    ax.set_title("Function Learned by Perceptron")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    plt.savefig(
        "./graphs/Tutorial 2 - Question 4 (GD Perceptron)) (hyperplane).png"
    )
    # plt.show()


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": "{:6.2f}".format})
    # generate training data
    SEED = 10
    np.random.seed(SEED)  # for reproducibility

    no_epoch = 2000
    lr = 0.01

    X, d = generate_datapoints()
    # print(f"X: {torch.transpose(torch.tensor(X), 0, 1)}")
    # print(f"d: {torch.transpose(torch.tensor(d), 0, 1)}")
    model = Perceptron()

    og_w = copy.deepcopy(model.w)
    og_b = copy.deepcopy(model.b)

    mse = []

    for epoch in range(no_epoch):
        u, y_ = model(X, d)

        loss_ = train(model, X, d, lr)
        mse.append(loss_)

        if epoch == 0:
            print(f"Epoch: {epoch}")
            print(f"y: {y_.detach().numpy()}")
            print(f"mse: {loss_:5.2f}")
            print(f"w: \n{model.w.detach().numpy()}")
            print(f"b: {model.b.detach().numpy()}")
            print()

        if epoch > no_epoch - 10:
            print(f"Epoch: {epoch}\loss: {loss_:5.2f}")
            print(f"w: \n{model.w.detach().numpy()}")
            print(f"b: {model.b.detach().numpy()}")

    # print(f"MSE: {mse}")

    print(f"Mean Square Error: {mean_square_error(mse)}")
    print(f"Final w:\t\tOriginal w:")
    for i in range(len(model.w.detach().numpy())):
        print(f"{model.w.detach().numpy()[i]}\t\t{og_w.detach().numpy()[i]}")

    print(f"Final b:\tOriginal b:")
    print(f"{model.b.detach().numpy():5.2f}\t\t{og_b.detach().numpy():5.2f}")

    plot_learning_curve(no_epoch, mse)
    plot_target_vs_prediction(X, d)
    plot_hyperplane(model.w, model.b, d)

from typing import Any
import torch
import numpy as np
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import copy
import math

np.set_printoptions(formatter={"float": "{:6.2f}".format})


def generate_datapoints():
    """
    Generate my X matrix from range 0-1 for both x and y as X>=0, y<=1.
    Calculates targets based on given formula.

    Returns:
        X: Matrix of input feature
        d: Output label
    """
    x = np.random.uniform(0, 1, 100).reshape(100, 1)
    y = np.random.uniform(0, 1, 100).reshape(100, 1)
    d = 1.5 + (3.3 * x) - (2.5 * y) + (1.2 * x * y)
    # print(f"x: {x.shape}, y: {y.shape}")
    X = np.column_stack((x, y))
    # print(f"X: {X.shape}, {type(X)}")
    return X, d


class Linear:
    def __init__(self) -> None:
        self.w = torch.tensor(
            np.random.rand(2, 1), dtype=torch.double, requires_grad=True
        )
        self.b = torch.tensor(0.0, dtype=torch.double, requires_grad=True)

    def __call__(self, x):
        pass


class Perceptron:
    def __init__(self) -> None:
        self.w = torch.tensor(
            np.random.rand(2, 1), dtype=torch.double, requires_grad=True
        )
        self.b = torch.tensor(0.0, dtype=torch.double, requires_grad=True)

    def __call__(self, x, d):
        """
        u = Xw + b1p
        f(u) = 1 / (1 + e^-u), with adjustments to amplitude and graph shifting.
        I took the ceiling of the max value for the amplitude,
        and the floor of the min value for how much the graph shifts downwards.

        Args:
            x (matrix): input features
            d (vector): target labels
        """
        u = torch.matmul(torch.tensor(x), self.w) + self.b
        # print(f"d max: {d.max(), math.ceil(d.max())}, \tmin: {d.min(), math.floor(d.min())}")
        y = math.ceil(d.max()) * torch.sigmoid(u) - math.floor(d.min())
        return u, y


def loss(predicted, target):
    """
    J = (1 / P) * Summation from 1 to P((dp - yp)^2)
    """
    return torch.mean(torch.square(predicted - torch.tensor(target)))


def train(model, x, d, learning_rate):
    """
    Training goes through here.
    Calculate loss function J in this training function.
    loss_.backward() gives the gradient of J.
    model.w and model.b are updated with the formula of
        w = w - a*(gradient of J w.r.t. w),
        b = b - a*(gradient of J w.r.t. b)

    Returns:
        loss_ (J): The value of the cost
    """
    u, y = model(x, d)
    loss_ = loss(y_, d)
    loss_.backward()
    # grad_w = -torch.matmul(
    #     torch.transpose(torch.tensor(x), 0, 1), (torch.tensor(d) - y) * model.w.grad
    # )
    # grad_b = -torch.sum((torch.tensor(d) - y) * model.b.grad)
    with torch.no_grad():
        model.w -= learning_rate * model.w.grad
        model.b -= learning_rate * model.b.grad

    model.w.grad = None
    model.b.grad = None

    return loss_


def mean_square_error(mse):
    """
    Calculate final mean square error
    """
    return torch.mean(torch.tensor(mse))


def plot_learning_curve(no_epochs, mse):
    """
    Plot learning curve of epoch vs mean square error.
    """
    plt.figure(1)
    plt.plot(range(no_epochs), torch.tensor(mse).detach().numpy())
    plt.xlabel("epochs")
    plt.ylabel("mse")
    plt.savefig("./graphs/Tutorial 2 - Question 4 (GD Perceptron) (epoch vs mse).png")
    # plt.show()


def plot_target_vs_prediction(X, Y):
    """
    Plot target vs prediction in a 3D graph.
    """
    u, y = model(X, Y)
    y = y.detach().numpy()[:, 0]

    fig = plt.figure(2)
    ax = plt.axes(projection="3d")
    ax.scatter(X[:, 0], X[:, 1], Y[:, 0], color="blue", marker="x", label="targets")
    ax.scatter(X[:, 0], X[:, 1], y, color="red", marker=".", label="predicted")
    ax.set_title("Targets and Predictions")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.legend()
    plt.savefig(
        "./graphs/Tutorial 2 - Question 4 (GD Perceptron)) (targets vs prediction).png"
    )
    # plt.show()


def plot_hyperplane(w, b, targets):
    """
    Plot the hyperplane.
    first Z is actually u = w^Tx + b
    second Z calculates the sigmoid function with adjustments to the amplitude and graph shift. 
    """
    fig = plt.figure(3)
    ax = plt.axes(projection="3d")
    X1 = np.arange(0, 1, 0.05)
    X2 = np.arange(0, 1, 0.05)
    X1, X2 = np.meshgrid(X1, X2)
    Z = w.detach().numpy()[0] * X1 + w.detach().numpy()[1] * X2 + b.detach().numpy()
    # Z = 6/(1+np.exp(-Z))-1.0
    Z = math.ceil(targets.max()) * torch.sigmoid(torch.tensor(Z)) - math.floor(
        targets.min()
    )
    regression_plane = ax.plot_surface(X1, X2, Z)
    ax.xaxis.set_major_locator(ticker.IndexLocator(base=0.2, offset=0.0))
    ax.yaxis.set_major_locator(ticker.IndexLocator(base=0.2, offset=0.0))
    ax.set_title("Function Learned by Perceptron")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    plt.savefig(
        "./graphs/Tutorial 2 - Question 4 (GD Perceptron)) (hyperplane).png"
    )
    # plt.show()


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": "{:6.2f}".format})
    # generate training data
    SEED = 10
    np.random.seed(SEED)  # for reproducibility

    no_epoch = 2000
    lr = 0.01

    X, d = generate_datapoints()
    # print(f"X: {torch.transpose(torch.tensor(X), 0, 1)}")
    # print(f"d: {torch.transpose(torch.tensor(d), 0, 1)}")
    model = Perceptron()

    og_w = copy.deepcopy(model.w)
    og_b = copy.deepcopy(model.b)

    mse = []

    for epoch in range(no_epoch):
        u, y_ = model(X, d)

        loss_ = train(model, X, d, lr)
        mse.append(loss_)

        if epoch == 0:
            print(f"Epoch: {epoch}")
            print(f"y: {y_.detach().numpy()}")
            print(f"mse: {loss_:5.2f}")
            print(f"w: \n{model.w.detach().numpy()}")
            print(f"b: {model.b.detach().numpy()}")
            print()

        if epoch > no_epoch - 10:
            print(f"Epoch: {epoch}\loss: {loss_:5.2f}")
            print(f"w: \n{model.w.detach().numpy()}")
            print(f"b: {model.b.detach().numpy()}")

    # print(f"MSE: {mse}")

    print(f"Mean Square Error: {mean_square_error(mse)}")
    print(f"Final w:\t\tOriginal w:")
    for i in range(len(model.w.detach().numpy())):
        print(f"{model.w.detach().numpy()[i]}\t\t{og_w.detach().numpy()[i]}")

    print(f"Final b:\tOriginal b:")
    print(f"{model.b.detach().numpy():5.2f}\t\t{og_b.detach().numpy():5.2f}")

    plot_learning_curve(no_epoch, mse)
    plot_target_vs_prediction(X, d)
    plot_hyperplane(model.w, model.b, d)
